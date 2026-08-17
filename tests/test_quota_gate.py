"""Deterministic coverage for the Session A request-quota gate."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from async_batch_llm import TokenEstimate, TokenEstimateExceedsLimit
from async_batch_llm._internal.admission import AdmissionGateClosed, QuotaGate


@dataclass
class _Sleeper:
    deadline: float
    future: asyncio.Future[None]


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleepers: list[_Sleeper] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.sleepers.append(_Sleeper(self.now + delay, future))
        await future

    async def advance(self, seconds: float) -> None:
        self.now += seconds
        for sleeper in tuple(self.sleepers):
            if sleeper.deadline <= self.now and not sleeper.future.done():
                sleeper.future.set_result(None)
                self.sleepers.remove(sleeper)
        # Let the wake task refill the gate and resume granted waiters.
        await asyncio.sleep(0)
        await asyncio.sleep(0)


async def _consume(gate: QuotaGate) -> None:
    reservation = await gate.reserve()
    reservation.mark_provider_started()
    assert reservation.finalize()


@pytest.mark.asyncio
async def test_continuous_refill_uses_burst_capacity_and_one_wake_task() -> None:
    clock = _ManualClock()
    gate = QuotaGate(2.0, clock=clock, sleep=clock.sleep)

    await _consume(gate)
    await _consume(gate)
    assert gate.request_available == 0
    assert not gate.has_wake_task

    third = asyncio.create_task(gate.reserve())
    fourth = asyncio.create_task(gate.reserve())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert gate.waiter_count == 2
    assert gate.has_wake_task
    assert len(clock.sleepers) == 1

    await clock.advance(29.0)
    assert not third.done()
    await clock.advance(1.0)
    third_reservation = await third
    assert third_reservation.wait_seconds == 30.0
    third_reservation.mark_provider_started()
    third_reservation.finalize()
    assert not fourth.done()

    await clock.advance(30.0)
    fourth_reservation = await fourth
    fourth_reservation.mark_provider_started()
    fourth_reservation.finalize()
    await gate.shutdown()


@pytest.mark.asyncio
async def test_fractional_rpm_supports_sub_one_rate() -> None:
    clock = _ManualClock()
    gate = QuotaGate(0.5, clock=clock, sleep=clock.sleep)
    await _consume(gate)

    waiting = asyncio.create_task(gate.reserve())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await clock.advance(119.0)
    assert not waiting.done()
    await clock.advance(1.0)
    reservation = await waiting
    assert reservation.wait_seconds == 120.0
    reservation.mark_provider_started()
    reservation.finalize()
    await gate.shutdown()


@pytest.mark.asyncio
async def test_fifo_cancellation_and_pre_provider_refund() -> None:
    clock = _ManualClock()
    gate = QuotaGate(1.0, clock=clock, sleep=clock.sleep)
    held = await gate.reserve()

    cancelled = asyncio.create_task(gate.reserve())
    next_waiter = asyncio.create_task(gate.reserve())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert gate.waiter_count == 2

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert gate.waiter_count == 1

    # No provider work began, so finalization refunds immediately to the FIFO
    # head instead of waiting a minute for refill.
    assert held.finalize()
    assert not held.finalize()
    reservation = await next_waiter
    assert reservation.provider_started is False
    reservation.mark_provider_started()
    reservation.finalize()
    await gate.shutdown()


@pytest.mark.asyncio
async def test_post_provider_finalization_consumes_request() -> None:
    clock = _ManualClock()
    gate = QuotaGate(1.0, clock=clock, sleep=clock.sleep)
    held = await gate.reserve()
    held.mark_provider_started()
    held.finalize()

    waiting = asyncio.create_task(gate.reserve())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not waiting.done()
    await clock.advance(60.0)
    reservation = await waiting
    reservation.mark_provider_started()
    reservation.finalize()
    await gate.shutdown()


@pytest.mark.asyncio
async def test_fast_path_has_no_timer_and_shutdown_wakes_waiters() -> None:
    clock = _ManualClock()
    gate = QuotaGate(1.0, clock=clock, sleep=clock.sleep)
    await _consume(gate)
    assert not gate.has_wake_task

    waiting = asyncio.create_task(gate.reserve())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    wake_task = gate._wake_task
    assert wake_task is not None

    await gate.shutdown()
    with pytest.raises(AdmissionGateClosed):
        await waiting
    assert wake_task.done()
    assert gate.waiter_count == 0
    assert not gate.has_wake_task
    await gate.shutdown()


@pytest.mark.asyncio
async def test_disabled_gate_is_immediate_and_reservation_is_noop() -> None:
    gate = QuotaGate(None)
    reservation = await gate.reserve()
    assert reservation.request_reserved is False
    reservation.mark_provider_started()
    assert reservation.finalize()
    assert not gate.has_wake_task
    await gate.shutdown()


@pytest.mark.asyncio
async def test_tpm_only_refills_and_reserves_estimate_atomically() -> None:
    clock = _ManualClock()
    gate = QuotaGate(None, 120, clock=clock, sleep=clock.sleep)

    first = await gate.reserve(TokenEstimate(80, 40))
    assert first.request_reserved is False
    assert first.reserved_tokens == 120
    assert gate.token_available == 0
    first.mark_provider_started()
    assert first.reconcile(120) is not None

    waiting = asyncio.create_task(gate.reserve(TokenEstimate(30)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not waiting.done()
    await clock.advance(14.9)
    assert not waiting.done()
    await clock.advance(0.1)
    second = await waiting
    assert second.wait_seconds == pytest.approx(15.0)
    assert second.limited_by == "tpm"
    second.mark_provider_started()
    second.reconcile(30)
    await gate.shutdown()


@pytest.mark.asyncio
async def test_combined_gate_is_fifo_and_cancellation_wakes_smaller_next_waiter() -> None:
    clock = _ManualClock()
    gate = QuotaGate(1.0, 60, clock=clock, sleep=clock.sleep)
    held = await gate.reserve(TokenEstimate(60))
    held.mark_provider_started()
    held.reconcile(60)

    large = asyncio.create_task(gate.reserve(TokenEstimate(60)))
    small = asyncio.create_task(gate.reserve(TokenEstimate(1)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await clock.advance(1.0)
    assert not large.done()
    assert not small.done(), "a smaller request must not bypass the FIFO head"

    large.cancel()
    with pytest.raises(asyncio.CancelledError):
        await large
    await clock.advance(59.0)
    reservation = await small
    assert reservation.limited_by == "rpm"
    reservation.mark_provider_started()
    reservation.reconcile(1)
    await gate.shutdown()


@pytest.mark.asyncio
async def test_reconciliation_refund_debt_known_zero_unknown_and_exactly_once() -> None:
    clock = _ManualClock()
    gate = QuotaGate(None, 100, clock=clock, sleep=clock.sleep)

    over = await gate.reserve(TokenEstimate(80))
    over.mark_provider_started()
    final = over.reconcile(20)
    assert final is not None and final.delta_tokens == -60
    assert gate.token_available == pytest.approx(80)
    assert over.reconcile(0) is None

    under = await gate.reserve(TokenEstimate(80))
    under.mark_provider_started()
    final = under.reconcile(250)
    assert final is not None and final.delta_tokens == 170
    assert gate.token_available == pytest.approx(-170)

    waiting = asyncio.create_task(gate.reserve(TokenEstimate(1)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await clock.advance(102.6)
    reservation = await waiting
    reservation.mark_provider_started()
    reservation.reconcile(0)
    assert gate.token_available == pytest.approx(1)
    await clock.advance(59.4)
    assert gate.token_available == pytest.approx(100)

    unknown = await gate.reserve(TokenEstimate(40))
    unknown.mark_provider_started()
    final = unknown.finalize_unknown()
    assert final is not None and final.reported_tokens is None
    assert gate.token_available == pytest.approx(60)
    await gate.shutdown()


@pytest.mark.asyncio
async def test_tpm_rejects_zero_missing_and_impossible_estimates_immediately() -> None:
    gate = QuotaGate(None, 10)
    with pytest.raises(ValueError, match="positive TokenEstimate"):
        await gate.reserve()
    with pytest.raises(ValueError, match="positive TokenEstimate"):
        await gate.reserve(TokenEstimate(0))
    with pytest.raises(TokenEstimateExceedsLimit, match="11 > 10"):
        await gate.reserve(TokenEstimate(11))
    assert gate.waiter_count == 0
    assert not gate.has_wake_task
    await gate.shutdown()


@pytest.mark.asyncio
async def test_prestart_finalization_refunds_request_and_tokens_once() -> None:
    gate = QuotaGate(1.0, 100)
    reservation = await gate.reserve(TokenEstimate(75))
    assert gate.request_available == pytest.approx(0, abs=1e-5)
    assert gate.token_available == pytest.approx(25, abs=1e-3)
    final = reservation.finalize_before_start()
    assert final is not None and final.disposition == "refunded_before_start"
    assert gate.request_available == pytest.approx(1)
    assert gate.token_available == pytest.approx(100)
    assert reservation.finalize_before_start() is None
    await gate.shutdown()


@pytest.mark.asyncio
async def test_combined_gate_reports_each_limiting_dimension() -> None:
    clock = _ManualClock()

    rpm_gate = QuotaGate(1, 120, clock=clock, sleep=clock.sleep)
    first = await rpm_gate.reserve(TokenEstimate(1))
    first.mark_provider_started()
    first.reconcile(0)
    rpm_waiter = asyncio.create_task(rpm_gate.reserve(TokenEstimate(1)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await clock.advance(60)
    rpm = await rpm_waiter
    assert rpm.limited_by == "rpm"
    rpm.finalize_before_start()
    await rpm_gate.shutdown()

    tpm_gate = QuotaGate(120, 60, clock=clock, sleep=clock.sleep)
    first = await tpm_gate.reserve(TokenEstimate(60))
    first.mark_provider_started()
    first.reconcile(60)
    tpm_waiter = asyncio.create_task(tpm_gate.reserve(TokenEstimate(60)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await clock.advance(60)
    tpm = await tpm_waiter
    assert tpm.limited_by == "tpm"
    tpm.finalize_before_start()
    await tpm_gate.shutdown()

    both_gate = QuotaGate(1, 60, clock=clock, sleep=clock.sleep)
    first = await both_gate.reserve(TokenEstimate(60))
    first.mark_provider_started()
    first.reconcile(60)
    both_waiter = asyncio.create_task(both_gate.reserve(TokenEstimate(60)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await clock.advance(60)
    both = await both_waiter
    assert both.limited_by == "both"
    both.finalize_before_start()
    await both_gate.shutdown()


@pytest.mark.asyncio
async def test_refunds_cap_at_burst_capacity_and_repeated_cancel_does_not_leak() -> None:
    clock = _ManualClock()
    gate = QuotaGate(1, 100, clock=clock, sleep=clock.sleep)
    reservation = await gate.reserve(TokenEstimate(10))
    await clock.advance(60)
    assert reservation.finalize_before_start() is not None
    assert gate.request_available == pytest.approx(1)
    assert gate.token_available == pytest.approx(100)

    held = await gate.reserve(TokenEstimate(100))
    cancelled = asyncio.create_task(gate.reserve(TokenEstimate(1)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    cancelled.cancel()
    assert gate.waiter_count == 0
    held.finalize_before_start()
    assert gate.request_available == pytest.approx(1)
    assert gate.token_available == pytest.approx(100)
    await gate.shutdown()
