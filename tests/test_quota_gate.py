"""Deterministic coverage for the Session A request-quota gate."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

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
