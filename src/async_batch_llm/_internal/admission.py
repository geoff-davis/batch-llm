"""Quota-scoped cooldown and proactive request admission."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..llm_strategies import LLMCallStrategy
from ..strategies import RateLimitStrategy
from .event_dispatcher import EventDispatcher
from .rate_limit_coordinator import RateLimitCoordinator

_REQUEST_UNITS = 1.0
_FLOAT_TOLERANCE = 1e-9


class AdmissionGateClosed(RuntimeError):
    """An internal quota gate was shut down while an attempt was waiting."""


@dataclass
class _Waiter:
    started_at: float
    future: asyncio.Future[QuotaReservation]


class QuotaReservation:
    """Exactly-once lifecycle for one request reservation.

    Session A reserves request capacity only. The same private object is the
    extension seam for Session B token reservation and reconciliation.
    """

    __slots__ = (
        "_finalized",
        "_gate",
        "_provider_started",
        "request_reserved",
        "wait_seconds",
    )

    def __init__(
        self,
        gate: QuotaGate,
        *,
        request_reserved: bool,
        wait_seconds: float,
    ) -> None:
        self._gate = gate
        self.request_reserved = request_reserved
        self.wait_seconds = wait_seconds
        self._provider_started = False
        self._finalized = False

    @property
    def provider_started(self) -> bool:
        return self._provider_started

    @property
    def finalized(self) -> bool:
        return self._finalized

    def mark_provider_started(self) -> None:
        if self._finalized:
            raise RuntimeError("Cannot start provider work after reservation finalization")
        self._provider_started = True

    def finalize(self) -> bool:
        """Finalize once, refunding only when provider work never started."""
        if self._finalized:
            return False
        self._finalized = True
        if self.request_reserved and not self._provider_started:
            self._gate._refund_request()
        return True


class QuotaGate:
    """FIFO monotonic token bucket for atomic quota reservations.

    In Session A the only configured dimension is RPM. The gate deliberately
    owns a single wake task per scope, not one sleeping task per waiter.
    """

    def __init__(
        self,
        max_requests_per_minute: float | None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.max_requests_per_minute = max_requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._request_capacity = (
            max(1.0, max_requests_per_minute) if max_requests_per_minute is not None else None
        )
        self._request_refill_per_second = (
            max_requests_per_minute / 60.0 if max_requests_per_minute is not None else None
        )
        self._request_available = self._request_capacity
        self._last_refill = clock()
        self._waiters: deque[_Waiter] = deque()
        self._wake_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self._request_capacity is not None

    @property
    def waiter_count(self) -> int:
        return len(self._waiters)

    @property
    def has_wake_task(self) -> bool:
        return self._wake_task is not None and not self._wake_task.done()

    @property
    def request_available(self) -> float | None:
        self._refill()
        return self._request_available

    async def reserve(self) -> QuotaReservation:
        """Reserve one request unit in strict FIFO order."""
        if self._closed:
            raise AdmissionGateClosed("Quota gate is closed")
        if not self.enabled:
            return QuotaReservation(self, request_reserved=False, wait_seconds=0.0)

        started_at = self._clock()
        self._refill()
        if not self._waiters and self._request_ready():
            self._consume_request()
            return QuotaReservation(self, request_reserved=True, wait_seconds=0.0)

        future: asyncio.Future[QuotaReservation] = asyncio.get_running_loop().create_future()
        waiter = _Waiter(started_at=started_at, future=future)
        self._waiters.append(waiter)
        self._process_waiters()

        try:
            # Shield keeps a granted reservation inspectable if cancellation
            # lands between future completion and task resumption.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                with contextlib.suppress(AdmissionGateClosed):
                    future.result().finalize()
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(waiter)
                future.cancel()
                self._process_waiters()
            raise

    def _refill(self) -> None:
        if not self.enabled:
            return
        assert self._request_capacity is not None
        assert self._request_refill_per_second is not None
        assert self._request_available is not None
        now = self._clock()
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now
        self._request_available = min(
            self._request_capacity,
            self._request_available + elapsed * self._request_refill_per_second,
        )

    def _request_ready(self) -> bool:
        assert self._request_available is not None
        return self._request_available + _FLOAT_TOLERANCE >= _REQUEST_UNITS

    def _consume_request(self) -> None:
        assert self._request_available is not None
        self._request_available -= _REQUEST_UNITS
        if abs(self._request_available) <= _FLOAT_TOLERANCE:
            self._request_available = 0.0

    def _refund_request(self) -> None:
        if not self.enabled:
            return
        self._refill()
        assert self._request_available is not None
        assert self._request_capacity is not None
        self._request_available = min(
            self._request_capacity,
            self._request_available + _REQUEST_UNITS,
        )
        self._process_waiters()

    def _process_waiters(self) -> None:
        if self._closed or not self.enabled:
            return
        self._refill()
        while self._waiters:
            waiter = self._waiters[0]
            if waiter.future.cancelled():
                self._waiters.popleft()
                continue
            if not self._request_ready():
                self._schedule_wake()
                return
            self._waiters.popleft()
            self._consume_request()
            reservation = QuotaReservation(
                self,
                request_reserved=True,
                wait_seconds=max(0.0, self._clock() - waiter.started_at),
            )
            waiter.future.set_result(reservation)
        self._cancel_wake_task()

    def _schedule_wake(self) -> None:
        if self._wake_task is not None and not self._wake_task.done():
            return
        assert self._request_available is not None
        assert self._request_refill_per_second is not None
        deficit = max(0.0, _REQUEST_UNITS - self._request_available)
        delay = max(0.0, deficit / self._request_refill_per_second)
        self._wake_task = asyncio.create_task(self._wake_after(delay))

    async def _wake_after(self, delay: float) -> None:
        try:
            await self._sleep(delay)
        except asyncio.CancelledError:
            raise
        finally:
            if self._wake_task is asyncio.current_task():
                self._wake_task = None
        if not self._closed:
            self._process_waiters()

    def _cancel_wake_task(self) -> asyncio.Task[None] | None:
        task = self._wake_task
        self._wake_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        return task

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        task = self._cancel_wake_task()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.future.done():
                waiter.future.set_exception(AdmissionGateClosed("Quota gate was shut down"))


@dataclass(frozen=True)
class ScopeAdmissionState:
    ordinal: int
    scope: object
    cooldown: RateLimitCoordinator
    quota_gate: QuotaGate


@dataclass(frozen=True)
class _StrategyScopeEntry:
    strategy: LLMCallStrategy
    state: ScopeAdmissionState


class AdmissionRegistry:
    """Own one cooldown and quota gate per quota-scope object identity."""

    def __init__(
        self,
        *,
        rate_limit_strategy: RateLimitStrategy,
        events: EventDispatcher[Any, Any, Any],
        max_requests_per_minute: float | None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._rate_limit_strategy = rate_limit_strategy
        self._events = events
        self._max_requests_per_minute = max_requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._scope_entries: dict[int, ScopeAdmissionState] = {}
        self._strategy_entries: dict[int, _StrategyScopeEntry] = {}
        self._next_ordinal = 1
        self._closed = False

    @staticmethod
    def _quota_scope(strategy: LLMCallStrategy) -> object:
        try:
            scope = strategy.quota_scope
        except Exception:
            scope = strategy
        return strategy if scope is None else scope

    def resolve(self, strategy: LLMCallStrategy) -> ScopeAdmissionState:
        if self._closed:
            raise AdmissionGateClosed("Admission registry is closed")
        strategy_id = id(strategy)
        strategy_entry = self._strategy_entries.get(strategy_id)
        if strategy_entry is not None and strategy_entry.strategy is strategy:
            return strategy_entry.state

        scope = self._quota_scope(strategy)
        scope_id = id(scope)
        state = self._scope_entries.get(scope_id)
        if state is None or state.scope is not scope:
            ordinal = self._next_ordinal
            self._next_ordinal += 1
            state = ScopeAdmissionState(
                ordinal=ordinal,
                scope=scope,
                cooldown=RateLimitCoordinator(
                    rate_limit_strategy=self._rate_limit_strategy,
                    events=self._events,
                    quota_scope_id=ordinal,
                ),
                quota_gate=QuotaGate(
                    self._max_requests_per_minute,
                    clock=self._clock,
                    sleep=self._sleep,
                ),
            )
            self._scope_entries[scope_id] = state
        self._strategy_entries[strategy_id] = _StrategyScopeEntry(strategy, state)
        return state

    @property
    def entry_count(self) -> int:
        return len(self._scope_entries)

    @property
    def states(self) -> tuple[ScopeAdmissionState, ...]:
        return tuple(self._scope_entries.values())

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for state in tuple(self._scope_entries.values()):
            try:
                await state.cooldown.shutdown()
            except BaseException as exc:  # cleanup every scope before surfacing one error
                errors.append(exc)
            try:
                await state.quota_gate.shutdown()
            except BaseException as exc:
                errors.append(exc)
        self._strategy_entries.clear()
        self._scope_entries.clear()
        if errors:
            raise errors[0]


__all__ = [
    "AdmissionGateClosed",
    "AdmissionRegistry",
    "QuotaGate",
    "QuotaReservation",
    "ScopeAdmissionState",
]
