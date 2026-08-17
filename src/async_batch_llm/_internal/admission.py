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
from ..strategies import RateLimitStrategy, TokenEstimateExceedsLimit
from ..token_estimation import TokenEstimate
from .event_dispatcher import EventDispatcher
from .rate_limit_coordinator import RateLimitCoordinator

_REQUEST_UNITS = 1.0
_FLOAT_TOLERANCE = 1e-9


class AdmissionGateClosed(RuntimeError):
    """An internal quota gate was shut down while an attempt was waiting."""


@dataclass
class _Waiter:
    started_at: float
    estimate: TokenEstimate | None
    future: asyncio.Future[QuotaReservation]
    blocked_by_rpm: bool = False
    blocked_by_tpm: bool = False


@dataclass(frozen=True)
class QuotaFinalization:
    """One immutable exactly-once reservation disposition."""

    provider_started: bool
    known_usage: bool
    reported_tokens: int | None
    delta_tokens: int | None
    disposition: str


class QuotaReservation:
    """Exactly-once lifecycle for one atomic request/token reservation."""

    __slots__ = (
        "_finalized",
        "_finalization",
        "_gate",
        "_provider_started",
        "estimated_input_tokens",
        "estimated_output_tokens",
        "limited_by",
        "request_reserved",
        "reserved_tokens",
        "wait_seconds",
    )

    def __init__(
        self,
        gate: QuotaGate,
        *,
        request_reserved: bool,
        estimate: TokenEstimate | None,
        wait_seconds: float,
        limited_by: str | None = None,
    ) -> None:
        self._gate = gate
        self.request_reserved = request_reserved
        self.estimated_input_tokens = estimate.input_tokens if estimate is not None else None
        self.estimated_output_tokens = estimate.output_tokens if estimate is not None else None
        self.reserved_tokens = estimate.total_tokens if estimate is not None else 0
        self.wait_seconds = wait_seconds
        self.limited_by = limited_by
        self._provider_started = False
        self._finalized = False
        self._finalization: QuotaFinalization | None = None

    @property
    def provider_started(self) -> bool:
        return self._provider_started

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def finalization(self) -> QuotaFinalization | None:
        return self._finalization

    def mark_provider_started(self) -> None:
        if self._finalized:
            raise RuntimeError("Cannot start provider work after reservation finalization")
        self._provider_started = True

    def finalize_before_start(self) -> QuotaFinalization | None:
        """Refund both dimensions when provider work never began."""
        if self._finalized:
            return None
        if self._provider_started:
            raise RuntimeError("Cannot finalize before start after provider work began")
        self._finalized = True
        self._gate._refund(self.request_reserved, self.reserved_tokens)
        self._finalization = QuotaFinalization(
            provider_started=False,
            known_usage=False,
            reported_tokens=None,
            delta_tokens=(-self.reserved_tokens if self.reserved_tokens else None),
            disposition="refunded_before_start",
        )
        return self._finalization

    def reconcile(self, reported_tokens: int) -> QuotaFinalization | None:
        """Finalize a started reservation against known provider usage."""
        if self._finalized:
            return None
        if not self._provider_started:
            raise RuntimeError("Cannot reconcile before provider work starts")
        if isinstance(reported_tokens, bool) or not isinstance(reported_tokens, int):
            raise TypeError("reported_tokens must be an integer")
        if reported_tokens < 0:
            raise ValueError("reported_tokens must be non-negative")
        self._finalized = True
        delta = self._gate._reconcile_tokens(self.reserved_tokens, reported_tokens)
        disposition = "exact"
        if delta < 0:
            disposition = "refunded"
        elif delta > 0:
            disposition = "debt"
        self._finalization = QuotaFinalization(
            provider_started=True,
            known_usage=True,
            reported_tokens=reported_tokens,
            delta_tokens=delta,
            disposition=disposition,
        )
        return self._finalization

    def finalize_unknown(self) -> QuotaFinalization | None:
        """Finalize conservatively, retaining the token estimate."""
        if self._finalized:
            return None
        if not self._provider_started:
            return self.finalize_before_start()
        self._finalized = True
        self._finalization = QuotaFinalization(
            provider_started=True,
            known_usage=False,
            reported_tokens=None,
            delta_tokens=None,
            disposition="retained_unknown",
        )
        return self._finalization

    def finalize_request_only(self) -> QuotaFinalization | None:
        """Consume a started RPM reservation when TPM is disabled."""
        if self._finalized:
            return None
        if not self._provider_started:
            return self.finalize_before_start()
        self._finalized = True
        self._finalization = QuotaFinalization(
            provider_started=True,
            known_usage=False,
            reported_tokens=None,
            delta_tokens=None,
            disposition="request_consumed",
        )
        return self._finalization

    def finalize(self) -> bool:
        """Compatibility helper: refund pre-start, retain unknown post-start."""
        if self._provider_started:
            return self.finalize_unknown() is not None
        return self.finalize_before_start() is not None


class QuotaGate:
    """Strict-FIFO monotonic gate that atomically reserves RPM and TPM."""

    def __init__(
        self,
        max_requests_per_minute: float | None,
        max_tokens_per_minute: int | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._request_capacity = (
            max(1.0, max_requests_per_minute) if max_requests_per_minute is not None else None
        )
        self._request_refill_per_second = (
            max_requests_per_minute / 60.0 if max_requests_per_minute is not None else None
        )
        self._request_available = self._request_capacity
        self._token_capacity = (
            float(max_tokens_per_minute) if max_tokens_per_minute is not None else None
        )
        self._token_refill_per_second = (
            max_tokens_per_minute / 60.0 if max_tokens_per_minute is not None else None
        )
        self._token_available = self._token_capacity
        self._last_refill = clock()
        self._waiters: deque[_Waiter] = deque()
        self._wake_task: asyncio.Task[None] | None = None
        self._wake_deadline: float | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self.rpm_enabled or self.tpm_enabled

    @property
    def rpm_enabled(self) -> bool:
        return self._request_capacity is not None

    @property
    def tpm_enabled(self) -> bool:
        return self._token_capacity is not None

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

    @property
    def token_available(self) -> float | None:
        self._refill()
        return self._token_available

    async def reserve(self, estimate: TokenEstimate | None = None) -> QuotaReservation:
        """Atomically reserve configured request and token dimensions."""
        if self._closed:
            raise AdmissionGateClosed("Quota gate is closed")
        self._validate_estimate(estimate)
        if not self.enabled:
            return QuotaReservation(
                self,
                request_reserved=False,
                estimate=None,
                wait_seconds=0.0,
            )

        started_at = self._clock()
        self._refill()
        if not self._waiters and self._ready(estimate):
            self._consume(estimate)
            return QuotaReservation(
                self,
                request_reserved=self.rpm_enabled,
                estimate=estimate if self.tpm_enabled else None,
                wait_seconds=0.0,
            )

        future: asyncio.Future[QuotaReservation] = asyncio.get_running_loop().create_future()
        waiter = _Waiter(started_at=started_at, estimate=estimate, future=future)
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
        now = self._clock()
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now
        if self.rpm_enabled:
            assert self._request_capacity is not None
            assert self._request_refill_per_second is not None
            assert self._request_available is not None
            self._request_available = min(
                self._request_capacity,
                self._request_available + elapsed * self._request_refill_per_second,
            )
        if self.tpm_enabled:
            assert self._token_capacity is not None
            assert self._token_refill_per_second is not None
            assert self._token_available is not None
            self._token_available = min(
                self._token_capacity,
                self._token_available + elapsed * self._token_refill_per_second,
            )

    def _request_ready(self) -> bool:
        if not self.rpm_enabled:
            return True
        assert self._request_available is not None
        return self._request_available + _FLOAT_TOLERANCE >= _REQUEST_UNITS

    def _tokens_ready(self, estimate: TokenEstimate | None) -> bool:
        if not self.tpm_enabled:
            return True
        assert estimate is not None
        assert self._token_available is not None
        return self._token_available + _FLOAT_TOLERANCE >= estimate.total_tokens

    def _ready(self, estimate: TokenEstimate | None) -> bool:
        return self._request_ready() and self._tokens_ready(estimate)

    def _consume_request(self) -> None:
        if not self.rpm_enabled:
            return
        assert self._request_available is not None
        self._request_available -= _REQUEST_UNITS
        if abs(self._request_available) <= _FLOAT_TOLERANCE:
            self._request_available = 0.0

    def _consume(self, estimate: TokenEstimate | None) -> None:
        self._consume_request()
        if self.tpm_enabled:
            assert estimate is not None
            assert self._token_available is not None
            self._token_available -= estimate.total_tokens
            if abs(self._token_available) <= _FLOAT_TOLERANCE:
                self._token_available = 0.0

    def _refund(self, request_reserved: bool, reserved_tokens: int) -> None:
        if not self.enabled:
            return
        self._refill()
        if request_reserved:
            assert self._request_available is not None
            assert self._request_capacity is not None
            self._request_available = min(
                self._request_capacity,
                self._request_available + _REQUEST_UNITS,
            )
        if reserved_tokens:
            assert self._token_available is not None
            assert self._token_capacity is not None
            self._token_available = min(
                self._token_capacity,
                self._token_available + reserved_tokens,
            )
        self._process_waiters()

    def _refund_request(self) -> None:
        """Session A compatibility alias for tests inspecting gate internals."""
        self._refund(True, 0)

    def _reconcile_tokens(self, reserved_tokens: int, reported_tokens: int) -> int:
        delta = reported_tokens - reserved_tokens
        if self.tpm_enabled and delta:
            self._refill()
            assert self._token_available is not None
            assert self._token_capacity is not None
            self._token_available = min(self._token_capacity, self._token_available - delta)
            self._process_waiters()
        return delta

    def _process_waiters(self) -> None:
        if self._closed or not self.enabled:
            return
        self._refill()
        while self._waiters:
            waiter = self._waiters[0]
            if waiter.future.cancelled():
                self._waiters.popleft()
                continue
            request_ready = self._request_ready()
            tokens_ready = self._tokens_ready(waiter.estimate)
            if not request_ready or not tokens_ready:
                waiter.blocked_by_rpm = waiter.blocked_by_rpm or not request_ready
                waiter.blocked_by_tpm = waiter.blocked_by_tpm or not tokens_ready
                self._schedule_wake(waiter.estimate)
                return
            self._waiters.popleft()
            self._consume(waiter.estimate)
            limited_by: str | None = None
            if waiter.blocked_by_rpm and waiter.blocked_by_tpm:
                limited_by = "both"
            elif waiter.blocked_by_rpm:
                limited_by = "rpm"
            elif waiter.blocked_by_tpm:
                limited_by = "tpm"
            reservation = QuotaReservation(
                self,
                request_reserved=self.rpm_enabled,
                estimate=waiter.estimate if self.tpm_enabled else None,
                wait_seconds=max(0.0, self._clock() - waiter.started_at),
                limited_by=limited_by,
            )
            waiter.future.set_result(reservation)
        self._cancel_wake_task()

    def _schedule_wake(self, estimate: TokenEstimate | None) -> None:
        delay = 0.0
        if self.rpm_enabled:
            assert self._request_available is not None
            assert self._request_refill_per_second is not None
            request_deficit = max(0.0, _REQUEST_UNITS - self._request_available)
            delay = max(delay, request_deficit / self._request_refill_per_second)
        if self.tpm_enabled:
            assert estimate is not None
            assert self._token_available is not None
            assert self._token_refill_per_second is not None
            token_deficit = max(0.0, estimate.total_tokens - self._token_available)
            delay = max(delay, token_deficit / self._token_refill_per_second)
        deadline = self._clock() + delay
        if self._wake_task is not None and not self._wake_task.done():
            if (
                self._wake_deadline is not None
                and self._wake_deadline <= deadline + _FLOAT_TOLERANCE
            ):
                return
            self._cancel_wake_task()
        self._wake_deadline = deadline
        self._wake_task = asyncio.create_task(self._wake_after(delay))

    async def _wake_after(self, delay: float) -> None:
        try:
            await self._sleep(delay)
        except asyncio.CancelledError:
            raise
        finally:
            if self._wake_task is asyncio.current_task():
                self._wake_task = None
                self._wake_deadline = None
        if not self._closed:
            self._process_waiters()

    def _cancel_wake_task(self) -> asyncio.Task[None] | None:
        task = self._wake_task
        self._wake_task = None
        self._wake_deadline = None
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

    def _validate_estimate(self, estimate: TokenEstimate | None) -> None:
        if not self.tpm_enabled:
            return
        if estimate is None or estimate.total_tokens <= 0:
            raise ValueError("A positive TokenEstimate is required when TPM admission is enabled")
        assert self.max_tokens_per_minute is not None
        if estimate.total_tokens > self.max_tokens_per_minute:
            raise TokenEstimateExceedsLimit(
                "Token estimate cannot fit in the configured per-minute token limit "
                f"({estimate.total_tokens} > {self.max_tokens_per_minute})."
            )


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
        max_tokens_per_minute: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._rate_limit_strategy = rate_limit_strategy
        self._events = events
        self._max_requests_per_minute = max_requests_per_minute
        self._max_tokens_per_minute = max_tokens_per_minute
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
                    self._max_tokens_per_minute,
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
    "QuotaFinalization",
    "QuotaReservation",
    "ScopeAdmissionState",
]
