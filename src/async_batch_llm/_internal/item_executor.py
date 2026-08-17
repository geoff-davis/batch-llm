"""Per-item execution engine: retries, error classification, rate-limit
coordination, and token accounting for a single work item.

Extracted from :class:`ParallelBatchProcessor` so the exact same execution
semantics can back three surfaces: the batch worker loop, the single-call
helper (:mod:`async_batch_llm.single`), and the rate-limited gateway
(:mod:`async_batch_llm.gateway`). The processor delegates its per-item methods
here; the queue-less surfaces drive :meth:`ItemExecutor.execute` directly.

The executor reads its dependencies live from a *host* (the processor passes
``self``; the gateway passes a lightweight host) because processor stats are
rebound by ``start()`` and admission/classifier ownership lives on the host.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, cast

from ..base import (
    AttemptTiming,
    LLMWorkItem,
    RetryState,
    TContext,
    TInput,
    TokenUsage,
    TOutput,
    WorkItemResult,
    WorkItemTiming,
    _unpack_strategy_result,
)
from ..observers import ProcessingEvent
from ..strategies import (
    BatchAbortedError,
    BatchDeadlineExceeded,
    ErrorClassifier,
    ErrorInfo,
    FrameworkTimeoutError,
    ItemDeadlineExceeded,
    RateLimitRetriesExceeded,
    TokenEstimateExceedsLimit,
    TokenEstimationError,
    TokenEstimatorRequired,
)
from ..token_estimation import TokenEstimate
from ..token_extractor import TokenUsageObservation
from .admission import (
    AdmissionRegistry,
    QuotaFinalization,
    QuotaReservation,
    ScopeAdmissionState,
)
from .capacity import CapacityLimiter
from .classifier_resolver import StrategyClassifierResolver
from .error_logging import log_retryable_error, log_validation_error
from .guardrails import AbortController, await_with_guardrails, remaining_seconds

if TYPE_CHECKING:
    from ..base import ProcessingStats
    from ..core import ProcessorConfig
    from ..llm_strategies import LLMCallStrategy
    from ..token_extractor import TokenExtractor
    from .event_dispatcher import EventDispatcher
    from .rate_limit_coordinator import RateLimitCoordinator
    from .strategy_lifecycle import StrategyLifecycle

logger = logging.getLogger(__name__)

# Kept in sync with parallel.py (single source would create an import cycle).
ERROR_MESSAGE_MAX_LENGTH = 200
ERROR_MESSAGE_DETAILED_LENGTH = 500
_ADMISSION_WAIT_STATE_KEY = "_abl_admission_wait_seconds"
_ADMISSION_WAIT_EXCEPTION_KEY = "_abl_admission_wait_seconds"
_TIMING_EXCEPTION_KEY = "_abl_work_item_timing"
_LAST_ADMISSION_KEY = "_abl_last_admission_wait_seconds"
_LAST_STARTUP_RAMP_KEY = "_abl_last_startup_ramp_wait_seconds"
_LAST_EXECUTION_KEY = "_abl_last_execution_seconds"
_LAST_PROVIDER_KEY = "_abl_last_provider_seconds"
_LAST_COOLDOWN_KEY = "_abl_last_cooldown_wait_seconds"
_LAST_QUOTA_WAIT_KEY = "_abl_last_quota_wait_seconds"
_LAST_ESTIMATED_INPUT_KEY = "_abl_last_estimated_input_tokens"
_LAST_ESTIMATED_OUTPUT_KEY = "_abl_last_estimated_output_tokens"
_LAST_RESERVED_TOKENS_KEY = "_abl_last_reserved_tokens"
_LAST_REPORTED_TOKENS_KEY = "_abl_last_reported_tokens"
_LAST_RECONCILIATION_DELTA_KEY = "_abl_last_reconciliation_delta_tokens"
_LAST_QUOTA_SCOPE_KEY = "_abl_last_quota_scope_id"
_PHYSICAL_TRY_KEY = "_abl_physical_try_number"
_LAST_TIMEOUT_KEY = "_abl_last_timeout_category"
_LAST_ERROR_CATEGORY_KEY = "_abl_last_error_category"
_TOTAL_DEADLINE_KEY = "_abl_total_item_deadline"
_ERROR_INFO_EXCEPTION_KEY = "_abl_error_info"

_E = TypeVar("_E", bound=BaseException)


def _state_float(state: RetryState, key: str) -> float:
    value = state.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _state_optional_int(state: RetryState, key: str) -> int | None:
    value = state.get(key)
    return value if not isinstance(value, bool) and isinstance(value, int) else None


def _is_async_callable(callback: object) -> bool:
    return inspect.iscoroutinefunction(callback) or (
        callable(callback) and inspect.iscoroutinefunction(type(callback).__call__)
    )


def _attempt_timing(
    state: RetryState,
    *,
    attempt: int,
    try_number: int,
    total_seconds: float,
    success: bool,
    error_type: str | None = None,
    error_category: str | None = None,
) -> AttemptTiming:
    provider_value = state.get(_LAST_PROVIDER_KEY)
    provider_seconds = float(provider_value) if isinstance(provider_value, (int, float)) else None
    timeout_value = state.get(_LAST_TIMEOUT_KEY)
    return AttemptTiming(
        attempt=attempt,
        try_number=try_number,
        total_seconds=total_seconds,
        admission_wait_seconds=_state_float(state, _LAST_ADMISSION_KEY),
        startup_ramp_wait_seconds=_state_float(state, _LAST_STARTUP_RAMP_KEY),
        execution_seconds=_state_float(state, _LAST_EXECUTION_KEY),
        provider_seconds=provider_seconds,
        cooldown_wait_seconds=_state_float(state, _LAST_COOLDOWN_KEY),
        quota_wait_seconds=_state_float(state, _LAST_QUOTA_WAIT_KEY),
        estimated_input_tokens=_state_optional_int(state, _LAST_ESTIMATED_INPUT_KEY),
        estimated_output_tokens=_state_optional_int(state, _LAST_ESTIMATED_OUTPUT_KEY),
        reserved_tokens=_state_optional_int(state, _LAST_RESERVED_TOKENS_KEY) or 0,
        reported_tokens=_state_optional_int(state, _LAST_REPORTED_TOKENS_KEY),
        reconciliation_delta_tokens=_state_optional_int(state, _LAST_RECONCILIATION_DELTA_KEY),
        quota_scope_id=_state_optional_int(state, _LAST_QUOTA_SCOPE_KEY),
        success=success,
        error_type=error_type,
        error_category=error_category,
        timeout_category=timeout_value if isinstance(timeout_value, str) else None,
    )


def _work_item_timing(started: float, attempts: list[AttemptTiming]) -> WorkItemTiming:
    timeout_category = next(
        (attempt.timeout_category for attempt in reversed(attempts) if attempt.timeout_category),
        None,
    )
    return WorkItemTiming(
        total_seconds=max(0.0, time.perf_counter() - started),
        attempts=list(attempts),
        timeout_category=timeout_category,
    )


def _detach_traceback(exc: _E) -> _E:
    """Clear tracebacks along an exception's cause/context chain before it is
    stored on a ``WorkItemResult``.

    A traceback pins every frame's locals — strategies, clients, raw responses —
    for as long as the result is held, which for a large accumulated batch of
    failures can retain far more memory than before ``WorkItemResult.exception``
    existed. The full failure (type, message, stack) is already logged at the
    point it happens, so the stored exception keeps its type/message/args (enough
    for ``call()`` to re-raise the provider's type) but drops the frame-pinning
    tracebacks. A re-raise gets a fresh traceback from the raise site.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        cur.__traceback__ = None
        cur = cur.__cause__ or cur.__context__
    return exc


def _classify_error(exception: Exception, classifier: ErrorClassifier) -> ErrorInfo:
    """Classify an exception once and reuse that exact decision downstream."""
    cached = getattr(exception, "__dict__", {}).get(_ERROR_INFO_EXCEPTION_KEY)
    if isinstance(cached, ErrorInfo):
        return cached
    if isinstance(exception, TokenEstimationError):
        error_info = ErrorInfo(
            is_retryable=False,
            is_rate_limit=False,
            is_timeout=False,
            error_category=exception.error_category,
        )
    else:
        error_info = classifier.classify(exception)
    if hasattr(exception, "__dict__"):
        exception.__dict__[_ERROR_INFO_EXCEPTION_KEY] = error_info
    return error_info


class ExecutorHostProtocol(Protocol[TInput, TOutput, TContext]):
    """The surface :class:`ItemExecutor` reads from its host.

    Both hosts — :class:`~async_batch_llm.parallel.ParallelBatchProcessor` and
    the lightweight :class:`~async_batch_llm._internal.executor_host.ExecutorHost`
    — expose exactly these. Dependencies are read live rather than snapshotted.

    ``_extract_token_usage``, ``_process_item``, and ``_handle_execution_error``
    are invoked *through the host* (not as the executor's own methods) so that a
    ``ParallelBatchProcessor`` subclass overriding any of them — ``_process_item``
    is abstract on ``BatchProcessor``; ``_extract_token_usage`` is a documented
    override point — still takes effect during batch runs. The processor's
    versions delegate back to the executor's implementations, so the base case is
    one extra hop with no behavior change.
    """

    config: ProcessorConfig
    error_classifier: ErrorClassifier
    _classifier_resolver: StrategyClassifierResolver
    _admission_registry: AdmissionRegistry
    _token_extractor: TokenExtractor
    _rate_limit_coord: RateLimitCoordinator
    _events: EventDispatcher[TInput, TOutput, TContext]
    _stats: ProcessingStats
    _stats_lock: asyncio.Lock
    _strategy_lifecycle: StrategyLifecycle[TOutput]
    _capacity_limiter: CapacityLimiter
    _abort_controller: AbortController | None

    def _extract_token_usage(self, exception: Exception) -> dict[str, int]: ...

    async def _process_item(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int = 1,
        strategy: LLMCallStrategy[TOutput] | None = None,
        retry_state: RetryState | None = None,
    ) -> WorkItemResult[TOutput, TContext]: ...

    async def _handle_execution_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
    ) -> WorkItemResult[TOutput, TContext]: ...

    async def _process_item_with_retries(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        deadline: float | None = None,
    ) -> WorkItemResult[TOutput, TContext]: ...


class ItemExecutor(Generic[TInput, TOutput, TContext]):
    """Executes one work item with the full resilience pipeline.

    ``host`` must satisfy :class:`ExecutorHostProtocol`.
    """

    def __init__(self, host: ExecutorHostProtocol[TInput, TOutput, TContext]) -> None:
        self._host = host

    # ── Dependencies (read live from host) ───────────────────────
    @property
    def config(self) -> ProcessorConfig:
        return self._host.config

    @property
    def error_classifier(self) -> ErrorClassifier:
        return self._host.error_classifier

    @property
    def _classifier_resolver(self) -> StrategyClassifierResolver:
        return self._host._classifier_resolver

    @property
    def _admission_registry(self) -> AdmissionRegistry:
        return self._host._admission_registry

    @property
    def _token_extractor(self) -> TokenExtractor:
        return self._host._token_extractor

    @property
    def _rate_limit_coord(self) -> RateLimitCoordinator:
        return self._host._rate_limit_coord

    @property
    def _events(self) -> EventDispatcher[TInput, TOutput, TContext]:
        return self._host._events

    @property
    def _stats(self) -> ProcessingStats:
        return self._host._stats

    @property
    def _stats_lock(self) -> asyncio.Lock:
        return self._host._stats_lock

    @property
    def _strategy_lifecycle(self) -> StrategyLifecycle[TOutput]:
        return self._host._strategy_lifecycle

    @property
    def _capacity_limiter(self) -> CapacityLimiter:
        return self._host._capacity_limiter

    @property
    def _abort_controller(self) -> AbortController | None:
        return self._host._abort_controller

    # ── Thin delegators (so moved bodies stay verbatim) ──────────
    async def _emit_event(self, event: ProcessingEvent, data: dict | None = None) -> None:
        await self._events.emit(event, data)

    async def _run_middlewares_before(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMWorkItem[TInput, TOutput, TContext] | None:
        return await self._events.run_before(work_item)

    async def _run_middlewares_after(
        self, result: WorkItemResult[TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        return await self._events.run_after(result)

    async def _run_middlewares_on_error(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], error: Exception
    ) -> WorkItemResult[TOutput, TContext] | None:
        return await self._events.run_on_error(work_item, error)

    async def _ensure_strategy_prepared(self, strategy) -> None:
        await self._strategy_lifecycle.ensure_prepared(strategy)

    async def _handle_rate_limit(
        self,
        state: ScopeAdmissionState,
        worker_id: int,
        observed_generation: int | None = None,
        suggested_wait: float | None = None,
        strategy_type: str | None = None,
    ) -> None:
        await state.cooldown.handle_rate_limit(
            worker_id,
            observed_generation,
            suggested_wait,
            strategy_type=strategy_type,
        )

    def _log_retryable_error(
        self, exception, work_item, attempt_number, failed_token_usage
    ) -> None:
        log_retryable_error(exception, work_item.item_id, attempt_number, failed_token_usage)

    def _log_validation_error(self, exception, work_item, attempt_number, token_msg) -> None:
        log_validation_error(exception, work_item.item_id, attempt_number, token_msg)

    async def _resolve_token_estimate(
        self,
        *,
        prompt: str,
        strategy: LLMCallStrategy[TOutput],
        attempt: int,
        retry_state: RetryState | None,
        deadline: float | None,
        item_id: str,
    ) -> TokenEstimate:
        estimator = self.config.token_estimator

        async def invoke() -> object:
            value: object
            if estimator is not None:
                if _is_async_callable(estimator):
                    value = estimator(
                        prompt,
                        strategy=strategy,
                        attempt=attempt,
                        state=retry_state,
                    )
                else:
                    value = await asyncio.to_thread(
                        estimator,
                        prompt,
                        strategy=strategy,
                        attempt=attempt,
                        state=retry_state,
                    )
            else:
                estimate_tokens = strategy.estimate_tokens
                if _is_async_callable(estimate_tokens):
                    value = estimate_tokens(prompt, attempt, retry_state)
                else:
                    value = await asyncio.to_thread(
                        estimate_tokens,
                        prompt,
                        attempt,
                        retry_state,
                    )
            if inspect.isawaitable(value):
                return await value
            return value

        try:
            value = await await_with_guardrails(
                invoke(),
                item_deadline=deadline,
                item_id=item_id,
                abort_controller=self._abort_controller,
            )
            if value is None:
                raise TokenEstimatorRequired(
                    "TPM admission requires ProcessorConfig.token_estimator or a "
                    "strategy-level token estimator."
                )
            if not isinstance(value, TokenEstimate):
                raise TokenEstimationError(
                    f"Token estimator must return TokenEstimate (got {type(value).__name__})."
                )
            if value.total_tokens <= 0:
                raise TokenEstimationError(
                    "Token estimate total must be greater than zero when TPM admission is enabled."
                )
            limit = self.config.max_tokens_per_minute
            assert limit is not None
            if value.total_tokens > limit:
                raise TokenEstimateExceedsLimit(
                    "Token estimate cannot fit in the configured per-minute token limit "
                    f"({value.total_tokens} > {limit})."
                )
            return value
        except asyncio.CancelledError:
            raise
        except (ItemDeadlineExceeded, BatchDeadlineExceeded, BatchAbortedError):
            raise
        except TokenEstimationError:
            async with self._stats_lock:
                self._stats.token_estimation_failures += 1
            raise
        except Exception:
            async with self._stats_lock:
                self._stats.token_estimation_failures += 1
            raise TokenEstimationError(
                "Token estimation failed; verify the configured local estimator."
            ) from None

    def _observe_exception_usage(self, exception: Exception) -> TokenUsageObservation:
        """Preserve processor extraction overrides while retaining known/unknown."""
        default = self._token_extractor.observe_exception(exception)
        try:
            override_usage = self._host._extract_token_usage(exception)
        except asyncio.CancelledError:
            raise
        except Exception:
            return default
        nonzero_override = any(
            not isinstance(value, bool) and isinstance(value, int) and value != 0
            for value in override_usage.values()
        )
        try:
            observed_override = self._token_extractor.observe_result(override_usage)
        except (TypeError, ValueError):
            return default
        if default.known:
            return TokenUsageObservation(
                usage=cast(TokenUsage, dict(override_usage)),
                known=True,
                reported_tokens=(
                    observed_override.reported_tokens
                    if nonzero_override
                    else default.reported_tokens
                ),
            )
        if nonzero_override:
            return TokenUsageObservation(
                usage=cast(TokenUsage, dict(override_usage)),
                known=True,
                reported_tokens=observed_override.reported_tokens,
            )
        return TokenUsageObservation(
            usage=cast(TokenUsage, dict(override_usage)),
            known=False,
            reported_tokens=None,
        )

    async def _record_quota_admitted(
        self,
        reservation: QuotaReservation,
        state: ScopeAdmissionState,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
        retry_state: RetryState | None,
    ) -> None:
        estimated_input = reservation.estimated_input_tokens or 0
        estimated_output = reservation.estimated_output_tokens or 0
        async with self._stats_lock:
            self._stats.record_quota_admission(
                wait_seconds=reservation.wait_seconds,
                estimated_input_tokens=estimated_input,
                estimated_output_tokens=estimated_output,
                scope_count=self._admission_registry.entry_count,
            )
        if self._events.observers:
            await self._emit_event(
                ProcessingEvent.QUOTA_ADMITTED,
                {
                    "item_id": work_item.item_id,
                    "worker_id": worker_id,
                    "attempt": attempt_number,
                    "try_number": (
                        retry_state.get(_PHYSICAL_TRY_KEY) if retry_state is not None else None
                    ),
                    "quota_scope_id": state.ordinal,
                    "wait_seconds": reservation.wait_seconds,
                    "request_reserved": int(reservation.request_reserved),
                    "estimated_input_tokens": estimated_input,
                    "estimated_output_tokens": estimated_output,
                    "estimated_total_tokens": reservation.reserved_tokens,
                    "reserved_tokens": reservation.reserved_tokens,
                    "rpm_configured": state.quota_gate.rpm_enabled,
                    "tpm_configured": state.quota_gate.tpm_enabled,
                    "limited_by": reservation.limited_by,
                },
            )

    @staticmethod
    def _store_quota_timing(
        retry_state: RetryState | None,
        reservation: QuotaReservation,
        scope_id: int,
    ) -> None:
        if retry_state is None:
            return
        retry_state.set(_LAST_QUOTA_WAIT_KEY, reservation.wait_seconds)
        retry_state.set(_LAST_QUOTA_SCOPE_KEY, scope_id)
        retry_state.set(_LAST_RESERVED_TOKENS_KEY, reservation.reserved_tokens)
        if reservation.estimated_input_tokens is not None:
            retry_state.set(_LAST_ESTIMATED_INPUT_KEY, reservation.estimated_input_tokens)
        if reservation.estimated_output_tokens is not None:
            retry_state.set(_LAST_ESTIMATED_OUTPUT_KEY, reservation.estimated_output_tokens)

    async def _record_quota_finalization(
        self,
        *,
        reservation: QuotaReservation,
        finalization: QuotaFinalization,
        state: ScopeAdmissionState,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
        retry_state: RetryState | None,
    ) -> None:
        async with self._stats_lock:
            self._stats.record_quota_finalization(
                provider_started=finalization.provider_started,
                reserved_tokens=reservation.reserved_tokens,
                reported_tokens=finalization.reported_tokens,
                delta_tokens=finalization.delta_tokens,
            )
        if retry_state is not None:
            if finalization.provider_started and finalization.reported_tokens is not None:
                retry_state.set(_LAST_REPORTED_TOKENS_KEY, finalization.reported_tokens)
            if finalization.delta_tokens is not None:
                retry_state.set(_LAST_RECONCILIATION_DELTA_KEY, finalization.delta_tokens)
        if finalization.provider_started and self._events.observers:
            await self._emit_event(
                ProcessingEvent.QUOTA_RECONCILED,
                {
                    "item_id": work_item.item_id,
                    "worker_id": worker_id,
                    "attempt": attempt_number,
                    "try_number": (
                        retry_state.get(_PHYSICAL_TRY_KEY) if retry_state is not None else None
                    ),
                    "quota_scope_id": state.ordinal,
                    "reserved_tokens": reservation.reserved_tokens,
                    "reported_tokens": finalization.reported_tokens,
                    "known_usage": finalization.known_usage,
                    "delta_tokens": finalization.delta_tokens,
                    "disposition": finalization.disposition,
                },
            )

    # ── Queue-less entry points (used by gateway + single) ───────
    async def wait_for_capacity(
        self,
        *,
        admission_state: ScopeAdmissionState | None = None,
        deadline: float | None = None,
        retry_state: RetryState | None = None,
        item_id: str | None = None,
    ) -> None:
        """Respect one quota scope's cooldown + slow-start ramp."""
        coordinator = (
            admission_state.cooldown if admission_state is not None else self._rate_limit_coord
        )
        cooldown_started = time.perf_counter()
        await await_with_guardrails(
            coordinator.wait_if_paused(),
            item_deadline=deadline,
            item_id=item_id,
            abort_controller=self._abort_controller,
        )
        if retry_state is not None:
            retry_state.set(
                _LAST_COOLDOWN_KEY,
                max(0.0, time.perf_counter() - cooldown_started),
            )
        delay = await await_with_guardrails(
            coordinator.apply_slow_start(),
            item_deadline=deadline,
            item_id=item_id,
            abort_controller=self._abort_controller,
        )
        if delay > 0:
            ramp_started = time.perf_counter()
            await await_with_guardrails(
                asyncio.sleep(delay),
                item_deadline=deadline,
                item_id=item_id,
                abort_controller=self._abort_controller,
            )
            if retry_state is not None:
                retry_state.set(
                    _LAST_STARTUP_RAMP_KEY,
                    max(0.0, time.perf_counter() - ramp_started),
                )

    async def execute(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], worker_id: int = 0
    ) -> WorkItemResult[TOutput, TContext]:
        """Run one item end-to-end, always returning a WorkItemResult.

        Waits out any active cooldown, runs the retry pipeline, and converts an
        exhausted/unhandled failure into a failed result (never raises for
        business errors; CancelledError still propagates).
        """
        timeout = self.config.guardrails.total_timeout_per_item
        deadline = time.perf_counter() + timeout if timeout is not None else None
        try:
            result = await self._host._process_item_with_retries(work_item, worker_id, deadline)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            result = await self.build_failure_result(work_item, e, worker_id)
        result.submission_index = work_item.submission_index
        return result

    async def build_failure_result(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        e: Exception,
        worker_id: int = 0,
    ) -> WorkItemResult[TOutput, TContext]:
        """Build the failed result for an exhausted/unhandled error.

        Relocated verbatim from the worker loop so both the batch worker and the
        queue-less surfaces produce identical failure results and ITEM_FAILED
        events.
        """
        # All retries exhausted or unhandled exception
        # Create a failed result so the item is recorded

        # Extract token usage from exception if available
        failed_tokens = {}
        if hasattr(e, "__dict__") and "_failed_token_usage" in e.__dict__:
            failed_tokens = e.__dict__["_failed_token_usage"]
        admission_wait_seconds = float(
            getattr(e, "__dict__", {}).get(_ADMISSION_WAIT_EXCEPTION_KEY, 0.0)
        )
        timing = getattr(e, "__dict__", {}).get(_TIMING_EXCEPTION_KEY)
        if not isinstance(timing, WorkItemTiming):
            timing = WorkItemTiming()
        classifier = self._classifier_resolver.resolve(work_item.strategy)
        error_info = _classify_error(e, classifier)

        token_msg = ""
        if failed_tokens.get("total_tokens", 0) > 0:
            token_msg = f" (consumed {failed_tokens['total_tokens']} tokens across all attempts)"

        logger.error(
            f"[FAIL]Worker {worker_id} failed to process {work_item.item_id} after all retries: "
            f"{type(e).__name__}: {str(e)[:ERROR_MESSAGE_MAX_LENGTH]}{token_msg}"
        )

        # Controlled guardrail termination is already the final framework
        # outcome. Do not let a recovery hook delay or rewrite it.
        middleware_result = (
            None
            if isinstance(e, (ItemDeadlineExceeded, BatchDeadlineExceeded, BatchAbortedError))
            else await self._run_middlewares_on_error(work_item, e)
        )
        result: WorkItemResult[TOutput, TContext]
        if middleware_result is not None:
            result = middleware_result
        else:
            # Annotated above: ty infers unannotated constructions against
            # the PEP 696 defaults ([Any, None]) instead of the executor's
            # type parameters.
            result = WorkItemResult(
                item_id=work_item.item_id,
                success=False,
                error=f"{type(e).__name__}: {str(e)[:ERROR_MESSAGE_MAX_LENGTH]}",
                context=work_item.context,
                token_usage=cast(TokenUsage, failed_tokens),
                exception=_detach_traceback(e),
                admission_wait_seconds=admission_wait_seconds,
                timing=timing,
                error_category=error_info.error_category,
            )
        result.admission_wait_seconds = admission_wait_seconds
        result.timing = timing
        if not result.success and result.error_category is None:
            result.error_category = error_info.error_category

        # Emit ITEM_FAILED here too. Items that exhaust retries reach
        # this fallback (the exception propagates out of
        # _process_item_with_retries) rather than the non-retryable
        # branch in _handle_execution_error, so without this emit a
        # MetricsObserver would undercount failures vs BatchResult.
        if not result.success:
            await self._emit_event(
                ProcessingEvent.ITEM_FAILED,
                {
                    "item_id": work_item.item_id,
                    "error_type": type(e).__name__,
                    "error_category": result.error_category,
                },
            )

        return result

    async def _process_item_with_retries(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        deadline: float | None = None,
    ) -> WorkItemResult[TOutput, TContext]:
        """Wrapper that applies retry logic and strategy lifecycle."""
        item_started = time.perf_counter()
        attempt_timings: list[AttemptTiming] = []
        try_number = 0
        # Track cumulative token usage across all failed attempts
        cumulative_failed_tokens = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }

        # Get the strategy
        strategy = self._get_strategy(work_item)
        classifier = self._classifier_resolver.resolve(strategy)

        # Create retry state for this work item (v0.3.0)
        # This state persists across all retry attempts for multi-stage strategies
        retry_state = RetryState()
        if deadline is None and self.config.guardrails.total_timeout_per_item is not None:
            deadline = time.perf_counter() + self.config.guardrails.total_timeout_per_item
        retry_state.set(_TOTAL_DEADLINE_KEY, deadline)

        # Ensure strategy is prepared (framework ensures this is called only once per unique strategy instance)
        # (v0.4.0: cleanup now happens in __aexit__, not per-item)
        try:
            await await_with_guardrails(
                self._ensure_strategy_prepared(strategy),
                item_deadline=deadline,
                item_id=work_item.item_id,
                abort_controller=self._abort_controller,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[FAIL]Strategy prepare() failed for {work_item.item_id}: {e}")
            raise

        # Two independent counters: `attempt` is the *logical* attempt number
        # (what execute()/on_error() see, and what model-escalation strategies
        # key off). `rate_limit_retries` bounds throttling retries separately.
        # Rate-limit errors retry the SAME logical attempt — they don't consume
        # the max_attempts budget — so a busy endpoint can't trigger escalation.
        attempt = 1
        rate_limit_retries = 0
        max_attempts = self.config.retry.max_attempts
        max_rate_limit_retries = self.config.retry.max_rate_limit_retries

        while True:
            try:
                remaining_seconds(deadline, item_id=work_item.item_id)
                if self._abort_controller is not None:
                    self._abort_controller.raise_if_aborted(work_item.item_id)
            except (ItemDeadlineExceeded, BatchDeadlineExceeded, BatchAbortedError) as exc:
                self._attach_failed_tokens(exc, cumulative_failed_tokens)
                exc.__dict__[_TIMING_EXCEPTION_KEY] = _work_item_timing(
                    item_started, attempt_timings
                )
                raise
            try_number += 1
            try_started = time.perf_counter()
            for key in (
                _LAST_ADMISSION_KEY,
                _LAST_STARTUP_RAMP_KEY,
                _LAST_EXECUTION_KEY,
                _LAST_PROVIDER_KEY,
                _LAST_COOLDOWN_KEY,
                _LAST_QUOTA_WAIT_KEY,
                _LAST_ESTIMATED_INPUT_KEY,
                _LAST_ESTIMATED_OUTPUT_KEY,
                _LAST_RESERVED_TOKENS_KEY,
                _LAST_REPORTED_TOKENS_KEY,
                _LAST_RECONCILIATION_DELTA_KEY,
                _LAST_QUOTA_SCOPE_KEY,
                _LAST_TIMEOUT_KEY,
                _LAST_ERROR_CATEGORY_KEY,
            ):
                retry_state.delete(key)
            retry_state.set(_PHYSICAL_TRY_KEY, try_number)
            try:
                # Through the host so a processor subclass override takes effect.
                result = await self._host._process_item(
                    work_item,
                    worker_id,
                    attempt_number=attempt,
                    strategy=strategy,
                    retry_state=retry_state,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Accumulate token usage across every attempt (including
                # rate-limit retries) so users see the true cost of a failure.
                attempt_tokens = self._host._extract_token_usage(e)
                self._token_extractor.accumulate(cumulative_failed_tokens, attempt_tokens)
                admission_wait_seconds = float(retry_state.get(_ADMISSION_WAIT_STATE_KEY, 0.0))
                if hasattr(e, "__dict__"):
                    e.__dict__[_ADMISSION_WAIT_EXCEPTION_KEY] = admission_wait_seconds

                error_info = _classify_error(e, classifier)
                error_snippet = str(e)[:ERROR_MESSAGE_MAX_LENGTH]
                error_type = type(e).__name__
                attempt_timing = _attempt_timing(
                    retry_state,
                    attempt=attempt,
                    try_number=try_number,
                    total_seconds=max(0.0, time.perf_counter() - try_started),
                    success=False,
                    error_type=error_type,
                    error_category=error_info.error_category,
                )
                attempt_timings.append(attempt_timing)

                def attach_timing(exception: Exception) -> None:
                    if hasattr(exception, "__dict__"):
                        exception.__dict__[_TIMING_EXCEPTION_KEY] = _work_item_timing(
                            item_started, attempt_timings
                        )

                if not error_info.is_retryable:
                    # Surface an operator hint (e.g. a 402 insufficient-balance
                    # remediation) at WARNING so a misconfiguration doesn't read
                    # like a generic API/code bug; otherwise a quiet debug line.
                    if error_info.hint:
                        logger.warning(
                            f"[FAIL]Non-retryable error for {work_item.item_id}: {error_info.hint}"
                        )
                    else:
                        logger.debug(f"Error not retryable: {error_type}")
                    self._attach_failed_tokens(e, cumulative_failed_tokens)
                    attach_timing(e)
                    raise

                if error_info.is_rate_limit:
                    # Rate limits do NOT consume the max_attempts budget — they're
                    # "wait and try again", not a failed attempt. The coordinated
                    # cooldown already ran inside _handle_rate_limit(), so we retry
                    # the SAME logical attempt immediately. A separate counter
                    # bounds this so a permanently-throttled endpoint can't hang.
                    rate_limit_retries += 1
                    if rate_limit_retries > max_rate_limit_retries:
                        token_summary = self._cumulative_token_summary(cumulative_failed_tokens)
                        logger.error(
                            f"[FAIL]EXCEEDED {max_rate_limit_retries} RATE-LIMIT RETRIES "
                            f"for {work_item.item_id}:\n"
                            f"  Last error type: {error_type}\n"
                            f"  Last error message: "
                            f"{str(e)[:ERROR_MESSAGE_DETAILED_LENGTH]}{token_summary}"
                        )
                        exhausted = RateLimitRetriesExceeded(
                            f"Exceeded {max_rate_limit_retries} rate-limit retries for "
                            f"{work_item.item_id} (last error: {error_type}: {error_snippet})",
                            item_id=work_item.item_id,
                            rate_limit_retries=rate_limit_retries,
                        )
                        exhausted.__dict__["_failed_token_usage"] = cumulative_failed_tokens
                        exhausted.__dict__[_ADMISSION_WAIT_EXCEPTION_KEY] = admission_wait_seconds
                        attach_timing(exhausted)
                        raise exhausted from e
                    logger.warning(
                        f"[WARN]Rate-limit retry {rate_limit_retries} for {work_item.item_id} "
                        f"(attempt {attempt}/{max_attempts} budget unchanged): "
                        f"{error_type} - {error_snippet}. "
                        f"Retrying immediately (cooldown already applied)..."
                    )
                    continue  # attempt unchanged; no extra backoff (cooldown done)

                # --- Non-rate-limit retryable error: consumes the budget. ---
                if attempt >= max_attempts:
                    token_summary = self._cumulative_token_summary(cumulative_failed_tokens)
                    logger.error(
                        f"[FAIL]ALL {max_attempts} ATTEMPTS EXHAUSTED "
                        f"for {work_item.item_id}:\n"
                        f"  Final error type: {error_type}\n"
                        f"  Final error message: "
                        f"{str(e)[:ERROR_MESSAGE_DETAILED_LENGTH]}{token_summary}"
                    )
                    self._attach_failed_tokens(e, cumulative_failed_tokens)
                    attach_timing(e)
                    raise

                # Validation errors retry immediately — the strategy adjusts on
                # retry; other transient errors get exponential backoff keyed off
                # the logical attempt number. (PydanticAI wraps validation errors
                # in UnexpectedModelBehavior.)
                error_msg_for_check = str(e)
                is_validation_error = (
                    "validation" in error_type.lower()
                    or "parse" in error_type.lower()
                    or "unexpectedmodelbehavior" in error_type.lower()
                    or "result validation" in error_msg_for_check.lower()
                    or error_info.error_category == "validation_error"
                )

                if is_validation_error:
                    wait_time = 0.0
                else:
                    wait_time = min(
                        self.config.retry.initial_wait
                        * (self.config.retry.exponential_base ** (attempt - 1)),
                        self.config.retry.max_wait,
                    )
                    if self.config.retry.jitter:
                        import random

                        # Jitter to 50-100% of the computed wait to spread retries.
                        wait_time = wait_time * (0.5 + random.random() * 0.5)

                retry_desc = "immediately" if wait_time == 0 else f"in {wait_time:.1f}s"
                logger.warning(
                    f"[WARN]Attempt {attempt}/{max_attempts} failed for "
                    f"{work_item.item_id}: {error_type} - {error_snippet}. "
                    f"Retrying {retry_desc}..."
                )

                attempt += 1
                if wait_time > 0:
                    backoff_started = time.perf_counter()
                    try:
                        await await_with_guardrails(
                            asyncio.sleep(wait_time),
                            item_deadline=deadline,
                            item_id=work_item.item_id,
                            abort_controller=self._abort_controller,
                        )
                    except (ItemDeadlineExceeded, BatchDeadlineExceeded, BatchAbortedError) as exc:
                        attempt_timing.retry_backoff_seconds = max(
                            0.0, time.perf_counter() - backoff_started
                        )
                        self._attach_failed_tokens(exc, cumulative_failed_tokens)
                        attach_timing(exc)
                        raise
                    else:
                        attempt_timing.retry_backoff_seconds = max(
                            0.0, time.perf_counter() - backoff_started
                        )
            else:
                # _process_item returned without raising (success, or a result
                # produced by middleware / non-retryable handling). Fold in the
                # tokens consumed by any earlier failed attempts so cost
                # reporting is aggregated across retries, not just the final
                # attempt (see README "aggregated across retries").
                self._merge_failed_tokens(result, cumulative_failed_tokens)
                result.admission_wait_seconds = float(
                    retry_state.get(_ADMISSION_WAIT_STATE_KEY, 0.0)
                )
                final_error_type: str | None = None
                if not result.success and result.error:
                    final_error_type = result.error.split(":", 1)[0]
                category_value = retry_state.get(_LAST_ERROR_CATEGORY_KEY)
                if (
                    not result.success
                    and result.error_category is None
                    and isinstance(category_value, str)
                ):
                    result.error_category = category_value
                attempt_timings.append(
                    _attempt_timing(
                        retry_state,
                        attempt=attempt,
                        try_number=try_number,
                        total_seconds=max(0.0, time.perf_counter() - try_started),
                        success=result.success,
                        error_type=final_error_type,
                        error_category=(
                            category_value if isinstance(category_value, str) else None
                        ),
                    )
                )
                result.timing = _work_item_timing(item_started, attempt_timings)
                result.admission_wait_seconds = result.timing.admission_wait_seconds
                return result

    @staticmethod
    def _attach_failed_tokens(exception: Exception, tokens: dict[str, int]) -> None:
        """Stamp cumulative failed-attempt tokens onto an exception for the worker
        to surface in the failed ``WorkItemResult``. No-op if the exception has
        no writable ``__dict__``."""
        if hasattr(exception, "__dict__"):
            exception.__dict__["_failed_token_usage"] = tokens

    @staticmethod
    def _cumulative_token_summary(tokens: dict[str, int]) -> str:
        """Format the cross-attempt token total for final-failure logs (or '')."""
        if tokens.get("total_tokens", 0) > 0:
            return f"\n  Total tokens consumed across all attempts: {tokens['total_tokens']}"
        return ""

    @staticmethod
    def _merge_failed_tokens(
        result: WorkItemResult[TOutput, TContext], failed_tokens: dict[str, int]
    ) -> None:
        """Add tokens consumed by prior failed attempts into ``result.token_usage``.

        Mutates ``result.token_usage`` in place. Keys that would stay zero and
        weren't already present are left out so a clean success keeps its tidy
        ``{input, output, total}`` shape.
        """
        existing = cast("dict[str, int]", result.token_usage)
        usage: dict[str, int] = {}
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
            combined = existing.get(key, 0) + failed_tokens.get(key, 0)
            if combined or key in existing:
                usage[key] = combined
        result.token_usage = cast(TokenUsage, usage)

    def _get_strategy(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMCallStrategy[TOutput]:
        """Get the LLM call strategy for this work item."""
        return work_item.strategy

    async def _process_item(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int = 1,
        strategy: LLMCallStrategy[TOutput] | None = None,
        retry_state: RetryState | None = None,
    ) -> WorkItemResult[TOutput, TContext]:
        """Process a single work item using the provided strategy."""
        start_time = time.time()
        deadline_value = retry_state.get(_TOTAL_DEADLINE_KEY) if retry_state is not None else None
        deadline = float(deadline_value) if isinstance(deadline_value, (int, float)) else None

        # Store original item_id before middleware might return None
        original_item_id = work_item.item_id
        classifier: ErrorClassifier | None = None
        admission_state: ScopeAdmissionState | None = None
        known_provider_token_usage: TokenUsage | None = None

        # Skip building the event payload entirely when nobody is listening.
        if self._events.observers:
            await self._emit_event(
                ProcessingEvent.ITEM_STARTED,
                {"item_id": original_item_id, "worker_id": worker_id},
            )

        try:
            # Run before middlewares
            processed_item = await await_with_guardrails(
                self._run_middlewares_before(work_item),
                item_deadline=deadline,
                item_id=work_item.item_id,
                abort_controller=self._abort_controller,
            )
            if processed_item is None:
                logger.debug("Skipping %s (filtered by middleware)", original_item_id)
                return WorkItemResult(
                    item_id=original_item_id,
                    success=False,
                    error="Skipped by middleware",
                    context=work_item.context,
                )
            work_item = processed_item

            # Middleware may replace a work item's strategy. Admission and
            # classification always follow the effective strategy identity.
            effective_strategy = work_item.strategy
            if strategy is not effective_strategy:
                await await_with_guardrails(
                    self._ensure_strategy_prepared(effective_strategy),
                    item_deadline=deadline,
                    item_id=work_item.item_id,
                    abort_controller=self._abort_controller,
                )
            strategy = effective_strategy
            classifier = self._classifier_resolver.resolve(strategy)
            admission_state = self._admission_registry.resolve(strategy)

            # Execute the strategy
            if attempt_number > 1:
                logger.debug(
                    "[Worker %s] Retry attempt %s for %s",
                    worker_id,
                    attempt_number,
                    work_item.item_id,
                )
            # Resolved to a float in ProcessorConfig.__post_init__.
            attempt_timeout = cast(float, self.config.attempt_timeout)
            logger.debug(
                "[STRATEGY] Starting strategy.execute() for %s (attempt %s, timeout=%ss)",
                work_item.item_id,
                attempt_number,
                attempt_timeout,
            )
            # Ensure strategy is not None (it shouldn't be since we always pass it)
            if strategy is None:
                raise RuntimeError("Strategy is None in _process_item - this should not happen")

            # Dry-run mode has no physical provider attempt, so it bypasses
            # cooldown, request quota, and provider-capacity admission.
            if self.config.dry_run:
                logger.debug("[DRY-RUN] Skipping API call for %s", work_item.item_id)
                llm_start_time = time.time()
                execution_started = time.perf_counter()
                try:
                    output, token_usage = await await_with_guardrails(
                        strategy.dry_run(work_item.prompt),
                        item_deadline=deadline,
                        item_id=work_item.item_id,
                        abort_controller=self._abort_controller,
                    )
                finally:
                    if retry_state is not None:
                        retry_state.set(
                            _LAST_EXECUTION_KEY,
                            max(0.0, time.perf_counter() - execution_started),
                        )
                response_metadata = None
            else:
                # Every physical try re-enters scoped admission. RPM is never
                # held during cooldown and provider capacity is never held
                # while waiting for RPM.
                await self.wait_for_capacity(
                    admission_state=admission_state,
                    deadline=deadline,
                    retry_state=retry_state,
                    item_id=work_item.item_id,
                )
                estimate: TokenEstimate | None = None
                if admission_state.quota_gate.tpm_enabled:
                    estimate = await self._resolve_token_estimate(
                        prompt=work_item.prompt,
                        strategy=strategy,
                        attempt=attempt_number,
                        retry_state=retry_state,
                        deadline=deadline,
                        item_id=work_item.item_id,
                    )
                if admission_state.quota_gate.enabled:
                    quota_wait_started = time.perf_counter()
                    reservation_task = asyncio.create_task(
                        admission_state.quota_gate.reserve(estimate)
                    )
                    try:
                        reservation = await await_with_guardrails(
                            reservation_task,
                            item_deadline=deadline,
                            item_id=work_item.item_id,
                            abort_controller=self._abort_controller,
                        )
                    except BaseException:
                        # Guardrail awaiting uses a child task. If cancellation
                        # lands after that child was granted quota but before
                        # this task receives it, refund the unseen grant.
                        if reservation_task.done() and not reservation_task.cancelled():
                            try:
                                unseen_reservation = reservation_task.result()
                            except BaseException:
                                pass
                            else:
                                unseen_reservation.finalize()
                        if retry_state is not None:
                            retry_state.set(
                                _LAST_QUOTA_WAIT_KEY,
                                max(0.0, time.perf_counter() - quota_wait_started),
                            )
                            retry_state.set(_LAST_QUOTA_SCOPE_KEY, admission_state.ordinal)
                        raise
                else:
                    # Disabled mode is an allocation-light synchronous fast
                    # path: no waiter or task is created.
                    reservation = await admission_state.quota_gate.reserve(estimate)
                try:
                    if admission_state.quota_gate.enabled:
                        self._store_quota_timing(
                            retry_state,
                            reservation,
                            admission_state.ordinal,
                        )
                        await self._record_quota_admitted(
                            reservation,
                            admission_state,
                            work_item,
                            worker_id,
                            attempt_number,
                            retry_state,
                        )
                    async with self._capacity_limiter.admit(
                        strategy,
                        deadline=deadline,
                        abort_controller=self._abort_controller,
                        item_id=work_item.item_id,
                    ) as admission:
                        previous_wait = (
                            float(retry_state.get(_ADMISSION_WAIT_STATE_KEY, 0.0))
                            if retry_state is not None
                            else 0.0
                        )
                        total_admission_wait = previous_wait + admission.wait_seconds
                        if retry_state is not None:
                            retry_state.set(_ADMISSION_WAIT_STATE_KEY, total_admission_wait)
                            retry_state.set(_LAST_ADMISSION_KEY, admission.wait_seconds)
                            retry_state.set(
                                _LAST_STARTUP_RAMP_KEY,
                                _state_float(retry_state, _LAST_STARTUP_RAMP_KEY)
                                + admission.startup_ramp_wait_seconds,
                            )
                        if self._events.observers:
                            await self._emit_event(
                                ProcessingEvent.ITEM_ADMITTED,
                                {
                                    "item_id": work_item.item_id,
                                    "worker_id": worker_id,
                                    "attempt": attempt_number,
                                    "wait_seconds": admission.wait_seconds,
                                    "capacity": admission.capacity,
                                    "startup_ramp_wait_seconds": (
                                        admission.startup_ramp_wait_seconds
                                    ),
                                },
                            )

                        llm_start_time = time.time()
                        execution_started = time.perf_counter()
                        # _unpack_strategy_result accepts both legacy 2-tuples
                        # and current 3-tuples (output, tokens, metadata).
                        try:
                            try:
                                remaining = remaining_seconds(deadline, item_id=work_item.item_id)
                                effective_timeout = attempt_timeout
                                if remaining is not None:
                                    effective_timeout = min(effective_timeout, remaining)
                                raw_result = await await_with_guardrails(
                                    strategy.execute(
                                        work_item.prompt,
                                        attempt_number,
                                        effective_timeout,
                                        retry_state,
                                    ),
                                    item_deadline=deadline,
                                    item_id=work_item.item_id,
                                    abort_controller=self._abort_controller,
                                    operation_timeout=attempt_timeout,
                                    active_provider=True,
                                    on_start=reservation.mark_provider_started,
                                )
                            except ItemDeadlineExceeded:
                                if retry_state is not None:
                                    retry_state.set(
                                        _LAST_TIMEOUT_KEY, "framework_total_item_timeout"
                                    )
                                raise
                            except (BatchDeadlineExceeded, BatchAbortedError):
                                raise
                            except (TimeoutError, asyncio.TimeoutError) as timeout_exc:
                                elapsed = time.time() - llm_start_time
                                if retry_state is not None:
                                    retry_state.set(
                                        _LAST_TIMEOUT_KEY, "framework_execution_timeout"
                                    )
                                logger.error(
                                    f"⏱ FRAMEWORK TIMEOUT for {work_item.item_id} "
                                    f"after {elapsed:.1f}s (limit: {attempt_timeout}s, "
                                    f"attempt {attempt_number}). Consider increasing "
                                    "config.attempt_timeout if this error persists."
                                )
                                framework_timeout = FrameworkTimeoutError(
                                    f"Framework timeout after {elapsed:.1f}s "
                                    f"(limit: {attempt_timeout}s)",
                                    item_id=work_item.item_id,
                                    elapsed=elapsed,
                                    timeout_limit=attempt_timeout,
                                )
                                if (
                                    hasattr(timeout_exc, "__dict__")
                                    and "_failed_token_usage" in timeout_exc.__dict__
                                ):
                                    framework_timeout.__dict__["_failed_token_usage"] = (
                                        timeout_exc.__dict__["_failed_token_usage"]
                                    )
                                raise framework_timeout from timeout_exc
                            output, token_usage, response_metadata = _unpack_strategy_result(
                                raw_result
                            )
                            usage_observation = self._token_extractor.observe_result(token_usage)
                            if (
                                usage_observation.known
                                and usage_observation.reported_tokens is not None
                            ):
                                normalized_usage = cast(
                                    "dict[str, int]", usage_observation.usage
                                ).copy()
                                normalized_usage["total_tokens"] = usage_observation.reported_tokens
                                known_provider_token_usage = cast(TokenUsage, normalized_usage)
                        except BaseException as provider_error:
                            if reservation.provider_started and not reservation.finalized:
                                if admission_state.quota_gate.tpm_enabled:
                                    observation = (
                                        self._observe_exception_usage(provider_error)
                                        if isinstance(provider_error, Exception)
                                        else TokenUsageObservation(
                                            usage=cast(TokenUsage, {}),
                                            known=False,
                                            reported_tokens=None,
                                        )
                                    )
                                    finalization = (
                                        reservation.reconcile(observation.reported_tokens)
                                        if observation.known
                                        and observation.reported_tokens is not None
                                        else reservation.finalize_unknown()
                                    )
                                else:
                                    finalization = reservation.finalize_request_only()
                                if finalization is not None:
                                    await self._record_quota_finalization(
                                        reservation=reservation,
                                        finalization=finalization,
                                        state=admission_state,
                                        work_item=work_item,
                                        worker_id=worker_id,
                                        attempt_number=attempt_number,
                                        retry_state=retry_state,
                                    )
                            raise
                        else:
                            if admission_state.quota_gate.tpm_enabled:
                                finalization = (
                                    reservation.reconcile(usage_observation.reported_tokens)
                                    if usage_observation.known
                                    and usage_observation.reported_tokens is not None
                                    else reservation.finalize_unknown()
                                )
                            else:
                                finalization = reservation.finalize_request_only()
                            if finalization is not None:
                                await self._record_quota_finalization(
                                    reservation=reservation,
                                    finalization=finalization,
                                    state=admission_state,
                                    work_item=work_item,
                                    worker_id=worker_id,
                                    attempt_number=attempt_number,
                                    retry_state=retry_state,
                                )
                        finally:
                            if retry_state is not None:
                                retry_state.set(
                                    _LAST_EXECUTION_KEY,
                                    max(0.0, time.perf_counter() - execution_started),
                                )
                finally:
                    if not reservation.finalized:
                        finalization = reservation.finalize_unknown()
                        if finalization is not None:
                            await self._record_quota_finalization(
                                reservation=reservation,
                                finalization=finalization,
                                state=admission_state,
                                work_item=work_item,
                                worker_id=worker_id,
                                attempt_number=attempt_number,
                                retry_state=retry_state,
                            )

            llm_duration = time.time() - llm_start_time
            logger.debug(
                "[STRATEGY] Completed strategy.execute() for %s in %.1fs",
                work_item.item_id,
                llm_duration,
            )

            # Log success after previous failures
            if attempt_number > 1:
                logger.debug(
                    "SUCCESS on attempt %s for %s (after %s failure(s), took %.1fs)",
                    attempt_number,
                    work_item.item_id,
                    attempt_number - 1,
                    llm_duration,
                )

            # Log first few results for debugging (lazy: the big banner string is
            # only built when DEBUG is actually enabled).
            if self._stats.succeeded < 3:
                logger.debug(
                    "\n%s\nRESULT for %s:\n%s\n%s\n%s",
                    "=" * 80,
                    work_item.item_id,
                    "=" * 80,
                    output,
                    "=" * 80,
                )

            # Create result (annotated: ty infers unannotated constructions
            # against the PEP 696 defaults instead of the executor's params)
            work_result: WorkItemResult[TOutput, TContext] = WorkItemResult(
                item_id=work_item.item_id,
                success=True,
                output=output,
                context=work_item.context,
                token_usage=token_usage,
                metadata=response_metadata,
            )

            # Run after middlewares
            work_result = await self._run_middlewares_after(work_result)
            work_result.admission_wait_seconds = (
                float(retry_state.get(_ADMISSION_WAIT_STATE_KEY, 0.0))
                if retry_state is not None
                else 0.0
            )

            # Skip the duration calc + payload dict when nobody is observing.
            if self._events.observers:
                duration = time.time() - start_time
                await self._emit_event(
                    ProcessingEvent.ITEM_COMPLETED,
                    {
                        "item_id": work_item.item_id,
                        "duration": duration,
                        "tokens": token_usage.get("total_tokens", 0),
                        "admission_wait_seconds": work_result.admission_wait_seconds,
                        "structured_output_recovered": (work_result.structured_output_recovered),
                        "structured_output_recovery_reason": (
                            work_result.structured_output_recovery_reason
                        ),
                        "structured_output_retries_avoided": (
                            work_result.structured_output_retries_avoided
                        ),
                    },
                )

            # Only a successful live provider attempt advances this scope's
            # cooldown recovery. Dry-run must remain admission-state-neutral.
            if not self.config.dry_run:
                assert admission_state is not None
                await admission_state.cooldown.on_item_success()

            return work_result

        except asyncio.CancelledError:
            raise
        except Exception as e:
            if (
                known_provider_token_usage is not None
                and hasattr(e, "__dict__")
                and "_failed_token_usage" not in e.__dict__
            ):
                e.__dict__["_failed_token_usage"] = dict(known_provider_token_usage)
            # Notify strategy about the error before handling it
            # This allows strategy to adjust behavior for next retry (v0.3.0: now includes retry_state)
            if strategy is not None and not isinstance(e, TokenEstimationError):
                try:
                    await await_with_guardrails(
                        strategy.on_error(e, attempt_number, retry_state),
                        item_deadline=deadline,
                        item_id=work_item.item_id,
                        abort_controller=self._abort_controller,
                    )
                except asyncio.CancelledError:
                    raise
                except (
                    ItemDeadlineExceeded,
                    BatchDeadlineExceeded,
                    BatchAbortedError,
                ) as guard_exc:
                    failed_tokens = self._host._extract_token_usage(e)
                    if failed_tokens:
                        guard_exc.__dict__["_failed_token_usage"] = failed_tokens
                    raise
                except Exception as callback_error:
                    # Log but don't fail if on_error callback has bugs
                    logger.warning(
                        f"Strategy.on_error callback failed for {work_item.item_id}: {callback_error}"
                    )

            # Delegate error handling to separate method. Rate-limit handling
            # includes the coordinated cooldown wait; record it separately from
            # provider execution and retry backoff.
            effective_strategy = strategy or work_item.strategy
            if classifier is None:
                classifier = self._classifier_resolver.resolve(effective_strategy)
            if admission_state is None:
                admission_state = self._admission_registry.resolve(effective_strategy)
            error_info = _classify_error(e, classifier)
            if retry_state is not None:
                retry_state.set(_LAST_ERROR_CATEGORY_KEY, error_info.error_category)
            if (
                retry_state is not None
                and not isinstance(e, FrameworkTimeoutError)
                and "timeout" in type(e).__name__.lower()
            ):
                retry_state.set(_LAST_TIMEOUT_KEY, "provider_or_transport_timeout")
            cooldown_started = time.perf_counter()
            try:
                return await await_with_guardrails(
                    self._host._handle_execution_error(e, work_item, worker_id, attempt_number),
                    item_deadline=deadline,
                    item_id=work_item.item_id,
                    abort_controller=self._abort_controller,
                )
            except (ItemDeadlineExceeded, BatchDeadlineExceeded, BatchAbortedError) as guard_exc:
                failed_tokens = self._host._extract_token_usage(e)
                if failed_tokens:
                    guard_exc.__dict__["_failed_token_usage"] = failed_tokens
                raise
            finally:
                if retry_state is not None and error_info.is_rate_limit:
                    retry_state.set(
                        _LAST_COOLDOWN_KEY,
                        _state_float(retry_state, _LAST_COOLDOWN_KEY)
                        + max(0.0, time.perf_counter() - cooldown_started),
                    )

    async def _handle_execution_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
    ) -> WorkItemResult[TOutput, TContext]:
        """
        Handle exceptions from LLM execution.

        This method classifies errors, extracts token usage, handles rate limits,
        and determines whether errors should be retried or treated as permanent failures.

        Args:
            exception: The exception that was raised during execution
            work_item: The work item being processed
            worker_id: ID of the worker processing this item
            attempt_number: Current attempt number (for logging)

        Returns:
            WorkItemResult for permanent failures

        Raises:
            RateLimitException: If rate limit detected (for re-queueing)
            Exception: If error is retryable (for retry logic to handle)
        """
        # Try to extract token usage from failed LLM calls using robust extraction
        # Even if validation fails, the LLM consumed tokens
        failed_token_usage = self._host._extract_token_usage(exception)
        if failed_token_usage and failed_token_usage.get("total_tokens", 0) > 0:
            logger.debug(
                f"Extracted token usage from failed attempt for {work_item.item_id}: "
                f"{failed_token_usage['total_tokens']} tokens"
            )

        strategy = work_item.strategy
        classifier = self._classifier_resolver.resolve(strategy)
        admission_state = self._admission_registry.resolve(strategy)
        error_info = _classify_error(exception, classifier)

        # Check if it's a rate limit
        if error_info.is_rate_limit:
            # Update stats (thread-safe)
            async with self._stats_lock:
                self._stats.rate_limit_count += 1

            await self._emit_event(
                ProcessingEvent.RATE_LIMIT_HIT,
                {
                    "item_id": work_item.item_id,
                    "worker_id": worker_id,
                    "quota_scope_id": admission_state.ordinal,
                    "strategy_type": type(strategy).__name__,
                },
            )

            # Handle rate limit (cooldown) - this will pause all workers.
            # Pass the classifier's suggested_wait (e.g. a parsed Retry-After)
            # as a floor on the cooldown duration.
            observed_generation = admission_state.cooldown.current_generation
            await self._handle_rate_limit(
                admission_state,
                worker_id,
                observed_generation,
                error_info.suggested_wait,
                type(strategy).__name__,
            )

            # Re-raise the original exception to trigger retry logic
            # The retry loop will increment attempt and try again after cooldown
            raise exception

        # If error is retryable, re-raise to trigger retry in _process_item_with_retries
        # Note: Cache invalidation is automatic because retries use different temperatures,
        # which creates different cache keys and bypasses any cached bad responses
        if error_info.is_retryable:
            self._log_retryable_error(exception, work_item, attempt_number, failed_token_usage)
            raise exception

        # Try middleware error handlers
        middleware_result = (
            None
            if isinstance(exception, TokenEstimationError)
            else await self._run_middlewares_on_error(work_item, exception)
        )
        if middleware_result is not None:
            return middleware_result

        # Log non-retryable error with full details
        error_name = type(exception).__name__
        error_msg = str(exception)

        token_summary = ""
        if failed_token_usage:
            token_summary = f"\n  Tokens consumed: {failed_token_usage.get('total_tokens', 0)}"

        logger.error(
            f"[FAIL]PERMANENT FAILURE for {work_item.item_id}:\n"
            f"  Error type: {error_name}\n"
            f"  Error message: {error_msg[:ERROR_MESSAGE_DETAILED_LENGTH]}\n"
            f"  This error will NOT be retried (not retryable){token_summary}"
        )

        await self._emit_event(
            ProcessingEvent.ITEM_FAILED,
            {
                "item_id": work_item.item_id,
                "error_type": error_name,
                "error_category": error_info.error_category,
            },
        )

        return WorkItemResult(
            item_id=work_item.item_id,
            success=False,
            error=f"{error_name}: {error_msg[:ERROR_MESSAGE_DETAILED_LENGTH]}",
            context=work_item.context,
            token_usage=cast(TokenUsage, failed_token_usage),
            exception=_detach_traceback(exception),
            error_category=error_info.error_category,
        )
