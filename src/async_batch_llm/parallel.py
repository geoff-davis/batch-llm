"""Parallel batch processor"""

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Generic, cast

if TYPE_CHECKING:
    from types import TracebackType

from ._internal.admission import AdmissionRegistry
from ._internal.capacity import CapacityLimiter, warn_if_worker_capacity_exceeded
from ._internal.classifier_resolver import StrategyClassifierResolver
from ._internal.event_dispatcher import EventDispatcher
from ._internal.guardrails import (
    AbortCause,
    AbortController,
    BatchAdmissionStopped,
)
from ._internal.item_executor import ItemExecutor
from ._internal.rate_limit_coordinator import RateLimitCoordinator
from ._internal.strategy_lifecycle import StrategyLifecycle
from .artifacts import ArtifactError, ArtifactStore, ResumePolicy
from .base import (
    BatchProcessor,
    BatchTermination,
    LLMWorkItem,
    PostProcessorFunc,
    ProgressCallbackFunc,
    RetryState,
    TContext,
    TInput,
    TOutput,
    WorkItemResult,
)
from .core import ProcessorConfig
from .llm_strategies import LLMCallStrategy
from .middleware import Middleware
from .observers import ProcessingEvent, ProcessorObserver
from .strategies import (
    ErrorClassifier,
    ExponentialBackoffStrategy,
    RateLimitStrategy,
)
from .token_extractor import TokenExtractor

logger = logging.getLogger(__name__)

# Maximum length for error messages in logs / result payloads.
# Longer errors get truncated to keep logs readable.
ERROR_MESSAGE_MAX_LENGTH = 200
# Larger truncation used for detailed diagnostic logs (final attempt, validation traces).
ERROR_MESSAGE_DETAILED_LENGTH = 500

# Warn when the soft open-file limit leaves less than this many fds of headroom
# above max_workers (each in-flight request typically holds a socket).
_FD_LIMIT_HEADROOM = 128
# Docs section explaining the open-file-limit footgun and how to fix it.
_FD_LIMIT_DOCS_URL = (
    "https://geoff-davis.github.io/async-batch-llm/getting-started/"
    "#open-file-limits-and-high-concurrency"
)


def _warn_if_fd_limit_low(max_workers: int) -> None:
    """Warn if ``max_workers`` is close to the process's soft open-file limit.

    Each in-flight request typically holds a socket (a file descriptor), so a
    ``max_workers`` near the OS soft limit (``RLIMIT_NOFILE`` — 256 by default
    on macOS) risks ``OSError: [Errno 24] Too many open files`` once the
    connection pools, workers, and the app's own fds are all drawing from it.

    This only *warns* — raising the limit mutates process-global state, which is
    the operator's call, not the library's. No-op on non-Unix platforms (no
    ``RLIMIT_NOFILE``) or when the limit is unlimited / comfortably high.
    """
    try:
        import resource
    except ImportError:
        return  # non-Unix (e.g. Windows): no RLIMIT_NOFILE
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        return
    if soft == resource.RLIM_INFINITY or soft - max_workers >= _FD_LIMIT_HEADROOM:
        return

    import warnings

    warnings.warn(
        f"max_workers={max_workers} is close to this process's soft open-file "
        f"limit (RLIMIT_NOFILE={soft}); high-concurrency runs may fail with "
        f"'OSError: [Errno 24] Too many open files'. Raise the limit "
        f"(e.g. `ulimit -n {max(8192, max_workers * 4)}`, or resource.setrlimit "
        f"in-process) or lower max_workers. See {_FD_LIMIT_DOCS_URL}",
        UserWarning,
        stacklevel=3,
    )


class ParallelBatchProcessor(
    BatchProcessor[TInput, TOutput, TContext], Generic[TInput, TOutput, TContext]
):
    """
    Batch processor that executes items in parallel as individual agent calls.

    This refactored version uses:
    - Pluggable error classification (provider-agnostic)
    - Pluggable rate limit strategies
    - Middleware pipeline for extensibility
    - Observer pattern for monitoring
    - Configuration objects for easier setup
    """

    def __init__(
        self,
        max_workers: int | None = None,
        post_processor: PostProcessorFunc[TOutput, TContext] | None = None,
        timeout_per_item: float | None = None,
        rate_limit_cooldown: float | None = None,  # Deprecated, use config
        # New parameters
        config: ProcessorConfig | None = None,
        error_classifier: ErrorClassifier | None = None,
        rate_limit_strategy: RateLimitStrategy | None = None,
        middlewares: list[Middleware[TInput, TOutput, TContext]] | None = None,
        observers: list[ProcessorObserver] | None = None,
        progress_callback: "ProgressCallbackFunc | None" = None,
        artifact_store: ArtifactStore | None = None,
        resume: ResumePolicy = ResumePolicy.NONE,
    ):
        """
        Initialize the parallel batch processor.

        Args:
            max_workers: Maximum concurrent workers (deprecated, use config)
            post_processor: Optional async function called after each successful item
            timeout_per_item: Timeout per item in seconds (deprecated, use config)
            rate_limit_cooldown: Cooldown duration (deprecated, use config)
            config: Processor configuration object (recommended)
            error_classifier: Optional global classifier override. When omitted,
                each strategy's recommendation is resolved independently.
            rate_limit_strategy: Strategy for handling rate limits
            middlewares: List of middleware to apply
            observers: List of observers for events
            progress_callback: Optional callback(completed, total, current_item_id) for progress updates
        """
        import warnings

        # Emit deprecation warnings for legacy parameters
        if max_workers is not None:
            warnings.warn(
                "The 'max_workers' parameter is deprecated. "
                "Use ProcessorConfig(max_workers=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if timeout_per_item is not None:
            warnings.warn(
                "The 'timeout_per_item' parameter is deprecated. "
                "Use ProcessorConfig(attempt_timeout=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if rate_limit_cooldown is not None:
            warnings.warn(
                "The 'rate_limit_cooldown' parameter is deprecated. "
                "Use ProcessorConfig(rate_limit=RateLimitConfig(cooldown_seconds=...)) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Handle backward compatibility
        if config is None:
            from .core import RateLimitConfig

            config = ProcessorConfig(
                max_workers=max_workers or 5,
                attempt_timeout=timeout_per_item or 120.0,
                rate_limit=RateLimitConfig(cooldown_seconds=rate_limit_cooldown or 300.0),
            )
        else:
            # Override config with explicit legacy parameters if provided — but
            # build a NEW config via dataclasses.replace rather than mutating the
            # caller's object (and its nested RateLimitConfig). Callers may reuse
            # the same ProcessorConfig across processors and shouldn't see it
            # silently rewritten under them.
            import dataclasses

            overrides: dict = {}
            if max_workers is not None:
                overrides["max_workers"] = max_workers
            if timeout_per_item is not None:
                overrides["attempt_timeout"] = timeout_per_item
            if rate_limit_cooldown is not None:
                overrides["rate_limit"] = dataclasses.replace(
                    config.rate_limit, cooldown_seconds=rate_limit_cooldown
                )
            if overrides:
                config = dataclasses.replace(config, **overrides)

        config.validate()
        # Always an int after ProcessorConfig.__post_init__ resolution.
        resolved_max_workers = cast(int, config.max_workers)

        super().__init__(
            resolved_max_workers,
            post_processor,
            max_queue_size=config.max_queue_size,
            progress_callback=progress_callback,
            progress_callback_timeout=config.progress_callback_timeout,
            max_result_queue_size=config.max_result_queue_size,
        )
        self.config = config
        self.artifact_store = artifact_store
        self.resume = ResumePolicy(resume)
        self._abort_controller: AbortController | None = AbortController(
            config.guardrails.abort_mode
        )
        self._guardrails_started = False
        self._batch_timeout_task: asyncio.Task[None] | None = None

        # Diagnostic: high max_workers can outrun the OS open-file limit.
        _warn_if_fd_limit_low(resolved_max_workers)

        # Automatic classifiers are resolved per actual strategy. Keep the old
        # host-wide attribute only as a compatibility/debug alias.
        self._classifier_resolver = StrategyClassifierResolver(error_classifier)
        self.error_classifier: ErrorClassifier = self._classifier_resolver.compatibility_classifier
        self._capacity_checked_strategy_ids: set[int] = set()
        self.rate_limit_strategy = rate_limit_strategy or ExponentialBackoffStrategy(
            initial_cooldown=config.rate_limit.cooldown_seconds,
            max_cooldown=config.rate_limit.max_cooldown_seconds,
            backoff_multiplier=config.rate_limit.backoff_multiplier,
            slow_start_items=config.rate_limit.slow_start_items,
            slow_start_initial_delay=config.rate_limit.slow_start_initial_delay,
            slow_start_final_delay=config.rate_limit.slow_start_final_delay,
        )

        # Set up middleware and observers
        self.middlewares = middlewares or []
        self.observers = observers or []

        # Event + middleware dispatch. Delegates observer emits and the
        # middleware chain (before/after/on_error) to a stateless helper.
        self._events: EventDispatcher[TInput, TOutput, TContext] = EventDispatcher(
            observers=self.observers, middlewares=self.middlewares
        )

        self._admission_registry = AdmissionRegistry(
            rate_limit_strategy=self.rate_limit_strategy,
            events=self._events,
            max_requests_per_minute=config.max_requests_per_minute,
            max_tokens_per_minute=config.max_tokens_per_minute,
        )
        # A private compatibility coordinator serves direct legacy calls made
        # before a strategy exists. Once the first item is admitted, the old
        # `_rate_limit_coord` alias points at that item's real scoped state.
        self._compatibility_rate_limit_coord = RateLimitCoordinator(
            rate_limit_strategy=self.rate_limit_strategy,
            events=self._events,
        )
        self._rate_limit_coord = self._compatibility_rate_limit_coord
        self._compatibility_scope_bound = False

        # Thread-safety locks (_stats_lock / _results_lock) live on the base
        # class so both batch and streaming modes share them.

        # Strategy lifecycle management (v0.2.0, extracted in v0.7.0).
        # Tracks prepared strategies via a WeakSet so sharing one instance
        # across work items invokes prepare() exactly once.
        self._strategy_lifecycle: StrategyLifecycle[TOutput] = StrategyLifecycle()
        # Back-compat aliases used by existing private methods and tests.
        self._prepared_strategies = self._strategy_lifecycle._prepared
        self._strategy_lock = self._strategy_lifecycle._lock
        self._capacity_limiter = CapacityLimiter(
            config.max_provider_concurrency,
            max_workers=resolved_max_workers,
            startup_ramp=config.startup_ramp,
        )

        # Centralized token-usage extraction across all exception shapes.
        self._token_extractor = TokenExtractor()

        # Per-item execution engine (extracted so the single-call helper and the
        # gateway can share the exact same retry/rate-limit/token pipeline). The
        # processor is one host for the executor; it reads its deps live from
        # `self`, so it must be built here, before start(), and the worker keeps
        # calling the instance methods below (which delegate to it) so tests that
        # monkeypatch them still take effect.
        self._executor: ItemExecutor[TInput, TOutput, TContext] = ItemExecutor(self)

    @property
    def _strategies_cleaned_up(self) -> bool:
        return self._strategy_lifecycle._cleaned_up

    @_strategies_cleaned_up.setter
    def _strategies_cleaned_up(self, value: bool) -> None:
        self._strategy_lifecycle._cleaned_up = value

    # Back-compat attribute accessors for tests and subclasses that read
    # the rate-limit coordinator's state directly.

    @property
    def _rate_limit_event(self) -> asyncio.Event:
        return self._rate_limit_coord._rate_limit_event

    @property
    def _current_generation_event(self) -> asyncio.Event:
        return self._rate_limit_coord._current_generation_event

    @property
    def _rate_limit_lock(self) -> asyncio.Lock:
        return self._rate_limit_coord._lock

    @property
    def _in_cooldown(self) -> bool:
        return self._rate_limit_coord._in_cooldown

    @property
    def _cooldown_generation(self) -> int:
        return self._rate_limit_coord._cooldown_generation

    @property
    def _cooldown_complete_generation(self) -> int:
        return self._rate_limit_coord._cooldown_complete_generation

    @property
    def _consecutive_rate_limits(self) -> int:
        return self._rate_limit_coord._consecutive_rate_limits

    @property
    def _slow_start_active(self) -> bool:
        return self._rate_limit_coord._slow_start_active

    @property
    def _items_since_resume(self) -> int:
        return self._rate_limit_coord._items_since_resume

    async def _cleanup_strategies(self) -> None:
        """Delegate to StrategyLifecycle (see _internal/strategy_lifecycle.py)."""
        await self._strategy_lifecycle.cleanup_all()

    async def _ensure_strategy_prepared(self, strategy: LLMCallStrategy[TOutput]) -> None:
        """Delegate to StrategyLifecycle.ensure_prepared."""
        await self._strategy_lifecycle.ensure_prepared(strategy)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> bool:
        """
        Context manager exit - ensures cleanup of strategies and resources.

        Calls cleanup() on all prepared strategies, then delegates to parent cleanup.

        Args:
            exc_type: Exception type (if any exception occurred)
            exc_val: Exception value (if any exception occurred)
            exc_tb: Exception traceback (if any exception occurred)

        Returns:
            False to indicate exceptions should not be suppressed
        """
        # Stop workers before closing strategies or their artifact sink: a
        # cancelled/early stream may still have an in-flight terminal write.
        errors: list[BaseException] = []
        for cleanup in (self.cleanup, self._cleanup_strategies):
            try:
                await cleanup()
            except BaseException as error:
                errors.append(error)
        if self.artifact_store is not None:
            try:
                await self.artifact_store.close()
            except BaseException as error:
                errors.append(error)
        if errors and exc_val is None:
            raise errors[0]
        return False  # Don't suppress exceptions

    def _start_guardrail_run(self) -> None:
        if self._guardrails_started:
            return
        self._guardrails_started = True
        timeout = self.config.guardrails.batch_timeout
        if timeout is not None:
            self._batch_timeout_task = asyncio.create_task(self._expire_batch_after(timeout))

    async def _expire_batch_after(self, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
            await self._trip_abort(
                AbortCause(
                    kind="batch_timeout",
                    reason=f"Batch deadline exceeded after {timeout:g}s",
                    error_category="batch_deadline_exceeded",
                )
            )
        except asyncio.CancelledError:
            raise

    async def _trip_abort(self, cause: AbortCause) -> bool:
        controller = self._abort_controller
        assert controller is not None
        tripped = await controller.trip(cause)
        if tripped:
            self.termination = controller.termination()
            await self._emit_event(
                ProcessingEvent.BATCH_ABORTED,
                {
                    "kind": cause.kind,
                    "reason": cause.reason,
                    "error_category": cause.error_category,
                    "triggering_item_id": cause.triggering_item_id,
                    "abort_mode": self.config.guardrails.abort_mode.value,
                },
            )
        return tripped

    async def wait_for_abort(self) -> None:
        """Wait until a configured batch deadline or fail-fast abort trips."""
        assert self._abort_controller is not None
        await self._abort_controller.event.wait()

    @property
    def aborted(self) -> bool:
        """Whether this run has entered controlled guardrail termination."""
        return self._abort_controller is not None and self._abort_controller.aborted

    async def _cancel_batch_timeout(self) -> None:
        task = self._batch_timeout_task
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def cleanup(self) -> None:
        """Cancel workers, timers, and every quota-scoped admission resource."""
        errors: list[BaseException] = []
        for cleanup in (
            self._cancel_batch_timeout,
            super().cleanup,
            self._admission_registry.shutdown,
            self._compatibility_rate_limit_coord.shutdown,
        ):
            try:
                await cleanup()
            except BaseException as error:
                errors.append(error)
        self._classifier_resolver.clear()
        if errors:
            raise errors[0]

    def start(self) -> None:
        """Start streaming workers and the batch deadline clock."""
        self._start_guardrail_run()
        super().start()

    async def get_stats(self) -> dict:
        """
        Get processor statistics (thread-safe).

        Returns:
            Dictionary containing processing statistics including:
            - processed: Number of items processed
            - succeeded: Number of successful items
            - failed: Number of failed items
            - rate_limit_count: Number of rate limit errors encountered
            - error_counts: Dictionary of error types and their counts
            - total: Total number of items queued
            - start_time: Timestamp when processing started
        """
        async with self._stats_lock:
            return self._stats.copy()

    async def add_work(self, work_item: LLMWorkItem[TInput, TOutput, TContext]) -> None:
        """Queue a work item and register its identity-scoped admission state."""
        if self.artifact_store is not None:
            work_item._artifact_key = await self.artifact_store.prepare_item(work_item)

        strategy_id = id(work_item.strategy)
        if strategy_id not in self._capacity_checked_strategy_ids:
            if self.config.concurrency is not None:
                # Let built-in models right-size
                # their connection pools before the first request. Runs before
                # the capacity check so a successful resize (advertised
                # capacity == concurrency == workers) produces no warning;
                # an explicit smaller max_connections refuses the resize and
                # the warning below surfaces the real contradiction.
                hook = getattr(work_item.strategy, "request_concurrency", None)
                if hook is not None:
                    try:
                        await hook(self.config.concurrency)
                    except Exception as e:
                        logger.warning(
                            "request_concurrency(%s) failed for %s: %s",
                            self.config.concurrency,
                            type(work_item.strategy).__name__,
                            e,
                        )
            warn_if_worker_capacity_exceeded(
                strategy=work_item.strategy,
                max_workers=cast(int, self.config.max_workers),
                surface="ParallelBatchProcessor",
                stacklevel=3,
            )
            self._capacity_checked_strategy_ids.add(strategy_id)
        assert self._abort_controller is not None
        if self._abort_controller.aborted:
            raise BatchAdmissionStopped("Batch is no longer accepting work")
        admission_state = self._admission_registry.resolve(work_item.strategy)
        if not self._compatibility_scope_bound:
            self._rate_limit_coord = admission_state.cooldown
            self._compatibility_scope_bound = True
        if self._streaming and self._guardrails_started:
            acceptance = asyncio.create_task(super().add_work(work_item))
            abort_wait = asyncio.create_task(self._abort_controller.event.wait())
            try:
                done, _ = await asyncio.wait(
                    {acceptance, abort_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                # An acceptance that completed concurrently wins: the item is
                # now owned by the queue and must receive a terminal result.
                if acceptance in done:
                    await acceptance
                else:
                    acceptance.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await acceptance
                    raise BatchAdmissionStopped("Batch stopped accepting work")
            finally:
                abort_wait.cancel()
                await asyncio.gather(abort_wait, return_exceptions=True)
        else:
            await super().add_work(work_item)

    async def _on_batch_started(self) -> None:
        """Emit batch start event with initial stats snapshot."""
        self._start_guardrail_run()

        async with self._stats_lock:
            stats_snapshot = self._stats.copy()

        await self._emit_event(
            ProcessingEvent.BATCH_STARTED,
            {
                "total": stats_snapshot["total"],
                "max_workers": self.max_workers,
                "start_time": stats_snapshot["start_time"],
            },
        )

    async def _on_batch_completed(self) -> None:
        """Emit batch completion event with final stats snapshot."""
        await self._cancel_batch_timeout()
        async with self._stats_lock:
            stats_snapshot = self._stats.copy()

        duration = 0.0
        if stats_snapshot.get("start_time"):
            duration = time.time() - float(stats_snapshot["start_time"])

        await self._emit_event(
            ProcessingEvent.BATCH_COMPLETED,
            {
                "processed": stats_snapshot["processed"],
                "succeeded": stats_snapshot["succeeded"],
                "failed": stats_snapshot["failed"],
                "total": stats_snapshot["total"],
                "total_tokens": stats_snapshot["total_tokens"],
                "cached_input_tokens": stats_snapshot.get("cached_input_tokens", 0),
                "total_admission_wait_seconds": stats_snapshot.get(
                    "total_admission_wait_seconds", 0.0
                ),
                "max_admission_wait_seconds": stats_snapshot.get("max_admission_wait_seconds", 0.0),
                "total_quota_wait_seconds": stats_snapshot.get("total_quota_wait_seconds", 0.0),
                "max_quota_wait_seconds": stats_snapshot.get("max_quota_wait_seconds", 0.0),
                "quota_wait_p50_seconds": stats_snapshot.get("quota_wait_p50_seconds", 0.0),
                "quota_wait_p95_seconds": stats_snapshot.get("quota_wait_p95_seconds", 0.0),
                "quota_wait_p99_seconds": stats_snapshot.get("quota_wait_p99_seconds", 0.0),
                "estimated_input_tokens": stats_snapshot.get("estimated_input_tokens", 0),
                "estimated_output_tokens": stats_snapshot.get("estimated_output_tokens", 0),
                "reserved_tokens": stats_snapshot.get("reserved_tokens", 0),
                "reported_reconciliation_tokens": stats_snapshot.get(
                    "reported_reconciliation_tokens", 0
                ),
                "refunded_tokens": stats_snapshot.get("refunded_tokens", 0),
                "underestimated_tokens": stats_snapshot.get("underestimated_tokens", 0),
                "unknown_usage_attempts": stats_snapshot.get("unknown_usage_attempts", 0),
                "known_zero_usage_attempts": stats_snapshot.get("known_zero_usage_attempts", 0),
                "token_estimation_failures": stats_snapshot.get("token_estimation_failures", 0),
                "quota_scope_count": stats_snapshot.get("quota_scope_count", 0),
                "admission_wait_p50_seconds": stats_snapshot.get("admission_wait_p50_seconds", 0.0),
                "admission_wait_p95_seconds": stats_snapshot.get("admission_wait_p95_seconds", 0.0),
                "admission_wait_p99_seconds": stats_snapshot.get("admission_wait_p99_seconds", 0.0),
                "execution_p50_seconds": stats_snapshot.get("execution_p50_seconds", 0.0),
                "execution_p95_seconds": stats_snapshot.get("execution_p95_seconds", 0.0),
                "execution_p99_seconds": stats_snapshot.get("execution_p99_seconds", 0.0),
                "structured_output_recoveries": stats_snapshot.get(
                    "structured_output_recoveries", 0
                ),
                "structured_output_retries_avoided": stats_snapshot.get(
                    "structured_output_retries_avoided", 0
                ),
                "structured_output_recovery_reasons": stats_snapshot.get(
                    "structured_output_recovery_reasons", {}
                ),
                "input_queue_high_water_mark": stats_snapshot["input_queue_high_water_mark"],
                "result_queue_high_water_mark": stats_snapshot["result_queue_high_water_mark"],
                "duration": duration,
            },
        )
        if self.artifact_store is not None:
            await self.artifact_store.close()

    async def _emit_event(self, event: ProcessingEvent, data: dict | None = None) -> None:
        """Delegate to EventDispatcher (see _internal/event_dispatcher.py)."""
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

    async def _worker(self, worker_id: int):
        """Process accepted items with exactly-once queue accounting."""
        logger.debug("[Worker %s] started and waiting for work", worker_id)
        if self._events.observers:
            await self._emit_event(ProcessingEvent.WORKER_STARTED, {"worker_id": worker_id})

        while True:
            try:
                work_item = await self._queue.get()
            except asyncio.CancelledError:
                logger.debug("[Worker %s] cancelled while waiting for work", worker_id)
                raise
            try:
                if work_item is None:
                    logger.debug("[Worker %s] finished (no more work)", worker_id)
                    if self._events.observers:
                        await self._emit_event(
                            ProcessingEvent.WORKER_STOPPED,
                            {"worker_id": worker_id},
                        )
                    return
                try:
                    await self._process_accepted_item(work_item, worker_id)
                except ArtifactError as exc:
                    # Artifact failures remain exceptional, but streaming users
                    # can still inspect why the low-level processor terminated.
                    self.termination = BatchTermination(
                        kind="artifact_error",
                        reason=str(exc),
                    )
                    raise
            finally:
                # The only task_done() site for every successful queue.get().
                self._queue.task_done()

    async def _process_accepted_item(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
    ) -> None:
        """Resolve replay/execution, checkpoint, then publish one terminal result."""
        logger.debug("[Worker %s] Picked up %s from queue", worker_id, work_item.item_id)
        result: WorkItemResult[TOutput, TContext] | None = None
        generated_from_abort = False
        controller = self._abort_controller
        assert controller is not None
        if controller.aborted:
            result = cast(
                WorkItemResult[TOutput, TContext],
                controller.result_for(work_item),
            )
            generated_from_abort = True
        elif self.artifact_store is not None:
            replayed = await self.artifact_store.lookup(
                work_item,
                work_item._artifact_key,
                self.resume,
            )
            result = cast(WorkItemResult[TOutput, TContext] | None, replayed)

        if result is None:
            result = await self._executor.execute(work_item, worker_id)

        result.submission_index = work_item.submission_index

        # Persist newly executed/aborted terminal state before it becomes
        # visible. Replayed records are deliberately not duplicated.
        if self.artifact_store is not None and not result.replayed_from_artifact:
            await self.artifact_store.append(work_item, work_item._artifact_key, result)

        # Fail-fast is triggered only by a terminal failure and only after its
        # checkpoint is complete. The controller retains the first cause.
        if (
            not result.success
            and result.error_category in self.config.guardrails.abort_on_error_categories
        ):
            await self._trip_abort(
                AbortCause(
                    kind="fail_fast",
                    reason=(
                        f"Fail-fast triggered by item {work_item.item_id!r} "
                        f"with category {result.error_category!r}"
                    ),
                    error_category=result.error_category,
                    triggering_item_id=work_item.item_id,
                )
            )

        if self._events.observers:
            if result.replayed_from_artifact:
                await self._emit_event(
                    ProcessingEvent.ITEM_REPLAYED,
                    {
                        "item_id": work_item.item_id,
                        "submission_index": work_item.submission_index,
                        "success": result.success,
                        "error_category": result.error_category,
                    },
                )
            if result.error_category == "framework_total_item_timeout":
                await self._emit_event(
                    ProcessingEvent.ITEM_DEADLINE_EXCEEDED,
                    {"item_id": work_item.item_id},
                )
            if generated_from_abort:
                await self._emit_event(
                    ProcessingEvent.ITEM_FAILED,
                    {
                        "item_id": work_item.item_id,
                        "error_type": type(result.exception).__name__,
                        "error_category": result.error_category,
                    },
                )

        completed, total, current_item, stats_snapshot = await self._record_terminal_stats(
            work_item, result
        )
        # User callbacks fire for every completed item. The private bundled
        # reporter observes every exact count too, but coalesces terminal
        # rendering by time. progress_interval gates only the log line below.
        # Progress runs before publication so finalization cannot lose the
        # terminal count when a consumer stops after the final result.
        if self.progress_callback:
            await self._run_progress_callback(completed, total, current_item)
        if stats_snapshot is not None:
            self._log_progress(stats_snapshot)

        if self._streaming:
            assert self._result_stream is not None
            await self._publish_stream_result(result)
        else:
            async with self._results_lock:
                self._results.append(result)

        if self.config.concurrent_post_processing:
            self._spawn_post_processor(result, self.config.post_processor_timeout)
        else:
            await self._run_post_processor(result, timeout=self.config.post_processor_timeout)

        logger.debug(
            "[Worker %s] Completed %s (%s)",
            worker_id,
            work_item.item_id,
            "success" if result.success else "failed",
        )

    async def _record_terminal_stats(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        result: WorkItemResult[TOutput, TContext],
    ) -> tuple[int, int, str, dict | None]:
        async with self._stats_lock:
            self._stats.processed += 1
            if result.success:
                self._stats.succeeded += 1
            else:
                self._stats.failed += 1
                if result.error:
                    error_type = result.error.split(":")[0]
                    self._stats.error_counts[error_type] = (
                        self._stats.error_counts.get(error_type, 0) + 1
                    )
            if result.replayed_from_artifact:
                self._stats.replayed += 1
            if result.error_category in {"batch_aborted", "batch_deadline_exceeded"}:
                self._stats.aborted += 1

            # Replayed tokens remain on the result for historical audit but are
            # not counted as newly consumed provider tokens in live stats.
            if result.token_usage and not result.replayed_from_artifact:
                self._stats.total_input_tokens += result.token_usage.get("input_tokens", 0)
                self._stats.total_output_tokens += result.token_usage.get("output_tokens", 0)
                self._stats.total_tokens += result.token_usage.get("total_tokens", 0)
                self._stats.cached_input_tokens += result.token_usage.get("cached_input_tokens", 0)
            self._stats.total_admission_wait_seconds += result.admission_wait_seconds
            self._stats.max_admission_wait_seconds = max(
                self._stats.max_admission_wait_seconds,
                result.admission_wait_seconds,
            )
            self._stats.record_timing(result.timing)
            if result.structured_output_recovered:
                self._stats.structured_output_recoveries += 1
                self._stats.structured_output_retries_avoided += (
                    result.structured_output_retries_avoided
                )
                reason = result.structured_output_recovery_reason
                if reason is not None:
                    self._stats.structured_output_recovery_reasons[reason] = (
                        self._stats.structured_output_recovery_reasons.get(reason, 0) + 1
                    )

            at_interval = self._stats.processed % self.config.progress_interval == 0
            return (
                self._stats.processed,
                self._stats.total,
                work_item.item_id,
                self._stats.copy() if at_interval else None,
            )

    @staticmethod
    def _log_progress(stats_snapshot: dict) -> None:
        start_time = stats_snapshot.get("start_time")
        elapsed = time.time() - start_time if isinstance(start_time, (int, float)) else 0.0
        calls_per_sec = stats_snapshot["processed"] / elapsed if elapsed > 0 else 0.0
        errors = ", ".join(
            f"{error}: {count}" for error, count in stats_snapshot["error_counts"].items()
        )
        error_breakdown = f" | Errors: {errors}" if errors else ""
        token_summary = ""
        if stats_snapshot["total_tokens"] > 0:
            cached = stats_snapshot.get("cached_input_tokens", 0)
            cached_info = f", {cached:,} cached" if cached > 0 else ""
            token_summary = (
                f" | Tokens: {stats_snapshot['total_tokens']:,} "
                f"({stats_snapshot['total_input_tokens']:,} in, "
                f"{stats_snapshot['total_output_tokens']:,} out{cached_info})"
            )
        total = stats_snapshot["total"]
        percent = stats_snapshot["processed"] / total * 100 if total else 100.0
        logger.info(
            f"[INFO]Progress: {stats_snapshot['processed']}/{total} ({percent:.1f}%) | "
            f"Succeeded: {stats_snapshot['succeeded']}, Failed: {stats_snapshot['failed']}"
            f"{error_breakdown} | {calls_per_sec:.2f} calls/sec{token_summary}"
        )

    async def _handle_rate_limit(
        self,
        worker_id: int,
        observed_generation: int | None = None,
        suggested_wait: float | None = None,
    ) -> None:
        """Delegate to RateLimitCoordinator."""
        await self._rate_limit_coord.handle_rate_limit(
            worker_id, observed_generation, suggested_wait
        )

    async def _finalize_cooldown(self, start_time: float, error: Exception | None) -> None:
        """Delegate to RateLimitCoordinator._finalize_cooldown (kept for subclass overrides)."""
        await self._rate_limit_coord._finalize_cooldown(start_time, error)

    def _extract_token_usage(self, exception: Exception) -> dict[str, int]:
        """Extract token usage from a failed LLM call exception.

        Thin wrapper around :class:`TokenExtractor`; retained as a method
        for subclass override points. See ``token_extractor.py`` for the
        three extraction strategies.
        """
        return self._token_extractor.extract_from_exception(exception)

    async def _process_item_with_retries(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        deadline: float | None = None,
    ) -> WorkItemResult[TOutput, TContext]:
        """Apply retry logic and strategy lifecycle (delegates to ItemExecutor).

        Kept as an instance method (not inlined to the executor call) because
        tests monkeypatch *this* method and call it directly on a non-started
        processor; the worker must keep routing through it.
        """
        return await self._executor._process_item_with_retries(work_item, worker_id, deadline)

    async def _process_item(  # type: ignore[override]  # ty:ignore[invalid-method-override]
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int = 1,
        strategy: LLMCallStrategy[TOutput] | None = None,
        retry_state: RetryState | None = None,
    ) -> WorkItemResult[TOutput, TContext]:
        """Process a single work item (delegates to ItemExecutor)."""
        return await self._executor._process_item(
            work_item, worker_id, attempt_number, strategy, retry_state
        )

    async def _handle_execution_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
    ) -> WorkItemResult[TOutput, TContext]:
        """Handle exceptions from LLM execution (delegates to ItemExecutor).

        Classifies errors, extracts token usage, handles rate limits, and either
        re-raises (retryable / rate-limited) or returns a permanent-failure
        result. See :class:`ItemExecutor` for the implementation.
        """
        return await self._executor._handle_execution_error(
            exception, work_item, worker_id, attempt_number
        )

    async def shutdown(self):
        """Clean up resources: flush observers and cancel pending tasks."""
        errors: list[BaseException] = []
        for cleanup in (self.cleanup, self._cleanup_strategies):
            try:
                await cleanup()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
