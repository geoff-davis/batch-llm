"""A lightweight host that backs :class:`ItemExecutor` without a worker pool.

The :class:`~async_batch_llm.parallel.ParallelBatchProcessor` is one host for
the executor; this is the other. It builds the same per-item dependencies
(rate-limit coordinator, strategy lifecycle, token extractor, classifier,
stats) but no queue, workers, or result stream — so the single-call helper and
the gateway can run ``executor.execute(work_item)`` directly.

Observers and middlewares are intentionally empty here: these surfaces are the
request path, not a batch with progress reporting. One registry per shared host
makes concurrent callers coordinate by quota-scope identity.
"""

from __future__ import annotations

import asyncio
from typing import Generic, cast

from ..base import (
    LLMWorkItem,
    ProcessingStats,
    RetryState,
    TContext,
    TInput,
    TOutput,
    WorkItemResult,
)
from ..core import ProcessorConfig
from ..llm_strategies import LLMCallStrategy
from ..strategies import (
    ErrorClassifier,
    ExponentialBackoffStrategy,
    RateLimitStrategy,
)
from ..token_extractor import TokenExtractor
from .admission import AdmissionRegistry
from .capacity import CapacityLimiter
from .classifier_resolver import StrategyClassifierResolver
from .event_dispatcher import EventDispatcher
from .guardrails import AbortController
from .item_executor import ItemExecutor
from .rate_limit_coordinator import RateLimitCoordinator
from .strategy_lifecycle import StrategyLifecycle


class ExecutorHost(Generic[TInput, TOutput, TContext]):
    """Owns the per-item dependencies for queue-less execution.

    Exposes exactly the attribute surface :class:`ItemExecutor` reads, plus a
    ready-built ``executor``. Call :meth:`aclose` to run strategy ``cleanup()``.
    """

    def __init__(
        self,
        config: ProcessorConfig,
        *,
        strategy: LLMCallStrategy | None = None,
        error_classifier: ErrorClassifier | None = None,
        rate_limit_strategy: RateLimitStrategy | None = None,
    ) -> None:
        self.config = config
        self._classifier_resolver = StrategyClassifierResolver(error_classifier)
        # Compatibility/debug alias. Item execution resolves per strategy.
        self.error_classifier = self._classifier_resolver.compatibility_classifier
        self.rate_limit_strategy = rate_limit_strategy or ExponentialBackoffStrategy(
            initial_cooldown=config.rate_limit.cooldown_seconds,
            max_cooldown=config.rate_limit.max_cooldown_seconds,
            backoff_multiplier=config.rate_limit.backoff_multiplier,
            slow_start_items=config.rate_limit.slow_start_items,
            slow_start_initial_delay=config.rate_limit.slow_start_initial_delay,
            slow_start_final_delay=config.rate_limit.slow_start_final_delay,
        )

        # No observers/middlewares on the request path. (Parameterized
        # explicitly: with PEP 696 defaults a bare `EventDispatcher` would
        # resolve to [str, Any, None] and break ExecutorHostProtocol
        # conformance.)
        self._events: EventDispatcher[TInput, TOutput, TContext] = EventDispatcher(
            observers=[], middlewares=[]
        )
        self._admission_registry = AdmissionRegistry(
            rate_limit_strategy=self.rate_limit_strategy,
            events=self._events,
            max_requests_per_minute=config.max_requests_per_minute,
        )
        # Queue-less hosts are constructed for one strategy, so this old
        # private alias can point at that strategy's real scoped coordinator.
        if strategy is not None:
            initial_state = self._admission_registry.resolve(strategy)
            self._rate_limit_coord = initial_state.cooldown
            self._owns_compatibility_coordinator = False
        else:
            self._rate_limit_coord = RateLimitCoordinator(
                rate_limit_strategy=self.rate_limit_strategy,
                events=self._events,
            )
            self._owns_compatibility_coordinator = True
        self._strategy_lifecycle: StrategyLifecycle[TOutput] = StrategyLifecycle()
        self._capacity_limiter = CapacityLimiter(
            config.max_provider_concurrency,
            max_workers=cast(int, config.max_workers),
            startup_ramp=config.startup_ramp,
        )
        self._stats = ProcessingStats()
        self._stats_lock = asyncio.Lock()
        self._token_extractor = TokenExtractor()
        # Queue-less call/gateway surfaces have item deadlines but no shared
        # batch abort controller.
        self._abort_controller: AbortController | None = None

        self.executor: ItemExecutor[TInput, TOutput, TContext] = ItemExecutor(self)

    # These three satisfy ExecutorHostProtocol's override-point hooks. On the
    # queue-less path there's no subclass to override them, so they delegate
    # straight back to the executor (the processor's versions do the same).
    def _extract_token_usage(self, exception: Exception) -> dict[str, int]:
        return self._token_extractor.extract_from_exception(exception)

    async def _process_item(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int = 1,
        strategy: LLMCallStrategy[TOutput] | None = None,
        retry_state: RetryState | None = None,
    ) -> WorkItemResult[TOutput, TContext]:
        return await self.executor._process_item(
            work_item, worker_id, attempt_number, strategy, retry_state
        )

    async def _process_item_with_retries(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        deadline: float | None = None,
    ) -> WorkItemResult[TOutput, TContext]:
        return await self.executor._process_item_with_retries(work_item, worker_id, deadline)

    async def _handle_execution_error(
        self,
        exception: Exception,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        worker_id: int,
        attempt_number: int,
    ) -> WorkItemResult[TOutput, TContext]:
        return await self.executor._handle_execution_error(
            exception, work_item, worker_id, attempt_number
        )

    async def aclose(self) -> None:
        """Run cleanup() on every strategy this host prepared."""
        errors: list[BaseException] = []
        try:
            await self._strategy_lifecycle.cleanup_all()
        except BaseException as exc:
            errors.append(exc)
        try:
            await self._admission_registry.shutdown()
        except BaseException as exc:
            errors.append(exc)
        if self._owns_compatibility_coordinator:
            try:
                await self._rate_limit_coord.shutdown()
            except BaseException as exc:
                errors.append(exc)
        self._classifier_resolver.clear()
        if errors:
            raise errors[0]
