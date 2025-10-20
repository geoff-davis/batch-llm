"""Parallel batch processor"""

import asyncio
import logging
import time
from typing import Generic

from .base import (
    BatchProcessor,
    LLMWorkItem,
    PostProcessorFunc,
    TContext,
    TInput,
    TOutput,
    WorkItemResult,
)
from .classifiers import GeminiErrorClassifier
from .core import ProcessorConfig
from .middleware import Middleware
from .observers import ProcessingEvent, ProcessorObserver
from .strategies import ErrorClassifier, ExponentialBackoffStrategy, RateLimitStrategy

logger = logging.getLogger(__name__)


class RateLimitException(Exception):
    """Exception raised when rate limit is detected. Triggers cooldown and item re-queue."""

    pass


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
    ):
        """
        Initialize the parallel batch processor.

        Args:
            max_workers: Maximum concurrent workers (deprecated, use config)
            post_processor: Optional async function called after each successful item
            timeout_per_item: Timeout per item in seconds (deprecated, use config)
            rate_limit_cooldown: Cooldown duration (deprecated, use config)
            config: Processor configuration object (recommended)
            error_classifier: Strategy for classifying errors (default: GeminiErrorClassifier)
            rate_limit_strategy: Strategy for handling rate limits
            middlewares: List of middleware to apply
            observers: List of observers for events
        """
        # Handle backward compatibility
        if config is None:
            from .core import RateLimitConfig

            config = ProcessorConfig(
                max_workers=max_workers or 5,
                timeout_per_item=timeout_per_item or 120.0,
                rate_limit=RateLimitConfig(
                    cooldown_seconds=rate_limit_cooldown or 300.0
                ),
            )
        else:
            # Override config with explicit parameters if provided
            if max_workers is not None:
                config.max_workers = max_workers
            if timeout_per_item is not None:
                config.timeout_per_item = timeout_per_item
            if rate_limit_cooldown is not None:
                config.rate_limit.cooldown_seconds = rate_limit_cooldown

        config.validate()

        super().__init__(
            config.max_workers,
            post_processor,
            max_queue_size=config.max_queue_size
        )
        self.config = config

        # Set up strategies
        self.error_classifier = error_classifier or GeminiErrorClassifier()
        self.rate_limit_strategy = rate_limit_strategy or ExponentialBackoffStrategy(
            initial_cooldown=config.rate_limit.cooldown_seconds,
            slow_start_items=config.rate_limit.slow_start_items,
            slow_start_initial_delay=config.rate_limit.slow_start_initial_delay,
            slow_start_final_delay=config.rate_limit.slow_start_final_delay,
        )

        # Set up middleware and observers
        self.middlewares = middlewares or []
        self.observers = observers or []

        # Rate limit coordination
        self._rate_limit_event = asyncio.Event()
        self._rate_limit_event.set()  # Start in "not paused" state
        self._in_cooldown = False
        self._items_since_resume = 0
        self._consecutive_rate_limits = 0

        # Thread safety locks
        self._rate_limit_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()
        self._results_lock = asyncio.Lock()

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

    async def _emit_event(
        self, event: ProcessingEvent, data: dict | None = None
    ) -> None:
        """Emit event to all observers."""
        if not self.observers:
            return

        event_data = data or {}
        for observer in self.observers:
            try:
                await asyncio.wait_for(
                    observer.on_event(event, event_data),
                    timeout=5.0  # 5 second timeout for observer callbacks
                )
            except TimeoutError:
                logger.warning(
                    f"⚠️  Observer callback timed out after 5s for event {event.name}"
                )
            except Exception as e:
                logger.warning(f"⚠️  Observer error: {e}")

    async def _run_middlewares_before(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMWorkItem[TInput, TOutput, TContext] | None:
        """Run before_process on all middlewares."""
        current_item = work_item
        for middleware in self.middlewares:
            try:
                result = await middleware.before_process(current_item)
                if result is None:
                    return None  # Skip this item
                current_item = result
            except Exception as e:
                logger.warning(
                    f"⚠️  Middleware before_process error for {work_item.item_id}: {e}"
                )
        return current_item

    async def _run_middlewares_after(
        self, result: WorkItemResult[TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        """Run after_process on all middlewares in reverse order (onion pattern)."""
        current_result = result
        # Run in reverse order to maintain onion-style middleware pattern
        for middleware in reversed(self.middlewares):
            try:
                current_result = await middleware.after_process(current_result)
            except Exception as e:
                logger.warning(
                    f"⚠️  Middleware after_process error for {result.item_id}: {e}"
                )
        return current_result

    async def _run_middlewares_on_error(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], error: Exception
    ) -> WorkItemResult[TOutput, TContext] | None:
        """Run on_error on all middlewares."""
        for middleware in self.middlewares:
            try:
                result = await middleware.on_error(work_item, error)
                if result is not None:
                    return result  # Middleware handled the error
            except Exception as e:
                logger.warning(
                    f"⚠️  Middleware on_error error for {work_item.item_id}: {e}"
                )
        return None

    async def _worker(self, worker_id: int):
        """Worker coroutine that processes items from the queue."""
        logger.info(f"✓ Worker {worker_id} started and waiting for work")
        await self._emit_event(ProcessingEvent.WORKER_STARTED, {"worker_id": worker_id})

        while True:
            try:
                work_item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                if work_item is None:  # Sentinel value
                    self._queue.task_done()
                    logger.info(f"✓ Worker {worker_id} finished (no more work)")
                    await self._emit_event(
                        ProcessingEvent.WORKER_STOPPED, {"worker_id": worker_id}
                    )
                    return
            except TimeoutError:
                continue

            logger.info(f"ℹ️  [Worker {worker_id}] Picked up {work_item.item_id} from queue")

            # Wait if we're in rate limit cooldown
            await self._rate_limit_event.wait()

            # Slow start after rate limit recovery (thread-safe)
            async with self._rate_limit_lock:
                should_delay, delay = self.rate_limit_strategy.should_apply_slow_start(
                    self._items_since_resume
                )
                if should_delay:
                    self._items_since_resume += 1

            if should_delay:
                await asyncio.sleep(delay)

            # Process the item
            try:
                result = await self._process_item_with_retries(work_item, worker_id)
            except RateLimitException:
                # Item was re-queued during rate limit handling
                self._queue.task_done()
                continue
            except Exception as e:
                # All retries exhausted or unhandled exception
                # Create a failed result so the item is recorded

                # Extract token usage from exception if available
                failed_tokens = {}
                if hasattr(e, '__dict__') and '_failed_token_usage' in e.__dict__:
                    failed_tokens = e.__dict__['_failed_token_usage']

                token_msg = ""
                if failed_tokens.get("total_tokens", 0) > 0:
                    token_msg = f" (consumed {failed_tokens['total_tokens']} tokens across all attempts)"

                logger.error(
                    f"✗ Worker {worker_id} failed to process {work_item.item_id} after all retries: "
                    f"{type(e).__name__}: {str(e)[:200]}{token_msg}"
                )

                # Try middleware error handlers
                middleware_result = await self._run_middlewares_on_error(work_item, e)
                if middleware_result is not None:
                    result = middleware_result
                else:
                    result = WorkItemResult(
                        item_id=work_item.item_id,
                        success=False,
                        error=f"{type(e).__name__}: {str(e)[:200]}",
                        context=work_item.context,
                        token_usage=failed_tokens,
                    )
                # Fall through to store result and call task_done()

            # Store result (thread-safe)
            async with self._results_lock:
                self._results.append(result)

            # Update stats (thread-safe)
            async with self._stats_lock:
                self._stats["processed"] += 1
                if result.success:
                    self._stats["succeeded"] += 1
                else:
                    self._stats["failed"] += 1
                    if result.error:
                        error_type = result.error.split(":")[0]
                        self._stats["error_counts"][error_type] = (
                            self._stats["error_counts"].get(error_type, 0) + 1
                        )

            # Run post-processor for both success AND failure
            # Note: Post-processors should check result.success and handle accordingly
            # Most post-processors return early for failures, but some may want to
            # save failed items (e.g., dedupe_authors saves failed clusters as singletons)
            try:
                await asyncio.wait_for(
                    self._run_post_processor(result), timeout=90.0
                )
            except TimeoutError:
                logger.error(
                    f"⏱ Post-processor exceeded timeout for {work_item.item_id}"
                )

            self._queue.task_done()

            # Log completion
            status = "✓" if result.success else "✗"
            logger.info(f"{status} [Worker {worker_id}] Completed {work_item.item_id} ({'success' if result.success else 'failed'})")

            # Log progress (thread-safe read of stats)
            async with self._stats_lock:
                should_log = self._stats["processed"] % self.config.progress_interval == 0
                if should_log:
                    stats_snapshot = self._stats.copy()

            if should_log:
                elapsed = time.time() - stats_snapshot["start_time"]
                calls_per_sec = stats_snapshot["processed"] / elapsed if elapsed > 0 else 0

                error_breakdown = ""
                if stats_snapshot["error_counts"]:
                    error_strs = [
                        f"{err}: {count}"
                        for err, count in stats_snapshot["error_counts"].items()
                    ]
                    error_breakdown = f" | Errors: {', '.join(error_strs)}"

                logger.info(
                    f"ℹ️  Progress: {stats_snapshot['processed']}/{stats_snapshot['total']} "
                    f"({stats_snapshot['processed']/stats_snapshot['total']*100:.1f}%) | "
                    f"Succeeded: {stats_snapshot['succeeded']}, Failed: {stats_snapshot['failed']}"
                    f"{error_breakdown} | {calls_per_sec:.2f} calls/sec"
                )

    async def _handle_rate_limit(self, worker_id: int):
        """Handle rate limit by pausing all workers and cooling down (thread-safe)."""
        # Atomic check-and-set to prevent multiple workers from triggering cooldown
        async with self._rate_limit_lock:
            if self._in_cooldown:
                return  # Another worker is already handling it

            self._in_cooldown = True
            self._consecutive_rate_limits += 1
            self._rate_limit_event.clear()  # Pause all workers
            consecutive = self._consecutive_rate_limits

        # Get cooldown duration from strategy (outside lock - can be slow)
        cooldown = await self.rate_limit_strategy.on_rate_limit(
            worker_id, consecutive
        )

        await self._emit_event(
            ProcessingEvent.COOLDOWN_STARTED,
            {
                "worker_id": worker_id,
                "duration": cooldown,
                "consecutive": consecutive,
            },
        )

        logger.warning(
            f"🚫  Rate limit detected by worker {worker_id}. "
            f"Pausing all workers for {cooldown:.1f}s..."
        )

        start_time = time.time()
        await asyncio.sleep(cooldown)
        actual_duration = time.time() - start_time

        # Atomic reset
        async with self._rate_limit_lock:
            self._items_since_resume = 0
            self._in_cooldown = False
            self._rate_limit_event.set()  # Resume all workers

        await self._emit_event(
            ProcessingEvent.COOLDOWN_ENDED,
            {"duration": actual_duration},
        )

        logger.info("✓ Cooldown complete. Resuming with slow-start...")

    def _should_retry_error(self, exception: Exception) -> bool:
        """Determine if error should be retried using error classifier."""
        error_info = self.error_classifier.classify(exception)
        return error_info.is_retryable

    def _extract_token_usage(self, exception: Exception) -> dict[str, int]:
        """
        Extract token usage from a failed LLM call exception.

        Attempts multiple strategies to extract token usage from different provider
        exception structures. Returns empty dict if extraction fails.

        Args:
            exception: The exception from which to extract token usage

        Returns:
            Dictionary with input_tokens, output_tokens, total_tokens (or empty dict)
        """
        try:
            # Strategy 1: PydanticAI-style exception with result in __cause__
            if hasattr(exception, '__cause__') and exception.__cause__:
                cause = exception.__cause__
                if hasattr(cause, 'result'):
                    result = cause.result
                    if hasattr(result, 'usage') and callable(result.usage):
                        usage = result.usage()
                        if usage:
                            return {
                                "input_tokens": getattr(usage, 'request_tokens', 0),
                                "output_tokens": getattr(usage, 'response_tokens', 0),
                                "total_tokens": getattr(usage, 'total_tokens', 0),
                            }

            # Strategy 2: Direct usage attribute on exception
            if hasattr(exception, 'usage'):
                usage = exception.usage
                if callable(usage):
                    usage = usage()
                if usage:
                    return {
                        "input_tokens": getattr(usage, 'request_tokens', getattr(usage, 'input_tokens', 0)),
                        "output_tokens": getattr(usage, 'response_tokens', getattr(usage, 'output_tokens', 0)),
                        "total_tokens": getattr(usage, 'total_tokens', 0),
                    }

            # Strategy 3: Custom _failed_token_usage attribute (set by this framework)
            if hasattr(exception, '__dict__') and '_failed_token_usage' in exception.__dict__:
                return exception.__dict__['_failed_token_usage']

        except Exception:
            # Extraction failed - return empty dict
            pass

        return {}


    async def _process_item_with_retries(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], worker_id: int
    ) -> WorkItemResult[TOutput, TContext]:
        """Wrapper that applies retry logic."""
        # Track cumulative token usage across all failed attempts
        cumulative_failed_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for attempt in range(1, self.config.retry.max_attempts + 1):
            try:
                return await self._process_item(work_item, worker_id, attempt_number=attempt)
            except Exception as e:
                # Try to extract token usage from this failed attempt using robust extraction
                attempt_tokens = self._extract_token_usage(e)
                if attempt_tokens:
                    cumulative_failed_tokens["input_tokens"] += attempt_tokens.get("input_tokens", 0)
                    cumulative_failed_tokens["output_tokens"] += attempt_tokens.get("output_tokens", 0)
                    cumulative_failed_tokens["total_tokens"] += attempt_tokens.get("total_tokens", 0)

                if not self._should_retry_error(e):
                    logger.debug(f"Error not retryable: {type(e).__name__}")
                    # Attach token usage to exception so it can be included in failed result
                    if hasattr(e, '__dict__'):
                        e.__dict__['_failed_token_usage'] = cumulative_failed_tokens
                    raise
                if attempt >= self.config.retry.max_attempts:
                    error_msg = str(e)
                    token_summary = ""
                    if cumulative_failed_tokens["total_tokens"] > 0:
                        token_summary = f"\n  Total tokens consumed across all attempts: {cumulative_failed_tokens['total_tokens']}"
                    logger.error(
                        f"✗ ALL {self.config.retry.max_attempts} ATTEMPTS EXHAUSTED for {work_item.item_id}:\n"
                        f"  Final error type: {type(e).__name__}\n"
                        f"  Final error message: {error_msg[:500]}\n"
                        f"  All temperature variations tried (0.0, 0.25, 0.5){token_summary}"
                    )
                    # Attach token usage to exception so it can be included in failed result
                    if hasattr(e, '__dict__'):
                        e.__dict__['_failed_token_usage'] = cumulative_failed_tokens
                    raise

                # Classify error to determine if we should delay
                error_info = self.error_classifier.classify(e)

                # Only delay for network/timeout errors, not for validation errors
                # Validation errors are immediate - just need different temperature
                # PydanticAI wraps validation errors in UnexpectedModelBehavior
                error_msg_for_check = str(e)
                is_validation_error = (
                    'validation' in type(e).__name__.lower()
                    or 'parse' in type(e).__name__.lower()
                    or 'unexpectedmodelbehavior' in type(e).__name__.lower()
                    or 'result validation' in error_msg_for_check.lower()
                    or error_info.error_category == 'validation_error'
                )

                if is_validation_error:
                    wait_time = 0.0  # No delay for validation - retry immediately
                else:
                    # Calculate wait time with exponential backoff for network/timeout errors
                    wait_time = min(
                        self.config.retry.initial_wait
                        * (self.config.retry.exponential_base ** (attempt - 1)),
                        self.config.retry.max_wait,
                    )

                # Log retry with temperature info if using agent_factory
                if work_item.agent_factory:
                    next_attempt = attempt + 1
                    # Infer temperature based on attempt (matches dedupe_authors logic)
                    if next_attempt == 1:
                        next_temp = 0.0
                    elif next_attempt == 2:
                        next_temp = 0.25
                    else:
                        next_temp = 0.5

                    if is_validation_error:
                        error_snippet = str(e)[:150]
                        logger.warning(
                            f"⚠️  Attempt {attempt}/{self.config.retry.max_attempts} failed for {work_item.item_id}: {type(e).__name__} - {error_snippet}. "
                            f"Retrying immediately with temperature={next_temp}..."
                        )
                    else:
                        error_snippet = str(e)[:150]
                        logger.warning(
                            f"⚠️  Attempt {attempt}/{self.config.retry.max_attempts} failed for {work_item.item_id}: {type(e).__name__} - {error_snippet}. "
                            f"Retrying in {wait_time:.1f}s with temperature={next_temp}..."
                        )
                else:
                    error_snippet = str(e)[:150]
                    if is_validation_error:
                        logger.warning(
                            f"⚠️  Attempt {attempt}/{self.config.retry.max_attempts} failed for {work_item.item_id}: {type(e).__name__} - {error_snippet}. "
                            f"Retrying immediately..."
                        )
                    else:
                        logger.warning(
                            f"⚠️  Attempt {attempt}/{self.config.retry.max_attempts} failed for {work_item.item_id}: {type(e).__name__} - {error_snippet}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

    async def _process_item(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext], worker_id: int, attempt_number: int = 1
    ) -> WorkItemResult[TOutput, TContext]:
        """Process a single work item with retries."""
        start_time = time.time()

        # Store original item_id before middleware might return None
        original_item_id = work_item.item_id

        await self._emit_event(
            ProcessingEvent.ITEM_STARTED,
            {"item_id": original_item_id, "worker_id": worker_id},
        )

        try:
            # Run before middlewares
            processed_item = await self._run_middlewares_before(work_item)
            if processed_item is None:
                logger.info(f"ℹ️  Skipping {original_item_id} (filtered by middleware)")
                return WorkItemResult(
                    item_id=original_item_id,
                    success=False,
                    error="Skipped by middleware",
                    context=work_item.context,
                )
            work_item = processed_item

            # Determine processing method: direct_call or agent
            if work_item.direct_call:
                # Direct API call path
                # Infer temperature for logging
                if attempt_number == 1:
                    temp = 0.0
                elif attempt_number == 2:
                    temp = 0.25
                else:
                    temp = 0.5

                if attempt_number > 1:
                    logger.info(f"ℹ️  [Worker {worker_id}] Retry attempt {attempt_number} for {work_item.item_id} with temperature={temp}")
                logger.debug(f"[DIRECT CALL] Starting direct LLM call for {work_item.item_id} (attempt {attempt_number}, temperature={temp}, timeout={self.config.timeout_per_item}s)")
                llm_start_time = time.time()

                try:
                    # Call the direct function with attempt number and timeout
                    # Wrap in wait_for to enforce timeout at the framework level
                    output, token_usage = await asyncio.wait_for(
                        work_item.direct_call(attempt_number, self.config.timeout_per_item),
                        timeout=self.config.timeout_per_item
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    elapsed = time.time() - llm_start_time
                    logger.error(
                        f"⏱ LLM TIMEOUT for {work_item.item_id} after {elapsed:.1f}s "
                        f"(limit: {self.config.timeout_per_item}s, attempt {attempt_number})"
                    )
                    raise

                llm_duration = time.time() - llm_start_time
                logger.debug(f"[DIRECT CALL] Completed LLM call for {work_item.item_id} in {llm_duration:.1f}s")

                # Log success after previous failures
                if attempt_number > 1:
                    logger.info(f"✓ SUCCESS on attempt {attempt_number} for {work_item.item_id} (after {attempt_number-1} failure(s), took {llm_duration:.1f}s)")

                # Log first few results for debugging
                if self._stats["succeeded"] < 3:
                    logger.info(
                        f"ℹ️  \n{'='*80}\nRESULT for {work_item.item_id}:\n{'='*80}\n{output}\n{'='*80}"
                    )

                # Create result
                work_result = WorkItemResult(
                    item_id=work_item.item_id,
                    success=True,
                    output=output,
                    context=work_item.context,
                    token_usage=token_usage,
                )

            else:
                # Agent path (PydanticAI)
                # Get agent for this attempt (supports temperature progression)
                if work_item.agent_factory:
                    agent = work_item.agent_factory(attempt_number)
                    # Infer temperature for logging
                    if attempt_number == 1:
                        temp = 0.0
                    elif attempt_number == 2:
                        temp = 0.25
                    else:
                        temp = 0.5
                    if attempt_number > 1:
                        logger.info(f"ℹ️  Retry attempt {attempt_number} for {work_item.item_id} with temperature={temp}")
                    logger.debug(f"Created agent for attempt {attempt_number} with temperature={temp}")
                else:
                    agent = work_item.agent

                # Run agent
                logger.debug(f"[AGENT CALL] Starting agent.run() for {work_item.item_id} (attempt {attempt_number}, timeout={self.config.timeout_per_item}s)")
                llm_start_time = time.time()

                try:
                    result = await asyncio.wait_for(
                        agent.run(work_item.prompt),
                        timeout=self.config.timeout_per_item,
                    )
                except TimeoutError:
                    elapsed = time.time() - llm_start_time
                    logger.error(
                        f"⏱ LLM TIMEOUT for {work_item.item_id} after {elapsed:.1f}s "
                        f"(limit: {self.config.timeout_per_item}s, attempt {attempt_number})"
                    )
                    raise

                llm_duration = time.time() - llm_start_time
                logger.debug(f"[AGENT CALL] Completed agent.run() for {work_item.item_id} in {llm_duration:.1f}s")

                # Log success after previous failures
                if attempt_number > 1:
                    logger.info(f"✓ SUCCESS on attempt {attempt_number} for {work_item.item_id} (after {attempt_number-1} failure(s), took {llm_duration:.1f}s)")

                # Extract token usage
                token_usage = {
                    "input_tokens": result.usage().request_tokens if result.usage() else 0,
                    "output_tokens": (
                        result.usage().response_tokens if result.usage() else 0
                    ),
                    "total_tokens": result.usage().total_tokens if result.usage() else 0,
                }

                # Log first few results for debugging
                if self._stats["succeeded"] < 3:
                    logger.info(
                        f"ℹ️  \n{'='*80}\nRESULT for {work_item.item_id}:\n{'='*80}\n{result.output}\n{'='*80}"
                    )

                # Create result
                work_result = WorkItemResult(
                    item_id=work_item.item_id,
                    success=True,
                    output=result.output,
                    context=work_item.context,
                    token_usage=token_usage,
                )

            # Run after middlewares
            work_result = await self._run_middlewares_after(work_result)

            duration = time.time() - start_time
            await self._emit_event(
                ProcessingEvent.ITEM_COMPLETED,
                {
                    "item_id": work_item.item_id,
                    "duration": duration,
                    "tokens": token_usage.get("total_tokens", 0),
                },
            )

            # Reset consecutive rate limit counter on success (thread-safe)
            async with self._rate_limit_lock:
                self._consecutive_rate_limits = 0

            return work_result

        except Exception as e:
            # Try to extract token usage from failed LLM calls using robust extraction
            # Even if validation fails, the LLM consumed tokens
            failed_token_usage = self._extract_token_usage(e)
            if failed_token_usage and failed_token_usage.get("total_tokens", 0) > 0:
                logger.debug(
                    f"Extracted token usage from failed attempt for {work_item.item_id}: "
                    f"{failed_token_usage['total_tokens']} tokens"
                )

            # Classify the error
            error_info = self.error_classifier.classify(e)

            # Check if it's a rate limit
            if error_info.is_rate_limit:
                # Update stats (thread-safe)
                # Note: Don't increment 'total' for re-queued items as that inflates the count
                # The item will be re-processed, so it stays in the original total
                async with self._stats_lock:
                    self._stats["rate_limit_count"] += 1

                await self._emit_event(
                    ProcessingEvent.RATE_LIMIT_HIT,
                    {"item_id": work_item.item_id, "worker_id": worker_id},
                )

                # Re-queue the item
                await self._queue.put(work_item)

                # Handle rate limit (cooldown)
                await self._handle_rate_limit(worker_id)

                # Signal to worker to not count this as processed
                raise RateLimitException(str(e)) from e

            # If error is retryable, re-raise to trigger retry in _process_item_with_retries
            # Note: Cache invalidation is automatic because retries use different temperatures,
            # which creates different cache keys and bypasses any cached bad responses
            if error_info.is_retryable:
                # Log detailed error info for debugging
                error_name = type(e).__name__
                error_msg = str(e)

                # Log token usage for failed attempt if we have it
                token_msg = ""
                if failed_token_usage:
                    token_msg = f" ({failed_token_usage.get('total_tokens', 0)} tokens consumed)"

                # For validation errors, try to extract field-level details
                # PydanticAI wraps validation errors in UnexpectedModelBehavior
                is_validation_type = (
                    'validation' in error_name.lower()
                    or 'unexpectedmodelbehavior' in error_name.lower()
                    or 'result validation' in error_msg.lower()
                )

                if is_validation_type:
                    # Try to extract raw LLM response and underlying ValidationError
                    raw_response = None
                    underlying_validation_error = None

                    try:
                        # Walk the exception chain to find ValidationError and raw response
                        current = e
                        depth = 0
                        while current and depth < 10:
                            # Try to extract raw response
                            if hasattr(current, 'response'):
                                raw_response = str(current.response)[:1000]
                            if hasattr(current, 'messages'):
                                try:
                                    raw_response = str(current.messages)[:1000]
                                except Exception:
                                    pass

                            # Check if this is a ValidationError
                            from pydantic import ValidationError
                            if isinstance(current, ValidationError):
                                underlying_validation_error = current
                                break

                            # Move to cause
                            current = getattr(current, '__cause__', None)
                            depth += 1
                    except Exception:
                        pass

                    # Try to parse Pydantic ValidationError for field details
                    try:
                        from pydantic import ValidationError
                        if underlying_validation_error or isinstance(e, ValidationError):
                            validation_err = underlying_validation_error or e
                            error_details = []
                            for err in validation_err.errors():
                                field_path = ' -> '.join(str(loc) for loc in err['loc'])
                                error_details.append(
                                    f"    Field: {field_path}\n"
                                    f"      Type: {err['type']}\n"
                                    f"      Message: {err['msg']}\n"
                                    f"      Input: {str(err.get('input', 'N/A'))[:100]}"
                                )

                            log_msg = (
                                f"✗ Validation error on attempt {attempt_number} for {work_item.item_id}{token_msg}:\n"
                                f"  Error type: {error_name}\n"
                                f"  Field-level errors:\n" + "\n".join(error_details)
                            )
                            if raw_response:
                                log_msg += f"\n  Raw LLM response (first 1000 chars):\n{raw_response}"
                            logger.error(log_msg)
                        else:
                            # Not a Pydantic ValidationError, log full error with more context
                            log_msg = (
                                f"✗ Validation error on attempt {attempt_number} for {work_item.item_id}{token_msg}:\n"
                                f"  Error type: {error_name}\n"
                                f"  Full error message: {error_msg}\n"
                                f"  Exception chain:"
                            )
                            # Show exception chain for debugging
                            current = e
                            depth = 0
                            while current and depth < 5:
                                log_msg += f"\n    {depth}: {type(current).__name__}: {str(current)[:200]}"
                                current = getattr(current, '__cause__', None)
                                depth += 1
                            if raw_response:
                                log_msg += f"\n  Raw LLM response (first 1000 chars):\n{raw_response}"
                            logger.error(log_msg)
                    except Exception as parse_error:
                        # Fallback if we can't parse the error
                        log_msg = (
                            f"✗ Validation error on attempt {attempt_number} for {work_item.item_id}{token_msg}:\n"
                            f"  Error type: {error_name}\n"
                            f"  Full error: {error_msg}\n"
                            f"  (Failed to parse error details: {parse_error})"
                        )
                        if raw_response:
                            log_msg += f"\n  Raw LLM response (first 1000 chars):\n{raw_response}"
                        logger.error(log_msg)
                else:
                    logger.debug(
                        f"Retryable {error_name} on attempt {attempt_number} for {work_item.item_id}: {error_msg[:300]}"
                    )

                raise

            # Try middleware error handlers
            middleware_result = await self._run_middlewares_on_error(work_item, e)
            if middleware_result is not None:
                return middleware_result

            # Log non-retryable error with full details
            error_name = type(e).__name__
            error_msg = str(e)

            token_summary = ""
            if failed_token_usage:
                token_summary = f"\n  Tokens consumed: {failed_token_usage.get('total_tokens', 0)}"

            logger.error(
                f"✗ PERMANENT FAILURE for {work_item.item_id}:\n"
                f"  Error type: {error_name}\n"
                f"  Error message: {error_msg[:500]}\n"
                f"  This error will NOT be retried (not retryable){token_summary}"
            )

            await self._emit_event(
                ProcessingEvent.ITEM_FAILED,
                {"item_id": work_item.item_id, "error_type": error_name},
            )

            return WorkItemResult(
                item_id=work_item.item_id,
                success=False,
                error=f"{error_name}: {error_msg[:500]}",
                context=work_item.context,
                token_usage=failed_token_usage,
            )
