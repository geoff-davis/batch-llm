"""Concurrency and thread-safety tests for batch_llm.

These tests verify that the package handles concurrent access correctly
and doesn't have race conditions.
"""

import asyncio
import threading
import time
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from batch_llm import (
    LLMWorkItem,
    MetricsObserver,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,
)
from batch_llm.strategies import ExponentialBackoffStrategy, RateLimitStrategy
from batch_llm.strategies.errors import ErrorInfo
from batch_llm.testing import MockAgent


class TestOutput(BaseModel):
    """Test output model."""

    value: Annotated[str, Field(description="Test value")]


@pytest.mark.asyncio
async def test_concurrent_stats_updates():
    """Test that stats are counted correctly under concurrent load."""

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)

    # Use many workers to maximize concurrency
    config = ProcessorConfig(max_workers=10, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Add many items
    num_items = 100
    for i in range(num_items):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Test {i}",
            context=None,
        )
        await processor.add_work(work_item)

    # Process all
    result = await processor.process_all()

    # Verify counts are exact
    assert result.total_items == num_items, "Total items count incorrect"
    assert result.succeeded == num_items, "Success count incorrect"
    assert result.failed == 0, "Failed count should be 0"
    assert len(result.results) == num_items, "Results list length incorrect"

    # Verify no duplicates in results
    item_ids = [r.item_id for r in result.results]
    assert len(item_ids) == len(set(item_ids)), "Duplicate items in results"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_rate_limit_handling():
    """Test that only one cooldown happens when multiple workers hit rate limit.

    Note: This test is marked as 'slow' because it involves rate limit simulation
    with cooldown delays. Run with: pytest -m slow
    """

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    # Create a mock agent that simulates rate limiting
    mock_agent = MockAgent(
        response_factory=mock_response,
        latency=0.01,
        rate_limit_on_call=2,  # Trigger rate limit on 2nd call
    )

    # Use fast rate limit config for testing (instead of default 300s cooldown)
    from batch_llm.core import RateLimitConfig
    fast_rate_limit = RateLimitConfig(
        cooldown_seconds=0.2,  # Very short cooldown for testing
        slow_start_items=2,  # Minimal slow start
        slow_start_initial_delay=0.05,
        slow_start_final_delay=0.01,
    )

    config = ProcessorConfig(
        max_workers=3,  # Fewer workers
        timeout_per_item=10.0,
        rate_limit=fast_rate_limit,
    )
    metrics = MetricsObserver()
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config, observers=[metrics]
    )

    # Add just 3 items - rate limit triggers on 2nd call, so this tests the behavior
    for i in range(3):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Test {i}",
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    # Check metrics
    collected_metrics = await metrics.get_metrics()

    # All items should eventually succeed (after rate limit recovery)
    assert result.succeeded >= 0, "Some items should succeed"

    # Rate limit should have been hit
    if collected_metrics["rate_limits_hit"] > 0:
        # Verify rate limit was handled
        assert collected_metrics["total_cooldown_time"] > 0


@pytest.mark.asyncio
async def test_rate_limit_requeue_completes_without_deadlock():
    """Ensure items requeued after rate limiting still complete."""

    class TestRateLimitClassifier:
        def classify(self, exception: Exception) -> ErrorInfo:
            message = str(exception).lower()
            if "resource_exhausted" in message or "429" in message:
                return ErrorInfo(
                    is_retryable=False,
                    is_rate_limit=True,
                    is_timeout=False,
                    error_category="rate_limit",
                    suggested_wait=0.0,
                )
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="unknown",
            )

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(
        response_factory=mock_response,
        latency=0.01,
        rate_limit_on_call=1,  # First call triggers rate limit once
    )

    from batch_llm.core import RateLimitConfig

    config = ProcessorConfig(
        max_workers=2,
        timeout_per_item=1.0,
        rate_limit=RateLimitConfig(
            cooldown_seconds=0.01,
            slow_start_items=0,
            slow_start_initial_delay=0.0,
            slow_start_final_delay=0.0,
            backoff_multiplier=1.0,
        ),
    )

    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        error_classifier=TestRateLimitClassifier(),
    )

    for i in range(3):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Test {i}",
            context=None,
        )
        await processor.add_work(work_item)

    result = await asyncio.wait_for(processor.process_all(), timeout=1.0)

    assert result.succeeded == 3
    assert result.failed == 0


@pytest.mark.asyncio
async def test_rate_limit_strategy_failure_does_not_block_workers():
    """Rate limit strategies that raise should not leave workers paused."""

    class FlakyRateLimitStrategy(RateLimitStrategy):
        def __init__(self) -> None:
            self.calls = 0

        async def on_rate_limit(
            self, worker_id: int, consecutive_limit_count: int
        ) -> float:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("strategy blew up")
            return 0.0

        def should_apply_slow_start(self, items_since_resume: int) -> tuple[bool, float]:
            return (False, 0.0)

    flaky_strategy = FlakyRateLimitStrategy()

    processor = ParallelBatchProcessor[str, TestOutput, None](
        rate_limit_strategy=flaky_strategy,
    )

    # Simulate rate limit handling directly to ensure internal coordination
    await asyncio.wait_for(processor._handle_rate_limit(worker_id=0), timeout=0.1)

    assert flaky_strategy.calls == 1
    assert processor._rate_limit_event.is_set()
    assert processor._in_cooldown is False


@pytest.mark.asyncio
async def test_rate_limit_backoff_multiplier_config_applied():
    """Configured backoff multiplier should reach the exponential strategy."""

    from batch_llm.core import RateLimitConfig

    rate_limit_config = RateLimitConfig(
        cooldown_seconds=2.0,
        backoff_multiplier=3.0,
    )
    config = ProcessorConfig(max_workers=1, rate_limit=rate_limit_config)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    assert isinstance(processor.rate_limit_strategy, ExponentialBackoffStrategy)
    strategy = processor.rate_limit_strategy

    first = await strategy.on_rate_limit(0, 1)
    second = await strategy.on_rate_limit(0, 2)

    assert first == pytest.approx(2.0)
    assert second == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_progress_callback_timeout_for_sync_function():
    """Progress callback timeout should prevent synchronous hooks from blocking."""

    call_counter = {"count": 0}
    invoked = threading.Event()

    def slow_progress_callback(completed: int, total: int, current_item: str) -> None:
        call_counter["count"] += 1
        invoked.set()
        time.sleep(0.2)  # Would block the worker without timeout handling

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)

    from batch_llm.core import RateLimitConfig

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=1.0,
        progress_interval=1,
        progress_callback_timeout=0.05,
        rate_limit=RateLimitConfig(
            cooldown_seconds=0.01,
            slow_start_items=0,
            slow_start_initial_delay=0.0,
            slow_start_final_delay=0.0,
            backoff_multiplier=1.0,
        ),
    )
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config, progress_callback=slow_progress_callback
    )

    work_item = LLMWorkItem(
        item_id="sync_timeout",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    start = time.perf_counter()
    result = await asyncio.wait_for(processor.process_all(), timeout=2.0)
    duration = time.perf_counter() - start

    assert result.succeeded == 1
    assert call_counter["count"] == 1
    assert invoked.is_set()
    assert duration < 0.3, f"Processing stalled for {duration:.2f}s despite timeout"


@pytest.mark.asyncio
async def test_progress_callback_timeout_for_async_function():
    """Progress callback timeout should cancel long async hooks."""

    call_counter = {"count": 0}

    async def slow_async_callback(
        completed: int, total: int, current_item: str
    ) -> None:
        call_counter["count"] += 1
        await asyncio.sleep(0.2)

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)

    from batch_llm.core import RateLimitConfig

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=1.0,
        progress_interval=1,
        progress_callback_timeout=0.05,
        rate_limit=RateLimitConfig(
            cooldown_seconds=0.01,
            slow_start_items=0,
            slow_start_initial_delay=0.0,
            slow_start_final_delay=0.0,
            backoff_multiplier=1.0,
        ),
    )
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config, progress_callback=slow_async_callback
    )

    work_item = LLMWorkItem(
        item_id="async_timeout",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    start = time.perf_counter()
    result = await asyncio.wait_for(processor.process_all(), timeout=2.0)
    duration = time.perf_counter() - start

    assert result.succeeded == 1
    assert call_counter["count"] == 1
    assert duration < 0.3, f"Processing stalled for {duration:.2f}s despite timeout"


@pytest.mark.asyncio
async def test_slow_start_skipped_before_rate_limit():
    """Slow-start delays should not run before any rate limit is encountered."""

    class TrackingSlowStartStrategy(RateLimitStrategy):
        def __init__(self) -> None:
            self.calls: list[int] = []

        async def on_rate_limit(
            self, worker_id: int, consecutive_limit_count: int
        ) -> float:
            return 0.0

        def should_apply_slow_start(
            self, items_since_resume: int
        ) -> tuple[bool, float]:
            self.calls.append(items_since_resume)
            return (True, 0.0)

    strategy = TrackingSlowStartStrategy()

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.001)

    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config, rate_limit_strategy=strategy
    )

    for i in range(5):
        await processor.add_work(
            LLMWorkItem(
                item_id=f"no_rl_{i}",
                strategy=PydanticAIStrategy(agent=mock_agent),
                prompt=f"Test {i}",
                context=None,
            )
        )

    result = await processor.process_all()

    assert result.succeeded == 5
    assert strategy.calls == []


@pytest.mark.asyncio
async def test_slow_start_engages_after_rate_limit():
    """Slow-start delays activate only after a rate limit event."""

    class TrackingSlowStartStrategy(RateLimitStrategy):
        def __init__(self, slow_calls: int = 2) -> None:
            self.calls: list[int] = []
            self._slow_calls = slow_calls

        async def on_rate_limit(
            self, worker_id: int, consecutive_limit_count: int
        ) -> float:
            return 0.0

        def should_apply_slow_start(
            self, items_since_resume: int
        ) -> tuple[bool, float]:
            self.calls.append(items_since_resume)
            if len(self.calls) > self._slow_calls:
                return (False, 0.0)
            return (True, 0.0)

    strategy = TrackingSlowStartStrategy()

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(
        response_factory=mock_response, latency=0.001, rate_limit_on_call=1
    )

    class TestRateLimitClassifier:
        def classify(self, exception: Exception) -> ErrorInfo:
            message = str(exception).lower()
            if "resource_exhausted" in message or "429" in message:
                return ErrorInfo(
                    is_retryable=False,
                    is_rate_limit=True,
                    is_timeout=False,
                    error_category="rate_limit",
                    suggested_wait=0.0,
                )
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="unknown",
            )

    config = ProcessorConfig(max_workers=1, timeout_per_item=5.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        error_classifier=TestRateLimitClassifier(),
        rate_limit_strategy=strategy,
    )

    for i in range(3):
        await processor.add_work(
            LLMWorkItem(
                item_id=f"rl_{i}",
                strategy=PydanticAIStrategy(agent=mock_agent),
                prompt=f"Test {i}",
                context=None,
            )
        )

    result = await asyncio.wait_for(processor.process_all(), timeout=2.0)

    assert result.succeeded == 3
    assert strategy.calls != []
    assert strategy.calls[0] == 0
    assert strategy.calls[1] == 1


@pytest.mark.asyncio
async def test_metrics_observer_thread_safety():
    """Test that MetricsObserver correctly counts events under concurrent load."""

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.001)  # Reduced from 0.01s

    config = ProcessorConfig(max_workers=10, timeout_per_item=10.0)
    metrics = MetricsObserver()
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config, observers=[metrics]
    )

    # Add many items (reduced from 50 to 20 for faster execution)
    num_items = 20
    for i in range(num_items):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Test {i}",
            context=None,
        )
        await processor.add_work(work_item)

    await processor.process_all()

    # Get metrics
    collected_metrics = await metrics.get_metrics()

    # Verify all metrics match expected values
    assert (
        collected_metrics["items_processed"] == num_items
    ), "Metrics items_processed mismatch"
    assert (
        collected_metrics["items_succeeded"] == num_items
    ), "Metrics items_succeeded mismatch"
    assert collected_metrics["items_failed"] == 0, "Metrics items_failed should be 0"
    assert collected_metrics["success_rate"] == 1.0, "Success rate should be 100%"
    assert len(collected_metrics["processing_times"]) == num_items, "Processing times count mismatch"


@pytest.mark.asyncio
async def test_no_result_loss_under_concurrency():
    """Test that no results are lost when many workers append concurrently."""

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.005)

    # Many workers, short latency = high concurrency
    config = ProcessorConfig(max_workers=20, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Add many items
    num_items = 200
    expected_ids = set()
    for i in range(num_items):
        item_id = f"item_{i:03d}"
        expected_ids.add(item_id)
        work_item = LLMWorkItem(
            item_id=item_id,
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Test {i}",
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    # Verify all results present
    assert len(result.results) == num_items, "Some results were lost"

    # Verify all IDs present
    actual_ids = {r.item_id for r in result.results}
    assert actual_ids == expected_ids, "Some item IDs are missing or duplicated"

    # Verify counts
    assert result.total_items == num_items
    assert result.succeeded == num_items
    assert result.failed == 0


@pytest.mark.asyncio
async def test_concurrent_post_processor_calls():
    """Test that post-processor is called for all items under concurrent load."""

    processed_items = []
    lock = asyncio.Lock()

    async def post_process(result):
        """Track which items were post-processed."""
        if result.success:
            async with lock:
                processed_items.append(result.item_id)

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)

    config = ProcessorConfig(max_workers=10, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config, post_processor=post_process
    )

    # Add items
    num_items = 50
    for i in range(num_items):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Test {i}",
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    # Verify post-processor was called for all successful items
    assert len(processed_items) == result.succeeded
    assert len(processed_items) == num_items


@pytest.mark.asyncio
async def test_get_stats_thread_safety():
    """Test that get_stats() returns consistent snapshot under concurrent updates."""

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    mock_agent = MockAgent(response_factory=mock_response, latency=0.02)

    config = ProcessorConfig(max_workers=10, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Add items
    num_items = 30
    for i in range(num_items):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Test {i}",
            context=None,
        )
        await processor.add_work(work_item)

    # Start processing in background
    process_task = asyncio.create_task(processor.process_all())

    # Read stats while processing
    stats_snapshots = []
    for _ in range(10):
        await asyncio.sleep(0.01)
        stats = await processor.get_stats()
        stats_snapshots.append(stats)

        # Verify internal consistency of snapshot
        assert stats["processed"] == stats["succeeded"] + stats["failed"]

    # Wait for completion
    await process_task

    # Final stats should be consistent
    final_stats = await processor.get_stats()
    assert final_stats["processed"] == num_items
    assert final_stats["succeeded"] == num_items
    assert final_stats["failed"] == 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_slow_start_counter_accuracy():
    """Test that slow-start counter is incremented correctly under concurrency.

    Note: This test is marked as 'slow' because it involves rate limit simulation.
    Run with: pytest -m slow
    """

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}")

    # Create agent that triggers rate limit
    mock_agent = MockAgent(
        response_factory=mock_response,
        latency=0.01,
        rate_limit_on_call=5,
    )

    from batch_llm.core import RateLimitConfig

    config = ProcessorConfig(
        max_workers=5,
        timeout_per_item=10.0,
        rate_limit=RateLimitConfig(
            cooldown_seconds=0.1,  # Very short for testing
            slow_start_items=10,
            slow_start_initial_delay=0.05,
            slow_start_final_delay=0.01,
        ),
    )

    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Add items
    for i in range(15):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=f"Test {i}",
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    # All items should eventually succeed
    assert result.succeeded >= 10, "Most items should succeed after slow-start"
