"""Tests for middleware and observer functionality."""

import asyncio
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,
)
from batch_llm.middleware import BaseMiddleware
from batch_llm.observers import BaseObserver, ProcessingEvent
from batch_llm.testing import MockAgent


class TestOutput(BaseModel):
    """Test output model."""

    value: Annotated[str, Field(description="Test value")]


@pytest.mark.asyncio
async def test_middleware_before_process():
    """Test middleware before_process hook."""

    modified_items = []

    class ModifyingMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            # Track what we modified
            modified_items.append(work_item.item_id)
            # Modify the prompt
            work_item.prompt = f"Modified: {work_item.prompt}"
            return work_item

    prompts_received = []

    def track_prompt(prompt: str) -> TestOutput:
        prompts_received.append(prompt)
        return TestOutput(value="ok")

    mock_agent = MockAgent(response_factory=track_prompt, latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[ModifyingMiddleware()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Original prompt",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Middleware should have modified the prompt
    assert result.succeeded == 1
    assert len(modified_items) == 1
    assert "item_1" in modified_items
    assert prompts_received[0] == "Modified: Original prompt"


@pytest.mark.asyncio
async def test_middleware_after_process():
    """Test middleware after_process hook."""

    modified_results = []

    class ResultModifyingMiddleware(BaseMiddleware):
        async def after_process(self, result):
            # Track and modify result
            modified_results.append(result.item_id)
            # Could modify result here if needed
            return result

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[ResultModifyingMiddleware()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Middleware should have seen the result
    assert result.succeeded == 1
    assert "item_1" in modified_results


@pytest.mark.asyncio
async def test_middleware_can_skip_items():
    """Test that middleware can skip items by returning None."""

    skipped_items = []

    class SkippingMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            if "skip" in work_item.prompt.lower():
                skipped_items.append(work_item.item_id)
                return None  # Skip this item
            return work_item

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[SkippingMiddleware()],
    )

    # Add items, one should be skipped
    for item_id, prompt in [("item_1", "Process this"), ("item_2", "SKIP this")]:
        work_item = LLMWorkItem(
            item_id=item_id,
            strategy=PydanticAIStrategy(agent=mock_agent),
            prompt=prompt,
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    # One should be skipped
    assert "item_2" in skipped_items
    assert len(result.results) == 2  # Both recorded
    # Skipped item should fail
    skipped_result = [r for r in result.results if r.item_id == "item_2"][0]
    assert not skipped_result.success


@pytest.mark.asyncio
async def test_middleware_on_error():
    """Test middleware on_error hook."""

    errors_handled = []

    class ErrorHandlingMiddleware(BaseMiddleware):
        async def on_error(self, work_item, error):
            errors_handled.append({
                "item_id": work_item.item_id,
                "error": str(error)
            })
            # Return None to let default error handling proceed
            return None

    def always_fail(prompt: str) -> TestOutput:
        raise ValueError("Simulated error")

    mock_agent = MockAgent(response_factory=always_fail, latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[ErrorHandlingMiddleware()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Middleware should have seen the error
    assert result.failed == 1
    assert len(errors_handled) > 0  # At least one error


@pytest.mark.asyncio
async def test_multiple_middlewares_execute_in_order():
    """Test that multiple middlewares execute in order."""

    execution_order = []

    class FirstMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            execution_order.append("first_before")
            return work_item

        async def after_process(self, result):
            execution_order.append("first_after")
            return result

    class SecondMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            execution_order.append("second_before")
            return work_item

        async def after_process(self, result):
            execution_order.append("second_after")
            return result

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[FirstMiddleware(), SecondMiddleware()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Verify execution order
    assert result.succeeded == 1
    assert execution_order == [
        "first_before",
        "second_before",
        "second_after",
        "first_after",
    ]


@pytest.mark.asyncio
async def test_observer_receives_all_events():
    """Test that observers receive all processing events."""

    events_received = []

    class TrackingObserver(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            events_received.append({
                "event": event.name,
                "data": data.copy()
            })

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        observers=[TrackingObserver()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Check we received expected events
    assert result.succeeded == 1
    event_names = [e["event"] for e in events_received]

    assert "WORKER_STARTED" in event_names
    assert "ITEM_STARTED" in event_names
    assert "ITEM_COMPLETED" in event_names
    assert "WORKER_STOPPED" in event_names


@pytest.mark.asyncio
async def test_observer_receives_failure_events():
    """Test that observers receive failure events."""

    events_received = []

    class TrackingObserver(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            events_received.append(event.name)

    def always_fail(prompt: str) -> TestOutput:
        raise ValueError("Simulated failure")

    # Create custom error classifier that marks as non-retryable
    class NonRetryableClassifier:
        def classify(self, exception):
            from batch_llm.strategies import ErrorInfo
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="test_error",
            )

    mock_agent = MockAgent(response_factory=always_fail, latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        observers=[TrackingObserver()],
        error_classifier=NonRetryableClassifier(),
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Check we received failure event
    assert result.failed == 1
    assert "ITEM_FAILED" in events_received


@pytest.mark.asyncio
async def test_multiple_observers_all_receive_events():
    """Test that multiple observers all receive events."""

    observer1_events = []
    observer2_events = []

    class Observer1(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            observer1_events.append(event.name)

    class Observer2(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            observer2_events.append(event.name)

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        observers=[Observer1(), Observer2()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Both observers should receive events
    assert result.succeeded == 1
    assert len(observer1_events) > 0
    assert len(observer2_events) > 0
    assert observer1_events == observer2_events  # Same events


@pytest.mark.asyncio
async def test_observer_timeout_doesnt_break_processing():
    """Test that slow observers don't break processing."""

    class SlowObserver(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            # Sleep longer than observer timeout (5s)
            await asyncio.sleep(10.0)

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        observers=[SlowObserver()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Processing should still succeed despite slow observer
    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_middleware_and_observers_work_together():
    """Test that middleware and observers work together correctly."""

    middleware_calls = []
    observer_events = []

    class TestMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            middleware_calls.append("before")
            return work_item

        async def after_process(self, result):
            middleware_calls.append("after")
            return result

    class TestObserver(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]):
            observer_events.append(event.name)

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[TestMiddleware()],
        observers=[TestObserver()],
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Both should have been called
    assert result.succeeded == 1
    assert "before" in middleware_calls
    assert "after" in middleware_calls
    assert "ITEM_STARTED" in observer_events
    assert "ITEM_COMPLETED" in observer_events


@pytest.mark.asyncio
async def test_middleware_returns_none_original_item_id_preserved():
    """Test that when middleware returns None, we still have access to the original item_id.

    This is a regression test for a bug where accessing work_item.item_id after
    middleware returned None would cause an AttributeError.
    """

    class FilterMiddleware(BaseMiddleware):
        async def before_process(self, work_item):
            # Return None to skip the item
            return None

        async def after_process(self, result):
            return result

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        middlewares=[FilterMiddleware()],
    )

    work_item = LLMWorkItem(
        item_id="filtered_item",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Item should be marked as failed with item_id preserved
    assert result.total_items == 1
    assert result.failed == 1
    assert result.results[0].item_id == "filtered_item"
    assert result.results[0].error == "Skipped by middleware"
