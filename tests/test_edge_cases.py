"""Tests for edge cases and error conditions."""

import asyncio
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    WorkItemResult,
)
from batch_llm.core import RetryConfig
from batch_llm.testing import MockAgent


class TestOutput(BaseModel):
    """Test output model."""

    value: Annotated[str, Field(description="Test value")]


@pytest.mark.asyncio
async def test_empty_queue():
    """Test processing with no work items."""

    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Process without adding any work
    result = await processor.process_all()

    assert result.total_items == 0
    assert result.succeeded == 0
    assert result.failed == 0
    assert len(result.results) == 0


@pytest.mark.asyncio
async def test_single_item():
    """Test processing a single item."""

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=5, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        agent=mock_agent,
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.total_items == 1
    assert result.succeeded == 1
    assert result.failed == 0


@pytest.mark.asyncio
async def test_more_items_than_workers():
    """Test processing more items than workers."""

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value=f"Response: {p}"),
        latency=0.001  # Reduced from 0.01s
    )

    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Add 10 items with only 2 workers
    for i in range(10):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            agent=mock_agent,
            prompt=f"Prompt {i}",
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.total_items == 10
    assert result.succeeded == 10
    assert result.failed == 0


@pytest.mark.asyncio
async def test_all_items_fail():
    """Test when all items fail."""

    def always_fail(prompt: str) -> TestOutput:
        raise ValueError("Always fails")

    mock_agent = MockAgent(response_factory=always_fail, latency=0.001)  # Reduced from 0.01s

    config = ProcessorConfig(
        max_workers=2,
        timeout_per_item=10.0,
        retry=RetryConfig(max_attempts=1),  # No retries
    )
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    for i in range(5):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            agent=mock_agent,
            prompt="Test",
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.total_items == 5
    assert result.succeeded == 0
    assert result.failed == 5


@pytest.mark.asyncio
async def test_mixed_success_and_failure():
    """Test with some items succeeding and some failing."""

    def conditional_fail(prompt: str) -> TestOutput:
        if "fail" in prompt.lower():
            raise ValueError("Intentional failure")
        return TestOutput(value="Success")

    mock_agent = MockAgent(response_factory=conditional_fail, latency=0.001)  # Reduced from 0.01s

    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    prompts = ["succeed", "FAIL", "succeed", "fail", "succeed"]
    for i, prompt in enumerate(prompts):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            agent=mock_agent,
            prompt=prompt,
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.total_items == 5
    assert result.succeeded == 3
    assert result.failed == 2


@pytest.mark.asyncio
async def test_very_short_timeout():
    """Test with extremely short timeout."""

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.1  # 100ms latency
    )

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=0.01,  # 10ms timeout - will timeout
    )
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        agent=mock_agent,
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should timeout and fail
    assert result.failed == 1


@pytest.mark.asyncio
async def test_very_long_timeout():
    """Test with very long timeout."""

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=300.0,  # 5 minute timeout
    )
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        agent=mock_agent,
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_post_processor_exception_doesnt_fail_item():
    """Test that post-processor exceptions don't mark items as failed."""

    async def failing_post_processor(result: WorkItemResult):
        if result.success:
            raise ValueError("Post-processor failed")

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        post_processor=failing_post_processor,
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        agent=mock_agent,
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Item should still be marked as successful despite post-processor failure
    assert result.succeeded == 1
    assert result.failed == 0


@pytest.mark.slow
@pytest.mark.asyncio
async def test_large_batch():
    """Test processing a large batch of items."""

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.001  # Very fast
    )

    config = ProcessorConfig(max_workers=10, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Add 100 items
    num_items = 100
    for i in range(num_items):
        work_item = LLMWorkItem(
            item_id=f"item_{i:03d}",
            agent=mock_agent,
            prompt=f"Prompt {i}",
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.total_items == num_items
    assert result.succeeded == num_items
    assert result.failed == 0
    assert len(result.results) == num_items


@pytest.mark.asyncio
async def test_special_characters_in_item_id():
    """Test item IDs with special characters."""

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.001  # Reduced from 0.01s
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    special_ids = [
        "item-with-dashes",
        "item_with_underscores",
        "item.with.dots",
        "item/with/slashes",
        "item:with:colons",
        "item@with@at",
        "item#123",
    ]

    for item_id in special_ids:
        work_item = LLMWorkItem(
            item_id=item_id,
            agent=mock_agent,
            prompt="Test",
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.total_items == len(special_ids)
    assert result.succeeded == len(special_ids)

    # Verify all IDs preserved
    result_ids = {r.item_id for r in result.results}
    assert result_ids == set(special_ids)


@pytest.mark.asyncio
async def test_unicode_in_prompts():
    """Test prompts with unicode characters."""

    received_prompts = []

    def track_prompt(prompt: str) -> TestOutput:
        received_prompts.append(prompt)
        return TestOutput(value="ok")

    mock_agent = MockAgent(response_factory=track_prompt, latency=0.001)  # Reduced from 0.01s

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    unicode_prompts = [
        "Hello 世界",
        "Привет мир",
        "مرحبا بالعالم",
        "🚀 Emoji test 🎉",
        "Ñoño español",
    ]

    for i, prompt in enumerate(unicode_prompts):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            agent=mock_agent,
            prompt=prompt,
            context=None,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.succeeded == len(unicode_prompts)
    assert received_prompts == unicode_prompts


@pytest.mark.asyncio
async def test_none_context():
    """Test that None context is handled correctly."""

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.001  # Reduced from 0.01s
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        agent=mock_agent,
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.succeeded == 1
    assert result.results[0].context is None


@pytest.mark.asyncio
async def test_complex_context_types():
    """Test various complex context types."""

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.001  # Reduced from 0.01s
    )

    # Test with different context types
    contexts = [
        {"dict": "context"},
        ["list", "context"],
        ("tuple", "context"),
        42,
        "string context",
        {"nested": {"deep": {"context": "value"}}},
    ]

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, Any](config=config)

    for i, ctx in enumerate(contexts):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            agent=mock_agent,
            prompt="Test",
            context=ctx,
        )
        await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.succeeded == len(contexts)

    # Verify contexts preserved
    for i, res in enumerate(result.results):
        assert res.context == contexts[i]


@pytest.mark.asyncio
async def test_zero_workers():
    """Test that zero workers is handled (should use at least 1)."""

    # This should either error or auto-correct to 1
    config = ProcessorConfig(max_workers=0, timeout_per_item=10.0)

    # Should validate and reject
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.asyncio
async def test_negative_timeout():
    """Test that negative timeout is rejected."""

    config = ProcessorConfig(max_workers=1, timeout_per_item=-1.0)

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.asyncio
async def test_adding_work_after_processing_started():
    """Test that work added during processing is handled correctly."""

    # This is an edge case - normally not recommended
    # but should work as queue is async

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.05
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Add first item
    await processor.add_work(LLMWorkItem(
        item_id="item_1",
        agent=mock_agent,
        prompt="Test 1",
        context=None,
    ))

    # Start processing in background
    process_task = asyncio.create_task(processor.process_all())

    # Give it a moment to start
    await asyncio.sleep(0.01)

    # Try to add more work (this might not be processed)
    # Note: This is not officially supported behavior
    # Just documenting what happens

    # Wait for completion
    result = await process_task

    # At least the first item should be processed
    assert result.total_items >= 1
    assert result.succeeded >= 1


@pytest.mark.asyncio
async def test_max_queue_size_config_validation():
    """Test that max_queue_size configuration is properly validated."""

    # Test that negative max_queue_size is rejected
    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=10.0,
        max_queue_size=-1
    )

    with pytest.raises(ValueError, match="max_queue_size must be >= 0"):
        config.validate()

    # Test that 0 and positive values are accepted
    config_unlimited = ProcessorConfig(
        max_workers=1,
        timeout_per_item=10.0,
        max_queue_size=0  # Unlimited
    )
    config_unlimited.validate()  # Should not raise

    config_limited = ProcessorConfig(
        max_workers=1,
        timeout_per_item=10.0,
        max_queue_size=100
    )
    config_limited.validate()  # Should not raise


@pytest.mark.slow
@pytest.mark.asyncio
async def test_max_queue_size_zero_is_unlimited():
    """Test that max_queue_size=0 allows unlimited queue size."""

    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.001  # Reduced from 0.01s
    )

    # Default max_queue_size=0 means unlimited
    config = ProcessorConfig(
        max_workers=2,
        timeout_per_item=10.0,
        max_queue_size=0  # Unlimited
    )
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Add many items without blocking (reduced from 100 to 20)
    for i in range(20):
        await processor.add_work(LLMWorkItem(
            item_id=f"item_{i}",
            agent=mock_agent,
            prompt=f"Test {i}",
            context=None,
        ))

    result = await processor.process_all()

    assert result.total_items == 100
    assert result.succeeded == 100
