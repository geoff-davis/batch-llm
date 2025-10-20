"""Tests for different LLMCallStrategy implementations and patterns."""

import asyncio
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from batch_llm import (
    LLMCallStrategy,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,
)
from batch_llm.testing import MockAgent


class TestOutput(BaseModel):
    """Test output model."""

    value: Annotated[str, Field(description="Test value")]
    temperature: Annotated[float | None, Field(description="Temperature used")] = None


@pytest.mark.asyncio
async def test_pydantic_ai_strategy():
    """Test using PydanticAIStrategy with a MockAgent."""

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}", temperature=0.0)

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)

    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Test with PydanticAIStrategy
    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test prompt",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.succeeded == 1
    assert result.results[0].output.value == "Response: Test prompt"


@pytest.mark.asyncio
async def test_custom_strategy_with_progressive_temperature():
    """Test using a custom strategy that adjusts temperature based on attempt."""

    call_log = []

    class ProgressiveTempStrategy(LLMCallStrategy[TestOutput]):
        """Strategy that increases temperature with each retry attempt."""

        def __init__(self):
            self.temps = [0.0, 0.25, 0.5]

        async def execute(
            self, prompt: str, attempt: int, timeout: float
        ) -> tuple[TestOutput, dict[str, int]]:
            temp = self.temps[min(attempt - 1, len(self.temps) - 1)]
            call_log.append({"attempt": attempt, "temp": temp})

            # Fail first two attempts to test retry logic
            if len(call_log) < 3:
                raise Exception(f"Failed attempt {len(call_log)}")

            return TestOutput(value="Success", temperature=temp), {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Test with custom strategy
    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=ProgressiveTempStrategy(),
        prompt="Test prompt",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should succeed on third attempt with temp=0.5
    assert result.succeeded == 1
    assert len(call_log) == 3
    assert call_log[0]["temp"] == 0.0
    assert call_log[1]["temp"] == 0.25
    assert call_log[2]["temp"] == 0.5


@pytest.mark.asyncio
async def test_custom_strategy_with_simulated_api_call():
    """Test a custom strategy that simulates direct API calls."""

    call_log = []

    class DirectAPIStrategy(LLMCallStrategy[TestOutput]):
        """Strategy that simulates calling an API directly."""

        async def execute(
            self, prompt: str, attempt: int, timeout: float
        ) -> tuple[TestOutput, dict[str, int]]:
            temps = [0.0, 0.25, 0.5]
            temp = temps[min(attempt - 1, len(temps) - 1)]

            call_log.append({"attempt": attempt, "temp": temp, "timeout": timeout})

            # Simulate API latency
            await asyncio.sleep(0.01)

            # Return result and token usage
            return (
                TestOutput(value=f"Direct call result (temp={temp})", temperature=temp),
                {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Test with custom strategy
    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=DirectAPIStrategy(),
        prompt="Test prompt",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should succeed
    assert result.succeeded == 1
    assert len(call_log) == 1
    assert call_log[0]["attempt"] == 1
    assert call_log[0]["temp"] == 0.0
    assert call_log[0]["timeout"] == 10.0

    # Check token usage
    assert result.results[0].token_usage["total_tokens"] == 150


@pytest.mark.asyncio
async def test_custom_strategy_with_retries():
    """Test that custom strategies respect retry logic."""

    call_log = []

    class FailingStrategy(LLMCallStrategy[TestOutput]):
        """Fail first attempt, succeed on second."""

        async def execute(
            self, prompt: str, attempt: int, timeout: float
        ) -> tuple[TestOutput, dict[str, int]]:
            temps = [0.0, 0.25, 0.5]
            temp = temps[min(attempt - 1, len(temps) - 1)]

            call_log.append({"attempt": attempt, "temp": temp})

            if attempt == 1:
                raise Exception("Simulated failure on first attempt")

            return (
                TestOutput(value=f"Success on attempt {attempt}", temperature=temp),
                {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=FailingStrategy(),
        prompt="Test prompt",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should succeed on second attempt with temp=0.25
    assert result.succeeded == 1
    assert len(call_log) == 2
    assert call_log[0]["attempt"] == 1
    assert call_log[0]["temp"] == 0.0
    assert call_log[1]["attempt"] == 2
    assert call_log[1]["temp"] == 0.25


@pytest.mark.asyncio
async def test_custom_strategy_timeout_handling():
    """Test that custom strategies respect timeout."""

    class SlowStrategy(LLMCallStrategy[TestOutput]):
        """Strategy that takes too long."""

        async def execute(
            self, prompt: str, attempt: int, timeout: float
        ) -> tuple[TestOutput, dict[str, int]]:
            # Sleep longer than timeout to ensure it triggers
            await asyncio.sleep(1.0)  # 1 second sleep
            return TestOutput(value="Should timeout"), {}

    # Use very short timeout and no retries to make test fast
    from batch_llm.core import RetryConfig
    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=0.05,  # Very short timeout (50ms)
        retry=RetryConfig(max_attempts=1),  # No retries
    )
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=SlowStrategy(),
        prompt="Test prompt",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should fail due to timeout
    assert result.failed == 1
    assert result.results[0].error is not None
    # Just check it's not successful
    assert not result.results[0].success


@pytest.mark.asyncio
async def test_context_preserved_across_strategies():
    """Test that context is preserved with different strategies."""

    context_data = {"user_id": 123, "request_id": "abc"}

    # Test PydanticAIStrategy
    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, dict](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=context_data,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.results[0].context == context_data


@pytest.mark.asyncio
async def test_strategy_lifecycle_prepare_and_cleanup():
    """Test that strategy prepare and cleanup are called."""

    lifecycle_log = []

    class LifecycleStrategy(LLMCallStrategy[TestOutput]):
        """Strategy that tracks lifecycle calls."""

        async def prepare(self):
            lifecycle_log.append("prepare")

        async def execute(
            self, prompt: str, attempt: int, timeout: float
        ) -> tuple[TestOutput, dict[str, int]]:
            lifecycle_log.append("execute")
            return TestOutput(value="Success"), {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }

        async def cleanup(self):
            lifecycle_log.append("cleanup")

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[str, TestOutput, None](config=config) as processor:
        work_item = LLMWorkItem(
            item_id="item_1",
            strategy=LifecycleStrategy(),
            prompt="Test",
            context=None,
        )
        await processor.add_work(work_item)
        result = await processor.process_all()

    # Verify lifecycle
    assert result.succeeded == 1
    assert "prepare" in lifecycle_log
    assert "execute" in lifecycle_log
    assert "cleanup" in lifecycle_log
    # Prepare should come before execute
    assert lifecycle_log.index("prepare") < lifecycle_log.index("execute")
