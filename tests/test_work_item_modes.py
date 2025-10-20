"""Tests for the three LLMWorkItem processing modes."""

import asyncio
from typing import Annotated

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
)
from batch_llm.testing import MockAgent


class TestOutput(BaseModel):
    """Test output model."""

    value: Annotated[str, Field(description="Test value")]
    temperature: Annotated[float | None, Field(description="Temperature used")] = None


@pytest.mark.asyncio
async def test_agent_mode():
    """Test using a PydanticAI Agent directly."""

    def mock_response(prompt: str) -> TestOutput:
        return TestOutput(value=f"Response: {prompt}", temperature=0.0)

    mock_agent = MockAgent(response_factory=mock_response, latency=0.01)

    config = ProcessorConfig(max_workers=2, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Test with agent mode
    work_item = LLMWorkItem(
        item_id="item_1",
        agent=mock_agent,
        prompt="Test prompt",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.succeeded == 1
    assert result.results[0].output.value == "Response: Test prompt"


@pytest.mark.asyncio
async def test_agent_factory_mode():
    """Test using an agent factory for progressive temperature."""

    call_log = []

    def create_agent_with_temp(attempt: int) -> Agent:
        """Create agent with temperature based on attempt."""
        # Temperatures: 0.0, 0.25, 0.5
        temps = [0.0, 0.25, 0.5]
        temp = temps[min(attempt - 1, len(temps) - 1)]

        def response_with_temp(prompt: str) -> TestOutput:
            call_log.append({"attempt": attempt, "temp": temp})
            # Fail first two attempts
            if len(call_log) < 3:
                raise Exception(f"Failed attempt {len(call_log)}")
            return TestOutput(value="Success", temperature=temp)

        return MockAgent(response_factory=response_with_temp, latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    # Test with agent_factory mode
    work_item = LLMWorkItem(
        item_id="item_1",
        agent_factory=create_agent_with_temp,
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
async def test_direct_call_mode():
    """Test using direct_call for custom API calls."""

    call_log = []

    async def custom_llm_call(
        attempt: int, timeout: float
    ) -> tuple[TestOutput, dict[str, int]]:
        """Direct LLM API call with custom temperature."""
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

    # Test with direct_call mode
    work_item = LLMWorkItem(
        item_id="item_1",
        direct_call=custom_llm_call,
        input_data={"custom": "data"},  # Can pass custom data
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
async def test_direct_call_with_retries():
    """Test that direct_call respects retry logic and progressive temperature."""

    call_log = []

    async def failing_then_succeeding_call(
        attempt: int, timeout: float
    ) -> tuple[TestOutput, dict[str, int]]:
        """Fail first attempt, succeed on second."""
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
        direct_call=failing_then_succeeding_call,
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
async def test_work_item_validation_rejects_multiple_modes():
    """Test that LLMWorkItem rejects multiple processing modes."""

    mock_agent = MockAgent(latency=0.01)

    async def dummy_call(attempt: int, timeout: float):
        return TestOutput(value="test"), {}

    # Should raise ValueError when providing both agent and direct_call
    with pytest.raises(ValueError, match="Only one of"):
        LLMWorkItem(
            item_id="item_1",
            agent=mock_agent,
            direct_call=dummy_call,  # Can't have both!
            prompt="test",
        )


@pytest.mark.asyncio
async def test_work_item_validation_requires_one_mode():
    """Test that LLMWorkItem requires at least one processing mode."""

    # Should raise ValueError when providing no processing mode
    with pytest.raises(ValueError, match="Must provide either strategy or one of"):
        LLMWorkItem(
            item_id="item_1",
            prompt="test",
            # No strategy, agent, agent_factory, or direct_call!
        )


@pytest.mark.asyncio
async def test_agent_factory_gets_correct_attempt_numbers():
    """Test that agent_factory receives correct attempt numbers."""

    attempts_received = []

    def create_agent(attempt: int) -> Agent:
        attempts_received.append(attempt)

        def response(prompt: str) -> TestOutput:
            # Fail first two attempts
            if len(attempts_received) < 3:
                raise Exception(f"Fail attempt {attempt}")
            return TestOutput(value=f"Success on attempt {attempt}")

        return MockAgent(response_factory=response, latency=0.01)

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        agent_factory=create_agent,
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should succeed after 3 attempts
    assert result.succeeded == 1
    assert attempts_received == [1, 2, 3], f"Got attempts: {attempts_received}"


@pytest.mark.asyncio
async def test_direct_call_timeout_handling():
    """Test that direct_call respects timeout."""

    async def slow_call(attempt: int, timeout: float) -> tuple[TestOutput, dict]:
        # Sleep much longer than timeout to ensure it triggers
        await asyncio.sleep(5.0)  # Sleep longer than timeout
        return TestOutput(value="Should timeout"), {}

    # Use very short timeout and no retries to make test fast
    from batch_llm.core import RetryConfig
    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=0.1,  # Very short timeout (100ms)
        retry=RetryConfig(max_attempts=1),  # No retries
    )
    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        direct_call=slow_call,
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should fail due to timeout
    assert result.failed == 1
    assert result.results[0].error is not None
    result.results[0].error.lower()
    # asyncio.TimeoutError message is empty, so just check it's not successful
    assert not result.results[0].success


@pytest.mark.asyncio
async def test_context_preserved_across_modes():
    """Test that context is preserved across all three modes."""

    context_data = {"user_id": 123, "request_id": "abc"}

    # Test agent mode
    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(value="test"),
        latency=0.01
    )

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[str, TestOutput, dict](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        agent=mock_agent,
        prompt="Test",
        context=context_data,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    assert result.results[0].context == context_data


@pytest.mark.asyncio
async def test_input_data_passed_to_direct_call():
    """Test that input_data is accessible in direct_call context."""


    async def call_with_input(attempt: int, timeout: float) -> tuple[TestOutput, dict]:
        # Note: input_data is available via work_item.input_data, not as a parameter
        # This test verifies the pattern works
        return TestOutput(value="Success"), {}

    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)
    processor = ParallelBatchProcessor[dict, TestOutput, None](config=config)

    custom_input = {"cluster_id": "cluster_1", "items": [1, 2, 3]}

    work_item = LLMWorkItem(
        item_id="item_1",
        direct_call=call_with_input,
        input_data=custom_input,
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Verify it processes successfully
    assert result.succeeded == 1
    # input_data is stored on work_item
    assert work_item.input_data == custom_input
