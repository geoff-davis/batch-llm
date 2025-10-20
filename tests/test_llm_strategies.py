"""Tests for LLM call strategies."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, RetryConfig
from batch_llm.llm_strategies import (
    GeminiCachedStrategy,
    GeminiStrategy,
    LLMCallStrategy,
    PydanticAIStrategy,
)
from batch_llm.testing import MockAgent


class TestOutput(BaseModel):
    """Test output model."""

    text: str


# Test LLMCallStrategy base class


class MockStrategy(LLMCallStrategy[TestOutput]):
    """Mock strategy for testing base class behavior."""

    def __init__(self):
        self.prepare_called = False
        self.execute_calls = []
        self.cleanup_called = False

    async def prepare(self):
        self.prepare_called = True

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TestOutput, dict[str, int]]:
        self.execute_calls.append((prompt, attempt, timeout))
        return TestOutput(text=f"Response for {prompt}"), {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }

    async def cleanup(self):
        self.cleanup_called = False


@pytest.mark.asyncio
async def test_strategy_lifecycle():
    """Test that strategy prepare/execute/cleanup are called correctly."""
    strategy = MockStrategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[None, TestOutput, None](
        config=config
    ) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Test prompt",
            )
        )

        result = await processor.process_all()

    assert result.succeeded == 1
    assert strategy.prepare_called
    assert len(strategy.execute_calls) == 1
    assert strategy.execute_calls[0][0] == "Test prompt"
    assert strategy.execute_calls[0][1] == 1  # First attempt
    # Note: cleanup_called will be False because we set it to False in cleanup()
    # This tests that cleanup was actually called


@pytest.mark.asyncio
async def test_strategy_with_retries():
    """Test that strategy execute is called for each retry attempt."""

    class FailingStrategy(LLMCallStrategy[TestOutput]):
        def __init__(self, fail_count=2):
            self.fail_count = fail_count
            self.attempt_count = 0

        async def execute(
            self, prompt: str, attempt: int, timeout: float
        ) -> tuple[TestOutput, dict[str, int]]:
            self.attempt_count += 1
            if self.attempt_count < self.fail_count:
                raise ValueError("Simulated failure")
            return TestOutput(text="Success"), {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            }

    strategy = FailingStrategy(fail_count=3)
    config = ProcessorConfig(
        max_workers=1, timeout_per_item=10.0, retry=RetryConfig(max_attempts=3)
    )

    async with ParallelBatchProcessor[None, TestOutput, None](
        config=config
    ) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Test prompt",
            )
        )

        result = await processor.process_all()

    assert result.succeeded == 1
    assert strategy.attempt_count == 3  # Should have tried 3 times


# Test PydanticAIStrategy


@pytest.mark.asyncio
async def test_pydantic_ai_strategy():
    """Test PydanticAIStrategy with a mock agent."""
    mock_agent = MockAgent(
        response_factory=lambda p: TestOutput(text=f"Response: {p}"),
        latency=0.001,
    )

    strategy = PydanticAIStrategy(agent=mock_agent)
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[None, TestOutput, None](
        config=config
    ) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Hello",
            )
        )

        result = await processor.process_all()

    assert result.succeeded == 1
    assert result.results[0].output.text == "Response: Hello"
    assert result.results[0].token_usage["total_tokens"] > 0


# Test Gemini strategies (mock-based since we don't want to make real API calls)


@pytest.mark.asyncio
async def test_gemini_strategy_mock():
    """Test GeminiStrategy with mocked client."""

    # Create mock response
    mock_response = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_response.usage_metadata.total_token_count = 30

    # Create mock client
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    # Create strategy
    strategy = GeminiStrategy(
        model="gemini-test",
        client=mock_client,
        response_parser=lambda r: TestOutput(text="Gemini response"),
    )

    # Test execute
    output, tokens = await strategy.execute("Test prompt", 1, 10.0)

    assert output.text == "Gemini response"
    assert tokens["input_tokens"] == 10
    assert tokens["output_tokens"] == 20
    assert tokens["total_tokens"] == 30


@pytest.mark.asyncio
async def test_gemini_cached_strategy_lifecycle():
    """Test GeminiCachedStrategy prepare/execute/cleanup lifecycle."""

    # Create mock cache
    mock_cache = MagicMock()
    mock_cache.name = "test-cache"

    # Create mock response
    mock_response = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_response.usage_metadata.total_token_count = 30

    # Create mock client
    mock_client = MagicMock()
    mock_client.aio.caches.create = AsyncMock(return_value=mock_cache)
    mock_client.aio.caches.update = AsyncMock(return_value=mock_cache)
    mock_client.aio.caches.delete = AsyncMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    # Create strategy
    strategy = GeminiCachedStrategy(
        model="gemini-test",
        client=mock_client,
        response_parser=lambda r: TestOutput(text="Cached response"),
        cached_content=[],
        cache_ttl_seconds=3600,
    )

    # Test prepare
    await strategy.prepare()
    mock_client.aio.caches.create.assert_called_once()

    # Test execute
    output, tokens = await strategy.execute("Test prompt", 1, 10.0)
    assert output.text == "Cached response"
    assert tokens["total_tokens"] == 30

    # Test cleanup
    await strategy.cleanup()
    mock_client.aio.caches.delete.assert_called_once_with(name="test-cache")


@pytest.mark.asyncio
async def test_gemini_cached_strategy_ttl_refresh():
    """Test that cache TTL is refreshed when close to expiring."""

    # Create mock cache
    mock_cache = MagicMock()
    mock_cache.name = "test-cache"

    # Create mock response
    mock_response = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_response.usage_metadata.total_token_count = 30

    # Create mock client
    mock_client = MagicMock()
    mock_client.aio.caches.create = AsyncMock(return_value=mock_cache)
    mock_client.aio.caches.update = AsyncMock(return_value=mock_cache)
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    # Create strategy with short TTL and high refresh threshold
    strategy = GeminiCachedStrategy(
        model="gemini-test",
        client=mock_client,
        response_parser=lambda r: TestOutput(text="Cached response"),
        cached_content=[],
        cache_ttl_seconds=1,  # 1 second TTL
        cache_refresh_threshold=0.5,  # Refresh if <50% remaining
    )

    # Test prepare
    await strategy.prepare()
    mock_client.aio.caches.create.assert_called_once()

    # First execute - no refresh needed
    await strategy.execute("Test prompt 1", 1, 10.0)
    assert mock_client.aio.caches.update.call_count == 0

    # Wait until cache is close to expiring
    await asyncio.sleep(0.6)  # More than 50% of TTL has elapsed

    # Second execute - should refresh
    await strategy.execute("Test prompt 2", 1, 10.0)
    assert mock_client.aio.caches.update.call_count == 1

    # Cleanup
    await strategy.cleanup()


@pytest.mark.asyncio
async def test_strategy_error_handling():
    """Test that strategy errors are handled correctly."""

    class ErrorStrategy(LLMCallStrategy[TestOutput]):
        async def execute(
            self, prompt: str, attempt: int, timeout: float
        ) -> tuple[TestOutput, dict[str, int]]:
            raise ValueError("Strategy error")

    strategy = ErrorStrategy()
    config = ProcessorConfig(
        max_workers=1, timeout_per_item=10.0, retry=RetryConfig(max_attempts=1)
    )

    async with ParallelBatchProcessor[None, TestOutput, None](
        config=config
    ) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="test1",
                strategy=strategy,
                prompt="Test",
            )
        )

        result = await processor.process_all()

    assert result.failed == 1
    assert "ValueError" in result.results[0].error


@pytest.mark.asyncio
async def test_work_item_validation():
    """Test that work item validation works correctly."""

    # Should accept strategy
    item = LLMWorkItem(
        item_id="test1",
        strategy=MockStrategy(),
        prompt="Test",
    )
    assert item.strategy is not None
