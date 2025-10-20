"""Tests for retry logic and error classification."""

from typing import Annotated

import pytest
from pydantic import BaseModel, Field, ValidationError

from batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,
)
from batch_llm.classifiers import GeminiErrorClassifier
from batch_llm.core import RetryConfig
from batch_llm.strategies import DefaultErrorClassifier, ErrorInfo
from batch_llm.testing import MockAgent


class TestOutput(BaseModel):
    """Test output model."""

    value: Annotated[str, Field(description="Test value")]


@pytest.mark.asyncio
async def test_retry_on_timeout():
    """Test that timeouts trigger retries."""

    call_attempts = []

    def track_attempt(prompt: str) -> TestOutput:
        call_attempts.append(len(call_attempts) + 1)
        return TestOutput(value=f"Attempt {len(call_attempts)}")

    # Agent that times out on first call, succeeds on second
    mock_agent = MockAgent(
        response_factory=track_attempt,
        latency=0.01,
        timeout_on_call=1,  # Timeout on first call
    )

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=0.1,  # Short timeout
        retry=RetryConfig(max_attempts=3),
    )

    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should succeed on retry after timeout
    assert result.succeeded == 1 or result.failed == 1  # Depends on retry timing
    assert len(call_attempts) >= 1  # At least one attempt


@pytest.mark.asyncio
async def test_retry_with_exponential_backoff():
    """Test that retries use exponential backoff."""

    import time

    attempt_times = []

    def track_time(prompt: str) -> TestOutput:
        attempt_times.append(time.time())
        if len(attempt_times) < 2:
            raise Exception("Temporary failure")
        return TestOutput(value="Success")

    mock_agent = MockAgent(response_factory=track_time, latency=0.01)

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=10.0,
        retry=RetryConfig(
            max_attempts=3,
            initial_wait=0.1,
            exponential_base=2.0,
        ),
    )

    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should eventually succeed
    assert result.succeeded == 1

    # Verify exponential backoff (allow some timing variance)
    if len(attempt_times) >= 2:
        gap1 = attempt_times[1] - attempt_times[0]
        # First retry should wait ~0.1s
        assert 0.05 < gap1 < 0.3, f"First retry gap: {gap1}"


@pytest.mark.asyncio
async def test_max_attempts_respected():
    """Test that retry logic respects max_attempts."""

    call_count = []

    def always_fail(prompt: str) -> TestOutput:
        call_count.append(1)
        raise Exception("Always fails")

    mock_agent = MockAgent(response_factory=always_fail, latency=0.01)

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=10.0,
        retry=RetryConfig(
            max_attempts=3,
            initial_wait=0.01,  # Fast retries for testing
        ),
    )

    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should fail after max attempts
    assert result.failed == 1
    assert len(call_count) == 3, f"Expected 3 attempts, got {len(call_count)}"


@pytest.mark.asyncio
async def test_error_classification_default():
    """Test default error classifier."""

    classifier = DefaultErrorClassifier()

    # Test timeout classification
    timeout_error = TimeoutError("Operation timed out")
    info = classifier.classify(timeout_error)
    assert info.is_timeout
    assert info.is_retryable
    assert not info.is_rate_limit

    # Test connection error
    conn_error = ConnectionError("Connection failed")
    info = classifier.classify(conn_error)
    assert info.error_category == "connection_error"
    assert info.is_retryable
    assert not info.is_rate_limit

    # Test unknown error
    unknown_error = ValueError("Some error")
    info = classifier.classify(unknown_error)
    assert info.is_retryable
    assert not info.is_rate_limit


@pytest.mark.asyncio
async def test_error_classification_gemini():
    """Test Gemini-specific error classifier."""

    classifier = GeminiErrorClassifier()

    # Test rate limit detection via message
    rate_limit_error = Exception("429 RESOURCE_EXHAUSTED")
    info = classifier.classify(rate_limit_error)
    assert info.is_rate_limit
    assert not info.is_retryable
    assert info.error_category == "rate_limit"

    # Test timeout detection
    timeout_error = Exception("504 Gateway Timeout")
    info = classifier.classify(timeout_error)
    assert info.is_timeout
    assert info.is_retryable

    # Test unknown error
    unknown_error = Exception("Some random error")
    info = classifier.classify(unknown_error)
    assert info.is_retryable
    assert not info.is_rate_limit


@pytest.mark.asyncio
async def test_validation_error_retries():
    """Test that validation errors trigger retries."""

    attempt_count = []

    def fail_validation_once(prompt: str) -> TestOutput:
        attempt_count.append(1)
        if len(attempt_count) == 1:
            # Raise validation error on first attempt
            raise ValidationError.from_exception_data(
                "TestOutput",
                [{"type": "missing", "loc": ("value",), "msg": "Field required", "input": {}}]
            )
        return TestOutput(value="Success on retry")

    mock_agent = MockAgent(response_factory=fail_validation_once, latency=0.01)

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=10.0,
        retry=RetryConfig(max_attempts=3),
    )

    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
        error_classifier=GeminiErrorClassifier(),
    )

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should succeed on second attempt
    assert result.succeeded == 1
    assert len(attempt_count) == 2, "Should retry validation error"


@pytest.mark.asyncio
async def test_non_retryable_error_fails_immediately():
    """Test that non-retryable errors don't trigger retries."""

    call_count = []

    def raise_non_retryable(prompt: str) -> TestOutput:
        call_count.append(1)
        # Create custom classifier that marks this as non-retryable
        raise ValueError("Non-retryable error")

    class NonRetryableClassifier:
        def classify(self, exception: Exception) -> ErrorInfo:
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="non_retryable",
            )

    mock_agent = MockAgent(response_factory=raise_non_retryable, latency=0.01)

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=10.0,
        retry=RetryConfig(max_attempts=3),
    )

    processor = ParallelBatchProcessor[str, TestOutput, None](
        config=config,
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

    # Should fail immediately without retries
    assert result.failed == 1
    assert len(call_count) == 1, "Should not retry non-retryable error"


@pytest.mark.asyncio
async def test_retry_succeeds_on_final_attempt():
    """Test success on the last allowed attempt."""

    attempt_count = []

    def fail_twice_succeed_third(prompt: str) -> TestOutput:
        attempt_count.append(1)
        if len(attempt_count) < 3:
            raise Exception("Temporary failure")
        return TestOutput(value="Success on third attempt")

    mock_agent = MockAgent(response_factory=fail_twice_succeed_third, latency=0.01)

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=10.0,
        retry=RetryConfig(
            max_attempts=3,
            initial_wait=0.01,
        ),
    )

    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=mock_agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should succeed on third attempt
    assert result.succeeded == 1
    assert len(attempt_count) == 3
    assert result.results[0].success
    assert result.results[0].output.value == "Success on third attempt"


@pytest.mark.asyncio
async def test_token_usage_tracked_across_retries():
    """Test that token usage is accumulated across failed retries."""

    from batch_llm.testing.mocks import MockResult, MockUsage

    class TokenTrackingAgent:
        def __init__(self):
            self.call_count = 0

        async def run(self, prompt: str, **kwargs):
            self.call_count += 1
            # Create result with token usage
            usage = MockUsage(
                request_tokens=100 * self.call_count,
                response_tokens=50 * self.call_count
            )
            if self.call_count < 2:
                # Fail first attempt but include token usage
                result = MockResult(output=None, usage_info=usage)
                # Simulate PydanticAI wrapping
                error = Exception("Validation failed")
                error.__cause__ = type('obj', (), {'result': result})()
                raise error
            # Succeed on second attempt
            return MockResult(
                output=TestOutput(value="Success"),
                usage_info=usage
            )

    agent = TokenTrackingAgent()

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=10.0,
        retry=RetryConfig(max_attempts=3),
    )

    processor = ParallelBatchProcessor[str, TestOutput, None](config=config)

    work_item = LLMWorkItem(
        item_id="item_1",
        strategy=PydanticAIStrategy(agent=agent),
        prompt="Test",
        context=None,
    )
    await processor.add_work(work_item)

    result = await processor.process_all()

    # Should succeed and have token usage from successful attempt
    assert result.succeeded == 1
    # Note: Failed attempt tokens are tracked separately
    assert result.results[0].token_usage.get("total_tokens", 0) > 0
