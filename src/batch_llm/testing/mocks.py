"""Mock agents for testing."""

import asyncio
import random
from collections.abc import Callable
from typing import Any


class MockUsage:
    """Mock usage information."""

    def __init__(self, request_tokens: int = 100, response_tokens: int = 50):
        """Initialize mock usage."""
        self.request_tokens = request_tokens
        self.response_tokens = response_tokens
        self.total_tokens = request_tokens + response_tokens


class MockResult:
    """Mock agent result."""

    def __init__(self, output: Any, usage_info: MockUsage | None = None):
        """
        Initialize mock result.

        Args:
            output: The output data
            usage_info: Token usage information
        """
        self.output = output
        self._usage = usage_info or MockUsage()

    def usage(self) -> MockUsage:
        """Get usage information."""
        return self._usage

    def all_messages(self) -> list[Any]:
        """Get all messages (empty for mock)."""
        return []


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        response_factory: Callable[[str], Any] | None = None,
        latency: float = 0.1,
        failure_rate: float = 0.0,
        rate_limit_on_call: int | None = None,
        timeout_on_call: int | None = None,
    ):
        """
        Initialize mock agent.

        Args:
            response_factory: Function to generate responses from prompts
            latency: Simulated latency in seconds
            failure_rate: Probability of random failures (0.0 to 1.0)
            rate_limit_on_call: Call number to simulate rate limit (1-indexed, only triggers once)
            timeout_on_call: Call number to simulate timeout (1-indexed, only triggers once)
        """
        self.response_factory = response_factory or self._default_response
        self.latency = latency
        self.failure_rate = failure_rate
        self.rate_limit_on_call = rate_limit_on_call
        self.timeout_on_call = timeout_on_call
        self.call_count = 0
        self._rate_limit_triggered = False
        self._timeout_triggered = False

    def _default_response(self, prompt: str) -> dict:
        """Default response generator."""
        return {"response": f"Mock response to: {prompt[:50]}"}

    async def run(self, prompt: str, **kwargs) -> MockResult:
        """
        Simulate agent.run().

        Args:
            prompt: The prompt to process
            **kwargs: Additional arguments (ignored)

        Returns:
            MockResult with generated output

        Raises:
            Exception: For simulated failures
        """
        self.call_count += 1

        # Simulate latency
        await asyncio.sleep(self.latency)

        # Simulate rate limit (only once)
        if (
            self.rate_limit_on_call is not None
            and self.call_count == self.rate_limit_on_call
            and not self._rate_limit_triggered
        ):
            self._rate_limit_triggered = True
            # Simulate Gemini rate limit error
            # Create a simple exception that looks like a rate limit error
            class MockRateLimitError(Exception):
                """Mock rate limit error that mimics Gemini ClientError."""
                pass

            # Make it look like a ClientError for the classifier
            error = MockRateLimitError("429 RESOURCE_EXHAUSTED")
            error.__class__.__name__ = "ClientError"
            raise error

        # Simulate timeout (only once)
        if (
            self.timeout_on_call is not None
            and self.call_count == self.timeout_on_call
            and not self._timeout_triggered
        ):
            self._timeout_triggered = True
            await asyncio.sleep(999)  # Will trigger timeout in processor

        # Simulate random failures
        if random.random() < self.failure_rate:
            raise Exception(f"Random failure on call {self.call_count}")

        # Generate response
        output = self.response_factory(prompt)
        return MockResult(output)
