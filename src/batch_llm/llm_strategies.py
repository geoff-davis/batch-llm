"""LLM call strategies for flexible model configuration and execution.

This module provides strategy classes that encapsulate how LLM calls are made,
including caching, model selection, and retry behavior.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

# Conditional imports for optional dependencies
if TYPE_CHECKING:
    from google import genai
    from google.genai.types import Content, GenerateContentConfig
    from pydantic_ai import Agent
else:
    try:
        from google import genai
        from google.genai.types import Content, GenerateContentConfig
    except ImportError:
        genai = None  # type: ignore[assignment]
        Content = Any  # type: ignore[misc,assignment]
        GenerateContentConfig = Any  # type: ignore[misc,assignment]

    try:
        from pydantic_ai import Agent
    except ImportError:
        Agent = Any  # type: ignore[misc,assignment]

TOutput = TypeVar("TOutput")


class LLMCallStrategy(ABC, Generic[TOutput]):
    """
    Abstract base class for LLM call strategies.

    A strategy encapsulates how LLM calls are made, including:
    - Resource initialization (caches, clients)
    - Call execution with retries
    - Resource cleanup

    The framework calls:
    1. prepare() once before any retries
    2. execute() for each attempt (including retries)
    3. cleanup() once after all attempts complete or fail
    """

    async def prepare(self) -> None:
        """
        Initialize resources before making any LLM calls.

        Called once per work item before any retry attempts.
        Use this to set up caches, initialize clients, etc.

        Default: no-op
        """
        pass

    @abstractmethod
    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, dict[str, int]]:
        """
        Execute an LLM call for the given attempt.

        Args:
            prompt: The prompt to send to the LLM
            attempt: Which retry attempt this is (1, 2, 3, ...)
            timeout: Maximum time to wait for response (seconds)

        Returns:
            Tuple of (output, token_usage)
            where token_usage is a dict with keys: input_tokens, output_tokens, total_tokens

        Raises:
            Any exception to trigger retry (if retryable) or failure
        """
        pass

    async def cleanup(self) -> None:
        """
        Clean up resources after all retry attempts complete.

        Called once per work item after processing finishes (success or failure).
        Use this to delete caches, close clients, etc.

        Default: no-op
        """
        pass


class GeminiStrategy(LLMCallStrategy[TOutput]):
    """
    Strategy for calling Google Gemini API directly.

    This strategy uses the google-genai SDK to make direct API calls
    without caching. Best for one-off calls or when caching isn't needed.
    """

    def __init__(
        self,
        model: str,
        client: "genai.Client",
        response_parser: Callable[[Any], TOutput],
        config: "GenerateContentConfig | None" = None,
    ):
        """
        Initialize Gemini strategy.

        Args:
            model: Model name (e.g., "gemini-2.0-flash-exp")
            client: Initialized Gemini client
            response_parser: Function to parse response into TOutput
            config: Optional generation config (temperature, etc.)
        """
        if genai is None:
            raise ImportError(
                "google-genai is required for GeminiStrategy. "
                "Install with: pip install 'batch-llm[gemini]'"
            )

        self.model = model
        self.client = client
        self.response_parser = response_parser
        self.config = config

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, dict[str, int]]:
        """Execute Gemini API call.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        # Make the call
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.config,
        )

        # Parse output
        output = self.response_parser(response)

        # Extract token usage
        usage = response.usage_metadata
        tokens = {
            "input_tokens": usage.prompt_token_count if usage else 0,
            "output_tokens": usage.candidates_token_count if usage else 0,
            "total_tokens": usage.total_token_count if usage else 0,
        }

        return output, tokens


class GeminiCachedStrategy(LLMCallStrategy[TOutput]):
    """
    Strategy for calling Google Gemini API with context caching.

    This strategy creates a Gemini cache for the system instruction and/or
    initial context, then uses it across all retry attempts. The cache is
    automatically refreshed if it's close to expiring, and deleted on cleanup.

    Best for: Repeated calls with large shared context (RAG, long documents).
    """

    def __init__(
        self,
        model: str,
        client: "genai.Client",
        response_parser: Callable[[Any], TOutput],
        cached_content: list["Content"],
        cache_ttl_seconds: int = 3600,
        cache_refresh_threshold: float = 0.1,  # Refresh if <10% TTL remaining
        config: "GenerateContentConfig | None" = None,
    ):
        """
        Initialize Gemini cached strategy.

        Args:
            model: Model name (e.g., "gemini-2.0-flash-exp")
            client: Initialized Gemini client
            response_parser: Function to parse response into TOutput
            cached_content: Content to cache (system instructions, documents)
            cache_ttl_seconds: Initial cache TTL in seconds
            cache_refresh_threshold: Refresh cache when TTL falls below this fraction
            config: Optional generation config
        """
        if genai is None:
            raise ImportError(
                "google-genai is required for GeminiCachedStrategy. "
                "Install with: pip install 'batch-llm[gemini]'"
            )

        self.model = model
        self.client = client
        self.response_parser = response_parser
        self.cached_content = cached_content
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_refresh_threshold = cache_refresh_threshold
        self.config = config

        self._cache = None
        self._cache_created_at = None

    async def prepare(self) -> None:
        """Create the Gemini cache."""
        self._cache = await self.client.aio.caches.create(
            model=self.model,
            contents=self.cached_content,
            ttl=f"{self.cache_ttl_seconds}s",
        )
        self._cache_created_at = time.time()

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, dict[str, int]]:
        """Execute Gemini API call with cache, refreshing TTL if needed.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        if self._cache is None:
            raise RuntimeError("Cache not initialized - prepare() was not called")

        # Check if cache is close to expiring and refresh if needed
        elapsed = time.time() - self._cache_created_at
        remaining = self.cache_ttl_seconds - elapsed

        if remaining < (self.cache_ttl_seconds * self.cache_refresh_threshold):
            # Refresh cache TTL
            self._cache = await self.client.aio.caches.update(
                name=self._cache.name, ttl=f"{self.cache_ttl_seconds}s"
            )
            self._cache_created_at = time.time()

        # Make the call using the cache
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.config,
            cached_content=self._cache.name,
        )

        # Parse output
        output = self.response_parser(response)

        # Extract token usage (cached tokens counted separately)
        usage = response.usage_metadata
        tokens = {
            "input_tokens": usage.prompt_token_count if usage else 0,
            "output_tokens": usage.candidates_token_count if usage else 0,
            "total_tokens": usage.total_token_count if usage else 0,
            "cached_input_tokens": usage.cached_content_token_count if usage and hasattr(usage, 'cached_content_token_count') else 0,
        }

        return output, tokens

    async def cleanup(self) -> None:
        """Delete the Gemini cache."""
        if self._cache:
            try:
                await self.client.aio.caches.delete(name=self._cache.name)
            except Exception:
                # Cache might already be expired/deleted - ignore errors
                pass


class PydanticAIStrategy(LLMCallStrategy[TOutput]):
    """
    Strategy for using PydanticAI agents.

    This strategy wraps a PydanticAI agent, providing a clean interface
    for batch processing. The agent handles all model interaction, validation,
    and parsing.

    Best for: Structured output with Pydantic models, using PydanticAI's features.
    """

    def __init__(self, agent: "Agent[None, TOutput]"):
        """
        Initialize PydanticAI strategy.

        Args:
            agent: Configured PydanticAI agent
        """
        if Agent is Any:
            raise ImportError(
                "pydantic-ai is required for PydanticAIStrategy. "
                "Install with: pip install 'batch-llm[pydantic-ai]'"
            )

        self.agent = agent

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, dict[str, int]]:
        """Execute PydanticAI agent call.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        result = await self.agent.run(prompt)

        # Extract token usage
        usage = result.usage()
        tokens = {
            "input_tokens": usage.request_tokens if usage else 0,
            "output_tokens": usage.response_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

        return result.output, tokens
