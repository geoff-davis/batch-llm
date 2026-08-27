"""Batch LLM processing utilities for handling bulk LLM requests.

This module provides a flexible framework for processing multiple LLM requests
efficiently using a strategy pattern for provider-agnostic LLM integration.

Key features:
- Strategy pattern for any LLM provider (OpenAI, Anthropic, Google, LangChain, custom)
- Built-in strategies: PydanticAIStrategy, GeminiStrategy, OpenAIStrategy, OpenRouterStrategy
- Built-in models: GeminiModel, GeminiCachedModel, OpenAIModel, OpenRouterModel
- Provider-agnostic error classification
- Pluggable rate limit strategies
- Middleware pipeline for extensibility
- Observer pattern for monitoring
- Configuration-based setup

Example:
    >>> from async_batch_llm import (
    ...     ParallelBatchProcessor,
    ...     ProcessorConfig,
    ...     LLMWorkItem,
    ...     PydanticAIStrategy,
    ... )
    >>> from pydantic_ai import Agent
    >>>
    >>> agent = Agent("openai:gpt-4o-mini", result_type=MyOutput)
    >>> strategy = PydanticAIStrategy(agent=agent)
    >>> config = ProcessorConfig(max_workers=5, attempt_timeout=60.0)
    >>>
    >>> async with ParallelBatchProcessor(config=config) as processor:
    ...     await processor.add_work(LLMWorkItem(
    ...         item_id="item_1",
    ...         strategy=strategy,
    ...         prompt="Process this",
    ...     ))
    ...     result = await processor.process_all()

Type Aliases:
    For convenience, type aliases are provided to reduce verbosity:

    - ``SimpleBatchProcessor[T]``: Processor with string input, output type T, no context
      Equivalent to ``ParallelBatchProcessor[str, T, None]``

    - ``SimpleWorkItem[T]``: Work item with string input, output type T, no context
      Equivalent to ``LLMWorkItem[str, T, None]``

    - ``SimpleResult[T]``: Result with output type T, no context
      Equivalent to ``WorkItemResult[T, None]``

    Example using type aliases:
        >>> from async_batch_llm import SimpleBatchProcessor, SimpleWorkItem
        >>>
        >>> async with SimpleBatchProcessor[MyOutput](config=config) as processor:
        ...     await processor.add_work(SimpleWorkItem[MyOutput](
        ...         item_id="item_1",
        ...         strategy=strategy,
        ...         prompt="Process this",
        ...     ))
"""

from typing import TypeVar

from .artifacts import (
    ArtifactError,
    ArtifactFormatError,
    ArtifactIdentity,
    ArtifactIOError,
    ArtifactSerializationError,
    ArtifactStore,
    JsonlArtifactStore,
    ResumePolicy,
)

# Core classes
from .base import (
    AttemptTiming,
    BatchProcessor,
    BatchResult,
    BatchTermination,
    CachedTokenRates,
    LLMResponse,
    LLMWorkItem,
    PostProcessorFunc,
    ProcessingStats,
    ProgressCallbackFunc,
    RetryState,
    TokenUsage,
    WorkItemResult,
    WorkItemTiming,
)
from .callable_strategy import CallableStrategy, CallOutcome

# Classifiers
from .classifiers import (
    GeminiErrorClassifier,
    OpenAIErrorClassifier,
    OpenRouterErrorClassifier,
)

# Configuration
from .core import (
    AbortMode,
    GuardrailConfig,
    ProcessorConfig,
    RateLimitConfig,
    RetryConfig,
    StartupRampConfig,
)

# Protocols
from .core.protocols import LLMModel, ManagedLLMModel

# String-based strategy factory: llm("openai:gpt-4o-mini")
from .factory import llm

# Queue-less convenience surfaces (single call + shared call pool), built on the
# same per-item resilience pipeline as the batch processor.
from .gateway import LLMCallPool, LLMGateway

# LLM call strategies
from .llm_strategies import (
    DeepSeekStrategy,
    GeminiStrategy,
    LLMCallStrategy,
    ModelStrategy,
    OpenAIStrategy,
    OpenRouterStrategy,
    PydanticAIStrategy,
)

# Middleware
from .middleware import BaseMiddleware, Middleware

# Concrete models
from .models import (
    DeepSeekModel,
    GeminiCachedModel,
    GeminiModel,
    MetadataExtractor,
    OpenAICompatibleModel,
    OpenAIModel,
    OpenRouterModel,
    grounding_metadata_extractor,
)

# Observers
from .observers import BaseObserver, MetricsObserver, ProcessingEvent, ProcessorObserver

# Main processor
from .parallel import ParallelBatchProcessor

# Structured-output parsing helpers
from .parsing import pydantic_json_parser, strip_code_fences

# Provider auxiliary output (typed metadata views)
from .provider_output import Grounding, GroundingSource, ToolCall
from .serialization import ResultSerializationError
from .single import LLMCallError, call, call_result
from .sqlite_artifacts import SqliteArtifactStore, SqliteDurability

# Error classification and rate limit strategies
from .strategies import (
    BatchAbortedError,
    BatchDeadlineExceeded,
    DefaultErrorClassifier,
    EmptyResponseError,
    ErrorClassifier,
    ErrorInfo,
    ExponentialBackoffStrategy,
    FixedDelayStrategy,
    FrameworkTimeoutError,
    ItemDeadlineExceeded,
    ProviderResponseError,
    RateLimitRetriesExceeded,
    RateLimitStrategy,
    StructuredOutputSchemaError,
    StructuredOutputValidationError,
    TokenEstimateExceedsLimit,
    TokenEstimationError,
    TokenEstimatorRequired,
    TokenTrackingError,
)

# High-level streaming API (built on the processor's streaming mode)
from .streaming import process_prompts, process_stream
from .token_estimation import CharacterTokenEstimator, TokenEstimate, TokenEstimator

# Type variable for output type in simplified aliases
_T = TypeVar("_T")

# Type aliases for common use cases
# These reduce verbosity for the most common pattern: string input, typed output, no context
SimpleBatchProcessor = ParallelBatchProcessor[str, _T, None]
"""Type alias for ParallelBatchProcessor[str, T, None].

Use when you have string prompts, a typed output, and no context.

Example:
    async with SimpleBatchProcessor[MyOutput](config=config) as processor:
        ...
"""

SimpleWorkItem = LLMWorkItem[str, _T, None]
"""Type alias for LLMWorkItem[str, T, None].

Use when creating work items with string prompts, typed output, and no context.

Example:
    item = SimpleWorkItem[MyOutput](item_id="1", strategy=strategy, prompt="Hello")
"""

SimpleResult = WorkItemResult[_T, None]
"""Type alias for WorkItemResult[T, None].

Use when working with results that have no context.

Example:
    result: SimpleResult[MyOutput] = results[0]
"""

__all__ = [
    # Core
    "BatchProcessor",
    "AttemptTiming",
    "BatchResult",
    "BatchTermination",
    "CachedTokenRates",
    "LLMWorkItem",
    "PostProcessorFunc",
    "ProcessingStats",
    "ProgressCallbackFunc",
    "RetryState",
    "TokenUsage",
    "TokenEstimate",
    "TokenEstimator",
    "CharacterTokenEstimator",
    "WorkItemResult",
    "WorkItemTiming",
    "ResultSerializationError",
    "CallOutcome",
    "CallableStrategy",
    # Audit/checkpoint artifacts
    "ArtifactError",
    "ArtifactFormatError",
    "ArtifactIOError",
    "ArtifactIdentity",
    "ArtifactSerializationError",
    "ArtifactStore",
    "JsonlArtifactStore",
    "ResumePolicy",
    "SqliteArtifactStore",
    "SqliteDurability",
    # Configuration
    "ProcessorConfig",
    "AbortMode",
    "GuardrailConfig",
    "RateLimitConfig",
    "RetryConfig",
    "StartupRampConfig",
    # High-level convenience API
    "process_prompts",
    "process_stream",
    # Single-call + shared-call surfaces
    "call",
    "call_result",
    "LLMCallError",
    "LLMCallPool",
    "LLMGateway",
    # String-based strategy factory
    "llm",
    # LLM Strategies
    "DeepSeekStrategy",
    "GeminiStrategy",
    "LLMCallStrategy",
    "ModelStrategy",
    "OpenAIStrategy",
    "OpenRouterStrategy",
    "PydanticAIStrategy",
    # Models
    "DeepSeekModel",
    "GeminiModel",
    "GeminiCachedModel",
    "OpenAICompatibleModel",
    "OpenAIModel",
    "OpenRouterModel",
    "MetadataExtractor",
    "grounding_metadata_extractor",
    # Provider auxiliary output (typed metadata views)
    "Grounding",
    "GroundingSource",
    "ToolCall",
    # Protocols
    "LLMModel",
    "LLMResponse",
    "ManagedLLMModel",
    # Error Classification Strategies
    "ErrorClassifier",
    "ErrorInfo",
    "DefaultErrorClassifier",
    "EmptyResponseError",
    "FrameworkTimeoutError",
    "ItemDeadlineExceeded",
    "BatchDeadlineExceeded",
    "BatchAbortedError",
    "ProviderResponseError",
    "RateLimitRetriesExceeded",
    "StructuredOutputSchemaError",
    "StructuredOutputValidationError",
    "TokenEstimationError",
    "TokenEstimatorRequired",
    "TokenEstimateExceedsLimit",
    "TokenTrackingError",
    "RateLimitStrategy",
    "ExponentialBackoffStrategy",
    "FixedDelayStrategy",
    # Middleware
    "Middleware",
    "BaseMiddleware",
    # Observers
    "ProcessorObserver",
    "BaseObserver",
    "MetricsObserver",
    "ProcessingEvent",
    # Classifiers
    "GeminiErrorClassifier",
    "OpenAIErrorClassifier",
    "OpenRouterErrorClassifier",
    # Processor
    "ParallelBatchProcessor",
    # Structured-output parsing helpers
    "pydantic_json_parser",
    "strip_code_fences",
    # Type aliases (convenience)
    "SimpleBatchProcessor",
    "SimpleWorkItem",
    "SimpleResult",
]

# Version is read from package metadata (single source of truth in pyproject.toml)
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("async-batch-llm")
except PackageNotFoundError:
    # Package not installed (e.g., running from source in development)
    __version__ = "0.0.0+dev"
