"""Batch LLM processing utilities for handling bulk LLM requests.

This module provides a flexible framework for processing multiple LLM requests
efficiently, with support for PydanticAI agents, direct API calls, and custom factories.

Key features:
- Three integration modes: PydanticAI agents, direct API calls, agent factories
- Provider-agnostic error classification
- Pluggable rate limit strategies
- Middleware pipeline for extensibility
- Observer pattern for monitoring
- Configuration-based setup

Example:
    >>> from batch_llm import ParallelBatchProcessor, ProcessorConfig
    >>> from batch_llm.classifiers import GeminiErrorClassifier
    >>> from batch_llm.observers import MetricsObserver
    >>>
    >>> config = ProcessorConfig(max_workers=5, timeout_per_item=60.0)
    >>> metrics = MetricsObserver()
    >>>
    >>> processor = ParallelBatchProcessor(
    ...     config=config,
    ...     error_classifier=GeminiErrorClassifier(),
    ...     observers=[metrics],
    ... )
"""

# Core classes
from .base import (
    BatchProcessor,
    BatchResult,
    LLMWorkItem,
    PostProcessorFunc,
    ProcessingStats,
    ProgressCallbackFunc,
    TokenUsage,
    WorkItemResult,
)

# Classifiers
from .classifiers import GeminiErrorClassifier

# Configuration
from .core import ProcessorConfig, RateLimitConfig, RetryConfig

# LLM call strategies
from .llm_strategies import (
    GeminiCachedStrategy,
    GeminiStrategy,
    LLMCallStrategy,
    PydanticAIStrategy,
)

# Middleware
from .middleware import BaseMiddleware, Middleware

# Observers
from .observers import BaseObserver, MetricsObserver, ProcessingEvent, ProcessorObserver

# Main processor
from .parallel import ParallelBatchProcessor

# Error classification and rate limit strategies
from .strategies import (
    DefaultErrorClassifier,
    ErrorClassifier,
    ErrorInfo,
    ExponentialBackoffStrategy,
    FixedDelayStrategy,
    FrameworkTimeoutError,
    RateLimitStrategy,
)

__all__ = [
    # Core
    "BatchProcessor",
    "BatchResult",
    "LLMWorkItem",
    "PostProcessorFunc",
    "ProcessingStats",
    "ProgressCallbackFunc",
    "TokenUsage",
    "WorkItemResult",
    # Configuration
    "ProcessorConfig",
    "RateLimitConfig",
    "RetryConfig",
    # LLM Strategies
    "GeminiCachedStrategy",
    "GeminiStrategy",
    "LLMCallStrategy",
    "PydanticAIStrategy",
    # Error Classification Strategies
    "ErrorClassifier",
    "ErrorInfo",
    "DefaultErrorClassifier",
    "FrameworkTimeoutError",
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
    # Processor
    "ParallelBatchProcessor",
]

__version__ = "0.1.0"
