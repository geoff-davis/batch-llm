"""Error classification for different LLM providers."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ErrorInfo:
    """Structured information about an error."""

    is_retryable: bool
    is_rate_limit: bool
    is_timeout: bool
    error_category: str
    suggested_wait: float | None = None


class ErrorClassifier(Protocol):
    """Protocol for classifying LLM provider errors."""

    def classify(self, exception: Exception) -> ErrorInfo:
        """
        Classify an exception and determine handling strategy.

        Args:
            exception: The exception to classify

        Returns:
            ErrorInfo with classification details
        """
        ...


class DefaultErrorClassifier:
    """Default error classifier that handles common error types."""

    def classify(self, exception: Exception) -> ErrorInfo:
        """Classify common errors with conservative defaults."""
        error_str = str(exception).lower()

        # Check for timeout
        if isinstance(exception, TimeoutError) or "timeout" in error_str:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="timeout",
            )

        # Check for connection errors
        if isinstance(exception, ConnectionError) or "connection" in error_str:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="connection_error",
            )

        # Default: treat as retryable but not a rate limit
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown",
        )
