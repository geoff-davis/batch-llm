"""Error classification for different LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Common error pattern constants
RATE_LIMIT_PATTERNS = ("429", "resource_exhausted", "quota", "rate limit")
DEFAULT_RATE_LIMIT_WAIT = 300.0  # 5 minutes


class FrameworkTimeoutError(TimeoutError):
    """
    Timeout enforced by the batch-llm framework (asyncio.wait_for).

    This distinguishes framework-level timeouts from API-level timeouts.
    Framework timeouts indicate the configured timeout_per_item was exceeded,
    whereas API timeouts indicate the LLM provider returned a timeout error.
    """

    pass


@dataclass
class ErrorInfo:
    """Structured information about an error."""

    is_retryable: bool
    is_rate_limit: bool
    is_timeout: bool
    error_category: str
    suggested_wait: float | None = None


class ErrorClassifier(ABC):
    """Abstract base class for classifying LLM provider errors."""

    @abstractmethod
    def classify(self, exception: Exception) -> ErrorInfo:
        """
        Classify an exception and determine handling strategy.

        Args:
            exception: The exception to classify

        Returns:
            ErrorInfo with classification details
        """
        pass


class DefaultErrorClassifier(ErrorClassifier):
    """Default error classifier that handles common error types."""

    def _matches_rate_limit(self, error_str: str) -> bool:
        """Return True if the error string looks like a rate limit."""
        lowered = error_str.lower()
        return any(pattern in lowered for pattern in RATE_LIMIT_PATTERNS)

    def classify(self, exception: Exception) -> ErrorInfo:
        """Classify common errors with conservative defaults."""
        error_str = str(exception).lower()

        # Detect rate limit errors from message patterns (works for simple Exception mocks)
        if self._matches_rate_limit(error_str):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=DEFAULT_RATE_LIMIT_WAIT,
            )

        # Check for framework timeout (retryable but indicates timeout config may need adjustment)
        if isinstance(exception, FrameworkTimeoutError):
            return ErrorInfo(
                is_retryable=True,  # Retry - might succeed if LLM is faster
                is_rate_limit=False,
                is_timeout=True,
                error_category="framework_timeout",
            )

        # Check for API timeout (retryable - might be transient)
        if isinstance(exception, TimeoutError) or "timeout" in error_str:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="api_timeout",
            )

        # Check for connection errors
        if isinstance(exception, ConnectionError) or "connection" in error_str:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="connection_error",
            )

        # Check for Pydantic validation errors (retryable - LLM might generate valid output on retry)
        try:
            from pydantic import ValidationError

            if isinstance(exception, ValidationError):
                return ErrorInfo(
                    is_retryable=True,
                    is_rate_limit=False,
                    is_timeout=False,
                    error_category="validation_error",
                )
        except ImportError:
            pass

        # Check for logic bugs (deterministic errors that won't be fixed by retrying)
        # These are usually programming errors, not transient failures
        logic_bug_types = (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            NameError,
            ZeroDivisionError,
            AssertionError,
        )
        if isinstance(exception, logic_bug_types):
            return ErrorInfo(
                is_retryable=False,  # Don't retry logic bugs (deterministic failures)
                is_rate_limit=False,
                is_timeout=False,
                error_category="logic_error",
            )

        # Default: treat unknown generic exceptions as retryable
        # This allows custom transient errors and test mocks to work
        # Users with non-retryable custom errors should implement a custom ErrorClassifier
        return ErrorInfo(
            is_retryable=True,  # Retry unknown exceptions (might be transient)
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown",
        )
