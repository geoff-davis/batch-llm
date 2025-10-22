"""Google Gemini-specific error classification."""

from ..strategies.errors import ErrorClassifier, ErrorInfo, FrameworkTimeoutError

# Error pattern constants
RATE_LIMIT_PATTERNS = ("429", "resource_exhausted", "quota", "rate limit")
TIMEOUT_PATTERNS = ("timeout", "504", "deadline")

# Default wait time for rate limit errors (seconds)
DEFAULT_RATE_LIMIT_WAIT = 300.0  # 5 minutes


class GeminiErrorClassifier(ErrorClassifier):
    """Google Gemini-specific error classification."""

    def _matches_any_pattern(self, error_str: str, patterns: tuple[str, ...]) -> bool:
        """Check if error string matches any of the given patterns (case-insensitive)."""
        error_lower = error_str.lower()
        return any(pattern in error_lower for pattern in patterns)

    def classify(self, exception: Exception) -> ErrorInfo:
        """Classify Gemini-specific errors."""
        # Check for framework timeout first (highest priority)
        if isinstance(exception, FrameworkTimeoutError):
            return ErrorInfo(
                is_retryable=True,  # Retry - might succeed if LLM is faster
                is_rate_limit=False,
                is_timeout=True,
                error_category="framework_timeout",
            )

        try:
            from google.genai.errors import ClientError, ServerError
        except ImportError:
            # If google.genai not installed, fall back to default classification
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="unknown",
            )

        if isinstance(exception, ClientError):
            error_str = str(exception)
            is_rate_limit = self._matches_any_pattern(error_str, RATE_LIMIT_PATTERNS)
            return ErrorInfo(
                is_retryable=not is_rate_limit,  # Don't retry rate limits with tenacity
                is_rate_limit=is_rate_limit,
                is_timeout=False,
                error_category="rate_limit" if is_rate_limit else "client_error",
                suggested_wait=DEFAULT_RATE_LIMIT_WAIT if is_rate_limit else None,
            )

        if isinstance(exception, ServerError):
            error_str = str(exception)
            is_timeout = self._matches_any_pattern(error_str, TIMEOUT_PATTERNS)
            return ErrorInfo(
                is_retryable=is_timeout,  # Only retry server timeouts
                is_rate_limit=False,
                is_timeout=is_timeout,
                error_category="server_timeout" if is_timeout else "server_error",
            )

        # Check for PydanticAI validation errors
        try:
            from pydantic_ai.exceptions import UnexpectedModelBehavior

            if isinstance(exception, UnexpectedModelBehavior):
                return ErrorInfo(
                    is_retryable=True,  # Retry validation errors
                    is_rate_limit=False,
                    is_timeout=False,
                    error_category="validation_error",
                )
        except ImportError:
            pass

        # Fallback: Check error message for common patterns
        error_str = str(exception)

        # Check if it looks like a rate limit error (for mocks and other providers)
        if self._matches_any_pattern(error_str, RATE_LIMIT_PATTERNS):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=DEFAULT_RATE_LIMIT_WAIT,
            )

        # Check if it looks like a timeout
        if self._matches_any_pattern(error_str, TIMEOUT_PATTERNS):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="timeout",
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
        return ErrorInfo(
            is_retryable=True,  # Retry unknown exceptions (might be transient)
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown",
        )
