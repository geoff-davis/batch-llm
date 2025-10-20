"""Google Gemini-specific error classification."""

from ..strategies.errors import ErrorClassifier, ErrorInfo


class GeminiErrorClassifier(ErrorClassifier):
    """Google Gemini-specific error classification."""

    def classify(self, exception: Exception) -> ErrorInfo:
        """Classify Gemini-specific errors."""
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
            error_str = str(exception).lower()
            is_rate_limit = (
                "429" in error_str
                or "resource_exhausted" in error_str
                or "quota" in error_str
                or "rate limit" in error_str
            )
            return ErrorInfo(
                is_retryable=not is_rate_limit,  # Don't retry rate limits with tenacity
                is_rate_limit=is_rate_limit,
                is_timeout=False,
                error_category="rate_limit" if is_rate_limit else "client_error",
                suggested_wait=300.0 if is_rate_limit else None,  # 5 min default
            )

        if isinstance(exception, ServerError):
            error_str = str(exception).lower()
            is_timeout = "504" in error_str or "deadline" in error_str
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
        error_str = str(exception).lower()

        # Check if it looks like a rate limit error (for mocks and other providers)
        if "429" in error_str or "resource_exhausted" in error_str or "rate limit" in error_str:
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=300.0,
            )

        # Check if it looks like a timeout
        if "timeout" in error_str or "504" in error_str or "deadline" in error_str:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="timeout",
            )

        # Default: treat as retryable
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown",
        )
