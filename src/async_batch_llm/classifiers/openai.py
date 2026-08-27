"""OpenAI-specific error classification.

Handles exceptions raised by the ``openai`` Python SDK (RateLimitError,
APITimeoutError, APIConnectionError, APIStatusError) plus the generic
fallbacks used across the library (FrameworkTimeoutError, validation errors,
logic bugs).

Added in v0.9.0.
"""

from __future__ import annotations

from ..strategies.errors import (
    BatchAbortedError,
    BatchDeadlineExceeded,
    ErrorClassifier,
    ErrorInfo,
    FrameworkTimeoutError,
    ItemDeadlineExceeded,
    StructuredOutputSchemaError,
    StructuredOutputValidationError,
    _retry_after_seconds,
    matches_any_pattern,
)

RATE_LIMIT_PATTERNS = (
    "429",
    "rate limit",
    "rate_limit_exceeded",
    "too many requests",
    "quota",
)
TIMEOUT_PATTERNS = ("timeout", "504", "deadline", "request timed out")
NETWORK_PATTERNS = ("connection", "network", "econnreset", "broken pipe")
# 402 Payment Required: balance/credits exhausted. DeepSeek in particular
# returns "402 Insufficient Balance" on a prepaid account that's run dry.
INSUFFICIENT_BALANCE_PATTERNS = ("402", "insufficient balance", "insufficient_quota")

# Operator-facing hint attached to 402 errors so an exhausted balance doesn't
# read like a generic API/code bug. Auth has already passed at this point.
_INSUFFICIENT_BALANCE_HINT = (
    "402 Payment Required — your account balance/credits are exhausted "
    "(e.g. top up your prepaid DeepSeek balance at "
    "https://platform.deepseek.com/). Not retryable."
)


class OpenAIErrorClassifier(ErrorClassifier):
    """Classifier for OpenAI SDK exceptions plus generic fallbacks.

    Designed to be subclassed: provider-specific classifiers (e.g.
    :class:`OpenRouterErrorClassifier`) override :meth:`classify` to handle
    extra cases first and delegate to ``super().classify()`` for the rest.
    """

    # Status codes that should be retried (transient server-side issues).
    _RETRYABLE_STATUS = frozenset({408, 425, 500, 502, 503, 504})
    # Status codes that should NOT be retried (deterministic client errors).
    _NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 405, 409, 410, 422})

    def _matches_any_pattern(self, error_str: str, patterns: tuple[str, ...]) -> bool:
        # Numeric codes ("429", "402", "504") match on word boundaries so an
        # unrelated number (e.g. "4290 tokens") doesn't trip a pattern. SDK
        # exception types and HTTP status codes are still preferred over this
        # string sniffing — see _classify_openai_exception, which runs first.
        return matches_any_pattern(error_str, patterns)

    def classify(self, exception: Exception) -> ErrorInfo:
        if isinstance(exception, ItemDeadlineExceeded):
            return ErrorInfo(False, False, True, "framework_total_item_timeout")
        if isinstance(exception, BatchDeadlineExceeded):
            return ErrorInfo(False, False, True, "batch_deadline_exceeded")
        if isinstance(exception, BatchAbortedError):
            return ErrorInfo(False, False, False, "batch_aborted")
        # Framework timeout takes priority over everything else.
        if isinstance(exception, FrameworkTimeoutError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="framework_timeout",
            )

        if isinstance(exception, StructuredOutputSchemaError):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="structured_output_schema_rejected",
            )

        if isinstance(exception, StructuredOutputValidationError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="structured_output_validation_error",
            )

        # Try to dispatch on the openai SDK's exception types when available.
        info = self._classify_openai_exception(exception)
        if info is not None:
            return info

        # Pydantic validation — the LLM produced output that failed schema.
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

        # Generic timeout/connection by exception type or message.
        error_str = str(exception)

        if isinstance(exception, TimeoutError) or self._matches_any_pattern(
            error_str, TIMEOUT_PATTERNS
        ):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="api_timeout",
            )

        if isinstance(exception, ConnectionError) or self._matches_any_pattern(
            error_str, NETWORK_PATTERNS
        ):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="network_error",
            )

        # Insufficient balance / payment required (string fallback for when the
        # SDK isn't installed or for mocked exceptions). Checked before the
        # rate-limit fallback so "402" doesn't get swept up by a stray pattern.
        if self._matches_any_pattern(error_str, INSUFFICIENT_BALANCE_PATTERNS):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="insufficient_balance",
                hint=_INSUFFICIENT_BALANCE_HINT,
            )

        # String-pattern fallback for rate limits when the SDK isn't installed
        # or for mocked test exceptions. No response object to parse a
        # Retry-After from, so no server-suggested wait.
        if self._matches_any_pattern(error_str, RATE_LIMIT_PATTERNS):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
            )

        # Logic bugs — deterministic; don't retry.
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
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="logic_error",
            )

        # Default: unknown but retryable (likely transient).
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown",
        )

    def _classify_openai_exception(self, exception: Exception) -> ErrorInfo | None:
        """Return ErrorInfo for openai-SDK exceptions, or None to defer."""
        try:
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                RateLimitError,
            )
        except ImportError:
            return None

        if isinstance(exception, RateLimitError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=_retry_after_seconds(exception),
            )

        if isinstance(exception, APITimeoutError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="api_timeout",
            )

        if isinstance(exception, APIConnectionError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="network_error",
            )

        if isinstance(exception, APIStatusError):
            return self._classify_status_error(exception)

        return None

    def _classify_status_error(self, exception: Exception) -> ErrorInfo:
        """Branch on ``APIStatusError.status_code``."""
        status_code = getattr(exception, "status_code", None)

        if status_code == 429:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=_retry_after_seconds(exception),
            )

        if status_code == 402:
            # Payment required / balance exhausted — deterministic, don't retry.
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="insufficient_balance",
                hint=_INSUFFICIENT_BALANCE_HINT,
            )

        if status_code == 401:
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="authentication",
            )

        if status_code == 403:
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="permission_denied",
            )

        if isinstance(status_code, int) and status_code in self._RETRYABLE_STATUS:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=status_code == 504,
                error_category="server_error",
            )

        if isinstance(status_code, int) and status_code in self._NON_RETRYABLE_STATUS:
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="client_error",
            )

        # Unrecognized status — be conservative and retry.
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=False,
            is_timeout=False,
            error_category="api_error",
        )
