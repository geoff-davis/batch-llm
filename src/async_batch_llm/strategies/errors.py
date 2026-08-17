"""Error classification for different LLM providers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from async_batch_llm.base import TokenUsage

# Common error pattern constants
RATE_LIMIT_PATTERNS = ("429", "resource_exhausted", "quota", "rate limit")


@lru_cache(maxsize=64)
def _word_boundary_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(pattern)}\b")


def pattern_in(text_lower: str, pattern: str) -> bool:
    """Return True if ``pattern`` appears in already-lowercased ``text_lower``.

    Purely-numeric patterns (HTTP status codes like ``"429"``, ``"503"``,
    ``"402"``) are matched on **word boundaries** so an unrelated number such
    as ``"Expected 4290 tokens"`` doesn't get mistaken for a ``429`` rate
    limit and trigger a coordinated cooldown of every worker. Non-numeric
    patterns (``"quota"``, ``"rate limit"``) use plain substring containment.
    """
    if pattern.isdigit():
        return _word_boundary_regex(pattern).search(text_lower) is not None
    return pattern in text_lower


def matches_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    """Case-insensitively test ``text`` against ``patterns`` (see :func:`pattern_in`)."""
    lowered = text.lower()
    return any(pattern_in(lowered, pattern) for pattern in patterns)


def _retry_after_seconds(exception: Exception) -> float | None:
    """Parse a ``Retry-After`` header off an SDK exception, if present.

    Both the ``openai`` and ``google-genai`` SDKs attach the underlying HTTP
    response (with headers) to their exceptions as ``.response``.
    ``Retry-After`` may be either a number of seconds or an HTTP-date; we
    handle both and return the delay in seconds, or ``None`` when no usable
    header is present (including malformed or non-positive values).
    """
    response = getattr(exception, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        delay = float(raw)
        return delay if delay > 0 else None
    except (TypeError, ValueError):
        pass
    # HTTP-date form: compute the delay relative to now.
    try:
        import time
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(raw)
        delay = when.timestamp() - time.time()
        return delay if delay > 0 else None
    except (TypeError, ValueError):
        return None


class TokenTrackingError(Exception):
    """
    Wrapper exception that preserves token usage from failed LLM calls.

    When an LLM call fails (e.g., validation error), we still want to track
    the tokens that were consumed. This wrapper attaches token usage to
    exceptions that don't natively support it (e.g., built-in exceptions
    without __dict__).

    Attributes:
        token_usage: Dictionary with input_tokens, output_tokens, total_tokens,
            and optionally cached_input_tokens.

    Example:
        >>> try:
        ...     # LLM call that fails validation
        ...     output = parse_response(response)
        ... except Exception as e:
        ...     wrapped = TokenTrackingError(str(e), token_usage=tokens)
        ...     wrapped.__cause__ = e
        ...     raise wrapped from e
    """

    def __init__(self, message: str, *, token_usage: TokenUsage | dict[str, int] | None = None):
        """
        Initialize TokenTrackingError.

        Args:
            message: Human-readable error message
            token_usage: Token usage dict to preserve (input_tokens, output_tokens, etc.)
        """
        super().__init__(message)
        self._failed_token_usage = token_usage or {}


class TokenEstimationError(Exception):
    """Framework-owned, non-retryable token-estimation failure."""

    error_category = "token_estimation_error"


class TokenEstimatorRequired(TokenEstimationError):
    """TPM admission was enabled but no estimator resolved for an item."""

    error_category = "token_estimator_required"


class TokenEstimateExceedsLimit(TokenEstimationError):
    """One attempt's estimate cannot fit in the configured TPM bucket."""

    error_category = "token_estimate_exceeds_limit"


class EmptyResponseError(ValueError):
    """The provider returned a billed response with no usable text.

    Raised by the built-in models when the API call succeeded (and was
    billed) but produced no content — e.g. a Gemini safety block, or an
    OpenAI response whose ``finish_reason`` is ``length``/``content_filter``
    or a tool call.

    Subclasses ``ValueError`` so existing handlers and classifiers keep
    treating it as a deterministic, non-retryable failure. The tokens the
    provider already billed are attached as ``_failed_token_usage`` so
    failed-attempt accounting (``WorkItemResult.token_usage``) reflects the
    real spend.

    Added in v0.16.0.
    """

    def __init__(self, message: str, *, token_usage: TokenUsage | dict[str, int] | None = None):
        super().__init__(message)
        if token_usage:
            self._failed_token_usage = dict(token_usage)


class ProviderResponseError(Exception):
    """Provider signaled failure inside an HTTP-200 response body.

    Some gateways (notably OpenRouter) report upstream failures — no
    provider available, upstream 5xx, upstream rate limits — as HTTP 200
    with an ``error`` object in the body and no choices, so the SDK never
    raises. These are typically transient routing failures: classifiers
    treat them as retryable, and as rate limits when the embedded code
    is 429.

    Attributes:
        code: Numeric error code embedded in the body, if any.
        provider_error: The raw error payload from the response body.

    Added in v0.16.0.
    """

    def __init__(self, message: str, *, code: int | None = None, provider_error: Any = None):
        super().__init__(message)
        self.code = code
        self.provider_error = provider_error


class FrameworkTimeoutError(TimeoutError):
    """
    Timeout enforced by the batch-llm framework (asyncio.wait_for).

    This distinguishes framework-level timeouts from API-level timeouts.
    Framework timeouts indicate the configured attempt_timeout was exceeded,
    whereas API timeouts indicate the LLM provider returned a timeout error.

    Attributes:
        item_id: ID of the work item that timed out (if available)
        elapsed: Actual time elapsed in seconds
        timeout_limit: Configured timeout limit in seconds
    """

    def __init__(
        self,
        message: str,
        *,
        item_id: str | None = None,
        elapsed: float | None = None,
        timeout_limit: float | None = None,
    ):
        """
        Initialize FrameworkTimeoutError with structured context (v0.4.0).

        Args:
            message: Human-readable error message
            item_id: ID of the work item that timed out
            elapsed: Actual time elapsed before timeout
            timeout_limit: The timeout limit that was exceeded
        """
        super().__init__(message)
        self.item_id = item_id
        self.elapsed = elapsed
        self.timeout_limit = timeout_limit


class ItemDeadlineExceeded(TimeoutError):
    """The end-to-end monotonic deadline for one logical item expired."""

    def __init__(self, message: str, *, item_id: str | None = None) -> None:
        super().__init__(message)
        self.item_id = item_id


class BatchDeadlineExceeded(TimeoutError):
    """The batch deadline stopped an accepted item before completion."""

    def __init__(self, message: str, *, item_id: str | None = None) -> None:
        super().__init__(message)
        self.item_id = item_id


class BatchAbortedError(RuntimeError):
    """An accepted collateral item was stopped by configured fail-fast."""

    def __init__(self, message: str, *, item_id: str | None = None) -> None:
        super().__init__(message)
        self.item_id = item_id


class RateLimitRetriesExceeded(Exception):
    """A work item was retried after rate limits more than ``max_rate_limit_retries``.

    Rate-limit errors don't consume the ``max_attempts`` budget (they're a "wait
    and try again", not a failed attempt), but they ARE bounded separately by
    ``RetryConfig.max_rate_limit_retries`` so an endpoint that throttles forever
    can't hang the batch indefinitely. When that bound is exceeded the framework
    fails the item with this exception.

    Attributes:
        item_id: ID of the work item that exhausted its rate-limit retries.
        rate_limit_retries: How many rate-limit retries were attempted.
    """

    def __init__(
        self,
        message: str,
        *,
        item_id: str | None = None,
        rate_limit_retries: int | None = None,
    ):
        super().__init__(message)
        self.item_id = item_id
        self.rate_limit_retries = rate_limit_retries


@dataclass
class ErrorInfo:
    """Structured information about an error.

    Attributes:
        is_retryable: Whether the framework should retry the call.
        is_rate_limit: Whether this is a rate-limit error (triggers a
            coordinated cooldown rather than per-item backoff).
        is_timeout: Whether this is a timeout.
        error_category: A short label for stats/logging.
        suggested_wait: For rate limits, a server-suggested minimum wait in
            seconds (e.g. parsed from a ``Retry-After`` header). The
            ``RateLimitCoordinator`` honors this as a *floor* on the cooldown:
            the backoff strategy may wait longer, but never shorter. ``None``
            means "no suggestion; use the strategy's value as-is".
        hint: Optional human-readable remediation hint for the operator
            (e.g. "top up your prepaid balance" for a 402). Surfaced in the
            logs when the error is non-retryable so a misconfiguration doesn't
            look like a generic API bug. ``None`` means no extra guidance.
    """

    is_retryable: bool
    is_rate_limit: bool
    is_timeout: bool
    error_category: str
    suggested_wait: float | None = None
    hint: str | None = None


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
        """Return True if the error string looks like a rate limit.

        Numeric codes ("429") match on word boundaries via
        :func:`matches_any_pattern`, so "Expected 4290 tokens" is not
        misread as a rate limit.
        """
        return matches_any_pattern(error_str, RATE_LIMIT_PATTERNS)

    def classify(self, exception: Exception) -> ErrorInfo:
        """Classify common errors with conservative defaults."""
        error_str = str(exception).lower()

        if isinstance(exception, ItemDeadlineExceeded):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=True,
                error_category="framework_total_item_timeout",
            )
        if isinstance(exception, BatchDeadlineExceeded):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=True,
                error_category="batch_deadline_exceeded",
            )
        if isinstance(exception, BatchAbortedError):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="batch_aborted",
            )

        # Detect rate limit errors from message patterns (works for simple Exception mocks)
        if self._matches_rate_limit(error_str):
            return ErrorInfo(
                is_retryable=True,  # Rate limits are retryable - framework handles cooldown
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
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
