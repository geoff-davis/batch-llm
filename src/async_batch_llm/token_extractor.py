"""Centralized token-usage extraction for LLM responses and exceptions.

The framework needs to account for tokens consumed even by failed attempts
so users see accurate cost/usage telemetry. Different providers surface
usage in different ways:

1. **Custom framework attribute** — strategies attach `_failed_token_usage`
   to exceptions via `__dict__` when they know the count. Checked first:
   it's an exact per-attempt count, so it must win over the heuristics.
2. **PydanticAI-style** — exception's `__cause__` has a `.result` with a
   usage property (or legacy callable `.usage()`).
3. **Direct `.usage` attribute** on the exception (OpenAI-style wrappers).

Previously this logic lived inline on `ParallelBatchProcessor`. Extracting
it makes each path testable in isolation and keeps the processor lean.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from .base import TokenUsage

logger = logging.getLogger(__name__)


_EMPTY_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "cached_input_tokens": 0,
}
_USAGE_KEYS = frozenset(_EMPTY_USAGE)


@dataclass(frozen=True)
class TokenUsageObservation:
    """Internal usage plus whether a provider supplied an exact observation.

    ``reported_tokens=None`` means usage was unavailable. Zero is deliberately
    retained as a distinct known value for full reservation refunds.
    """

    usage: TokenUsage
    known: bool
    reported_tokens: int | None


class TokenExtractor:
    """Best-effort token-usage extraction from LLM exceptions."""

    def extract_from_exception(self, exception: BaseException) -> dict[str, int]:
        """Return a token-usage dict for a failed LLM call.

        Tries three strategies in order and returns the first match. Returns
        zeroed dict if no extraction succeeds. Never raises for normal
        extraction failures — only `asyncio.CancelledError` propagates.
        """
        return cast(dict[str, int], self.observe_exception(exception).usage).copy()

    def observe_exception(self, exception: BaseException) -> TokenUsageObservation:
        """Observe failed-attempt usage without collapsing unknown into zero."""
        try:
            # Strategy 1: Custom _failed_token_usage attribute (set by this
            # framework). Checked first — it carries the exact per-attempt
            # count and must not be shadowed by the heuristic paths below.
            exc_dict = getattr(exception, "__dict__", None)
            if isinstance(exc_dict, dict):
                failed = exc_dict.get("_failed_token_usage")
                if isinstance(failed, dict):
                    return _mapping_observation(failed, explicit=True, strict=False)

            # Strategy 2: PydanticAI-style exception with result in __cause__.
            # pydantic-ai 1.x exposes usage as a property; older versions and
            # test doubles expose a usage() method — call only bound methods.
            cause = getattr(exception, "__cause__", None)
            if cause is not None:
                result = getattr(cause, "result", None)
                if result is not None:
                    usage_attr = getattr(result, "usage", None)
                    if inspect.ismethod(usage_attr) or inspect.isfunction(usage_attr):
                        usage_attr = usage_attr()
                    if usage_attr is not None:
                        return _coerce_usage_observation(usage_attr)

            # Strategy 3: Direct .usage attribute on exception
            usage = getattr(exception, "usage", None)
            if usage is not None:
                if callable(usage):
                    usage = usage()
                if usage is not None:
                    return _coerce_usage_observation(usage)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Extraction is best-effort; log for debugging.
            logger.debug(
                "Failed to extract token usage from %s: %s. Returning zero tokens.",
                type(exception).__name__,
                e,
            )

        return TokenUsageObservation(
            usage=cast(TokenUsage, dict(_EMPTY_USAGE)),
            known=False,
            reported_tokens=None,
        )

    @staticmethod
    def observe_result(usage: object) -> TokenUsageObservation:
        """Validate and observe a successful strategy usage mapping."""
        if not isinstance(usage, Mapping):
            raise TypeError(f"Strategy token usage must be a mapping (got {type(usage).__name__})")
        return _mapping_observation(
            cast(Mapping[object, object], usage),
            explicit=False,
            strict=True,
        )

    @staticmethod
    def accumulate(cumulative: dict[str, int], attempt_tokens: dict[str, int]) -> None:
        """Add per-attempt token counts into a running cumulative total.

        Missing fields on `attempt_tokens` are treated as zero.
        """
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
            cumulative[key] = cumulative.get(key, 0) + attempt_tokens.get(key, 0)


def _first_attr(usage: Any, *names: str) -> Any:
    """Return the first present, non-None attribute in ``names`` (short-circuits).

    Short-circuiting matters: pydantic-ai 1.x keeps ``request_tokens`` /
    ``response_tokens`` as *deprecated* aliases that emit a DeprecationWarning
    when touched, so we ask for the 1.x names (``input_tokens`` /
    ``output_tokens``) first and never read the deprecated ones when the new
    ones are present.
    """
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return value
    return 0


def _coerce_usage(usage: Any) -> dict[str, int]:
    """Convert a provider-specific usage object into our dict shape.

    Field-name aliasing covers the common providers:

    - PydanticAI 1.x / Anthropic / our normalized shape: ``input_tokens`` /
      ``output_tokens`` (PydanticAI 0.x ``request_tokens`` / ``response_tokens``
      are still read as a fallback).
    - OpenAI / OpenRouter: ``prompt_tokens`` / ``completion_tokens``
    """
    return cast(dict[str, int], _coerce_usage_observation(usage).usage).copy()


def _coerce_usage_observation(usage: Any) -> TokenUsageObservation:
    input_tokens, input_valid = _nonnegative_int(
        _first_attr(usage, "input_tokens", "request_tokens", "prompt_tokens")
    )
    output_tokens, output_valid = _nonnegative_int(
        _first_attr(usage, "output_tokens", "response_tokens", "completion_tokens")
    )
    cached, _ = _nonnegative_int(getattr(usage, "cached_input_tokens", 0))
    if not cached:
        # pydantic-ai v1 surfaces cache hits as cache_read_tokens.
        cached, _ = _nonnegative_int(getattr(usage, "cache_read_tokens", 0))
    if not cached:
        # OpenAI surfaces cached prompt tokens nested under prompt_tokens_details.
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached, _ = _nonnegative_int(getattr(details, "cached_tokens", 0))
    total_attr = getattr(usage, "total_tokens", None)
    total_tokens, total_valid = _nonnegative_int(total_attr)
    if total_attr is not None:
        reported_tokens = total_tokens if total_valid else None
    elif input_valid and output_valid:
        reported_tokens = input_tokens + output_tokens
    else:
        reported_tokens = None
    normalized = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens if total_attr is not None else 0,
        "cached_input_tokens": cached,
    }
    return TokenUsageObservation(
        usage=cast(TokenUsage, normalized),
        known=reported_tokens is not None,
        reported_tokens=reported_tokens,
    )


def _mapping_observation(
    usage: Mapping[object, object],
    *,
    explicit: bool,
    strict: bool,
) -> TokenUsageObservation:
    recognized = {key for key in usage if isinstance(key, str) and key in _USAGE_KEYS}
    normalized = dict(_EMPTY_USAGE)
    valid: dict[str, bool] = {}
    for key in recognized:
        value = usage[key]
        if strict and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(
                f"Strategy token usage[{key!r}] must be a non-negative integer (got {value!r})"
            )
        normalized[key], valid[key] = _nonnegative_int(value)

    known = explicit or bool(recognized)
    if not known:
        reported_tokens = None
    elif "total_tokens" in recognized:
        reported_tokens = normalized["total_tokens"] if valid["total_tokens"] else None
    elif "input_tokens" in recognized or "output_tokens" in recognized:
        input_valid = valid.get("input_tokens", True)
        output_valid = valid.get("output_tokens", True)
        reported_tokens = (
            normalized["input_tokens"] + normalized["output_tokens"]
            if input_valid and output_valid
            else None
        )
    else:
        # A recognized cached-token-only mapping is still an explicit known
        # observation, but cached tokens do not reduce TPM automatically.
        reported_tokens = 0
    return TokenUsageObservation(
        usage=cast(TokenUsage, normalized),
        known=known and reported_tokens is not None,
        reported_tokens=reported_tokens,
    )


def _nonnegative_int(v: Any) -> tuple[int, bool]:
    try:
        value = int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0, False
    if isinstance(v, bool) or value < 0:
        return 0, False
    return value, True


__all__ = ["TokenExtractor", "TokenUsageObservation"]
