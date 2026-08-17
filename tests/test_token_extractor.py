"""Tests for TokenExtractor — central place for pulling token usage out of
successful responses, custom framework attributes, and PydanticAI-style
exception chains.

These tests drive the extraction logic out of ParallelBatchProcessor
(currently parallel.py:_extract_token_usage) into a dedicated class so it
can be reused and tested in isolation.
"""

from __future__ import annotations

import logging

import pytest

from async_batch_llm.token_extractor import TokenExtractor


@pytest.fixture
def extractor() -> TokenExtractor:
    return TokenExtractor()


# ─── Custom framework attribute ───────────────────────────────────────


def test_extract_from_failed_token_usage_dict(extractor):
    """Strategies attach `_failed_token_usage` dict to exceptions so the
    framework can account for tokens consumed by failed attempts."""
    e = RuntimeError("boom")
    e.__dict__["_failed_token_usage"] = {
        "input_tokens": 10,
        "output_tokens": 3,
        "total_tokens": 13,
    }
    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 3
    assert result["total_tokens"] == 13


def test_extract_ignores_non_dict_failed_token_usage(extractor):
    e = RuntimeError("boom")
    e.__dict__["_failed_token_usage"] = "not-a-dict"  # corrupt / unexpected shape
    result = extractor.extract_from_exception(e)
    assert result == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
    }


# ─── Direct .usage attribute on the exception ─────────────────────────


class _Usage:
    request_tokens = 7
    response_tokens = 2
    total_tokens = 9


class _ExceptionWithUsage(Exception):
    usage = _Usage()


def test_extract_from_exception_usage_attribute(extractor):
    e = _ExceptionWithUsage("x")
    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 7
    assert result["output_tokens"] == 2
    assert result["total_tokens"] == 9


class _CallableUsage:
    def __call__(self):
        return _Usage()


class _ExceptionWithCallableUsage(Exception):
    usage = _CallableUsage()


def test_extract_from_callable_usage_attribute(extractor):
    e = _ExceptionWithCallableUsage("x")
    result = extractor.extract_from_exception(e)
    assert result["total_tokens"] == 9


# ─── PydanticAI-style exception chain (cause.result.usage()) ──────────


class _PydanticAICause(Exception):
    class _Result:
        @staticmethod
        def usage():
            return _Usage()

    result = _Result()


def test_extract_from_pydantic_ai_cause_chain(extractor):
    outer = RuntimeError("outer")
    outer.__cause__ = _PydanticAICause("inner")
    result = extractor.extract_from_exception(outer)
    assert result["input_tokens"] == 7
    assert result["output_tokens"] == 2
    assert result["total_tokens"] == 9


# ─── Missing info → zeros, DEBUG log ──────────────────────────────────


def test_extract_returns_zeros_for_plain_exception(extractor, caplog):
    caplog.set_level(logging.DEBUG, logger="async_batch_llm.token_extractor")
    result = extractor.extract_from_exception(ValueError("no tokens here"))
    assert result == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
    }


def test_extract_logs_debug_when_usage_shape_is_weird(extractor, caplog):
    """If a provider exception has a `usage` attr that blows up on access,
    we should log DEBUG and return zeros rather than crash."""

    class _BadUsage:
        @property
        def request_tokens(self):
            raise RuntimeError("simulated shape mismatch")

    class _ExcBadUsage(Exception):
        usage = _BadUsage()

    caplog.set_level(logging.DEBUG, logger="async_batch_llm.token_extractor")
    result = extractor.extract_from_exception(_ExcBadUsage("x"))
    assert result["total_tokens"] == 0
    debug_messages = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
    assert any("token usage" in m.lower() for m in debug_messages), debug_messages


# ─── Accumulation across retries ──────────────────────────────────────


def test_accumulate_adds_all_fields(extractor):
    acc = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_input_tokens": 0}
    extractor.accumulate(acc, {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8})
    extractor.accumulate(
        acc, {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "cached_input_tokens": 4}
    )
    assert acc == {
        "input_tokens": 7,
        "output_tokens": 4,
        "total_tokens": 11,
        "cached_input_tokens": 4,
    }


def test_accumulate_ignores_missing_fields(extractor):
    acc = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "cached_input_tokens": 0}
    extractor.accumulate(acc, {})  # no-op
    assert acc == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "cached_input_tokens": 0,
    }


# ─── OpenAI/OpenRouter usage shape ────────────────────────────────────


class _OpenAIPromptDetails:
    cached_tokens = 32


class _OpenAIUsage:
    """Mimics the openai SDK's CompletionUsage shape."""

    prompt_tokens = 12
    completion_tokens = 7
    total_tokens = 19
    prompt_tokens_details = _OpenAIPromptDetails()


class _ExceptionWithOpenAIUsage(Exception):
    usage = _OpenAIUsage()


def test_extract_from_openai_usage_shape(extractor):
    """OpenAI / OpenRouter usage uses prompt_tokens / completion_tokens names
    and surfaces cached counts via prompt_tokens_details.cached_tokens."""
    e = _ExceptionWithOpenAIUsage("x")
    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 12
    assert result["output_tokens"] == 7
    assert result["total_tokens"] == 19
    assert result["cached_input_tokens"] == 32


# ─── CancelledError propagates (never swallowed) ──────────────────────


def test_cancelled_error_propagates(extractor):
    """Internal try/except must not swallow CancelledError."""
    import asyncio

    class _CancelOnAccess(Exception):
        @property
        def __cause__(self):  # type: ignore[override]
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        extractor.extract_from_exception(_CancelOnAccess("x"))


# ─── Precedence and coercion fixes ────────────────────────────────────


def test_failed_token_usage_takes_precedence_over_heuristics(extractor):
    """The framework-stamped exact count must win over heuristic paths.

    Regression: _failed_token_usage was checked last, so an exception that
    also exposed .usage (or a cause chain) had its exact count shadowed."""

    class _HeuristicUsage:
        prompt_tokens = 999
        completion_tokens = 999
        total_tokens = 1998

    e = Exception("boom")
    e.usage = _HeuristicUsage()  # type: ignore[attr-defined]
    e.__dict__["_failed_token_usage"] = {
        "input_tokens": 5,
        "output_tokens": 1,
        "total_tokens": 6,
    }

    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 5
    assert result["output_tokens"] == 1
    assert result["total_tokens"] == 6


def test_failed_token_usage_coerces_non_int_numerics(extractor):
    """Float/str counts are coerced instead of silently dropped to zero."""
    e = Exception("boom")
    e.__dict__["_failed_token_usage"] = {
        "input_tokens": 5.0,
        "output_tokens": "7",
        "total_tokens": 12,
    }

    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 5
    assert result["output_tokens"] == 7
    assert result["total_tokens"] == 12


def test_observation_distinguishes_explicit_zero_from_unknown(extractor):
    known_zero = Exception("known")
    known_zero.__dict__["_failed_token_usage"] = {"total_tokens": 0}

    explicit = extractor.observe_exception(known_zero)
    missing = extractor.observe_exception(Exception("unknown"))
    empty_success = extractor.observe_result({})

    assert explicit.known and explicit.reported_tokens == 0
    assert not missing.known and missing.reported_tokens is None
    assert not empty_success.known and empty_success.reported_tokens is None


def test_malformed_exception_usage_is_unknown_and_never_negative(extractor):
    malformed = Exception("provider failure")
    malformed.__dict__["_failed_token_usage"] = {"total_tokens": -5}

    observation = extractor.observe_exception(malformed)

    assert not observation.known
    assert observation.reported_tokens is None
    assert observation.usage["total_tokens"] == 0


def test_extract_pydantic_ai_v1_usage_shape(extractor):
    """pydantic-ai v1 renamed fields to input_tokens/output_tokens and
    surfaces cache hits as cache_read_tokens."""

    class _V1Usage:
        input_tokens = 10
        output_tokens = 5
        total_tokens = 15
        cache_read_tokens = 4

    e = Exception("x")
    e.usage = _V1Usage()  # type: ignore[attr-defined]

    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5
    assert result["cached_input_tokens"] == 4


def test_property_style_usage_on_cause_result(extractor):
    """pydantic-ai 1.x exposes result.usage as a property (not callable);
    the __cause__ path must read it directly."""

    class _V1Usage:
        input_tokens = 10
        output_tokens = 5
        total_tokens = 15

    class _Result:
        usage = _V1Usage()  # property-style: plain attribute, not a method

    cause = Exception("cause")
    cause.result = _Result()  # type: ignore[attr-defined]
    e = Exception("wrapper")
    e.__cause__ = cause

    result = extractor.extract_from_exception(e)
    assert result["input_tokens"] == 10
    assert result["total_tokens"] == 15
