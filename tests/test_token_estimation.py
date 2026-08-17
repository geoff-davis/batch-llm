"""Public token-estimation API and configuration contracts."""

from __future__ import annotations

import asyncio
import math
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from async_batch_llm import (
    CallableStrategy,
    CallOutcome,
    CharacterTokenEstimator,
    LLMResponse,
    ModelStrategy,
    ProcessorConfig,
    RetryState,
    TokenEstimate,
)


def test_token_estimate_is_immutable_validated_and_derives_total() -> None:
    estimate = TokenEstimate(input_tokens=3, output_tokens=4)
    assert estimate.total_tokens == 7
    with pytest.raises(FrozenInstanceError):
        estimate.input_tokens = 9  # type: ignore[misc]
    for value in (-1, True, 1.5, "1"):
        with pytest.raises(ValueError):
            TokenEstimate(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TokenEstimate(input_tokens=1, total_tokens=1)  # type: ignore[call-arg]


@pytest.mark.parametrize("value", [0, -1, True, math.inf, math.nan])
def test_character_estimator_rejects_invalid_ratio(value: object) -> None:
    with pytest.raises(ValueError):
        CharacterTokenEstimator(characters_per_token=value)  # type: ignore[arg-type]


def test_character_estimator_empty_prompt_minimum_and_expected_output() -> None:
    estimator = CharacterTokenEstimator(
        characters_per_token=4,
        expected_output_tokens=6,
        minimum_input_tokens=2,
    )
    strategy = _Callable()
    assert estimator("", strategy=strategy, attempt=1, state=None) == TokenEstimate(2, 6)
    assert estimator("123456789", strategy=strategy, attempt=1, state=None) == TokenEstimate(3, 6)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_tpm_config_rejects_non_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="max_tokens_per_minute"):
        ProcessorConfig(max_tokens_per_minute=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, math.inf, -math.inf, math.nan])
def test_rpm_config_rejects_bool_and_non_finite(value: object) -> None:
    with pytest.raises(ValueError, match="max_requests_per_minute"):
        ProcessorConfig(max_requests_per_minute=value)  # type: ignore[arg-type]


def test_fractional_rpm_and_disabled_tpm_are_compatible() -> None:
    config = ProcessorConfig(max_requests_per_minute=0.25)
    assert config.max_requests_per_minute == 0.25
    assert config.max_tokens_per_minute is None
    assert config.token_estimator is None


class _Callable(CallableStrategy[str]):
    def __init__(self, **kwargs: Any) -> None:
        async def invoke(
            prompt: str,
            *,
            attempt: int,
            timeout: float,
            state: RetryState | None,
        ) -> CallOutcome[str]:
            return CallOutcome(prompt)

        super().__init__(invoke, **kwargs)


def test_callable_strategy_exposes_estimator_convenience() -> None:
    seen: list[tuple[str, int, RetryState | None]] = []

    def estimator(prompt: str, *, strategy: Any, attempt: int, state: RetryState | None):
        seen.append((prompt, attempt, state))
        return TokenEstimate(5, 2)

    strategy = _Callable(token_estimator=estimator)
    state = RetryState()
    assert strategy.estimate_tokens("p", 3, state) == TokenEstimate(5, 2)
    assert seen == [("p", 3, state)]

    with pytest.raises(TypeError, match="token_estimator"):
        _Callable(token_estimator=object())  # type: ignore[arg-type]


class _Model:
    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(prompt, 1, 1, 2)


@pytest.mark.asyncio
async def test_model_strategy_exposes_async_estimator_convenience() -> None:
    async def estimator(prompt: str, *, strategy: Any, attempt: int, state: RetryState | None):
        await asyncio.sleep(0)
        return TokenEstimate(len(prompt), attempt)

    strategy = ModelStrategy(_Model(), token_estimator=estimator)
    pending = strategy.estimate_tokens("abc", 2, None)
    assert pending is not None
    assert await pending == TokenEstimate(3, 2)

    with pytest.raises(TypeError, match="token_estimator"):
        ModelStrategy(_Model(), token_estimator=object())  # type: ignore[arg-type]
