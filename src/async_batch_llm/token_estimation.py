"""Public token-estimation contracts for proactive TPM admission."""

from __future__ import annotations

import math
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .base import RetryState
    from .llm_strategies import LLMCallStrategy


def _non_negative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer (got {value!r})")
    return value


@dataclass(frozen=True)
class TokenEstimate:
    """Estimated input and expected output tokens for one physical attempt."""

    input_tokens: int
    output_tokens: int = 0

    def __post_init__(self) -> None:
        _non_negative_integer(self.input_tokens, name="input_tokens")
        _non_negative_integer(self.output_tokens, name="output_tokens")

    @property
    def total_tokens(self) -> int:
        """Return the total token demand reserved by the quota gate."""
        return self.input_tokens + self.output_tokens


class TokenEstimator(Protocol):
    """Synchronous or asynchronous token estimate callable.

    The explicit processor estimator is invoked after middleware and cooldown,
    before quota or provider-capacity admission. ``attempt`` is ABL's logical
    attempt number; rate-limit retries therefore reuse it while still receiving
    a fresh estimate and reservation.
    """

    def __call__(
        self,
        prompt: str,
        *,
        strategy: LLMCallStrategy,
        attempt: int,
        state: RetryState | None,
    ) -> TokenEstimate | Awaitable[TokenEstimate]: ...


@dataclass(frozen=True)
class CharacterTokenEstimator:
    """Explicit, approximate character-count token estimator.

    This heuristic is intentionally never enabled automatically. It is useful
    for conservative local smoothing when a model-specific tokenizer is not
    available, but it does not claim provider- or model-specific accuracy.
    """

    characters_per_token: float = 4.0
    expected_output_tokens: int = 256
    minimum_input_tokens: int = 1

    def __post_init__(self) -> None:
        value = self.characters_per_token
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"characters_per_token must be finite and > 0 (got {self.characters_per_token!r})"
            )
        _non_negative_integer(self.expected_output_tokens, name="expected_output_tokens")
        _non_negative_integer(self.minimum_input_tokens, name="minimum_input_tokens")

    def __call__(
        self,
        prompt: str,
        *,
        strategy: LLMCallStrategy,
        attempt: int,
        state: RetryState | None,
    ) -> TokenEstimate:
        """Estimate tokens without making a provider or network call."""
        del strategy, attempt, state
        input_tokens = max(
            self.minimum_input_tokens,
            math.ceil(len(prompt) / float(self.characters_per_token)),
        )
        return TokenEstimate(
            input_tokens=input_tokens,
            output_tokens=self.expected_output_tokens,
        )


__all__ = ["CharacterTokenEstimator", "TokenEstimate", "TokenEstimator"]
