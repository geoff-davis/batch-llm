"""Seeded, credential-free fake provider used by every scenario.

Behavior is derived from ``sha256(seed:index)`` so a million-item run needs no
per-item behavior table. The strategy retains only counters and bounded
diagnostics — never one object per item.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field

from async_batch_llm import LLMCallStrategy, RetryState
from async_batch_llm.strategies import ErrorClassifier, ErrorInfo

TOKENS_OK = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
TOKENS_FAILED_ATTEMPT = {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}


class FakeRateLimitError(Exception):
    """Deterministic stand-in for a provider 429/overload response."""


class FakeTransportError(Exception):
    """Deterministic stand-in for a retryable connection/5xx failure."""


class FakeValidationError(Exception):
    """First billed response failed parsing; carries recoverable token usage."""


class FakePoisonError(Exception):
    """Non-retryable terminal error used to trigger category fail-fast."""


class ScaleSoakClassifier(ErrorClassifier):
    """Maps the fake exceptions onto the production retry categories."""

    def __init__(self, suggested_wait: float = 0.01) -> None:
        self.suggested_wait = suggested_wait

    def classify(self, exception: Exception) -> ErrorInfo:
        if isinstance(exception, FakeRateLimitError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=self.suggested_wait,
            )
        if isinstance(exception, FakeTransportError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="network",
            )
        if isinstance(exception, FakeValidationError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="validation",
            )
        if isinstance(exception, FakePoisonError):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="poison_pill",
            )
        return ErrorInfo(
            is_retryable=False,
            is_rate_limit=False,
            is_timeout=False,
            error_category="unexpected",
        )


def item_id_for(index: int) -> str:
    return f"i{index}"


def index_of(item_id: str) -> int:
    if not item_id.startswith("i"):
        raise ValueError(f"harness item ids look like 'i<index>' (got {item_id!r})")
    return int(item_id[1:])


def behavior_hash(seed: int, index: int) -> int:
    """Stable per-item behavior source; no per-item state retained."""
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class FakeProviderConfig:
    """Which deterministic failures the strategy injects."""

    seed: int
    latency_s: float = 0.0
    # First physical attempt fails with a 429 for indexes in [wave_start, wave_end).
    rate_limit_wave: tuple[int, int] | None = None
    # behavior_hash % transport_modulus == 0 -> first physical attempt fails
    # with a retryable transport error.
    transport_modulus: int | None = None
    # behavior_hash % validation_modulus == 1 -> first logical attempt raises a
    # validation failure carrying recoverable token usage (scenario E).
    validation_modulus: int | None = None
    # This index fails permanently with the poison category (scenario F run 1).
    poison_index: int | None = None


@dataclass(eq=False)
class FakeProviderStrategy(LLMCallStrategy[str]):
    """Shared strategy instance driving every worker in a scenario."""

    config: FakeProviderConfig
    calls: int = 0
    concurrent_calls: int = 0
    peak_concurrent_calls: int = 0
    rate_limit_failures: int = 0
    transport_failures: int = 0
    validation_failures: int = 0
    poison_failures: int = 0
    isolation_violations: list[str] = field(default_factory=list)
    prepared: int = 0
    cleaned_up: int = 0

    async def prepare(self) -> None:
        self.prepared += 1

    async def cleanup(self) -> None:
        self.cleaned_up += 1

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ) -> None:
        if isinstance(exception, FakeValidationError) and state is not None:
            # Item-specific corrective feedback: the next attempt must observe
            # exactly this marker and no other item's (scenario E).
            state.set("feedback_owner", str(exception.args[0]))
            state.set("feedback", f"correct:{exception.args[0]}")
            state.set("escalate", True)

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[str, dict[str, int], None]:
        item_id = prompt.rsplit(":", 1)[-1]
        index = index_of(item_id)
        cfg = self.config

        self.calls += 1
        self.concurrent_calls += 1
        self.peak_concurrent_calls = max(self.peak_concurrent_calls, self.concurrent_calls)
        physical_attempt = 1
        if state is not None:
            physical_attempt = state.get("physical_attempts", 0) + 1
            state.set("physical_attempts", physical_attempt)
        try:
            if cfg.latency_s > 0:
                await asyncio.sleep(cfg.latency_s)

            if cfg.poison_index is not None and index == cfg.poison_index:
                self.poison_failures += 1
                raise FakePoisonError(item_id)

            wave = cfg.rate_limit_wave
            if wave is not None and wave[0] <= index < wave[1] and physical_attempt == 1:
                self.rate_limit_failures += 1
                raise FakeRateLimitError(item_id)

            digest = behavior_hash(cfg.seed, index)
            if (
                cfg.transport_modulus is not None
                and digest % cfg.transport_modulus == 0
                and physical_attempt == 1
            ):
                self.transport_failures += 1
                raise FakeTransportError(item_id)

            if cfg.validation_modulus is not None and digest % cfg.validation_modulus == 1:
                if attempt == 1:
                    self.validation_failures += 1
                    error = FakeValidationError(item_id)
                    # The response was billed before parsing failed.
                    error._failed_token_usage = dict(TOKENS_FAILED_ATTEMPT)  # type: ignore[attr-defined]
                    raise error
                # Recovery attempt: the stored feedback must belong to this item.
                owner = state.get("feedback_owner") if state is not None else None
                feedback = state.get("feedback") if state is not None else None
                if owner != item_id or feedback != f"correct:{item_id}":
                    self.isolation_violations.append(f"{item_id} observed feedback for {owner!r}")
                return (f"ok:{item_id}:escalated", dict(TOKENS_OK), None)

            return (f"ok:{item_id}", dict(TOKENS_OK), None)
        finally:
            self.concurrent_calls -= 1
