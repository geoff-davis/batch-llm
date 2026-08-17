"""Seeded, credential-free fake provider used by every scenario.

Behavior is derived from ``sha256(seed:index)`` so a million-item run needs no
per-item behavior table. The strategy retains only counters and bounded
diagnostics — never one object per item.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import cast

from async_batch_llm import LLMCallStrategy, RetryState, TokenEstimate, TokenUsage
from async_batch_llm.strategies import ErrorClassifier, ErrorInfo

TOKENS_OK = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
TOKENS_FAILED_ATTEMPT = {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}

TOKEN_BANDS = ("small", "medium", "large")
TOKEN_ESTIMATES = {
    "small": TokenEstimate(input_tokens=4, output_tokens=4),
    "medium": TokenEstimate(input_tokens=30, output_tokens=20),
    "large": TokenEstimate(input_tokens=3_000, output_tokens=1_500),
}
TOKEN_ACTUAL = {
    "small": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    "medium": {"input_tokens": 30, "output_tokens": 20, "total_tokens": 50},
    "large": {"input_tokens": 3_200, "output_tokens": 1_800, "total_tokens": 5_000},
}
TOKEN_FAILED_KNOWN = {"input_tokens": 25, "output_tokens": 15, "total_tokens": 40}


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


@dataclass(frozen=True)
class TokenMixLayout:
    """Constant-space layout for the mixed-token scenario."""

    items: int
    large_count: int
    medium_count: int
    known_retry_index: int
    unknown_retry_index: int
    known_zero_index: int

    @classmethod
    def for_items(cls, items: int) -> TokenMixLayout:
        if items < 12:
            raise ValueError("token_quota_mixed requires at least 12 items")
        large_count = 6 if items < 10_000 else max(6, (items // 100) // 2 * 2)
        medium_count = max(4, items // 20)
        medium_count = min(medium_count, items - large_count - 2)
        return cls(
            items=items,
            large_count=large_count,
            medium_count=medium_count,
            known_retry_index=large_count,
            unknown_retry_index=large_count + 1,
            known_zero_index=large_count + 2,
        )

    def band_for(self, index: int) -> str:
        if not 0 <= index < self.items:
            raise ValueError(f"index {index} outside 0..{self.items - 1}")
        if index < self.large_count:
            return "large"
        if index < self.large_count + self.medium_count:
            return "medium"
        return "small"

    def band_counts(self) -> dict[str, int]:
        return {
            "small": self.items - self.large_count - self.medium_count,
            "medium": self.medium_count,
            "large": self.large_count,
        }


def token_mix_expected(layout: TokenMixLayout) -> dict[str, object]:
    """Independent aggregate oracle for reservation/reconciliation totals."""
    estimated_input = 0
    estimated_output = 0
    reserved = 0
    reported_input = 0
    reported_output = 0
    reported = 0
    refunded = 0
    debt = 0
    net_by_scope = [0, 0]
    for index in range(layout.items):
        band = layout.band_for(index)
        estimate = TOKEN_ESTIMATES[band]
        actual = TOKEN_ACTUAL[band]["total_tokens"]
        physical_calls = 2 if index in (layout.known_retry_index, layout.unknown_retry_index) else 1
        estimated_input += estimate.input_tokens * physical_calls
        estimated_output += estimate.output_tokens * physical_calls
        reserved += estimate.total_tokens * physical_calls

        if index == layout.known_zero_index:
            item_reported_input = 0
            item_reported_output = 0
            item_reported = 0
            item_refunded = estimate.total_tokens
            item_debt = 0
        else:
            item_reported_input = TOKEN_ACTUAL[band]["input_tokens"]
            item_reported_output = TOKEN_ACTUAL[band]["output_tokens"]
            item_reported = actual
            item_refunded = max(estimate.total_tokens - actual, 0)
            item_debt = max(actual - estimate.total_tokens, 0)

        if index == layout.known_retry_index:
            item_reported_input += TOKEN_FAILED_KNOWN["input_tokens"]
            item_reported_output += TOKEN_FAILED_KNOWN["output_tokens"]
            item_reported += TOKEN_FAILED_KNOWN["total_tokens"]
            item_refunded += estimate.total_tokens - TOKEN_FAILED_KNOWN["total_tokens"]
        elif index == layout.unknown_retry_index:
            # The first transport attempt has unknown usage, so its reservation
            # is retained conservatively and is part of the quota charge only.
            net_by_scope[index % 2] += estimate.total_tokens

        reported_input += item_reported_input
        reported_output += item_reported_output
        reported += item_reported
        refunded += item_refunded
        debt += item_debt
        net_by_scope[index % 2] += item_reported

    return {
        "band_counts": layout.band_counts(),
        "estimated_input_tokens": estimated_input,
        "estimated_output_tokens": estimated_output,
        "reserved_tokens": reserved,
        "reported_input_tokens": reported_input,
        "reported_output_tokens": reported_output,
        "reported_tokens": reported,
        "refunded_tokens": refunded,
        "underestimated_tokens": debt,
        "unknown_usage_attempts": 1,
        "known_zero_usage_attempts": 1,
        "provider_calls": layout.items + 2,
        "retries": 2,
        "net_quota_tokens_by_scope": net_by_scope,
    }


@dataclass
class TokenMixTracker:
    """Bounded provider-side oracle shared by the two quota strategies."""

    calls: int = 0
    concurrent_calls: int = 0
    peak_concurrent_calls: int = 0
    known_usage_failures: int = 0
    unknown_usage_failures: int = 0
    reported_tokens: int = 0
    calls_by_band: dict[str, int] = field(default_factory=lambda: dict.fromkeys(TOKEN_BANDS, 0))


@dataclass(eq=False)
class TokenMixStrategy(LLMCallStrategy[str]):
    """Deterministic mixed-size provider with exact success/failure usage."""

    layout: TokenMixLayout
    tracker: TokenMixTracker
    quota_scope_key: object
    concurrency_scope_key: object
    latency_s: float = 0.0
    prepared: int = 0
    cleaned_up: int = 0

    @property
    def quota_scope(self) -> object:
        return self.quota_scope_key

    @property
    def concurrency_scope(self) -> object:
        return self.concurrency_scope_key

    async def prepare(self) -> None:
        self.prepared += 1

    async def cleanup(self) -> None:
        self.cleaned_up += 1

    async def estimate_tokens(
        self, prompt: str, attempt: int, state: RetryState | None
    ) -> TokenEstimate:
        del attempt, state
        index = index_of(prompt.rsplit(":", 1)[-1])
        return TOKEN_ESTIMATES[self.layout.band_for(index)]

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[str, TokenUsage, None]:
        del attempt, timeout
        item_id = prompt.rsplit(":", 1)[-1]
        index = index_of(item_id)
        band = self.layout.band_for(index)
        physical_attempt = 1
        if state is not None:
            physical_attempt = state.get("token_mix_physical_attempts", 0) + 1
            state.set("token_mix_physical_attempts", physical_attempt)

        tracker = self.tracker
        tracker.calls += 1
        tracker.calls_by_band[band] += 1
        tracker.concurrent_calls += 1
        tracker.peak_concurrent_calls = max(tracker.peak_concurrent_calls, tracker.concurrent_calls)
        try:
            if self.latency_s > 0:
                await asyncio.sleep(self.latency_s)
            if index == self.layout.known_retry_index and physical_attempt == 1:
                tracker.known_usage_failures += 1
                tracker.reported_tokens += TOKEN_FAILED_KNOWN["total_tokens"]
                error = FakeValidationError("token-mix-known-usage")
                error._failed_token_usage = dict(TOKEN_FAILED_KNOWN)  # type: ignore[attr-defined]
                raise error
            if index == self.layout.unknown_retry_index and physical_attempt == 1:
                tracker.unknown_usage_failures += 1
                raise FakeTransportError("token-mix-unknown-usage")

            usage: TokenUsage = (
                {"total_tokens": 0}
                if index == self.layout.known_zero_index
                else cast(TokenUsage, dict(TOKEN_ACTUAL[band]))
            )
            tracker.reported_tokens += usage["total_tokens"]
            return f"token-ok:{item_id}", usage, None
        finally:
            tracker.concurrent_calls -= 1


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
    ) -> tuple[str, TokenUsage, None]:
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
                return (f"ok:{item_id}:escalated", cast(TokenUsage, dict(TOKENS_OK)), None)

            return (f"ok:{item_id}", cast(TokenUsage, dict(TOKENS_OK)), None)
        finally:
            self.concurrent_calls -= 1
