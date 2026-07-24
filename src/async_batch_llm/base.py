"""Base classes and interfaces for batch LLM processing."""

import asyncio
import contextlib
import inspect
import logging
import math
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, cast, overload  # noqa: F401

from typing_extensions import TypeVar  # PEP 696 defaults on Python < 3.13

from .provider_output import ProviderOutputViews

# Conditional imports for type checking
if TYPE_CHECKING:
    from pathlib import Path

    from .llm_strategies import LLMCallStrategy
    from .serialization import ValueDecoder, ValueEncoder

# Type variables for generic typing. PEP 696 defaults let common usage drop
# trailing parameters: ``ParallelBatchProcessor[str, MyOutput]`` (context
# defaults to None), or even ``ParallelBatchProcessor[str]``.
# Note: TInput is currently unused by the framework (prompts are always str);
# it is kept for backward compatibility and slated for removal in the next
# major release.
TInput = TypeVar("TInput", default=str)  # Input data type
TOutput = TypeVar("TOutput", default=Any)  # Agent output type
TContext = TypeVar("TContext", default=None)  # Optional context passed through

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class RetryState:
    """
    Mutable state that persists across retry attempts for a single work item.

    This allows strategies to maintain state between retry attempts, enabling
    multi-stage retry patterns where each attempt can build on previous attempts.

    Example use cases:
    - Partial recovery: Save successfully parsed fields from failed attempts
    - Progressive refinement: Track which parts of output need improvement
    - Smart retry prompts: Build targeted prompts based on what failed
    - Model escalation: Track validation errors vs network errors separately

    Usage:
        # In strategy's on_error():
        if state:
            state.set('last_error_type', type(exception).__name__)
            state.set('validation_failures', state.get('validation_failures', 0) + 1)

        # In strategy's execute():
        if state and state.get('validation_failures', 0) > 1:
            # Use smarter model or modified prompt
            pass

    Added in v0.3.0.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the state.

        Args:
            key: State key to retrieve
            default: Value to return if key doesn't exist

        Returns:
            Value associated with key, or default if not found
        """
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the state.

        Args:
            key: State key to set
            value: Value to store (can be any JSON-serializable type)
        """
        self.data[key] = value

    def delete(self, key: str, raise_if_missing: bool = False) -> None:
        """
        Delete a value from the state.

        Args:
            key: State key to delete
            raise_if_missing: If True, raise KeyError if key doesn't exist.
                             If False, silently ignore missing keys (default).

        Raises:
            KeyError: If key doesn't exist and raise_if_missing=True
        """
        if raise_if_missing:
            del self.data[key]
        else:
            self.data.pop(key, None)

    def clear(self) -> None:
        """Clear all state data."""
        self.data.clear()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in state."""
        return key in self.data

    def __repr__(self) -> str:
        """String representation showing all state data."""
        return f"RetryState({self.data!r})"


# Timeout constants (seconds)
WORKER_CANCELLATION_TIMEOUT = 2.0  # Time to wait for workers to cancel gracefully
WORKER_SHUTDOWN_TIMEOUT = 30.0  # Time to wait for workers to finish after queue is done
PROGRESS_TASK_CANCELLATION_TIMEOUT = 2.0  # Time to wait for progress callbacks to cancel


class TokenUsage(TypedDict, total=False):
    """
    Token usage statistics from LLM API calls.

    All fields are optional to accommodate different provider APIs.
    Different providers may return different subsets of these fields.

    Fields:
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/completion
        total_tokens: Total tokens used (input + output)
        cached_input_tokens: Number of input tokens served from cache (Gemini)
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int


@dataclass
class LLMResponse(ProviderOutputViews):
    """
    Normalized response from any LLM provider.

    Returned by LLMModel.generate(). Provides a provider-agnostic interface
    so strategies don't need to know about Gemini, OpenAI, etc. response formats.

    Attributes:
        text: The response text content.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        total_tokens: Total tokens used.
        cached_input_tokens: Input tokens served from cache (0 if no caching).
        metadata: Provider-specific metadata (safety ratings, finish reason, etc.).
            The keys ``'grounding'``, ``'reasoning'``, ``'tool_calls'``, and
            ``'logprobs'`` are reserved, with documented dict shapes readable
            through the typed views ``.grounding``/``.reasoning``/
            ``.tool_calls``/``.logprobs`` (see ``provider_output.py``).
        raw: The raw provider response object, for edge cases.

    Added in v0.6.0.
    """

    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0
    metadata: dict[str, Any] | None = None
    raw: Any = None

    @property
    def token_usage(self) -> TokenUsage:
        """Return token counts as a TokenUsage dict."""
        result: TokenUsage = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_input_tokens:
            result["cached_input_tokens"] = self.cached_input_tokens
        return result


@dataclass
class LLMWorkItem(Generic[TInput, TOutput, TContext]):
    """
    Represents a single work item to be processed by an LLM strategy.

    Attributes:
        item_id: Unique identifier for this work item
        strategy: LLM call strategy that encapsulates how to make the LLM call
        prompt: The prompt/input to pass to the LLM
        context: Optional context data passed through to results/post-processor
    """

    item_id: str
    strategy: "LLMCallStrategy[TOutput]"
    prompt: str = ""
    context: TContext | None = None
    # Assigned by BatchProcessor.add_work() when the item is accepted. Kept on
    # the work item so duplicate item IDs remain independently orderable.
    submission_index: int | None = field(default=None, compare=False)
    _artifact_key: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Validate work item fields."""
        if not self.item_id or not isinstance(self.item_id, str):
            raise ValueError(
                f"item_id must be a non-empty string (got {type(self.item_id).__name__}: {repr(self.item_id)}). "
                f"Provide a unique string identifier for this work item."
            )
        if not self.item_id.strip():
            raise ValueError(
                f"item_id cannot be whitespace only (got {repr(self.item_id)}). "
                f"Provide a non-whitespace string identifier."
            )
        if self.strategy is None:
            raise ValueError(
                "strategy must not be None. "
                "Pass an LLMCallStrategy instance (e.g., PydanticAIStrategy, GeminiStrategy, "
                "or your custom subclass)."
            )
        if not isinstance(self.prompt, str):
            raise TypeError(
                f"prompt must be a string (got {type(self.prompt).__name__}: {repr(self.prompt)[:80]}). "
                f"If you need to pass structured data, serialize it to a string first."
            )


@dataclass
class AttemptTiming:
    """Timing and outcome details for one physical strategy execution try."""

    attempt: int
    try_number: int
    total_seconds: float = 0.0
    admission_wait_seconds: float = 0.0
    startup_ramp_wait_seconds: float = 0.0
    execution_seconds: float = 0.0
    provider_seconds: float | None = None
    cooldown_wait_seconds: float = 0.0
    retry_backoff_seconds: float = 0.0
    success: bool = False
    error_type: str | None = None
    error_category: str | None = None
    timeout_category: str | None = None


@dataclass
class WorkItemTiming:
    """End-to-end timing for one item, including every retry try."""

    total_seconds: float = 0.0
    attempts: list[AttemptTiming] = field(default_factory=list)
    timeout_category: str | None = None

    @property
    def admission_wait_seconds(self) -> float:
        return sum(attempt.admission_wait_seconds for attempt in self.attempts)

    @property
    def execution_seconds(self) -> float:
        return sum(attempt.execution_seconds for attempt in self.attempts)

    @property
    def startup_ramp_wait_seconds(self) -> float:
        return sum(attempt.startup_ramp_wait_seconds for attempt in self.attempts)

    @property
    def provider_seconds(self) -> float:
        return sum(attempt.provider_seconds or 0.0 for attempt in self.attempts)

    @property
    def cooldown_wait_seconds(self) -> float:
        return sum(attempt.cooldown_wait_seconds for attempt in self.attempts)

    @property
    def retry_backoff_seconds(self) -> float:
        return sum(attempt.retry_backoff_seconds for attempt in self.attempts)


@dataclass
class WorkItemResult(ProviderOutputViews, Generic[TOutput, TContext]):
    """
    Result of processing a single work item.

    Attributes:
        item_id: ID of the work item
        success: Whether processing succeeded
        output: Agent output if successful, None if failed
        error: Error message if failed, None if successful
        context: Context data from the work item
        token_usage: Token usage stats (input_tokens, output_tokens, total_tokens)
        metadata: Provider-specific metadata returned alongside the response —
            e.g. ``{"provider": "Anthropic", "finish_reason": "stop",
            "model": "anthropic/claude-haiku-4-5"}``. Populated when the
            strategy returns a 3-tuple ``(output, tokens, metadata)`` from
            ``execute()``; ``None`` for legacy 2-tuple strategies.
            The keys ``'grounding'``, ``'reasoning'``, ``'tool_calls'``, and
            ``'logprobs'`` are reserved, with documented dict shapes readable
            through the typed views ``.grounding``/``.reasoning``/
            ``.tool_calls``/``.logprobs`` (see ``provider_output.py``).
            Added in v0.10.0. (For Gemini safety ratings specifically, this
            replaces the older ``gemini_safety_ratings`` field — see below.)
        gemini_safety_ratings: **Deprecated.** Use ``metadata['safety_ratings']``
            instead. Still populated when the underlying model surfaces them,
            for backward compat. To be removed in a future release.
        exception: The originating exception for a failed result, when one was
            raised (all retries exhausted, or a permanent non-retryable error).
            ``None`` for successes and for non-error outcomes such as a
            middleware filter-skip. ``call()`` / ``LLMCallPool.submit()`` re-raise
            this exact exception (preserving the provider's type) rather than a
            generic ``LLMCallError``. Its traceback is detached before storage
            (the full failure is already logged at the failure site) so
            accumulated failed results don't pin frame locals; a re-raise gets a
            fresh traceback. Excluded from equality so two failed results with
            distinct exception instances still compare equal.
        admission_wait_seconds: Total time this item spent waiting for provider
            capacity across all attempts. This wait occurs before the per-attempt
            execution timeout starts.
        timing: Structured end-to-end and per-attempt timing details.
    """

    item_id: str
    success: bool
    output: TOutput | None = None
    error: str | None = None
    context: TContext | None = None
    token_usage: TokenUsage = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    metadata: dict[str, Any] | None = None
    # Deprecated — derived from metadata; excluded from repr/eq so the
    # deprecation warning on read (see the property attached below the class)
    # doesn't fire on every repr()/comparison.
    gemini_safety_ratings: dict[str, str] | None = field(default=None, repr=False, compare=False)
    exception: Exception | None = field(default=None, compare=False)
    admission_wait_seconds: float = 0.0
    timing: WorkItemTiming = field(default_factory=WorkItemTiming)
    submission_index: int | None = field(default=None, compare=False)
    error_category: str | None = None
    replayed_from_artifact: bool = False

    def __post_init__(self):
        """Backfill ``gemini_safety_ratings`` from ``metadata['safety_ratings']``
        for backward compatibility. Once ``gemini_safety_ratings`` is removed,
        this method goes away. (Reads/writes go through ``__dict__`` directly
        so the framework itself never triggers the deprecation warning.)
        """
        if (
            self.__dict__.get("gemini_safety_ratings") is None
            and self.metadata is not None
            and "safety_ratings" in self.metadata
        ):
            ratings = self.metadata["safety_ratings"]
            if isinstance(ratings, dict):
                self.__dict__["gemini_safety_ratings"] = ratings

    def to_dict(self, *, encoder: "ValueEncoder | None" = None) -> dict[str, Any]:
        """Return a versioned, JSON-safe representation of this result.

        ``encoder`` may convert application-specific output/context values to
        supported JSON-safe values. Unsupported values raise
        :class:`~async_batch_llm.ResultSerializationError`.
        """
        from .serialization import work_item_result_to_dict

        return work_item_result_to_dict(self, encoder=encoder)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        output_decoder: "ValueDecoder | None" = None,
        context_decoder: "ValueDecoder | None" = None,
    ) -> "WorkItemResult[Any, Any]":
        """Restore a result from :meth:`to_dict` output.

        Without decoders, custom values (including Pydantic models and
        dataclasses) are restored as JSON-native mappings/lists.
        """
        from .serialization import work_item_result_from_dict

        return work_item_result_from_dict(
            data,
            output_decoder=output_decoder,
            context_decoder=context_decoder,
        )


def _get_gemini_safety_ratings(self: "WorkItemResult") -> dict[str, str] | None:
    # dataclasses.replace()/asdict()/astuple() read every field via getattr;
    # copying a result is not a use of the deprecated field, so don't warn on
    # reads coming from the dataclasses module itself.
    if sys._getframe(1).f_globals.get("__name__") != "dataclasses":
        warnings.warn(
            "WorkItemResult.gemini_safety_ratings is deprecated and will be removed "
            "in a future release; read result.metadata['safety_ratings'] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return self.__dict__.get("gemini_safety_ratings")


def _set_gemini_safety_ratings(self: "WorkItemResult", value: dict[str, str] | None) -> None:
    self.__dict__["gemini_safety_ratings"] = value


# Replace the plain dataclass attribute with a property AFTER the decorator
# has generated __init__: reads emit a DeprecationWarning, writes (including
# the generated __init__'s assignment and the __post_init__ backfill) go
# straight to the instance __dict__.
WorkItemResult.gemini_safety_ratings = property(  # type: ignore[assignment,method-assign]  # ty:ignore[invalid-assignment]
    _get_gemini_safety_ratings, _set_gemini_safety_ratings
)


class CachedTokenRates:
    """Named cached-token price rates for the providers we support.

    Each constant is the **fraction of the normal input-token price** you
    pay for tokens served from cache (so 0.10 = "10% of normal" =
    "90% discount"). Pass these to
    :meth:`BatchResult.effective_input_tokens` to compute provider-aware
    billable token estimates:

    .. code-block:: python

        result.effective_input_tokens(CachedTokenRates.OPENAI)

    Rates are accurate as of v0.9.0 (early 2026); confirm with each
    provider's pricing page before using these for invoicing.

    Notes:
        - **Anthropic prompt caching** is asymmetric: cache *reads* are at
          ``ANTHROPIC_READ`` (10% of normal), but cache *writes* are billed
          at a 25% premium over normal input price. This helper covers
          read-side savings only; the write premium is a separate cost
          best computed at billing time from your usage logs.
        - **OpenRouter** routes to many upstream providers, each with its
          own rate. Pick the constant for the upstream that actually
          served your request (visible in ``LLMResponse.metadata['provider']``).
    """

    GEMINI: float = 0.10
    """Gemini context cache: cached tokens cost 10% of normal."""

    OPENAI: float = 0.50
    """OpenAI prompt caching: cached tokens cost 50% of normal (chat completions)."""

    ANTHROPIC_READ: float = 0.10
    """Anthropic prompt cache reads: 10% of normal (cache writes are
    billed at 1.25× normal — not modeled here)."""

    DEEPSEEK: float = 0.02
    """DeepSeek context cache: cached tokens cost ~2% of normal (V4 Flash
    bills cache hits at $0.0028/M vs $0.14/M cache-miss input, a price drop
    that took effect April 2026; earlier the discount was ~10%)."""


class _CallableRate(float):
    """Transitional return type for ``BatchResult.cache_hit_rate``.

    Behaves as a plain float in every numeric/format context; calling it
    (the v0.18 method spelling) still works but warns. Remove together
    with the deprecation in the next major release.
    """

    def __call__(self) -> float:
        warnings.warn(
            "BatchResult.cache_hit_rate is now a property; drop the "
            "parentheses (batch.cache_hit_rate). The callable form will be "
            "removed in the next major release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return float(self)


@dataclass(frozen=True)
class BatchTermination:
    """Serializable reason a batch stopped accepting or executing work."""

    kind: Literal["completed", "batch_timeout", "fail_fast", "artifact_error"] = "completed"
    reason: str | None = None
    error_category: str | None = None
    triggering_item_id: str | None = None


@dataclass
class BatchResult(Generic[TOutput, TContext]):
    """
    Result of processing a batch of work items.

    Attributes:
        results: Individual work item results, in **completion order** — the
            order items finished, which (with parallel workers, retries, and
            rate-limit cooldowns) is generally NOT the order they were added.
            Use :meth:`by_id` to look results up by ``item_id``.
        total_items: Total number of items in the batch
        succeeded: Number of successful items
        failed: Number of failed items
        total_input_tokens: Sum of input tokens across all items
        total_output_tokens: Sum of output tokens across all items
        total_cached_tokens: Sum of cached input tokens across all items (v0.2.0)
    """

    results: list[WorkItemResult[TOutput, TContext]]
    # Derived fields — always recomputed from `results` in __post_init__.
    # init=False so the constructor signature doesn't advertise arguments
    # that would be silently discarded.
    total_items: int = field(init=False, default=0)
    succeeded: int = field(init=False, default=0)
    failed: int = field(init=False, default=0)
    total_input_tokens: int = field(init=False, default=0)
    total_output_tokens: int = field(init=False, default=0)
    total_cached_tokens: int = field(init=False, default=0)  # v0.2.0
    termination: BatchTermination = field(default_factory=BatchTermination)
    # Wall-clock duration of the run that produced this batch, stamped by the
    # execution surfaces (process_all / process_prompts). None for batches
    # assembled by hand or restored from records created before v0.20.
    wall_time_seconds: float | None = None

    def __post_init__(self):
        """Calculate summary statistics from results."""
        self.total_items = len(self.results)
        self.succeeded = sum(1 for r in self.results if r.success)
        self.failed = sum(1 for r in self.results if not r.success)
        self.total_input_tokens = sum(r.token_usage.get("input_tokens", 0) for r in self.results)
        self.total_output_tokens = sum(r.token_usage.get("output_tokens", 0) for r in self.results)
        # v0.2.0: Aggregate cached tokens
        self.total_cached_tokens = sum(
            r.token_usage.get("cached_input_tokens", 0) for r in self.results
        )

    @property
    def cache_hit_rate(self) -> float:
        """Percentage (0.0–100.0) of input tokens served from cache.

        A property since v0.20.0 (issue #93), consistent with the sibling
        zero-arg scalars (``total_cached_tokens``, ``succeeded``, ...).
        The v0.18 method spelling ``batch.cache_hit_rate()`` still works
        via a transitional callable-float return value and emits a
        ``DeprecationWarning``; drop the parentheses.
        """
        if self.total_input_tokens == 0:
            return _CallableRate(0.0)
        return _CallableRate((self.total_cached_tokens / self.total_input_tokens) * 100.0)

    def effective_input_tokens(self, cached_token_rate: float | None = None) -> int:
        """
        Estimate billable input tokens after the cache discount.

        ``cached_token_rate`` is the fraction of the normal input-token price
        you pay for tokens served from cache. For example, Gemini charges
        10% of the normal price (rate = 0.10), so 1000 cached tokens cost
        the same as 100 uncached tokens.

        Use the named constants on :class:`CachedTokenRates` to avoid
        hardcoding magic numbers:

        .. code-block:: python

            result.effective_input_tokens(CachedTokenRates.OPENAI)
            result.effective_input_tokens(CachedTokenRates.GEMINI)

        Args:
            cached_token_rate: Fraction (0.0–1.0) of the normal input price
                paid for cached tokens. When omitted (``None``) it defaults to
                ``CachedTokenRates.GEMINI`` (0.10) for backward compatibility —
                pre-v0.9.0 versions hardcoded this value. **Pass an explicit
                rate when working with non-Gemini providers** to get accurate
                numbers; relying on the implicit default while cached tokens
                are present emits a ``UserWarning``, since the Gemini rate is
                wrong for e.g. OpenAI (~0.50).

        Returns:
            Effective input tokens billed. The discount is computed by
            truncating ``cached_tokens * (1 - rate)`` toward zero with
            ``int()``, which means the returned billable estimate is
            rounded **up** when the discount would have a fractional
            part — a deliberately conservative choice for cost reporting
            (your real bill is at most this number, never more).

        Raises:
            ValueError: If ``cached_token_rate`` is not in [0.0, 1.0].
        """
        if cached_token_rate is None:
            # Implicit default. Only nudge when it actually changes the answer
            # (i.e. there are cached tokens to discount) — silent for the common
            # no-cache case so we don't cry wolf.
            if self.total_cached_tokens > 0:
                import warnings

                warnings.warn(
                    "effective_input_tokens() called without an explicit "
                    "cached_token_rate; defaulting to the Gemini rate "
                    "(CachedTokenRates.GEMINI = 0.10). This is wrong for other "
                    "providers (OpenAI is ~0.50). Pass an explicit "
                    "CachedTokenRates constant to silence this warning.",
                    UserWarning,
                    stacklevel=2,
                )
            cached_token_rate = CachedTokenRates.GEMINI

        if not 0.0 <= cached_token_rate <= 1.0:
            raise ValueError(
                f"cached_token_rate must be in [0.0, 1.0]; got {cached_token_rate}. "
                f"This is the fraction of normal price paid for cached tokens "
                f"(0.0 = free, 1.0 = no discount). For named provider rates, "
                f"use CachedTokenRates."
            )
        # cached_token_rate is what you PAY; (1 - rate) is the discount.
        # int() floors the discount toward zero -> conservative (over-)estimate
        # of effective billable tokens. See the Returns docstring.
        discount = int(self.total_cached_tokens * (1.0 - cached_token_rate))
        return self.total_input_tokens - discount

    @property
    def successes(self) -> list[WorkItemResult[TOutput, TContext]]:
        """The successful results only, in completion order."""
        return [r for r in self.results if r.success]

    @property
    def failures(self) -> list[WorkItemResult[TOutput, TContext]]:
        """The failed results only, in completion order."""
        return [r for r in self.results if not r.success]

    def by_id(self) -> dict[str, WorkItemResult[TOutput, TContext]]:
        """Map ``item_id`` -> result for direct lookup.

        Results are ordered by completion, so use this when you need to align
        outputs back to specific inputs. If two results somehow share an
        ``item_id``, the later-completed one wins.
        """
        return {r.item_id: r for r in self.results}

    @overload
    def outputs(self) -> Iterator[TOutput]: ...

    @overload
    def outputs(self, *, with_ids: Literal[True]) -> Iterator[tuple[str, TOutput]]: ...

    def outputs(
        self, *, with_ids: bool = False
    ) -> Iterator[TOutput] | Iterator[tuple[str, TOutput]]:
        """Iterate over the outputs of successful results, in result order.

        The happy-path accessor — no ``if item.success`` loop needed:

        .. code-block:: python

            for output in batch.outputs():
                print(output)
            for item_id, output in batch.outputs(with_ids=True):
                print(item_id, output)

        Args:
            with_ids: When True, yield ``(item_id, output)`` pairs instead of
                bare outputs.

        Added in v0.20.0. Failed results are skipped; iterate ``.failures``
        (or check ``batch.failed``) to handle them.
        """
        if with_ids:
            return ((r.item_id, cast(TOutput, r.output)) for r in self.results if r.success)
        return (cast(TOutput, r.output) for r in self.results if r.success)

    def summary(self) -> str:
        """Return a printable plain-text report of the whole run.

        ``print(batch.summary())`` is a complete post-run report: item
        counts, termination, retry totals, token totals (with replayed work
        accounted separately, consistent with v0.18 replay accounting),
        admission/execution percentiles, wall time, and failures grouped by
        error category. Works identically on collected batches and on
        batches restored via ``from_dict``/``from_json``/``from_jsonl``.

        Added in v0.20.0.
        """
        current = [r for r in self.results if not r.replayed_from_artifact]
        replayed = [r for r in self.results if r.replayed_from_artifact]

        def _tokens(results: list[WorkItemResult[TOutput, TContext]]) -> tuple[int, int, int]:
            return (
                sum(r.token_usage.get("input_tokens", 0) for r in results),
                sum(r.token_usage.get("cached_input_tokens", 0) for r in results),
                sum(r.token_usage.get("output_tokens", 0) for r in results),
            )

        def _fmt_seconds(seconds: float) -> str:
            if seconds >= 100:
                return f"{seconds:,.0f}s"
            if seconds >= 10:
                return f"{seconds:.1f}s"
            return f"{seconds:.2f}s"

        def _percentile_line(label: str, values: list[float]) -> str:
            ordered = sorted(values)
            picks = []
            for q in (0.50, 0.95, 0.99):
                # Nearest-rank percentile on the sorted sample.
                index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
                picks.append(_fmt_seconds(ordered[index]))
            return f"  {label:<15} p50 {picks[0]}  p95 {picks[1]}  p99 {picks[2]}"

        lines = ["Batch summary", "============="]

        replay_note = f" ({len(replayed)} replayed from artifact)" if replayed else ""
        lines.append(
            f"Items:     {self.total_items} total — {self.succeeded} succeeded, "
            f"{self.failed} failed{replay_note}"
        )

        termination = self.termination.kind
        if self.termination.kind != "completed":
            details = [
                part
                for part in (
                    self.termination.reason,
                    f"category={self.termination.error_category}"
                    if self.termination.error_category
                    else None,
                    f"item={self.termination.triggering_item_id}"
                    if self.termination.triggering_item_id
                    else None,
                )
                if part
            ]
            if details:
                termination += f" — {'; '.join(details)}"
        lines.append(f"Stopped:   {termination}")

        extra_tries = sum(max(0, len(r.timing.attempts) - 1) for r in current)
        retried_items = sum(1 for r in current if len(r.timing.attempts) > 1)
        lines.append(f"Retries:   {extra_tries} extra attempt(s) across {retried_items} item(s)")

        input_tokens, cached_tokens, output_tokens = _tokens(current)
        lines.append(
            f"Tokens:    in {input_tokens:,} (cached {cached_tokens:,}) · out {output_tokens:,}"
        )
        if replayed:
            r_in, r_cached, r_out = _tokens(replayed)
            lines.append(
                f"Replayed:  in {r_in:,} (cached {r_cached:,}) · out {r_out:,} "
                "(prior run; excluded above)"
            )

        wall = _fmt_seconds(self.wall_time_seconds) if self.wall_time_seconds is not None else "n/a"
        lines.append(f"Wall time: {wall}")

        timed = [r for r in current if r.timing.attempts]
        if timed:
            lines.append(
                _percentile_line("admission wait", [r.admission_wait_seconds for r in timed])
            )
            lines.append(_percentile_line("execution", [r.timing.execution_seconds for r in timed]))

        if self.failed:
            lines.append("Failures by category:")
            by_category: dict[str, list[str]] = {}
            for r in self.results:
                if not r.success:
                    by_category.setdefault(r.error_category or "uncategorized", []).append(
                        r.item_id
                    )
            for category in sorted(by_category):
                ids = by_category[category]
                shown = ", ".join(ids[:3]) + ("..." if len(ids) > 3 else "")
                lines.append(f"  {category:<15} {len(ids)} — {shown}")

        return "\n".join(lines)

    def in_input_order(self) -> "BatchResult[TOutput, TContext]":
        """Return a new batch whose results are sorted by submission order.

        The current batch and its result list are not mutated. Ordering is
        never inferred from ``item_id`` because IDs may be duplicated.
        """
        missing = [result.item_id for result in self.results if result.submission_index is None]
        if missing:
            preview = ", ".join(repr(item_id) for item_id in missing[:3])
            suffix = "..." if len(missing) > 3 else ""
            raise ValueError(
                "Cannot order results by input: "
                f"{len(missing)} result(s) lack submission_index ({preview}{suffix})."
            )
        ordered = sorted(self.results, key=lambda result: cast(int, result.submission_index))
        return cast(
            "BatchResult[TOutput, TContext]",
            BatchResult(
                results=ordered,
                termination=self.termination,
                wall_time_seconds=self.wall_time_seconds,
            ),
        )

    def to_dict(self, *, encoder: "ValueEncoder | None" = None) -> dict[str, Any]:
        """Return a versioned, JSON-safe representation of the batch."""
        from .serialization import batch_result_to_dict

        return batch_result_to_dict(self, encoder=encoder)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        output_decoder: "ValueDecoder | None" = None,
        context_decoder: "ValueDecoder | None" = None,
    ) -> "BatchResult[Any, Any]":
        """Restore a batch from :meth:`to_dict` output."""
        from .serialization import batch_result_from_dict

        return batch_result_from_dict(
            data,
            output_decoder=output_decoder,
            context_decoder=context_decoder,
        )

    def to_json(self, *, encoder: "ValueEncoder | None" = None, indent: int | None = 2) -> str:
        """Serialize this batch to a JSON string."""
        from .serialization import batch_result_to_json

        return batch_result_to_json(self, encoder=encoder, indent=indent)

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
        *,
        output_decoder: "ValueDecoder | None" = None,
        context_decoder: "ValueDecoder | None" = None,
    ) -> "BatchResult[Any, Any]":
        """Restore a batch from a JSON string or UTF-8 bytes."""
        from .serialization import batch_result_from_json

        return batch_result_from_json(
            value,
            output_decoder=output_decoder,
            context_decoder=context_decoder,
        )

    def to_jsonl(self, path: "str | Path", *, encoder: "ValueEncoder | None" = None) -> None:
        """Write one versioned result record per UTF-8 JSONL line."""
        from .serialization import batch_result_to_jsonl

        batch_result_to_jsonl(self, path, encoder=encoder)

    @classmethod
    def from_jsonl(
        cls,
        path: "str | Path",
        *,
        output_decoder: "ValueDecoder | None" = None,
        context_decoder: "ValueDecoder | None" = None,
    ) -> "BatchResult[Any, Any]":
        """Restore a batch from :meth:`to_jsonl` output."""
        from .serialization import batch_result_from_jsonl

        return batch_result_from_jsonl(
            path,
            output_decoder=output_decoder,
            context_decoder=context_decoder,
        )

    def estimated_cost(
        self,
        input_per_mtok: float,
        output_per_mtok: float,
        cached_token_rate: float | None = None,
    ) -> float:
        """Estimate total spend from per-million-token prices.

        Applies the cache discount to input tokens via
        :meth:`effective_input_tokens`, so cached tokens are billed at their
        reduced rate.

        Args:
            input_per_mtok: Price per 1,000,000 input tokens (in your currency).
            output_per_mtok: Price per 1,000,000 output tokens.
            cached_token_rate: Fraction of the normal input price paid for
                cached tokens (see :class:`CachedTokenRates`). When ``None`` it
                defaults to the Gemini rate and emits a ``UserWarning`` if cached
                tokens are present — pass an explicit rate for other providers.

        Returns:
            Estimated total cost: ``effective_input / 1e6 * input_per_mtok +
            output / 1e6 * output_per_mtok``.
        """
        billable_input = self.effective_input_tokens(cached_token_rate)
        input_cost = billable_input / 1_000_000 * input_per_mtok
        output_cost = self.total_output_tokens / 1_000_000 * output_per_mtok
        return input_cost + output_cost


def _unpack_strategy_result(
    result: Any,
) -> tuple[Any, "TokenUsage", dict[str, Any] | None]:
    """Compat-shim for the LLMCallStrategy.execute() return contract.

    Strategies may return either:

    - ``(output, token_usage)`` — legacy 2-tuple shape (pre-v0.10.0).
    - ``(output, token_usage, metadata)`` — current 3-tuple shape; ``metadata``
      is forwarded into ``WorkItemResult.metadata``.

    Returns the normalized 3-tuple. Raises ``ValueError`` for any other shape
    (clearer than letting Python's tuple-unpacking error reach the caller).

    The 2-tuple path will be removed in a future release; custom strategies
    should migrate to the 3-tuple shape.
    """
    if not isinstance(result, tuple):
        raise ValueError(
            f"Strategy.execute() must return a tuple of (output, tokens) or "
            f"(output, tokens, metadata); got {type(result).__name__}."
        )
    if len(result) == 3:
        output, tokens, metadata = result
        return output, tokens, metadata
    if len(result) == 2:
        output, tokens = result
        return output, tokens, None
    raise ValueError(
        f"Strategy.execute() must return a 2- or 3-tuple "
        f"(output, tokens [, metadata]); got a tuple of length {len(result)}."
    )


# Type alias for post-processor function
PostProcessorFunc = Callable[[WorkItemResult[TOutput, TContext]], Awaitable[None] | None]

# Type alias for progress callback function (completed, total, current_item_id)
ProgressCallbackFunc = Callable[[int, int, str], Awaitable[None] | None]


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""

    total: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    start_time: float | None = None
    error_counts: dict[str, int] = field(default_factory=dict)
    rate_limit_count: int = 0
    # Token usage tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0  # Tokens served from provider-side caches
    total_admission_wait_seconds: float = 0.0
    max_admission_wait_seconds: float = 0.0
    _admission_wait_samples: list[float] = field(default_factory=list, repr=False)
    _execution_samples: list[float] = field(default_factory=list, repr=False)
    structured_output_recoveries: int = 0
    structured_output_retries_avoided: int = 0
    structured_output_recovery_reasons: dict[str, int] = field(default_factory=dict)
    replayed: int = 0
    aborted: int = 0
    input_queue_high_water_mark: int = 0
    result_queue_high_water_mark: int = 0

    def copy(self) -> dict[str, Any]:
        """Return a dictionary copy of the stats for backwards compatibility."""
        return {
            "total": self.total,
            "processed": self.processed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "start_time": self.start_time,
            "error_counts": self.error_counts.copy(),
            "rate_limit_count": self.rate_limit_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "total_admission_wait_seconds": self.total_admission_wait_seconds,
            "max_admission_wait_seconds": self.max_admission_wait_seconds,
            "structured_output_recoveries": self.structured_output_recoveries,
            "structured_output_retries_avoided": self.structured_output_retries_avoided,
            "structured_output_recovery_reasons": self.structured_output_recovery_reasons.copy(),
            "replayed": self.replayed,
            "aborted": self.aborted,
            "input_queue_high_water_mark": self.input_queue_high_water_mark,
            "result_queue_high_water_mark": self.result_queue_high_water_mark,
            "admission_wait_p50_seconds": _percentile(self._admission_wait_samples, 50),
            "admission_wait_p95_seconds": _percentile(self._admission_wait_samples, 95),
            "admission_wait_p99_seconds": _percentile(self._admission_wait_samples, 99),
            "execution_p50_seconds": _percentile(self._execution_samples, 50),
            "execution_p95_seconds": _percentile(self._execution_samples, 95),
            "execution_p99_seconds": _percentile(self._execution_samples, 99),
            # Alias kept for backward compatibility: the duplicate field was
            # never incremented, so it always read 0. Both keys now come from
            # the single counter that is actually updated.
            "total_cached_tokens": self.cached_input_tokens,
        }

    def record_timing(self, timing: WorkItemTiming, *, sample_limit: int = 10_000) -> None:
        """Record bounded attempt samples for percentile summaries."""
        for attempt in timing.attempts:
            self._admission_wait_samples.append(attempt.admission_wait_seconds)
            self._execution_samples.append(attempt.execution_seconds)
        if len(self._admission_wait_samples) > sample_limit:
            del self._admission_wait_samples[:-sample_limit]
        if len(self._execution_samples) > sample_limit:
            del self._execution_samples[:-sample_limit]


def _percentile(samples: list[float], percentile: int) -> float:
    """Return a nearest-rank percentile for a bounded sample list."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, (len(ordered) * percentile + 99) // 100 - 1))
    return ordered[index]


class _EndOfStream:
    """Sentinel pushed onto the result stream once all work is processed."""


class _WorkerCrashed:
    """Sentinel carrying an unexpected worker exception for ``results()`` to re-raise.

    A worker that finishes normally returns ``None``; one that dies from a
    framework bug would otherwise hang the consumer forever, so a done-callback
    wraps the exception in this sentinel and pushes it onto the result stream.
    """

    __slots__ = ("exception",)

    def __init__(self, exception: BaseException) -> None:
        self.exception = exception


# Singleton end-of-stream marker.
_END_OF_STREAM = _EndOfStream()


class _TrackedWorkQueue(asyncio.Queue[Any]):
    """Queue that observes exact data-item ownership without sampling races."""

    def __init__(self, maxsize: int, observer: Callable[[int], None]) -> None:
        super().__init__(maxsize=maxsize)
        self._observer = observer
        self._data_count = 0

    def _put(self, item: Any) -> None:
        super()._put(item)
        if item is not None:
            self._data_count += 1
            self._observer(self._data_count)

    def _get(self) -> Any:
        item = super()._get()
        if item is not None:
            self._data_count -= 1
        return item

    @property
    def data_count(self) -> int:
        return self._data_count


class BatchProcessor(ABC, Generic[TInput, TOutput, TContext]):
    """
    Abstract base class for batch LLM processing strategies.

    Subclasses implement different strategies for processing batches:
    - ParallelBatchProcessor: Process items in parallel as individual requests
    - BatchAPIProcessor: Use Google's true batch API (future)

    Two usage modes:

    - **Batch** (:meth:`process_all`): add all work, then process to a
      ``BatchResult``. One-shot; ``add_work`` is rejected once it starts.
    - **Streaming** (:meth:`start` / :meth:`add_work` / :meth:`finish` /
      :meth:`results`): workers run *while* work is still being added, so a
      bounded ``max_queue_size`` becomes backpressure — letting you stream
      arbitrarily large inputs through constant memory — instead of a deadlock.
      Results are yielded in completion order.
    """

    def __init__(
        self,
        max_workers: int = 5,
        post_processor: PostProcessorFunc[TOutput, TContext] | None = None,
        max_queue_size: int = 0,
        progress_callback: ProgressCallbackFunc | None = None,
        progress_callback_timeout: float | None = None,
        max_result_queue_size: int = 0,
    ):
        """
        Initialize the batch processor.

        Args:
            max_workers: Maximum number of concurrent workers
            post_processor: Optional async function called after each successful item
            max_queue_size: Maximum queue size (0 = unlimited)
            progress_callback: Optional callback(completed, total, current_item_id) for progress updates
            progress_callback_timeout: Maximum seconds to wait for progress callback (None = no limit)
            max_result_queue_size: Completed results waiting for a streaming
                consumer (0 = unlimited)
        """
        self.max_workers = max_workers
        self.post_processor = post_processor
        self._post_processor_is_async = False
        if post_processor is not None:
            self._post_processor_is_async = inspect.iscoroutinefunction(post_processor) or (
                callable(post_processor) and inspect.iscoroutinefunction(post_processor.__call__)  # type: ignore[operator]
            )
        self.max_queue_size = max_queue_size
        self.max_result_queue_size = max_result_queue_size
        self.progress_callback = progress_callback
        self.progress_callback_timeout = progress_callback_timeout
        # The bundled reporter opts into a cheap inline path. This marker is
        # private so user callbacks retain the established per-item task,
        # timeout, and asyncio.to_thread behavior.
        self._progress_callback_inline = bool(
            getattr(progress_callback, "_abl_inline_progress", False)
        )
        self._progress_callback_is_async = False
        if progress_callback is not None:
            self._progress_callback_is_async = inspect.iscoroutinefunction(progress_callback) or (
                callable(progress_callback)
                and inspect.iscoroutinefunction(progress_callback.__call__)  # type: ignore[operator]
            )
        self._stats = ProcessingStats()
        self._queue: asyncio.Queue[LLMWorkItem[TInput, TOutput, TContext] | None] = (
            _TrackedWorkQueue(max_queue_size, self._observe_input_queue_size)
        )
        self._results: list[WorkItemResult[TOutput, TContext]] = []
        self._workers: list[asyncio.Task] = []
        self._is_processing = False
        self._progress_tasks: set[asyncio.Task[Any]] = set()

        # Thread-safety locks (shared by both modes; subclasses use these).
        self._stats_lock = asyncio.Lock()
        self._results_lock = asyncio.Lock()
        self._submission_lock = asyncio.Lock()
        self._next_submission_index = 0

        # Background post-processor tasks (only used when
        # ProcessorConfig.concurrent_post_processing is True). The semaphore is
        # created per-run in process_all()/start() so it binds to the active
        # event loop and is sized to max_workers.
        self._post_processor_tasks: set[asyncio.Task[Any]] = set()
        self._post_processor_semaphore: asyncio.Semaphore | None = None

        self._processing_started = False  # Prevent add_work() after process_all() starts

        # Streaming-mode state (see start()/finish()/results()).
        self._streaming = False
        self._finished = False
        self._result_stream: (
            asyncio.Queue[WorkItemResult[TOutput, TContext] | _EndOfStream | _WorkerCrashed] | None
        ) = None
        # The mixed queue stays unbounded so control-plane messages are always
        # deliverable. In streaming mode this optional semaphore bounds only
        # queued WorkItemResult objects.
        self._result_slots: asyncio.BoundedSemaphore | None = None
        self._current_queued_results = 0
        self._reported_worker_crash_task_ids: set[int] = set()
        self._finalize_task: asyncio.Task[None] | None = None
        self.termination = BatchTermination()

    async def __aenter__(self):
        """Context manager entry - returns self for use in async with."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup of resources."""
        await self.cleanup()
        return False  # Don't suppress exceptions

    async def cleanup(self):
        """
        Clean up resources: cancel pending workers and clear queue.

        This method should be called when you're done with the processor,
        or use the processor as an async context manager.
        """
        # Cancel the streaming finalize task (may be blocked on queue.join()
        # if a worker crashed) before tearing down the workers it awaits.
        if self._finalize_task is not None and not self._finalize_task.done():
            self._finalize_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._finalize_task

        # Cancel any running workers
        if self._workers:
            logger.debug(f"Cleaning up {len(self._workers)} workers")
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()

            # Wait briefly for cancellations
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=WORKER_CANCELLATION_TIMEOUT,
                )
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning("Some workers did not cancel within timeout")

        # Results abandoned by an early consumer own one permit each. Release
        # those permits after publishers are stopped; control messages own none.
        self._drain_result_stream()

        # Clear any remaining items in queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Drain pending progress callbacks. The final item's callback is
        # dispatched just before its result is published, so a consumer that
        # stops right after the last result routinely reaches cleanup while
        # that callback is still running — give callbacks a bounded chance to
        # finish (previously they were cancelled outright, which
        # silently dropped the last progress update) and cancel only
        # stragglers.
        if self._progress_tasks:
            pending = list(self._progress_tasks)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=PROGRESS_TASK_CANCELLATION_TIMEOUT,
                )
            except (TimeoutError, asyncio.TimeoutError):
                for task in pending:
                    if not task.done():
                        task.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=PROGRESS_TASK_CANCELLATION_TIMEOUT,
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    logger.warning("[WARN]Some progress callbacks did not cancel in time")
            finally:
                self._progress_tasks.clear()

        # Cancel any outstanding background post-processor tasks (concurrent mode)
        if self._post_processor_tasks:
            for task in list(self._post_processor_tasks):
                task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._post_processor_tasks, return_exceptions=True),
                    timeout=PROGRESS_TASK_CANCELLATION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("[WARN]Some post-processors did not cancel in time")
            finally:
                self._post_processor_tasks.clear()

    async def add_work(self, work_item: LLMWorkItem[TInput, TOutput, TContext]):
        """
        Add a work item to the processing queue.

        In **streaming mode** (after :meth:`start`) workers are already running,
        so this may be called freely while processing; a bounded
        ``max_queue_size`` then provides **backpressure** (``put`` awaits a free
        slot) rather than deadlocking. :meth:`finish` is the streaming boundary
        equivalent of the batch mode's "no add_work after process_all()" rule.

        In **batch mode** (used with :meth:`process_all`) add all work *before*
        processing starts.

        Args:
            work_item: Work item to process

        Raises:
            RuntimeError: In streaming mode, if called after :meth:`finish`. In
                batch mode, if called after :meth:`process_all` has started.
            ValueError: In batch mode, if a bounded queue (``max_queue_size > 0``)
                fills up — batch-mode workers don't start until ``process_all()``,
                so the queue can't drain while you add, and blocking would
                deadlock. Use streaming mode for bounded queues.
        """
        # Serialize acceptance so the index reflects stable admission order,
        # including when multiple producers call add_work concurrently.
        async with self._submission_lock:
            if work_item.submission_index is not None:
                raise ValueError(
                    "The same LLMWorkItem instance cannot be submitted more than once. "
                    "Create a new LLMWorkItem for each accepted submission."
                )
            work_item.submission_index = self._next_submission_index
            if self._streaming:
                if self._finished:
                    work_item.submission_index = None
                    raise RuntimeError(
                        "Cannot add work after finish() — the streaming batch is closed. "
                        "Create a new processor for additional work."
                    )
                # Workers are running and draining the queue, so a bounded queue
                # safely applies backpressure here instead of deadlocking.
                try:
                    await self._queue.put(work_item)
                except BaseException:
                    work_item.submission_index = None
                    raise
                self._next_submission_index += 1
                # Do not introduce a cancellation point after queue.put()
                # commits ownership. Otherwise an abort can report rejection
                # even though a worker already owns the accepted item.
                self._stats.total += 1
                return

            if self._processing_started:
                work_item.submission_index = None
                raise RuntimeError(
                    "Cannot add work after process_all() has started. "
                    "Create a new processor instance for additional batches."
                )

            # Batch mode: workers don't exist yet, so a full bounded queue can't
            # drain — blocking would hang forever. Point users at streaming mode.
            try:
                self._queue.put_nowait(work_item)
            except asyncio.QueueFull:
                work_item.submission_index = None
                raise ValueError(
                    f"Work queue is full (max_queue_size={self.max_queue_size}) in batch mode. "
                    "Bounded queues require streaming mode: start()/add_work()/finish() (or the "
                    "high-level process_stream/process_prompts), where running workers drain the "
                    "queue so it becomes backpressure. For batch mode (process_all), set "
                    "max_queue_size=0 (unlimited) or size it to fit the whole batch."
                ) from None
            self._next_submission_index += 1
            self._stats.total += 1

    def _observe_input_queue_size(self, current_size: int) -> None:
        """Record the exact post-enqueue data-item count without awaiting."""
        self._stats.input_queue_high_water_mark = max(
            self._stats.input_queue_high_water_mark,
            current_size,
        )

    # ── Streaming mode ───────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn workers immediately and enter streaming mode. Idempotent.

        Unlike :meth:`process_all`, this lets you :meth:`add_work` while workers
        run, so a bounded ``max_queue_size`` becomes backpressure (stream
        arbitrarily large inputs in constant memory) instead of a deadlock.
        Consume results via :meth:`results`; call :meth:`finish` when no more
        work will be added.
        """
        if self._workers:
            return  # already started (streaming or batch)

        self._streaming = True
        self._processing_started = True
        self._is_processing = True
        self._finished = False
        self.termination = BatchTermination()
        self._result_stream = asyncio.Queue()
        self._result_slots = (
            asyncio.BoundedSemaphore(self.max_result_queue_size)
            if self.max_result_queue_size > 0
            else None
        )
        self._reported_worker_crash_task_ids = set()
        # Count any work added before start(); add_work() increments thereafter.
        self._results = []
        self._stats = ProcessingStats(total=self._queue.qsize())
        tracked_queue = cast(_TrackedWorkQueue, self._queue)
        self._stats.input_queue_high_water_mark = tracked_queue.data_count
        self._stats.start_time = time.time()
        self._current_queued_results = 0
        self._post_processor_tasks = set()
        self._post_processor_semaphore = asyncio.Semaphore(self.max_workers)

        self._workers = [
            asyncio.create_task(self._worker(worker_id)) for worker_id in range(self.max_workers)
        ]
        # Safety net: a worker dying from a framework bug must not hang the
        # consumer — surface it on the result stream (see _on_worker_done).
        for worker in self._workers:
            worker.add_done_callback(self._on_worker_done)

    async def finish(self) -> None:
        """Signal that no more work will be added; :meth:`results` ends once the
        queue drains and workers stop. Idempotent."""
        if not self._streaming:
            raise RuntimeError("finish() is only valid in streaming mode (call start() first).")
        if self._finished:
            return
        self._finished = True
        self._finalize_task = asyncio.create_task(self._finalize_stream())

    async def _finalize_stream(self) -> None:
        """Drain the queue, stop workers, then close the result stream."""
        try:
            await self._queue.join()  # all queued items processed
            for _ in range(self.max_workers):  # release workers
                await self._queue.put(None)
            await asyncio.gather(*self._workers, return_exceptions=True)
            # Do not rely solely on task done-callback scheduling: report any
            # worker failure before end-of-stream is enqueued.
            for worker in self._workers:
                self._report_worker_crash(worker)
            # Let any concurrent (background) post-processors finish too.
            if self._post_processor_tasks:
                await asyncio.gather(*self._post_processor_tasks, return_exceptions=True)
            await self._on_batch_completed()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            # Hook/finalization failures (including artifact close errors) must
            # reach the stream consumer rather than becoming an unobserved task
            # exception followed by an apparently normal end-of-stream.
            if self._result_stream is not None:
                self._publish_control(_WorkerCrashed(exc))
        finally:
            # Always close the stream so results() terminates, even on error.
            self._is_processing = False
            if self._result_stream is not None:
                self._publish_control(_END_OF_STREAM)

    async def _publish_stream_result(self, result: WorkItemResult[TOutput, TContext]) -> None:
        """Publish one result while preserving result-slot ownership."""
        if self._result_stream is None:
            raise RuntimeError("Result publication requires streaming mode")
        slots = self._result_slots
        acquired = False
        if slots is not None:
            await slots.acquire()
            acquired = True
        try:
            # The mixed queue is deliberately unbounded, so this cannot block
            # and introduces no cancellation point after permit acquisition.
            self._result_stream.put_nowait(result)
            self._current_queued_results += 1
            self._stats.result_queue_high_water_mark = max(
                self._stats.result_queue_high_water_mark,
                self._current_queued_results,
            )
        except BaseException:
            if acquired and slots is not None:
                slots.release()
            raise

    def _publish_control(self, message: _EndOfStream | _WorkerCrashed) -> None:
        """Publish a control-plane message without consuming result capacity."""
        if self._result_stream is not None:
            self._result_stream.put_nowait(message)

    def _release_result_slot(self) -> None:
        if self._result_slots is not None:
            # BoundedSemaphore detects accidental over-release in tests and in
            # future changes to the queue ownership protocol.
            self._result_slots.release()

    def _drain_result_stream(self) -> None:
        if self._result_stream is None:
            return
        while True:
            try:
                item = self._result_stream.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(item, WorkItemResult):
                self._current_queued_results -= 1
                self._release_result_slot()

    async def results(self) -> AsyncIterator[WorkItemResult[TOutput, TContext]]:
        """Yield results in completion order until :meth:`finish` + queue drain.

        Results arrive in the order items *finish* (not the order added). If a
        worker dies unexpectedly, the exception is re-raised here.
        """
        if self._result_stream is None:
            raise RuntimeError("results() requires streaming mode (call start() first).")
        while True:
            item = await self._result_stream.get()
            if isinstance(item, WorkItemResult):
                # A result being handled by application code is outside the
                # configured queue bound. Release before yielding it.
                self._current_queued_results -= 1
                self._release_result_slot()
            if isinstance(item, _EndOfStream):
                return
            if isinstance(item, _WorkerCrashed):
                raise item.exception
            yield item

    def _on_worker_done(self, task: "asyncio.Task[Any]") -> None:
        """Done-callback: surface an unexpected worker death to the consumer.

        Workers that finish normally return ``None``; a cancelled worker is
        expected during shutdown. Anything else is a framework bug that would
        otherwise hang :meth:`results`, so push an error sentinel.
        """
        self._report_worker_crash(task)

    def _report_worker_crash(self, task: "asyncio.Task[Any]") -> None:
        """Publish one crash signal per failed worker task."""
        task_id = id(task)
        if task_id in self._reported_worker_crash_task_ids or task.cancelled():
            return
        exc = task.exception()
        if exc is not None and self._result_stream is not None:
            self._reported_worker_crash_task_ids.add(task_id)
            self._publish_control(_WorkerCrashed(exc))

    async def _on_batch_started(self) -> None:
        """Hook for subclasses to run logic when a batch starts."""
        return

    async def _on_batch_completed(self) -> None:
        """Hook for subclasses to run logic after a batch completes."""
        return

    async def process_all(self) -> BatchResult[TOutput, TContext]:
        """
        Process all work items in the queue.

        Processor instances are one-shot: after processing starts, add_work()
        raises RuntimeError. Create a new processor for additional batches.

        Returns:
            BatchResult containing all results and statistics
        """
        # Mark processing as started to prevent add_work() calls (v0.4.0)
        self._processing_started = True

        # Initialize result and stats containers for this one-shot batch.
        self._results = []
        self._stats = ProcessingStats(total=self._queue.qsize())
        tracked_queue = cast(_TrackedWorkQueue, self._queue)
        self._stats.input_queue_high_water_mark = tracked_queue.data_count
        self._current_queued_results = 0

        # Fresh post-processor concurrency state, bound to this run's loop.
        self._post_processor_tasks = set()
        self._post_processor_semaphore = asyncio.Semaphore(self.max_workers)

        # Record start time for rate calculation
        self._stats.start_time = time.time()
        self._is_processing = True

        await self._on_batch_started()

        # Start workers and store them for cleanup
        self._workers = [
            asyncio.create_task(self._worker(worker_id)) for worker_id in range(self.max_workers)
        ]

        async def wait_for_worker_failure() -> None:
            done, _ = await asyncio.wait(self._workers, return_when=asyncio.FIRST_EXCEPTION)
            for worker in done:
                if worker.cancelled():
                    raise asyncio.CancelledError
                exception = worker.exception()
                if exception is not None:
                    raise exception
                # Workers cannot return normally before queue sentinels are
                # installed below; treat that as a framework failure.
                raise RuntimeError("Batch worker stopped before the queue drained")

        # Race queue completion against an unexpected worker exit. Without
        # this, one dead worker (especially max_workers=1) can strand
        # queue.join() forever with accepted items still queued.
        queue_join = asyncio.create_task(self._queue.join())
        worker_failure = asyncio.create_task(wait_for_worker_failure())
        try:
            done, _ = await asyncio.wait(
                {queue_join, worker_failure}, return_when=asyncio.FIRST_COMPLETED
            )
            if worker_failure in done:
                await worker_failure
            await queue_join
        except BaseException:
            queue_join.cancel()
            worker_failure.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await queue_join
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker_failure
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
            await asyncio.gather(*self._workers, return_exceptions=True)
            raise
        else:
            worker_failure.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_failure
            # Unblock workers only after every accepted item called task_done().
            for _ in range(self.max_workers):
                await self._queue.put(None)

        logger.info("[OK]Queue processing complete, waiting for workers to finish...")

        # Wait for workers to finish with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers),
                timeout=WORKER_SHUTDOWN_TIMEOUT,
            )
            logger.info(f"[OK]All {len(self._workers)} workers finished successfully")
        except (TimeoutError, asyncio.TimeoutError):
            logger.error(
                f"[WARN]Workers did not finish within {WORKER_SHUTDOWN_TIMEOUT}s after queue.join(). "
                "Cancelling workers and proceeding..."
            )
            # Cancel any workers that are still running
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()
            # Wait briefly for cancellations to complete
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=WORKER_CANCELLATION_TIMEOUT * 2.5,  # Allow more time during shutdown
                )
            except (TimeoutError, asyncio.TimeoutError):
                logger.error("[WARN]Some workers could not be cancelled")

        self._is_processing = False

        # In concurrent_post_processing mode, post-processors run as background
        # tasks; wait for all of them to finish before returning so callers can
        # rely on every post-processor having completed (return_exceptions so a
        # single failure can't break the await — they log their own errors).
        if self._post_processor_tasks:
            await asyncio.gather(*self._post_processor_tasks, return_exceptions=True)

        await self._on_batch_completed()

        # Snapshot results before returning so callers receive an independent list.
        results_snapshot = list(self._results)
        wall_time = (
            time.time() - self._stats.start_time if self._stats.start_time is not None else None
        )
        return BatchResult(
            results=results_snapshot,
            termination=self.termination,
            wall_time_seconds=wall_time,
        )

    @abstractmethod
    async def _worker(self, worker_id: int):
        """
        Worker coroutine that processes items from the queue.

        Each implementation defines its own worker strategy.

        Args:
            worker_id: Unique identifier for this worker
        """
        pass

    @abstractmethod
    async def _process_item(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        """
        Process a single work item.

        Each implementation defines how to execute the agent and handle results.

        Args:
            work_item: Work item to process

        Returns:
            Result of processing the work item
        """
        pass

    def _spawn_post_processor(
        self,
        result: WorkItemResult[TOutput, TContext],
        timeout: float | None,
    ) -> None:
        """Run the post-processor as a tracked background task (concurrent mode).

        Bounded by ``_post_processor_semaphore`` (size ``max_workers``) so we
        don't spawn unbounded concurrency. The task is tracked so
        ``process_all()`` can await it before returning and ``cleanup()`` can
        cancel it. ``_run_post_processor`` swallows its own exceptions, so the
        task never surfaces an error to the event loop.
        """
        if self._post_processor_semaphore is None:
            # Lazily bind to the running loop if process_all() hasn't set it.
            self._post_processor_semaphore = asyncio.Semaphore(self.max_workers)
        semaphore = self._post_processor_semaphore

        async def _bounded_run() -> None:
            async with semaphore:
                await self._run_post_processor(result, timeout=timeout)

        task: asyncio.Task[None] = asyncio.create_task(_bounded_run())
        self._post_processor_tasks.add(task)
        task.add_done_callback(self._post_processor_tasks.discard)

    async def _run_post_processor(
        self,
        result: WorkItemResult[TOutput, TContext],
        timeout: float | None = None,
    ) -> None:
        """
        Run the post-processor callback if provided.

        Args:
            result: Work item result to post-process
            timeout: Maximum seconds to wait for a post-processor callback
                (``None`` = no limit). Threaded through from
                ``ProcessorConfig.post_processor_timeout`` so a single, caller-
                configured value governs — there is no separate hardcoded cap.
        """
        if self.post_processor is None:
            return
        post_processor = self.post_processor

        async def invoke() -> None:
            if self._post_processor_is_async:
                callback_result = post_processor(result)
            else:
                # A synchronous callback may perform blocking DB/file I/O. Keep
                # it off the event loop just as synchronous progress callbacks are.
                callback_result = await asyncio.to_thread(post_processor, result)
            # Also support callable adapters that return an awaitable without
            # being declared with ``async def``.
            if inspect.isawaitable(callback_result):
                await callback_result

        try:
            if timeout is not None:
                await asyncio.wait_for(invoke(), timeout=timeout)
            else:
                await invoke()
        except (TimeoutError, asyncio.TimeoutError):
            logger.error(
                f"[FAIL]Post-processor execution timed out after {timeout}s for {result.item_id}"
            )
        except Exception as e:
            # Log error with full details - this is critical for debugging
            import traceback

            logger.error(
                f"[FAIL]Post-processor failed for {result.item_id}:\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error message: {str(e)}\n"
                f"  Full traceback:\n{traceback.format_exc()}"
            )

    async def _run_progress_callback(self, completed: int, total: int, current_item: str) -> None:
        """Invoke progress callback with timeout and non-blocking handling."""
        if self.progress_callback is None:
            return
        if self._progress_callback_inline:
            try:
                callback_result = self.progress_callback(completed, total, current_item)
                if inspect.isawaitable(callback_result):
                    await callback_result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[WARN]Progress callback failed: {exc}")
            return
        if self._progress_callback_is_async:
            callback_awaitable = self.progress_callback(completed, total, current_item)
        else:
            # Cast to sync callable since we know it's not async at this point
            callback_awaitable = asyncio.to_thread(
                self.progress_callback,  # type: ignore[arg-type]
                completed,
                total,
                current_item,
            )

        callback_task: asyncio.Task[None] = asyncio.create_task(callback_awaitable)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        self._track_progress_task(callback_task, log_exceptions=True)

        if self.progress_callback_timeout is not None:

            async def monitor() -> None:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(callback_task),
                        timeout=self.progress_callback_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[WARN]Progress callback exceeded timeout of %.2fs; continuing without waiting.",
                        self.progress_callback_timeout,
                    )
                    callback_task.cancel()
                except asyncio.CancelledError:
                    pass

            monitor_task = asyncio.create_task(monitor())
            self._track_progress_task(monitor_task, log_exceptions=False)

    def _track_progress_task(self, task: asyncio.Task[Any], *, log_exceptions: bool) -> None:
        """Track background tasks for progress callbacks and clean up on completion."""
        self._progress_tasks.add(task)

        def _cleanup(completed_task: asyncio.Task[Any]) -> None:
            self._progress_tasks.discard(completed_task)
            try:
                completed_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                if log_exceptions:
                    logger.warning(f"[WARN]Progress callback failed: {exc}")

        task.add_done_callback(_cleanup)
