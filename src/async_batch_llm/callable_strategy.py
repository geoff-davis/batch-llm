"""Adapt an existing asynchronous operation to ABL's execution pipeline."""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast

from .artifacts import ArtifactIdentity
from .base import RetryState, TokenUsage
from .llm_strategies import LLMCallStrategy
from .strategies import ErrorClassifier
from .token_estimation import TokenEstimate, TokenEstimator

TOutput = TypeVar("TOutput")
TOutputCallback = TypeVar("TOutputCallback")

LifecycleCallback: TypeAlias = Callable[[], None | Awaitable[None]]
ErrorCallback: TypeAlias = Callable[
    [Exception, int, RetryState | None],
    None | Awaitable[None],
]
RequestConcurrencyCallback: TypeAlias = Callable[[int], bool | Awaitable[bool]]

_TOKEN_USAGE_KEYS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"}
)
_LAST_PROVIDER_SECONDS_KEY = "_abl_last_provider_seconds"


@dataclass(frozen=True)
class CallOutcome(Generic[TOutput]):
    """The output, reported token usage, and metadata from an async operation.

    Token usage and metadata are copied and validated when the outcome crosses
    the strategy boundary. An empty usage mapping means the upstream operation
    did not report usage; ABL does not estimate missing values.
    """

    output: TOutput
    token_usage: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] | None = None


class InvokeCallback(Protocol[TOutputCallback]):
    """Canonical callable accepted by :class:`CallableStrategy`."""

    def __call__(
        self,
        prompt: str,
        *,
        attempt: int,
        timeout: float,
        state: RetryState | None,
    ) -> Awaitable[CallOutcome[TOutputCallback]]: ...


class DryRunCallback(Protocol[TOutputCallback]):
    """Synchronous or asynchronous dry-run outcome producer."""

    def __call__(
        self, prompt: str
    ) -> CallOutcome[TOutputCallback] | Awaitable[CallOutcome[TOutputCallback]]: ...


def _is_async_callable(callback: object) -> bool:
    if inspect.iscoroutinefunction(callback):
        return True
    return inspect.iscoroutinefunction(type(callback).__call__)


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _normalize_token_usage(usage: Mapping[str, int]) -> TokenUsage:
    if not isinstance(usage, Mapping):
        raise TypeError("CallOutcome.token_usage must be a mapping")

    unknown = set(usage) - _TOKEN_USAGE_KEYS
    if unknown:
        keys = ", ".join(sorted(str(key) for key in unknown))
        supported = ", ".join(sorted(_TOKEN_USAGE_KEYS))
        raise ValueError(
            f"Unknown CallOutcome.token_usage key(s): {keys}. Supported keys: {supported}."
        )

    normalized: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"CallOutcome.token_usage[{key!r}] must be a non-negative integer (got {value!r})"
            )
        normalized[key] = value

    if "total_tokens" not in normalized and (
        "input_tokens" in normalized or "output_tokens" in normalized
    ):
        normalized["total_tokens"] = normalized.get("input_tokens", 0) + normalized.get(
            "output_tokens", 0
        )
    return cast(TokenUsage, normalized)


def _normalize_outcome(
    outcome: object,
) -> tuple[Any, TokenUsage, dict[str, Any] | None]:
    if not isinstance(outcome, CallOutcome):
        raise TypeError(
            f"CallableStrategy callbacks must resolve to CallOutcome (got {type(outcome).__name__})"
        )
    token_usage = _normalize_token_usage(outcome.token_usage)
    if outcome.metadata is None:
        metadata = None
    elif isinstance(outcome.metadata, Mapping):
        metadata = dict(outcome.metadata)
    else:
        raise TypeError("CallOutcome.metadata must be a mapping or None")
    return outcome.output, token_usage, metadata


class CallableStrategy(LLMCallStrategy[TOutput]):
    """Adapt an existing async client or application operation to ABL.

    The invocation callback has this canonical shape::

        async def invoke(
            prompt: str,
            *,
            attempt: int,
            timeout: float,
            state: RetryState | None,
        ) -> CallOutcome[TOutput]: ...

    ``attempt`` is ABL's logical attempt number, ``timeout`` is the effective
    executor-owned attempt budget, and ``state`` belongs exclusively to the
    current work item. This class adapts callbacks to ``LLMCallStrategy``; the
    existing ``ItemExecutor`` remains responsible for retries, timeouts,
    cooldowns, accounting, and cancellation.
    """

    def __init__(
        self,
        invoke: InvokeCallback[TOutput],
        *,
        identity: ArtifactIdentity | None = None,
        error_classifier: ErrorClassifier | None = None,
        prepare: LifecycleCallback | None = None,
        cleanup: LifecycleCallback | None = None,
        on_error: ErrorCallback | None = None,
        dry_run: DryRunCallback[TOutput] | None = None,
        max_concurrency: int | None = None,
        concurrency_scope: object | None = None,
        quota_scope: object | None = None,
        token_estimator: TokenEstimator | None = None,
        request_concurrency: RequestConcurrencyCallback | None = None,
    ) -> None:
        if not callable(invoke):
            raise TypeError("CallableStrategy invoke must be callable")
        if not _is_async_callable(invoke):
            raise TypeError(
                "CallableStrategy invoke must be declared with async def and return CallOutcome"
            )
        if max_concurrency is not None and (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency <= 0
        ):
            raise ValueError("max_concurrency must be a positive integer or None")
        if token_estimator is not None and not callable(token_estimator):
            raise TypeError("token_estimator must be callable or None")

        self._invoke = invoke
        self._artifact_identity = identity
        self._error_classifier = error_classifier
        self._prepare_callback = prepare
        self._cleanup_callback = cleanup
        self._on_error_callback = on_error
        self._dry_run_callback = dry_run
        self._max_concurrency = max_concurrency
        self._concurrency_scope = concurrency_scope
        self._quota_scope = quota_scope
        self._token_estimator = token_estimator
        self._request_concurrency_callback = request_concurrency

    @property
    def artifact_identity(self) -> ArtifactIdentity | None:
        """Stable artifact identity, or ``None`` when the caller omitted one."""
        return self._artifact_identity

    @property
    def max_concurrency(self) -> int | None:
        return self._max_concurrency

    @property
    def concurrency_scope(self) -> object:
        return self if self._concurrency_scope is None else self._concurrency_scope

    @property
    def quota_scope(self) -> object:
        """Identity whose RPM and coordinated cooldown this call consumes.

        Defaults to :attr:`concurrency_scope`; pass ``quota_scope=`` when
        transport capacity and provider/account quota have different owners.
        """
        return self.concurrency_scope if self._quota_scope is None else self._quota_scope

    def recommended_error_classifier(self) -> ErrorClassifier | None:
        return self._error_classifier

    def estimate_tokens(
        self,
        prompt: str,
        attempt: int,
        state: RetryState | None,
    ) -> TokenEstimate | Awaitable[TokenEstimate] | None:
        if self._token_estimator is None:
            return None
        return self._token_estimator(
            prompt,
            strategy=self,
            attempt=attempt,
            state=state,
        )

    async def prepare(self) -> None:
        if self._prepare_callback is not None:
            await _await_if_needed(self._prepare_callback())

    async def cleanup(self) -> None:
        if self._cleanup_callback is not None:
            await _await_if_needed(self._cleanup_callback())

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ) -> None:
        if self._on_error_callback is not None:
            await _await_if_needed(self._on_error_callback(exception, attempt, state))

    async def request_concurrency(self, concurrency: int) -> bool:
        if self._request_concurrency_callback is None:
            return False
        return bool(await _await_if_needed(self._request_concurrency_callback(concurrency)))

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[TOutput, TokenUsage, dict[str, Any] | None]:
        provider_started = time.perf_counter()
        try:
            pending = self._invoke(prompt, attempt=attempt, timeout=timeout, state=state)
            if not inspect.isawaitable(pending):
                raise TypeError(
                    "CallableStrategy invoke returned a non-awaitable result; "
                    "declare it with async def"
                )
            outcome = await pending
        finally:
            if state is not None:
                state.set(
                    _LAST_PROVIDER_SECONDS_KEY,
                    max(0.0, time.perf_counter() - provider_started),
                )

        output, usage, metadata = _normalize_outcome(outcome)
        return cast(TOutput, output), usage, metadata

    async def dry_run(self, prompt: str) -> tuple[TOutput, TokenUsage]:
        if self._dry_run_callback is None:
            return await super().dry_run(prompt)
        outcome = await _await_if_needed(self._dry_run_callback(prompt))
        output, usage, _ = _normalize_outcome(outcome)
        return cast(TOutput, output), usage


__all__ = [
    "CallOutcome",
    "CallableStrategy",
    "DryRunCallback",
    "ErrorCallback",
    "InvokeCallback",
    "LifecycleCallback",
    "RequestConcurrencyCallback",
]
