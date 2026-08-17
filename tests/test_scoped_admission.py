"""Identity-scoped admission and classifier integration tests for v0.22."""

from __future__ import annotations

import asyncio
import gc
import math
import weakref
from typing import Any

import pytest

from async_batch_llm import (
    CallableStrategy,
    CallOutcome,
    GuardrailConfig,
    LLMCallPool,
    LLMWorkItem,
    ModelStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    call_result,
)
from async_batch_llm._internal.admission import AdmissionRegistry
from async_batch_llm._internal.capacity import CapacityLimiter
from async_batch_llm._internal.classifier_resolver import StrategyClassifierResolver
from async_batch_llm._internal.event_dispatcher import EventDispatcher
from async_batch_llm._internal.executor_host import ExecutorHost
from async_batch_llm.base import LLMResponse, RetryState, TokenUsage
from async_batch_llm.core import RateLimitConfig, RetryConfig
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.middleware import BaseMiddleware
from async_batch_llm.observers import BaseObserver, ProcessingEvent
from async_batch_llm.strategies import (
    DefaultErrorClassifier,
    ErrorClassifier,
    ErrorInfo,
    FixedDelayStrategy,
)


class _CategoryClassifier(ErrorClassifier):
    def __init__(self, category: str) -> None:
        self.category = category
        self.calls = 0

    def classify(self, exception: Exception) -> ErrorInfo:
        self.calls += 1
        return ErrorInfo(
            is_retryable=False,
            is_rate_limit=False,
            is_timeout=False,
            error_category=self.category,
        )


class _Strategy(LLMCallStrategy[str]):
    def __init__(
        self,
        *,
        scope: object | None = None,
        classifier: ErrorClassifier | None = None,
        fail: Exception | None = None,
        raise_scope: bool = False,
        raise_classifier: bool = False,
    ) -> None:
        self._scope = scope
        self._classifier = classifier
        self._fail = fail
        self._raise_scope = raise_scope
        self._raise_classifier = raise_classifier
        self.recommendation_calls = 0
        self.execute_calls = 0

    @property
    def quota_scope(self) -> object:
        if self._raise_scope:
            raise RuntimeError("broken quota scope")
        return self._scope  # type: ignore[return-value]

    def recommended_error_classifier(self) -> ErrorClassifier | None:
        self.recommendation_calls += 1
        if self._raise_classifier:
            raise RuntimeError("broken recommendation")
        return self._classifier

    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[str, TokenUsage, None]:
        self.execute_calls += 1
        if self._fail is not None:
            raise self._fail
        return prompt, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None


class _EqualScope:
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EqualScope)

    def __hash__(self) -> int:
        return 1


class _Model:
    async def generate(
        self,
        prompt: str | list[Any],
        *,
        temperature: float | None = 0.0,
        system_instruction: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> LLMResponse:
        raise AssertionError("not executed")


class _MatrixStrategy(_Strategy):
    def __init__(self, *, quota: object, concurrency: object) -> None:
        super().__init__(scope=quota)
        self._concurrency = concurrency

    @property
    def concurrency_scope(self) -> object:
        return self._concurrency

    @property
    def max_concurrency(self) -> int:
        return 1


def _registry(
    *,
    events: EventDispatcher[Any, Any, Any] | None = None,
    rpm: float | None = 60.0,
) -> AdmissionRegistry:
    return AdmissionRegistry(
        rate_limit_strategy=FixedDelayStrategy(cooldown=0.01, delay_between_requests=0),
        events=events or EventDispatcher(observers=[], middlewares=[]),
        max_requests_per_minute=rpm,
    )


@pytest.mark.asyncio
async def test_quota_scopes_use_identity_not_equality_and_share_only_when_explicit() -> None:
    registry = _registry()
    first_scope = _EqualScope()
    equal_but_distinct = _EqualScope()
    first = _Strategy(scope=first_scope)
    shared = _Strategy(scope=first_scope)
    distinct = _Strategy(scope=equal_but_distinct)

    first_state = registry.resolve(first)
    assert registry.resolve(first) is first_state
    assert registry.resolve(shared) is first_state
    distinct_state = registry.resolve(distinct)
    assert distinct_state is not first_state
    assert first_state.ordinal == 1
    assert distinct_state.ordinal == 2
    assert registry.entry_count == 2
    await registry.shutdown()


@pytest.mark.asyncio
async def test_same_scope_shares_rpm_while_different_scope_is_independent() -> None:
    registry = _registry(rpm=1.0)
    shared_scope = object()
    first_state = registry.resolve(_Strategy(scope=shared_scope))
    assert registry.resolve(_Strategy(scope=shared_scope)) is first_state
    independent_state = registry.resolve(_Strategy(scope=object()))

    consumed = await first_state.quota_gate.reserve()
    consumed.mark_provider_started()
    consumed.finalize()

    same_scope_waiter = asyncio.create_task(first_state.quota_gate.reserve())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert first_state.quota_gate.waiter_count == 1

    independent = await independent_state.quota_gate.reserve()
    independent.mark_provider_started()
    independent.finalize()

    same_scope_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await same_scope_waiter
    await registry.shutdown()


@pytest.mark.asyncio
async def test_none_and_raising_quota_scope_fall_back_to_strategy_identity() -> None:
    registry = _registry()
    none_scope = _Strategy(scope=None)
    broken_scope = _Strategy(raise_scope=True)

    none_state = registry.resolve(none_scope)
    broken_state = registry.resolve(broken_scope)
    assert none_state.scope is none_scope
    assert broken_state.scope is broken_scope
    assert none_state is not broken_state
    await registry.shutdown()


def test_public_quota_scope_constructor_hooks() -> None:
    scope = object()
    model = _Model()
    assert ModelStrategy[str](model).quota_scope is model
    assert ModelStrategy[str](model, quota_scope=scope).quota_scope is scope

    async def invoke(
        prompt: str, *, attempt: int, timeout: float, state: RetryState | None
    ) -> CallOutcome[str]:
        return CallOutcome(prompt)

    callable_strategy = CallableStrategy(invoke, quota_scope=scope)
    assert callable_strategy.quota_scope is scope
    default_callable = CallableStrategy(invoke)
    assert default_callable.quota_scope is default_callable.concurrency_scope


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("same_quota", "same_concurrency", "quota_entries", "capacity_entries"),
    [
        (True, True, 1, 1),
        (True, False, 1, 2),
        (False, True, 2, 1),
        (False, False, 2, 2),
    ],
)
async def test_quota_and_concurrency_scope_matrix(
    same_quota: bool,
    same_concurrency: bool,
    quota_entries: int,
    capacity_entries: int,
) -> None:
    first_quota = object()
    first_concurrency = object()
    first = _MatrixStrategy(quota=first_quota, concurrency=first_concurrency)
    second = _MatrixStrategy(
        quota=first_quota if same_quota else object(),
        concurrency=first_concurrency if same_concurrency else object(),
    )
    registry = _registry()
    registry.resolve(first)
    registry.resolve(second)

    limiter = CapacityLimiter(max_workers=2)
    async with limiter.admit(first):
        pass
    async with limiter.admit(second):
        pass

    assert registry.entry_count == quota_entries
    assert len(limiter._entries) == capacity_entries
    await registry.shutdown()


@pytest.mark.asyncio
async def test_registry_retains_scope_identity_and_is_bounded_by_scopes() -> None:
    registry = _registry()
    shared_scope = object()
    strategies = [_Strategy(scope=shared_scope) for _ in range(100)]
    state = registry.resolve(strategies[0])
    for strategy in strategies[1:]:
        assert registry.resolve(strategy) is state
    assert state.scope is shared_scope
    assert registry.entry_count == 1
    await registry.shutdown()


@pytest.mark.asyncio
async def test_registry_strong_reference_prevents_scope_id_reuse() -> None:
    registry = _registry()
    scope = _EqualScope()
    scope_ref = weakref.ref(scope)
    strategy = _Strategy(scope=scope)
    state = registry.resolve(strategy)

    # Even if the strategy drops its own reference, the identity-keyed entry
    # retains the exact object so its ID cannot be recycled into stale state.
    strategy._scope = None
    del scope
    gc.collect()
    assert scope_ref() is state.scope
    await registry.shutdown()


def test_classifier_resolver_is_per_strategy_once_and_explicit_override_wins() -> None:
    first_classifier = _CategoryClassifier("first")
    second_classifier = _CategoryClassifier("second")
    first = _Strategy(classifier=first_classifier)
    second = _Strategy(classifier=second_classifier)
    resolver = StrategyClassifierResolver()

    assert resolver.resolve(first) is first_classifier
    assert resolver.resolve(first) is first_classifier
    assert resolver.resolve(second) is second_classifier
    assert first.recommendation_calls == 1
    assert second.recommendation_calls == 1
    assert resolver.entry_count == 2

    explicit = _CategoryClassifier("explicit")
    explicit_resolver = StrategyClassifierResolver(explicit)
    assert explicit_resolver.resolve(first) is explicit
    assert explicit_resolver.resolve(second) is explicit
    assert first.recommendation_calls == 1
    assert second.recommendation_calls == 1


def test_classifier_recommendation_failure_falls_back_once() -> None:
    strategy = _Strategy(raise_classifier=True)
    resolver = StrategyClassifierResolver()
    first = resolver.resolve(strategy)
    second = resolver.resolve(strategy)
    assert isinstance(first, DefaultErrorClassifier)
    assert second is first
    assert strategy.recommendation_calls == 1


@pytest.mark.asyncio
async def test_mixed_strategy_batch_uses_each_matching_classifier_once() -> None:
    first_classifier = _CategoryClassifier("provider_a")
    second_classifier = _CategoryClassifier("provider_b")
    first = _Strategy(classifier=first_classifier, fail=ValueError("first"))
    second = _Strategy(classifier=second_classifier, fail=ValueError("second"))
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(max_workers=2, retry=RetryConfig(max_attempts=1))
    )
    await processor.add_work(LLMWorkItem(item_id="a", strategy=first, prompt="a"))
    await processor.add_work(LLMWorkItem(item_id="b", strategy=second, prompt="b"))

    result = await processor.process_all()
    categories = {item.item_id: item.error_category for item in result.results}
    assert categories == {"a": "provider_a", "b": "provider_b"}
    assert first.recommendation_calls == second.recommendation_calls == 1
    # One classification decision is reused by attempt timing, retry logic,
    # permanent failure construction, and final result construction.
    assert first_classifier.calls == second_classifier.calls == 1


@pytest.mark.asyncio
async def test_explicit_classifier_applies_to_every_strategy() -> None:
    recommended = _CategoryClassifier("recommended")
    explicit = _CategoryClassifier("explicit")
    strategies = [
        _Strategy(classifier=recommended, fail=ValueError("a")),
        _Strategy(classifier=recommended, fail=ValueError("b")),
    ]
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(max_workers=2, retry=RetryConfig(max_attempts=1)),
        error_classifier=explicit,
    )
    for index, strategy in enumerate(strategies):
        await processor.add_work(LLMWorkItem(item_id=str(index), strategy=strategy, prompt="x"))

    result = await processor.process_all()
    assert {item.error_category for item in result.results} == {"explicit"}
    assert all(strategy.recommendation_calls == 0 for strategy in strategies)
    assert explicit.calls == 2


@pytest.mark.asyncio
async def test_strategy_429_classifier_triggers_only_its_scoped_coordinator() -> None:
    events: list[tuple[ProcessingEvent, dict[str, Any]]] = []

    class _Observer(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
            events.append((event, data.copy()))

    class _RateClassifier(ErrorClassifier):
        def classify(self, exception: Exception) -> ErrorInfo:
            return ErrorInfo(True, True, False, "provider_rate_limit")

    class _RateOnce(_Strategy):
        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> tuple[str, TokenUsage, None]:
            self.execute_calls += 1
            if self.execute_calls == 1:
                raise RuntimeError("provider throttle")
            return prompt, {"total_tokens": 0}, None

    throttled = _RateOnce(scope=object(), classifier=_RateClassifier())
    independent = _Strategy(scope=object())
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(
            max_workers=1,
            retry=RetryConfig(max_attempts=1, max_rate_limit_retries=1),
            rate_limit=RateLimitConfig(
                cooldown_seconds=0,
                max_cooldown_seconds=0,
                slow_start_items=0,
            ),
        ),
        observers=[_Observer()],
    )
    await processor.add_work(LLMWorkItem(item_id="throttled", strategy=throttled, prompt="x"))
    await processor.add_work(LLMWorkItem(item_id="independent", strategy=independent, prompt="y"))
    throttled_state = processor._admission_registry.resolve(throttled)
    independent_state = processor._admission_registry.resolve(independent)
    result = await processor.process_all()

    assert result.succeeded == 2
    assert throttled_state.cooldown.current_generation == 1
    assert independent_state.cooldown.current_generation == 0
    scoped = [
        data
        for event, data in events
        if event
        in {
            ProcessingEvent.RATE_LIMIT_HIT,
            ProcessingEvent.COOLDOWN_STARTED,
            ProcessingEvent.COOLDOWN_ENDED,
        }
    ]
    assert len(scoped) == 3
    assert all(data["quota_scope_id"] == throttled_state.ordinal for data in scoped)
    await processor.shutdown()


@pytest.mark.asyncio
async def test_cooldown_events_are_scoped_and_do_not_format_scope_values() -> None:
    captured: list[tuple[ProcessingEvent, dict[str, Any]]] = []

    class _Observer(BaseObserver):
        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
            captured.append((event, data.copy()))

    class _HostileRepr:
        def __repr__(self) -> str:
            raise AssertionError("quota scope must never be formatted")

    events: EventDispatcher[Any, Any, Any] = EventDispatcher(
        observers=[_Observer()], middlewares=[]
    )
    registry = _registry(events=events)
    shared_scope = _HostileRepr()
    first = registry.resolve(_Strategy(scope=shared_scope))
    shared = registry.resolve(_Strategy(scope=shared_scope))
    separate = registry.resolve(_Strategy(scope=_HostileRepr()))
    assert first is shared

    cooldown = asyncio.create_task(first.cooldown.handle_rate_limit(7, strategy_type="_Strategy"))
    await asyncio.sleep(0)
    assert first.cooldown._in_cooldown
    assert separate.cooldown._rate_limit_event.is_set()
    await cooldown

    scoped_events = [
        data
        for event, data in captured
        if event in {ProcessingEvent.COOLDOWN_STARTED, ProcessingEvent.COOLDOWN_ENDED}
    ]
    assert len(scoped_events) == 2
    assert all(data["quota_scope_id"] == first.ordinal for data in scoped_events)
    assert all(data["strategy_type"] == "_Strategy" for data in scoped_events)
    await registry.shutdown()


@pytest.mark.asyncio
async def test_same_scope_attempt_waits_for_active_cooldown_but_other_scope_does_not() -> None:
    shared_scope = object()
    first = _Strategy(scope=shared_scope)
    same_scope = _Strategy(scope=shared_scope)
    other_scope = _Strategy(scope=object())
    host: ExecutorHost[str, str, None] = ExecutorHost(
        ProcessorConfig(max_workers=2), strategy=first
    )
    shared_state = host._admission_registry.resolve(first)
    assert host._admission_registry.resolve(same_scope) is shared_state
    other_state = host._admission_registry.resolve(other_scope)

    # Model an in-flight generation deterministically without a real sleep.
    shared_state.cooldown._in_cooldown = True
    shared_state.cooldown._rate_limit_event.clear()
    same_task = asyncio.create_task(
        host.executor.execute(LLMWorkItem(item_id="same", strategy=same_scope, prompt="same"))
    )
    other_task = asyncio.create_task(
        host.executor.execute(LLMWorkItem(item_id="other", strategy=other_scope, prompt="other"))
    )
    try:
        other_result = await asyncio.wait_for(other_task, timeout=0.2)
        assert other_result.success
        assert same_scope.execute_calls == 0
        assert other_state.cooldown._slow_start_active is False

        shared_state.cooldown._in_cooldown = False
        shared_state.cooldown._rate_limit_event.set()
        same_result = await asyncio.wait_for(same_task, timeout=0.2)
        assert same_result.success
        assert same_scope.execute_calls == 1
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_registry_shutdown_wakes_all_scope_waiters() -> None:
    registry = _registry(rpm=1.0)
    states = [registry.resolve(_Strategy(scope=object())) for _ in range(2)]
    waiters: list[asyncio.Task[Any]] = []
    for state in states:
        consumed = await state.quota_gate.reserve()
        consumed.mark_provider_started()
        consumed.finalize()
        waiters.append(asyncio.create_task(state.quota_gate.reserve()))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert all(state.quota_gate.waiter_count == 1 for state in states)

    await registry.shutdown()
    results = await asyncio.gather(*waiters, return_exceptions=True)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert all(not state.quota_gate.has_wake_task for state in states)


@pytest.mark.asyncio
async def test_registry_cleanup_failure_does_not_skip_other_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    first = registry.resolve(_Strategy(scope=object()))
    second = registry.resolve(_Strategy(scope=object()))
    second_cleaned = False

    async def broken_shutdown() -> None:
        raise RuntimeError("first cleanup failed")

    async def tracked_shutdown() -> None:
        nonlocal second_cleaned
        second_cleaned = True

    monkeypatch.setattr(first.cooldown, "shutdown", broken_shutdown)
    monkeypatch.setattr(second.cooldown, "shutdown", tracked_shutdown)
    with pytest.raises(RuntimeError, match="first cleanup failed"):
        await registry.shutdown()
    assert second_cleaned


@pytest.mark.asyncio
async def test_every_rate_limit_retry_consumes_a_fresh_rpm_reservation() -> None:
    class _RateLimitThenSuccess(_Strategy):
        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> tuple[str, TokenUsage, None]:
            self.execute_calls += 1
            if self.execute_calls == 1:
                raise RuntimeError("429 rate limit")
            return prompt, {"total_tokens": 0}, None

    strategy = _RateLimitThenSuccess()
    config = ProcessorConfig(
        max_workers=1,
        max_requests_per_minute=2.0,
        retry=RetryConfig(max_attempts=1, max_rate_limit_retries=1),
        rate_limit=RateLimitConfig(
            cooldown_seconds=0,
            max_cooldown_seconds=0,
            slow_start_items=0,
        ),
    )
    host: ExecutorHost[str, str, None] = ExecutorHost(config, strategy=strategy)
    try:
        result = await host.executor.execute(
            LLMWorkItem(item_id="retry", strategy=strategy, prompt="x")
        )
        state = host._admission_registry.resolve(strategy)
        assert result.success
        assert strategy.execute_calls == 2
        assert state.quota_gate.request_available is not None
        assert state.quota_gate.request_available < 0.01
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_every_transport_retry_consumes_a_fresh_rpm_reservation() -> None:
    class _TransportThenSuccess(_Strategy):
        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> tuple[str, TokenUsage, None]:
            self.execute_calls += 1
            if self.execute_calls == 1:
                raise ConnectionError("temporary transport failure")
            return prompt, {"total_tokens": 0}, None

    strategy = _TransportThenSuccess()
    config = ProcessorConfig(
        max_workers=1,
        max_requests_per_minute=2.0,
        retry=RetryConfig(
            max_attempts=2,
            initial_wait=0.001,
            max_wait=0.001,
            jitter=False,
        ),
    )
    host: ExecutorHost[str, str, None] = ExecutorHost(config, strategy=strategy)
    try:
        result = await host.executor.execute(
            LLMWorkItem(item_id="retry", strategy=strategy, prompt="x")
        )
        state = host._admission_registry.resolve(strategy)
        assert result.success
        assert strategy.execute_calls == 2
        assert state.quota_gate.request_available is not None
        assert state.quota_gate.request_available < 0.01
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_item_deadline_removes_rpm_waiter_and_never_reaches_capacity() -> None:
    strategy = _Strategy()
    config = ProcessorConfig(
        max_workers=1,
        max_provider_concurrency=1,
        max_requests_per_minute=1.0,
        guardrails=GuardrailConfig(total_timeout_per_item=0.01),
    )
    host: ExecutorHost[str, str, None] = ExecutorHost(config, strategy=strategy)
    state = host._admission_registry.resolve(strategy)
    consumed = await state.quota_gate.reserve()
    consumed.mark_provider_started()
    consumed.finalize()
    try:
        result = await host.executor.execute(
            LLMWorkItem(item_id="deadline", strategy=strategy, prompt="x")
        )
        assert not result.success
        assert result.error_category == "framework_total_item_timeout"
        assert state.quota_gate.waiter_count == 0
        assert host._capacity_limiter._entries == {}
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_queue_less_surfaces_use_strategy_classifier_resolver() -> None:
    direct_classifier = _CategoryClassifier("direct")
    direct_strategy = _Strategy(
        classifier=direct_classifier,
        fail=ValueError("direct failure"),
    )
    result = await call_result(
        direct_strategy,
        "x",
        config=ProcessorConfig(retry=RetryConfig(max_attempts=1)),
    )
    assert result.error_category == "direct"
    assert direct_classifier.calls == 1
    assert direct_strategy.recommendation_calls == 1

    pool_classifier = _CategoryClassifier("pool")
    pool_strategy = _Strategy(
        classifier=pool_classifier,
        fail=ValueError("pool failure"),
    )
    async with LLMCallPool(
        pool_strategy,
        config=ProcessorConfig(retry=RetryConfig(max_attempts=1)),
    ) as pool:
        pool_result = await pool.submit_result("x")
    assert pool_result.error_category == "pool"
    assert pool_classifier.calls == 1
    assert pool_strategy.recommendation_calls == 1


async def _wait_until(predicate: Any) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


@pytest.mark.asyncio
async def test_rpm_wait_does_not_hold_provider_capacity() -> None:
    strategy = _Strategy()
    config = ProcessorConfig(
        max_workers=1,
        max_provider_concurrency=1,
        max_requests_per_minute=1.0,
    )
    host: ExecutorHost[str, str, None] = ExecutorHost(config, strategy=strategy)
    state = host._admission_registry.resolve(strategy)
    consumed = await state.quota_gate.reserve()
    consumed.mark_provider_started()
    consumed.finalize()
    task = asyncio.create_task(
        host.executor.execute(LLMWorkItem(item_id="waiting", strategy=strategy, prompt="x"))
    )
    try:
        await _wait_until(lambda: state.quota_gate.waiter_count == 1)
        assert host._capacity_limiter._entries == {}
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await host.aclose()


@pytest.mark.asyncio
async def test_capacity_cancellation_refunds_pre_provider_rpm() -> None:
    strategy = _Strategy()
    config = ProcessorConfig(
        max_workers=2,
        max_provider_concurrency=1,
        max_requests_per_minute=1.0,
    )
    host: ExecutorHost[str, str, None] = ExecutorHost(config, strategy=strategy)
    state = host._admission_registry.resolve(strategy)
    occupied = host._capacity_limiter.admit(strategy, item_id="holder")
    await occupied.__aenter__()
    task = asyncio.create_task(
        host.executor.execute(LLMWorkItem(item_id="waiting", strategy=strategy, prompt="x"))
    )
    try:
        await _wait_until(lambda: (state.quota_gate.request_available or 0) < 0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.quota_gate.request_available == pytest.approx(1.0)
    finally:
        await occupied.__aexit__(None, None, None)
        await host.aclose()


@pytest.mark.asyncio
async def test_dry_run_and_middleware_skip_bypass_live_admission() -> None:
    dry_strategy = _Strategy()
    dry_host: ExecutorHost[str, str, None] = ExecutorHost(
        ProcessorConfig(
            max_workers=1,
            dry_run=True,
            max_provider_concurrency=1,
            max_requests_per_minute=0.5,
        ),
        strategy=dry_strategy,
    )
    try:
        dry_state = dry_host._admission_registry.resolve(dry_strategy)
        dry_state.cooldown._slow_start_active = True
        dry_state.cooldown._items_since_resume = 2
        result = await dry_host.executor.execute(
            LLMWorkItem(item_id="dry", strategy=dry_strategy, prompt="x")
        )
        assert result.success
        assert dry_state.quota_gate.request_available == 1.0
        assert dry_state.cooldown._slow_start_active is True
        assert dry_state.cooldown._items_since_resume == 2
        assert dry_host._capacity_limiter._entries == {}
    finally:
        await dry_host.aclose()

    class _Skip(BaseMiddleware[str, str, None]):
        async def before_process(
            self, work_item: LLMWorkItem[str, str, None]
        ) -> LLMWorkItem[str, str, None] | None:
            return None

    skip_strategy = _Strategy()
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(
            max_workers=1,
            max_provider_concurrency=1,
            max_requests_per_minute=0.5,
        ),
        middlewares=[_Skip()],
    )
    skip_state = processor._admission_registry.resolve(skip_strategy)
    try:
        result = await processor._executor.execute(
            LLMWorkItem(item_id="skip", strategy=skip_strategy, prompt="x")
        )
        assert not result.success
        assert skip_state.quota_gate.request_available == 1.0
        assert processor._capacity_limiter._entries == {}
    finally:
        await processor.shutdown()


@pytest.mark.asyncio
async def test_artifact_replay_bypasses_quota_cooldown_and_capacity() -> None:
    class _ReplayStore:
        closed = False

        async def prepare_item(self, work_item: LLMWorkItem[Any, Any, Any]) -> str:
            return work_item.item_id

        async def lookup(
            self,
            work_item: LLMWorkItem[Any, Any, Any],
            prepared_item: Any,
            policy: Any,
        ) -> Any:
            return WorkItemResult(
                item_id=work_item.item_id,
                success=True,
                output="replayed",
                replayed_from_artifact=True,
            )

        async def append(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("replay must not append")

        async def close(self) -> None:
            self.closed = True

    from async_batch_llm import ResumePolicy, WorkItemResult

    strategy = _Strategy()
    store = _ReplayStore()
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(
            max_workers=1,
            max_provider_concurrency=1,
            max_requests_per_minute=0.5,
        ),
        artifact_store=store,  # type: ignore[arg-type]
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    await processor.add_work(LLMWorkItem(item_id="replay", strategy=strategy, prompt="x"))
    state = processor._admission_registry.resolve(strategy)
    try:
        batch = await processor.process_all()
        assert batch.results[0].replayed_from_artifact
        assert strategy.execute_calls == 0
        assert state.quota_gate.request_available == 1.0
        assert state.cooldown.current_generation == 0
        assert processor._capacity_limiter._entries == {}
    finally:
        await processor.__aexit__(None, None, None)
    assert store.closed


@pytest.mark.asyncio
async def test_batch_abort_removes_rpm_waiter() -> None:
    strategy = _Strategy()
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(
            max_workers=1,
            max_provider_concurrency=1,
            max_requests_per_minute=1.0,
            guardrails=GuardrailConfig(batch_timeout=0.01),
        )
    )
    await processor.add_work(LLMWorkItem(item_id="abort", strategy=strategy, prompt="x"))
    state = processor._admission_registry.resolve(strategy)
    consumed = await state.quota_gate.reserve()
    consumed.mark_provider_started()
    consumed.finalize()
    try:
        batch = await processor.process_all()
        assert batch.termination.kind == "batch_timeout"
        assert batch.results[0].error_category == "batch_deadline_exceeded"
        assert state.quota_gate.waiter_count == 0
        assert processor._capacity_limiter._entries == {}
    finally:
        await processor.shutdown()


@pytest.mark.parametrize("value", [0, -1, math.inf, -math.inf, math.nan, True, "1"])
def test_rpm_validation_rejects_nonpositive_nonfinite_and_non_numeric(value: object) -> None:
    with pytest.raises(ValueError, match="max_requests_per_minute must be > 0"):
        ProcessorConfig(max_requests_per_minute=value)  # type: ignore[arg-type]


def test_rpm_validation_accepts_positive_fractional_values() -> None:
    assert ProcessorConfig(max_requests_per_minute=0.25).max_requests_per_minute == 0.25
