"""Session B integration coverage for token reservation and reconciliation."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from async_batch_llm import (
    AbortMode,
    ArtifactIdentity,
    AttemptTiming,
    BaseMiddleware,
    BaseObserver,
    BatchResult,
    GuardrailConfig,
    JsonlArtifactStore,
    LLMCallPool,
    LLMCallStrategy,
    LLMGateway,
    LLMWorkItem,
    MetricsObserver,
    ParallelBatchProcessor,
    ProcessingEvent,
    ProcessingStats,
    ProcessorConfig,
    RateLimitConfig,
    ResumePolicy,
    RetryConfig,
    RetryState,
    TokenEstimate,
    TokenEstimationError,
    TokenUsage,
    WorkItemResult,
    WorkItemTiming,
    call_result,
    process_prompts,
    process_stream,
)
from async_batch_llm._internal.executor_host import ExecutorHost


class _SequenceStrategy(LLMCallStrategy[str]):
    def __init__(
        self,
        outcomes: list[Mapping[str, int] | BaseException],
        *,
        estimate: TokenEstimate | None = None,
        scope: object | None = None,
    ) -> None:
        self.outcomes = list(outcomes)
        self.estimate = estimate
        self.scope = scope
        self.calls = 0
        self.on_error_calls = 0
        self.estimate_calls: list[tuple[str, int, RetryState | None]] = []

    @property
    def quota_scope(self) -> object:
        return self if self.scope is None else self.scope

    def estimate_tokens(
        self, prompt: str, attempt: int, state: RetryState | None
    ) -> TokenEstimate | None:
        self.estimate_calls.append((prompt, attempt, state))
        return self.estimate

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        index = self.calls
        self.calls += 1
        outcome = self.outcomes[min(index, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return prompt, dict(outcome), None  # type: ignore[return-value]

    async def on_error(
        self, exception: Exception, attempt: int, state: RetryState | None = None
    ) -> None:
        self.on_error_calls += 1


class _Recorder(BaseObserver):
    def __init__(self) -> None:
        self.events: list[tuple[ProcessingEvent, dict[str, Any]]] = []

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        self.events.append((event, data))


def _config(**kwargs: Any) -> ProcessorConfig:
    values: dict[str, Any] = {
        "max_workers": 2,
        "max_tokens_per_minute": 1_000,
        "retry": RetryConfig(max_attempts=2, initial_wait=0.001, max_wait=0.001, jitter=False),
        "rate_limit": RateLimitConfig(
            cooldown_seconds=0,
            max_cooldown_seconds=0,
            slow_start_items=0,
            slow_start_initial_delay=0,
            slow_start_final_delay=0,
            backoff_multiplier=1,
        ),
    }
    values.update(kwargs)
    return ProcessorConfig(**values)


@pytest.mark.asyncio
async def test_missing_estimator_fails_before_provider_and_bypasses_strategy_recovery() -> None:
    strategy = _SequenceStrategy([{"total_tokens": 1}])
    result = await call_result(strategy, "secret prompt", config=_config())

    assert not result.success
    assert result.error_category == "token_estimator_required"
    assert strategy.calls == 0
    assert strategy.on_error_calls == 0
    assert result.timing.attempts[0].reserved_tokens == 0


@pytest.mark.asyncio
async def test_config_estimator_precedes_strategy_estimator_and_async_is_supported() -> None:
    strategy = _SequenceStrategy([{"input_tokens": 3, "output_tokens": 2}])
    seen: list[tuple[str, int, RetryState | None]] = []

    async def estimator(
        prompt: str, *, strategy: Any, attempt: int, state: RetryState | None
    ) -> TokenEstimate:
        await asyncio.sleep(0)
        seen.append((prompt, attempt, state))
        return TokenEstimate(7, 3)

    result = await call_result(strategy, "processed", config=_config(token_estimator=estimator))

    assert result.success
    assert strategy.estimate_calls == []
    assert len(seen) == 1 and seen[0][0:2] == ("processed", 1)
    attempt = result.timing.attempts[0]
    assert attempt.estimated_input_tokens == 7
    assert attempt.estimated_output_tokens == 3
    assert attempt.reserved_tokens == 10
    assert attempt.reported_tokens == 5
    assert attempt.reconciliation_delta_tokens == -5
    assert result.admission_wait_seconds == 0
    assert result.quota_wait_seconds == 0


@pytest.mark.asyncio
async def test_estimator_exception_is_non_retryable_and_redacted() -> None:
    strategy = _SequenceStrategy([{"total_tokens": 1}])

    def estimator(prompt: str, **kwargs: Any) -> TokenEstimate:
        raise RuntimeError(f"estimator leaked {prompt}")

    result = await call_result(
        strategy,
        "top-secret-value",
        config=_config(token_estimator=estimator),
    )

    assert not result.success
    assert result.error_category == "token_estimation_error"
    assert "top-secret-value" not in (result.error or "")
    assert "top-secret-value" not in str(result.to_dict())
    assert result.exception is not None and result.exception.__cause__ is None
    assert strategy.calls == 0


@pytest.mark.asyncio
async def test_zero_and_above_limit_estimates_fail_before_provider() -> None:
    for estimate, category in (
        (TokenEstimate(0), "token_estimation_error"),
        (TokenEstimate(1_001), "token_estimate_exceeds_limit"),
    ):
        strategy = _SequenceStrategy([{"total_tokens": 1}], estimate=estimate)
        result = await call_result(strategy, "x", config=_config())

        assert not result.success
        assert result.error_category == category
        assert strategy.calls == 0
        assert strategy.on_error_calls == 0


@pytest.mark.asyncio
async def test_async_estimator_cancellation_propagates_without_provider_call() -> None:
    strategy = _SequenceStrategy([{"total_tokens": 1}])
    entered = asyncio.Event()

    async def estimator(prompt: str, **kwargs: Any) -> TokenEstimate:
        entered.set()
        await asyncio.Event().wait()
        return TokenEstimate(1)

    task = asyncio.create_task(
        call_result(strategy, "x", config=_config(token_estimator=estimator))
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert strategy.calls == 0


@pytest.mark.asyncio
async def test_sync_estimator_is_bounded_by_item_deadline() -> None:
    strategy = _SequenceStrategy([{"total_tokens": 1}])
    entered = threading.Event()
    release = threading.Event()

    def estimator(prompt: str, **kwargs: Any) -> TokenEstimate:
        entered.set()
        release.wait(timeout=1)
        return TokenEstimate(1)

    try:
        result = await asyncio.wait_for(
            call_result(
                strategy,
                "x",
                config=_config(
                    token_estimator=estimator,
                    guardrails=GuardrailConfig(total_timeout_per_item=0.02),
                ),
            ),
            timeout=0.5,
        )
    finally:
        release.set()

    assert entered.is_set()
    assert not result.success
    assert result.error_category == "framework_total_item_timeout"
    assert strategy.calls == 0


@pytest.mark.asyncio
async def test_processor_stats_events_and_known_zero_unknown_are_distinct() -> None:
    scope = object()
    strategies = [
        _SequenceStrategy([{"total_tokens": 4}], estimate=TokenEstimate(10), scope=scope),
        _SequenceStrategy([{"total_tokens": 0}], estimate=TokenEstimate(10), scope=scope),
        _SequenceStrategy([{}], estimate=TokenEstimate(10), scope=scope),
        _SequenceStrategy([{"total_tokens": 15}], estimate=TokenEstimate(10), scope=scope),
    ]
    observer = _Recorder()
    processor = ParallelBatchProcessor[str, str, None](
        config=_config(max_workers=3),
        observers=[observer],
    )
    for index, strategy in enumerate(strategies):
        await processor.add_work(
            LLMWorkItem(item_id=str(index), strategy=strategy, prompt=f"p{index}")
        )
    try:
        batch = await processor.process_all()
        stats = await processor.get_stats()
    finally:
        await processor.cleanup()

    assert batch.total_items == batch.succeeded == 4
    assert stats["reserved_tokens"] == 40
    assert stats["reported_reconciliation_tokens"] == 19
    assert stats["refunded_tokens"] == 16
    assert stats["underestimated_tokens"] == 5
    assert stats["unknown_usage_attempts"] == 1
    assert stats["known_zero_usage_attempts"] == 1
    assert stats["quota_scope_count"] == 1
    assert stats["total_tokens"] == 19
    admitted = [data for event, data in observer.events if event is ProcessingEvent.QUOTA_ADMITTED]
    reconciled = [
        data for event, data in observer.events if event is ProcessingEvent.QUOTA_RECONCILED
    ]
    assert len(admitted) == len(reconciled) == 4
    assert {data["quota_scope_id"] for data in admitted} == {1}
    assert all("prompt" not in data for data in admitted + reconciled)
    assert all(data["estimated_total_tokens"] == 10 for data in admitted)
    assert {data["disposition"] for data in reconciled} == {
        "debt",
        "refunded",
        "retained_unknown",
    }


@pytest.mark.asyncio
async def test_rate_limit_and_transport_retries_receive_fresh_reservations() -> None:
    rate = _SequenceStrategy(
        [RuntimeError("rate limit exceeded"), {"total_tokens": 5}],
        estimate=TokenEstimate(10),
    )
    observer = _Recorder()
    processor = ParallelBatchProcessor[str, str, None](
        config=_config(retry=RetryConfig(max_attempts=1, max_rate_limit_retries=2)),
        observers=[observer],
    )
    await processor.add_work(LLMWorkItem(item_id="rate", strategy=rate, prompt="x"))
    try:
        batch = await processor.process_all()
        stats = await processor.get_stats()
    finally:
        await processor.cleanup()

    result = batch.results[0]
    assert result.success and rate.calls == 2
    assert [attempt.attempt for attempt in result.timing.attempts] == [1, 1]
    assert [call[1] for call in rate.estimate_calls] == [1, 1]
    assert stats["reserved_tokens"] == 20
    assert stats["unknown_usage_attempts"] == 1
    admitted = [data for event, data in observer.events if event is ProcessingEvent.QUOTA_ADMITTED]
    assert [data["try_number"] for data in admitted] == [1, 2]

    transport = _SequenceStrategy(
        [ConnectionError("temporary"), {"total_tokens": 5}],
        estimate=TokenEstimate(10),
    )
    result = await call_result(transport, "x", config=_config())
    assert result.success
    assert [attempt.attempt for attempt in result.timing.attempts] == [1, 2]
    assert [call[1] for call in transport.estimate_calls] == [1, 2]

    zero_failure = ConnectionError("known unbilled failure")
    zero_failure.__dict__["_failed_token_usage"] = {"total_tokens": 0}
    after_zero = _SequenceStrategy(
        [zero_failure, {"total_tokens": 5}],
        estimate=TokenEstimate(10),
    )
    result = await call_result(after_zero, "x", config=_config())
    assert result.success and after_zero.calls == 2
    assert result.timing.attempts[0].reported_tokens == 0
    assert result.timing.attempts[0].reconciliation_delta_tokens == -10
    assert result.timing.attempts[1].reserved_tokens == 10


@pytest.mark.asyncio
async def test_failed_known_usage_reconciles_before_retry_accounting() -> None:
    failure = ValueError("bad output")
    failure.__dict__["_failed_token_usage"] = {
        "input_tokens": 4,
        "output_tokens": 3,
        "total_tokens": 7,
    }
    strategy = _SequenceStrategy([failure], estimate=TokenEstimate(10))
    result = await call_result(
        strategy,
        "x",
        config=_config(retry=RetryConfig(max_attempts=1)),
    )

    assert not result.success
    assert result.token_usage["total_tokens"] == 7
    attempt = result.timing.attempts[0]
    assert attempt.reported_tokens == 7
    assert attempt.reconciliation_delta_tokens == -3


@pytest.mark.asyncio
async def test_provider_usage_object_reconciles_and_malformed_usage_stays_unknown() -> None:
    class Usage:
        input_tokens = 4
        output_tokens = 3
        total_tokens = 7

    known_failure = RuntimeError("provider failed")
    known_failure.usage = Usage()  # type: ignore[attr-defined]
    known = _SequenceStrategy([known_failure], estimate=TokenEstimate(10))
    known_result = await call_result(
        known,
        "x",
        config=_config(retry=RetryConfig(max_attempts=1)),
    )
    assert known_result.timing.attempts[0].reported_tokens == 7
    assert known_result.token_usage["total_tokens"] == 7

    malformed_failure = RuntimeError("original provider failure")
    malformed_failure.__dict__["_failed_token_usage"] = {"total_tokens": -4}
    malformed = _SequenceStrategy([malformed_failure], estimate=TokenEstimate(10))
    malformed_result = await call_result(
        malformed,
        "x",
        config=_config(retry=RetryConfig(max_attempts=1)),
    )
    assert not malformed_result.success
    assert "original provider failure" in (malformed_result.error or "")
    assert malformed_result.timing.attempts[0].reported_tokens is None
    assert malformed_result.timing.attempts[0].reconciliation_delta_tokens is None


@pytest.mark.asyncio
async def test_invalid_strategy_return_after_provider_start_retains_estimate() -> None:
    class InvalidReturn(LLMCallStrategy[str]):
        def estimate_tokens(
            self, prompt: str, attempt: int, state: RetryState | None
        ) -> TokenEstimate:
            return TokenEstimate(10)

        async def execute(
            self,
            prompt: str,
            attempt: int,
            timeout: float,
            state: RetryState | None = None,
        ) -> Any:
            return ("invalid",)

    result = await call_result(
        InvalidReturn(),
        "x",
        config=_config(retry=RetryConfig(max_attempts=1)),
    )

    assert not result.success
    assert result.timing.attempts[0].reserved_tokens == 10
    assert result.timing.attempts[0].reported_tokens is None
    assert result.timing.attempts[0].reconciliation_delta_tokens is None


class _BlockingStrategy(LLMCallStrategy[str]):
    def __init__(self) -> None:
        self.started_prompts: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.started_prompts.append(prompt)
        self.started.set()
        if prompt == "hold":
            await self.release.wait()
        return prompt, {"total_tokens": 0}, None


async def _wait_until(predicate: Any) -> None:
    deadline = asyncio.get_running_loop().time() + 1.0
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not reached")


@pytest.mark.asyncio
async def test_capacity_cancellation_before_provider_start_refunds_both_dimensions() -> None:
    strategy = _BlockingStrategy()

    def estimator(prompt: str, **kwargs: Any) -> TokenEstimate:
        return TokenEstimate(10 if prompt == "hold" else 20)

    host: ExecutorHost[str, str, None] = ExecutorHost(
        _config(max_provider_concurrency=1, token_estimator=estimator, max_tokens_per_minute=100),
        strategy=strategy,
    )
    gate = host._admission_registry.resolve(strategy).quota_gate
    first = asyncio.create_task(
        host.executor.execute(LLMWorkItem(item_id="hold", strategy=strategy, prompt="hold"))
    )
    await strategy.started.wait()
    second = asyncio.create_task(
        host.executor.execute(LLMWorkItem(item_id="waiting", strategy=strategy, prompt="waiting"))
    )
    await _wait_until(lambda: gate.token_available is not None and gate.token_available <= 70.1)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert strategy.started_prompts == ["hold"]
    assert gate.token_available == pytest.approx(90, abs=0.01)

    strategy.release.set()
    assert (await first).success
    assert gate.token_available == pytest.approx(100)
    await host.aclose()


@pytest.mark.asyncio
async def test_external_cancellation_after_provider_start_retains_unknown_estimate() -> None:
    strategy = _BlockingStrategy()
    host: ExecutorHost[str, str, None] = ExecutorHost(
        _config(
            token_estimator=lambda prompt, **kwargs: TokenEstimate(10), max_tokens_per_minute=100
        ),
        strategy=strategy,
    )
    gate = host._admission_registry.resolve(strategy).quota_gate
    task = asyncio.create_task(
        host.executor.execute(LLMWorkItem(item_id="hold", strategy=strategy, prompt="hold"))
    )
    await strategy.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gate.token_available == pytest.approx(90, abs=0.01)
    assert host._stats.unknown_usage_attempts == 1
    await host.aclose()


@pytest.mark.asyncio
async def test_attempt_timeout_after_provider_start_retains_unknown_estimate() -> None:
    strategy = _BlockingStrategy()
    host: ExecutorHost[str, str, None] = ExecutorHost(
        _config(
            attempt_timeout=0.01,
            retry=RetryConfig(max_attempts=1),
            token_estimator=lambda prompt, **kwargs: TokenEstimate(10),
            max_tokens_per_minute=100,
        ),
        strategy=strategy,
    )
    gate = host._admission_registry.resolve(strategy).quota_gate
    try:
        result = await host.executor.execute(
            LLMWorkItem(item_id="hold", strategy=strategy, prompt="hold")
        )
        assert not result.success
        assert result.error_category == "framework_timeout"
        assert result.timing.attempts[0].reported_tokens is None
        assert gate.token_available == pytest.approx(90, abs=0.1)
        assert host._stats.unknown_usage_attempts == 1
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_item_deadline_during_quota_wait_starts_no_second_provider_call() -> None:
    strategy = _SequenceStrategy(
        [{"total_tokens": 10}, {"total_tokens": 10}],
        estimate=TokenEstimate(10),
    )
    host: ExecutorHost[str, str, None] = ExecutorHost(
        _config(
            max_tokens_per_minute=10,
            guardrails=GuardrailConfig(total_timeout_per_item=0.02),
        ),
        strategy=strategy,
    )
    gate = host._admission_registry.resolve(strategy).quota_gate
    try:
        first = await host.executor.execute(
            LLMWorkItem(item_id="first", strategy=strategy, prompt="first")
        )
        second = await host.executor.execute(
            LLMWorkItem(item_id="second", strategy=strategy, prompt="second")
        )
        assert first.success
        assert not second.success
        assert second.error_category == "framework_total_item_timeout"
        assert second.quota_wait_seconds >= 0.01
        assert strategy.calls == 1
        assert gate.waiter_count == 0
        assert not gate.has_wake_task
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_batch_abort_during_quota_wait_starts_no_waiting_provider_call() -> None:
    strategy = _BlockingStrategy()
    result = await asyncio.wait_for(
        process_prompts(
            strategy,
            [("first", "hold"), ("second", "waiting")],
            config=_config(
                max_workers=2,
                max_tokens_per_minute=10,
                token_estimator=lambda prompt, **kwargs: TokenEstimate(10),
                guardrails=GuardrailConfig(
                    batch_timeout=0.02,
                    abort_mode=AbortMode.CANCEL_ACTIVE,
                ),
            ),
            preserve_order=True,
        ),
        timeout=0.5,
    )

    assert strategy.started_prompts == ["hold"]
    assert result.total_items == 2
    assert all(item.error_category == "batch_deadline_exceeded" for item in result.results)


class _ReplaceMiddleware(BaseMiddleware[str, str, None]):
    def __init__(self, strategy: LLMCallStrategy[str]) -> None:
        self.strategy = strategy

    async def before_process(
        self, work_item: LLMWorkItem[str, str, None]
    ) -> LLMWorkItem[str, str, None] | None:
        return LLMWorkItem(
            item_id=work_item.item_id,
            strategy=self.strategy,
            prompt="middleware prompt",
            context=work_item.context,
        )


@pytest.mark.asyncio
async def test_estimation_uses_middleware_prompt_and_effective_strategy() -> None:
    original = _SequenceStrategy([{"total_tokens": 1}], estimate=TokenEstimate(99))
    replacement = _SequenceStrategy([{"total_tokens": 1}], estimate=TokenEstimate(2))
    processor = ParallelBatchProcessor[str, str, None](
        config=_config(max_tokens_per_minute=10),
        middlewares=[_ReplaceMiddleware(replacement)],
    )
    await processor.add_work(LLMWorkItem(item_id="x", strategy=original, prompt="original"))
    try:
        batch = await processor.process_all()
    finally:
        await processor.cleanup()
    assert batch.results[0].success
    assert original.calls == 0 and original.estimate_calls == []
    assert replacement.calls == 1
    assert replacement.estimate_calls[0][0] == "middleware prompt"


@pytest.mark.asyncio
async def test_validation_retry_gets_fresh_reservation_and_estimator_sees_retry_state() -> None:
    class EscalatingStrategy(_SequenceStrategy):
        async def on_error(
            self, exception: Exception, attempt: int, state: RetryState | None = None
        ) -> None:
            assert state is not None
            state.set("model", "escalated")
            await super().on_error(exception, attempt, state)

    strategy = EscalatingStrategy([RuntimeError("result validation failed"), {"total_tokens": 3}])
    seen: list[tuple[int, object]] = []

    def estimator(
        prompt: str, *, strategy: Any, attempt: int, state: RetryState | None
    ) -> TokenEstimate:
        seen.append((attempt, state.get("model") if state is not None else None))
        return TokenEstimate(5)

    result = await call_result(strategy, "x", config=_config(token_estimator=estimator))

    assert result.success
    assert seen == [(1, None), (2, "escalated")]
    assert [attempt.reserved_tokens for attempt in result.timing.attempts] == [5, 5]
    assert result.timing.attempts[0].reported_tokens is None


class _SkipMiddleware(BaseMiddleware[str, str, None]):
    async def before_process(
        self, work_item: LLMWorkItem[str, str, None]
    ) -> LLMWorkItem[str, str, None] | None:
        return None


class _InvalidAfterMiddleware(BaseMiddleware[str, str, None]):
    async def after_process(self, result: WorkItemResult[str, None]) -> WorkItemResult[str, None]:
        return None  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_failure_after_provider_reconciliation_preserves_known_usage() -> None:
    strategy = _SequenceStrategy(
        [{"input_tokens": 4, "output_tokens": 3}],
        estimate=TokenEstimate(10),
    )
    processor = ParallelBatchProcessor[str, str, None](
        config=_config(retry=RetryConfig(max_attempts=1)),
        middlewares=[_InvalidAfterMiddleware()],
    )
    await processor.add_work(LLMWorkItem(item_id="x", strategy=strategy, prompt="x"))
    try:
        batch = await processor.process_all()
    finally:
        await processor.cleanup()

    result = batch.results[0]
    assert not result.success
    assert strategy.calls == 1
    assert result.token_usage["total_tokens"] == 7
    assert result.timing.attempts[0].reported_tokens == 7
    assert result.timing.attempts[0].reconciliation_delta_tokens == -3


@pytest.mark.asyncio
async def test_dry_run_filter_and_prepare_failure_do_not_estimate_or_reserve() -> None:
    calls = 0

    def estimator(prompt: str, **kwargs: Any) -> TokenEstimate:
        nonlocal calls
        calls += 1
        raise AssertionError("estimator must be bypassed")

    strategy = _SequenceStrategy([{"total_tokens": 1}])
    dry = await call_result(
        strategy,
        "x",
        config=_config(dry_run=True, token_estimator=estimator),
    )
    assert dry.success and calls == 0 and strategy.calls == 0

    processor = ParallelBatchProcessor[str, str, None](
        config=_config(token_estimator=estimator),
        middlewares=[_SkipMiddleware()],
    )
    await processor.add_work(LLMWorkItem(item_id="skip", strategy=strategy, prompt="x"))
    try:
        skipped = await processor.process_all()
    finally:
        await processor.cleanup()
    assert not skipped.results[0].success and calls == 0

    class BadPrepare(_SequenceStrategy):
        async def prepare(self) -> None:
            raise RuntimeError("prepare failed")

    bad = BadPrepare([{"total_tokens": 1}])
    prepared = await call_result(bad, "x", config=_config(token_estimator=estimator))
    assert not prepared.success and calls == 0 and bad.calls == 0


@pytest.mark.asyncio
async def test_replay_bypasses_estimator_quota_and_provider(tmp_path: Path) -> None:
    path = tmp_path / "quota-replay.jsonl"
    identity = ArtifactIdentity(provider="test", model="quota")
    first = _SequenceStrategy([{"total_tokens": 2}], estimate=TokenEstimate(3))
    initial = await process_prompts(
        first,
        [("x", "prompt")],
        config=_config(),
        artifact_store=JsonlArtifactStore(path, identity=identity),
    )
    assert initial.succeeded == 1 and first.calls == 1

    second = _SequenceStrategy([RuntimeError("provider must not run")])

    def forbidden(prompt: str, **kwargs: Any) -> TokenEstimate:
        raise AssertionError("replay must not estimate")

    replayed = await process_prompts(
        second,
        [("x", "prompt")],
        config=_config(max_tokens_per_minute=1, token_estimator=forbidden),
        artifact_store=JsonlArtifactStore(path, identity=identity),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replayed.succeeded == 1
    assert replayed.results[0].replayed_from_artifact
    assert second.calls == 0


@pytest.mark.asyncio
async def test_stream_single_and_call_pool_share_token_aware_executor() -> None:
    config = _config(token_estimator=lambda prompt, **kwargs: TokenEstimate(3))

    stream_strategy = _SequenceStrategy([{"total_tokens": 2}])
    streamed = [result async for result in process_stream(stream_strategy, ["x"], config=config)]
    assert streamed[0].timing.attempts[0].reserved_tokens == 3

    single_strategy = _SequenceStrategy([{"total_tokens": 2}])
    single = await call_result(single_strategy, "x", config=config)
    assert single.timing.attempts[0].reported_tokens == 2

    pool_strategy = _SequenceStrategy([{"total_tokens": 2}])
    async with LLMCallPool(pool_strategy, config=config) as pool:
        pooled = await pool.submit_result("x")
    assert pooled.timing.attempts[0].reconciliation_delta_tokens == -1

    gateway_strategy = _SequenceStrategy([{"total_tokens": 2}])
    async with LLMGateway(gateway_strategy, config=config) as gateway:
        gateway_result = await gateway.submit_result("x")
    assert gateway_result.timing.attempts[0].reserved_tokens == 3


def test_framework_estimation_errors_are_public_non_retryable_categories() -> None:
    assert issubclass(TokenEstimationError, Exception)


@pytest.mark.asyncio
async def test_metrics_observer_aggregates_quota_without_high_cardinality_labels() -> None:
    observer = MetricsObserver()
    await observer.on_event(
        ProcessingEvent.QUOTA_ADMITTED,
        {
            "item_id": "must-not-be-a-label",
            "quota_scope_id": 1,
            "wait_seconds": 2.0,
            "estimated_input_tokens": 6,
            "estimated_output_tokens": 4,
            "reserved_tokens": 10,
        },
    )
    await observer.on_event(
        ProcessingEvent.QUOTA_RECONCILED,
        {"reserved_tokens": 10, "reported_tokens": 7, "delta_tokens": -3},
    )
    await observer.on_event(
        ProcessingEvent.QUOTA_ADMITTED,
        {
            "item_id": "also-private",
            "quota_scope_id": 2,
            "wait_seconds": 0.0,
            "estimated_input_tokens": 5,
            "estimated_output_tokens": 0,
            "reserved_tokens": 5,
        },
    )
    await observer.on_event(
        ProcessingEvent.QUOTA_RECONCILED,
        {"reserved_tokens": 5, "reported_tokens": None, "delta_tokens": None},
    )

    metrics = await observer.get_metrics()
    assert metrics["quota_admitted_attempts"] == 2
    assert metrics["quota_wait_seconds_sum"] == 2
    assert metrics["quota_wait_seconds_max"] == 2
    assert metrics["reserved_tokens"] == 15
    assert metrics["reported_reconciliation_tokens"] == 7
    assert metrics["refunded_tokens"] == 3
    assert metrics["unknown_usage_attempts"] == 1
    assert metrics["quota_scope_count"] == 2

    exported = await observer.export_json()
    prometheus = await observer.export_prometheus()
    assert "must-not-be-a-label" not in exported + prometheus
    assert "also-private" not in exported + prometheus
    assert "quota_scope_id=" not in prometheus
    assert "async_batch_llm_reserved_tokens 15" in prometheus


def test_quota_wait_samples_are_bounded_and_summary_is_quiet_when_disabled() -> None:
    stats = ProcessingStats()
    for index in range(25):
        stats.record_quota_admission(
            wait_seconds=float(index),
            estimated_input_tokens=1,
            estimated_output_tokens=1,
            scope_count=3,
            sample_limit=10,
        )
    snapshot = stats.copy()
    assert len(stats._quota_wait_samples) == 10
    assert snapshot["quota_scope_count"] == 3
    assert snapshot["quota_wait_p50_seconds"] >= 15

    plain = BatchResult(
        results=[
            WorkItemResult(
                item_id="plain",
                success=True,
                timing=WorkItemTiming(attempts=[AttemptTiming(attempt=1, try_number=1)]),
            )
        ]
    ).summary()
    assert "quota wait" not in plain
    assert "quota tokens" not in plain

    quota = BatchResult(
        results=[
            WorkItemResult(
                item_id="quota",
                success=True,
                timing=WorkItemTiming(
                    attempts=[
                        AttemptTiming(
                            attempt=1,
                            try_number=1,
                            quota_wait_seconds=0.25,
                            reserved_tokens=10,
                            reported_tokens=7,
                            reconciliation_delta_tokens=-3,
                            quota_scope_id=1,
                        )
                    ]
                ),
            )
        ]
    ).summary()
    assert "quota wait" in quota
    assert "reserved 10" in quota
    assert "reported 7" in quota

    refunded_before_start = BatchResult(
        results=[
            WorkItemResult(
                item_id="cancelled",
                success=False,
                timing=WorkItemTiming(
                    attempts=[
                        AttemptTiming(
                            attempt=1,
                            try_number=1,
                            reserved_tokens=10,
                            reported_tokens=None,
                            reconciliation_delta_tokens=-10,
                            quota_scope_id=1,
                        )
                    ]
                ),
            )
        ]
    ).summary()
    assert "refunded 10" in refunded_before_start
    assert "unknown 0" in refunded_before_start
