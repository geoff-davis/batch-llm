"""Bounded completed-result handoff and control-plane reliability."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from async_batch_llm import (
    ArtifactError,
    ArtifactIdentity,
    GuardrailConfig,
    JsonlArtifactStore,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessingEvent,
    ProcessorConfig,
    ResumePolicy,
    process_prompts,
    process_stream,
)
from async_batch_llm.artifacts import ArtifactStore
from async_batch_llm.base import RetryState, TokenUsage, WorkItemResult
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.observers import BaseObserver
from async_batch_llm.strategies import ErrorClassifier, ErrorInfo

_TOKENS: TokenUsage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class _CountingStrategy(LLMCallStrategy[str]):
    def __init__(self, *, delays: dict[str, float] | None = None) -> None:
        self.calls = 0
        self.call_ids: list[str] = []
        self.delays = delays or {}
        self.reached: dict[int, asyncio.Event] = {}

    def event_for(self, count: int) -> asyncio.Event:
        return self.reached.setdefault(count, asyncio.Event())

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        self.calls += 1
        self.call_ids.append(prompt)
        for count, event in self.reached.items():
            if self.calls >= count:
                event.set()
        delay = self.delays.get(prompt, 0)
        if delay:
            await asyncio.sleep(delay)
        return prompt, _TOKENS, None


def _item(index: int, strategy: LLMCallStrategy[str], prompt: str | None = None) -> LLMWorkItem:
    return LLMWorkItem(item_id=f"item-{index}", strategy=strategy, prompt=prompt or str(index))


def _queued_results(processor: ParallelBatchProcessor[Any, Any, Any]) -> int:
    assert processor._result_stream is not None
    return sum(isinstance(item, WorkItemResult) for item in processor._result_stream._queue)


def _slot_value(processor: ParallelBatchProcessor[Any, Any, Any]) -> int | None:
    return None if processor._result_slots is None else processor._result_slots._value


def test_result_queue_configuration_defaults_and_validation() -> None:
    assert ProcessorConfig().max_result_queue_size == 0
    assert ProcessorConfig(max_result_queue_size=4).max_result_queue_size == 4
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="max_result_queue_size"):
            ProcessorConfig(max_result_queue_size=value)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_default_result_handoff_remains_unbounded() -> None:
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=2, max_result_queue_size=0)
    )
    async with processor:
        processor.start()
        assert processor._result_slots is None
        for index in range(10):
            await processor.add_work(_item(index, _CountingStrategy()))
        await processor.finish()
        results = [result async for result in processor.results()]
    assert len(results) == 10


@pytest.mark.asyncio
async def test_slow_consumer_bounds_results_pauses_and_resumes_workers() -> None:
    strategy = _CountingStrategy()
    fifth_call = strategy.event_for(5)
    sixth_call = strategy.event_for(6)
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=3, max_result_queue_size=2)
    )

    async with processor:
        processor.start()
        for index in range(20):
            await processor.add_work(_item(index, strategy))
        await processor.finish()

        await asyncio.wait_for(fifth_call.wait(), timeout=1)
        await asyncio.sleep(0)
        assert strategy.calls == 5
        assert _queued_results(processor) == 2
        assert _slot_value(processor) == 0
        stats = await processor.get_stats()
        assert stats["result_queue_high_water_mark"] == 2

        stream = processor.results()
        first = await anext(stream)
        await asyncio.wait_for(sixth_call.wait(), timeout=1)
        remaining = [result async for result in stream]

    results = [first, *remaining]
    assert len(results) == 20
    assert len({result.item_id for result in results}) == 20
    assert _slot_value(processor) == 2
    assert processor._queue._unfinished_tasks == 0
    assert processor._current_queued_results == 0


@pytest.mark.asyncio
async def test_completion_order_is_preserved_with_bounded_handoff() -> None:
    strategy = _CountingStrategy(delays={"slow": 0.03})
    results = [
        result
        async for result in process_stream(
            strategy,
            [("slow-id", "slow"), ("fast-id", "fast")],
            config=ProcessorConfig(max_workers=2, max_result_queue_size=1),
        )
    ]
    assert [result.item_id for result in results] == ["fast-id", "slow-id"]


@pytest.mark.asyncio
async def test_end_of_stream_bypasses_exhausted_result_capacity() -> None:
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, max_result_queue_size=1)
    )
    async with processor:
        processor.start()
        await processor.add_work(_item(0, _CountingStrategy()))
        await processor.finish()
        assert processor._finalize_task is not None
        await asyncio.wait_for(processor._finalize_task, timeout=1)

        assert _queued_results(processor) == 1
        # Result capacity is full, but the unbounded mixed queue also contains
        # the end-of-stream control message.
        assert processor._result_stream is not None
        assert processor._result_stream.qsize() == 2
        results = [result async for result in processor.results()]
    assert len(results) == 1


class _WorkerCrash(BaseException):
    pass


class _CrashAfterPublication(LLMCallStrategy[str]):
    def __init__(self, published: asyncio.Event) -> None:
        self.published = published

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        if prompt == "crash":
            await self.published.wait()
            raise _WorkerCrash("worker crashed at full result capacity")
        return prompt, _TOKENS, None


@pytest.mark.asyncio
async def test_worker_crash_signal_bypasses_exhausted_result_capacity() -> None:
    published = asyncio.Event()
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=2, max_result_queue_size=1)
    )
    original_publish = processor._publish_stream_result

    async def publish(result: WorkItemResult[str, None]) -> None:
        await original_publish(result)
        published.set()

    processor._publish_stream_result = publish  # type: ignore[method-assign]
    strategy = _CrashAfterPublication(published)

    with pytest.raises(_WorkerCrash, match="full result capacity"):
        async with processor:
            processor.start()
            await processor.add_work(_item(0, strategy, "ok"))
            await processor.add_work(_item(1, strategy, "crash"))
            await processor.finish()
            stream = processor.results()
            assert (await anext(stream)).output == "ok"
            await anext(stream)

    assert all(worker.done() for worker in processor._workers)
    assert processor._queue._unfinished_tasks == 0
    assert _slot_value(processor) == 1
    assert processor._current_queued_results == 0


@pytest.mark.asyncio
async def test_early_exit_cancels_blocked_publishers_without_slot_or_task_leaks() -> None:
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=3, max_queue_size=4, max_result_queue_size=1)
    )
    strategy = _CountingStrategy()
    before = {task for task in asyncio.all_tasks() if not task.done()}

    async with processor:
        processor.start()

        async def feed() -> None:
            for index in range(100):
                await processor.add_work(_item(index, strategy))
            await processor.finish()

        producer = asyncio.create_task(feed())
        async for _ in processor.results():
            break
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)

    await asyncio.sleep(0)
    after = {task for task in asyncio.all_tasks() if not task.done()}
    assert after - before == set()
    assert all(worker.done() for worker in processor._workers)
    assert processor._queue._unfinished_tasks == 0
    assert _slot_value(processor) == 1


@pytest.mark.asyncio
async def test_external_cancellation_while_publisher_waits_cleans_up() -> None:
    strategy = _CountingStrategy()
    third_call = strategy.event_for(3)

    async def consume() -> None:
        async for _ in process_stream(
            strategy,
            [str(index) for index in range(20)],
            config=ProcessorConfig(
                max_workers=1,
                max_queue_size=2,
                max_result_queue_size=1,
            ),
        ):
            await asyncio.Event().wait()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(third_call.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_producer_failure_drains_bounded_results_then_propagates() -> None:
    async def source() -> AsyncIterator[str]:
        for index in range(20):
            yield str(index)
        raise RuntimeError("source failed")

    seen: list[str] = []
    with pytest.raises(RuntimeError, match="source failed"):
        async for result in process_stream(
            _CountingStrategy(),
            source(),
            config=ProcessorConfig(max_workers=3, max_result_queue_size=1),
        ):
            seen.append(result.item_id)
    assert len(seen) == 20
    assert len(set(seen)) == 20


class _RecordingStore(ArtifactStore):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.persisted: set[str] = set()

    async def prepare_item(self, work_item: LLMWorkItem[Any, Any, Any]) -> object:
        return object()

    async def lookup(
        self, work_item: LLMWorkItem[Any, Any, Any], prepared_item: Any, policy: ResumePolicy
    ) -> None:
        return None

    async def append(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        result: WorkItemResult[Any, Any],
    ) -> None:
        if self.fail:
            raise ArtifactError("checkpoint failed")
        self.persisted.add(work_item.item_id)

    async def iter_results(
        self, *, successes_only: bool = False
    ) -> AsyncIterator[WorkItemResult[Any, Any]]:
        if False:
            yield WorkItemResult(item_id="unused", success=True)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_artifact_is_persisted_before_result_publication() -> None:
    store = _RecordingStore()
    async for result in process_stream(
        _CountingStrategy(),
        ["a", "b", "c"],
        config=ProcessorConfig(max_workers=3, max_result_queue_size=1),
        artifact_store=store,
    ):
        assert result.item_id in store.persisted


@pytest.mark.asyncio
async def test_artifact_failure_does_not_consume_or_publish_result_slot() -> None:
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, max_result_queue_size=1),
        artifact_store=_RecordingStore(fail=True),
    )
    with pytest.raises(ArtifactError, match="checkpoint failed"):
        async with processor:
            processor.start()
            await processor.add_work(_item(0, _CountingStrategy()))
            await processor.finish()
            async for _ in processor.results():
                pytest.fail("artifact failure must occur before result publication")
    assert _slot_value(processor) == 1
    assert processor._queue._unfinished_tasks == 0


@pytest.mark.asyncio
async def test_replayed_results_obey_same_capacity_rule(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    identity = ArtifactIdentity(provider="fake", model="local")
    first_strategy = _CountingStrategy()
    await process_prompts(
        first_strategy,
        [(f"id-{index}", str(index)) for index in range(8)],
        artifact_store=JsonlArtifactStore(path, identity=identity),
    )
    assert first_strategy.calls == 8

    replay_strategy = _CountingStrategy()
    results = [
        result
        async for result in process_stream(
            replay_strategy,
            [(f"id-{index}", str(index)) for index in range(8)],
            config=ProcessorConfig(max_workers=3, max_result_queue_size=1),
            artifact_store=JsonlArtifactStore(path, identity=identity),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
    ]
    assert len(results) == 8
    assert all(result.replayed_from_artifact for result in results)
    assert replay_strategy.calls == 0


@pytest.mark.asyncio
async def test_batch_deadline_while_publishers_wait_does_not_deadlock() -> None:
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(
            max_workers=2,
            max_result_queue_size=1,
            guardrails=GuardrailConfig(batch_timeout=0.02),
        )
    )
    async with processor:
        processor.start()
        strategy = _CountingStrategy()
        for index in range(10):
            await processor.add_work(_item(index, strategy))
        await processor.finish()
        await asyncio.wait_for(processor.wait_for_abort(), timeout=1)
        assert strategy.calls >= 3
        assert _queued_results(processor) == 1
        results = [result async for result in processor.results()]

    assert len(results) == 10
    assert len({result.item_id for result in results}) == 10
    assert processor.termination.kind == "batch_timeout"
    assert processor._queue._unfinished_tasks == 0


class _FatalError(Exception):
    pass


class _FatalClassifier(ErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        if isinstance(exception, _FatalError):
            return ErrorInfo(False, False, False, "fatal")
        return ErrorInfo(True, False, False, "other")


class _FailFastStrategy(LLMCallStrategy[str]):
    def __init__(self, published: asyncio.Event, allow_failure: asyncio.Event) -> None:
        self.published = published
        self.allow_failure = allow_failure

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        if prompt == "trigger":
            await self.published.wait()
            await self.allow_failure.wait()
            raise _FatalError("stop the batch")
        return prompt, _TOKENS, None


@pytest.mark.asyncio
async def test_fail_fast_while_publishers_wait_does_not_deadlock_or_duplicate() -> None:
    published = asyncio.Event()
    allow_failure = asyncio.Event()
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(
            max_workers=3,
            max_result_queue_size=1,
            guardrails=GuardrailConfig(abort_on_error_categories=frozenset({"fatal"})),
        ),
        error_classifier=_FatalClassifier(),
    )
    original_publish = processor._publish_stream_result

    async def publish(result: WorkItemResult[str, None]) -> None:
        await original_publish(result)
        published.set()

    processor._publish_stream_result = publish  # type: ignore[method-assign]
    strategy = _FailFastStrategy(published, allow_failure)
    prompts = ["ok", "trigger", *[f"later-{index}" for index in range(8)]]

    async with processor:
        processor.start()
        for index, prompt in enumerate(prompts):
            await processor.add_work(_item(index, strategy, prompt))
        await processor.finish()
        allow_failure.set()
        await asyncio.wait_for(processor.wait_for_abort(), timeout=1)
        results = [result async for result in processor.results()]

    assert len(results) == len(prompts)
    assert len({result.item_id for result in results}) == len(prompts)
    assert any(result.error_category == "fatal" for result in results)
    assert processor.termination.kind == "fail_fast"
    assert processor._queue._unfinished_tasks == 0


@pytest.mark.asyncio
async def test_collection_api_remains_behaviorally_unchanged_and_soaks_thousands() -> None:
    strategy = _CountingStrategy()
    batch = await process_prompts(
        strategy,
        [str(index) for index in range(2_000)],
        config=ProcessorConfig(
            max_workers=8,
            max_queue_size=16,
            max_result_queue_size=3,
        ),
    )
    assert batch.total_items == 2_000
    assert batch.succeeded == 2_000
    assert len({result.item_id for result in batch.results}) == 2_000
    assert strategy.calls == 2_000


@pytest.mark.asyncio
async def test_input_queue_high_water_is_exact_at_enqueue_ownership_point() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingStrategy(_CountingStrategy):
        async def execute(
            self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
        ) -> tuple[str, TokenUsage, None]:
            started.set()
            await release.wait()
            return prompt, _TOKENS, None

    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, max_queue_size=2, max_result_queue_size=2)
    )
    async with processor:
        processor.start()
        strategy = BlockingStrategy()
        await processor.add_work(_item(0, strategy))
        await started.wait()
        await processor.add_work(_item(1, strategy))
        await processor.add_work(_item(2, strategy))
        stats = await processor.get_stats()
        assert stats["input_queue_high_water_mark"] == 2
        assert processor._queue.qsize() == 2

        release.set()
        await processor.finish()
        results = [result async for result in processor.results()]
        stats = await processor.get_stats()

    assert len(results) == 3
    assert stats["input_queue_high_water_mark"] == 2
    assert stats["result_queue_high_water_mark"] <= 2
    assert processor._current_queued_results == 0


@pytest.mark.asyncio
async def test_unbounded_queues_report_observed_high_water_and_ignore_control_messages() -> None:
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, max_queue_size=0, max_result_queue_size=0)
    )
    async with processor:
        strategy = _CountingStrategy()
        for index in range(5):
            await processor.add_work(_item(index, strategy))
        processor.start()
        await processor.finish()
        assert processor._finalize_task is not None
        await asyncio.wait_for(processor._finalize_task, timeout=1)

        stats = await processor.get_stats()
        assert stats["input_queue_high_water_mark"] == 5
        assert stats["result_queue_high_water_mark"] == 5
        assert processor._current_queued_results == 5
        assert processor._result_stream is not None
        assert processor._result_stream.qsize() == 6  # five data rows plus end-of-stream
        results = [result async for result in processor.results()]

    assert len(results) == 5
    assert processor._current_queued_results == 0


@pytest.mark.asyncio
async def test_queue_high_water_stats_copy_and_batch_completed_payload() -> None:
    class RecordingObserver(BaseObserver):
        def __init__(self) -> None:
            self.completed: dict[str, Any] | None = None

        async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
            if event is ProcessingEvent.BATCH_COMPLETED:
                self.completed = data

    observer = RecordingObserver()
    processor = ParallelBatchProcessor[None, str, None](
        config=ProcessorConfig(max_workers=1, max_queue_size=0, max_result_queue_size=1),
        observers=[observer],
    )
    async with processor:
        for index in range(3):
            await processor.add_work(_item(index, _CountingStrategy()))
        result = await processor.process_all()
        stats = await processor.get_stats()

    assert result.total_items == 3
    assert stats["input_queue_high_water_mark"] == 3
    assert stats["result_queue_high_water_mark"] == 0  # process_all has no handoff queue
    assert observer.completed is not None
    assert observer.completed["input_queue_high_water_mark"] == 3
    assert observer.completed["result_queue_high_water_mark"] == 0

    fresh = ParallelBatchProcessor[None, str, None](config=ProcessorConfig())
    fresh_stats = await fresh.get_stats()
    assert fresh_stats["input_queue_high_water_mark"] == 0
    assert fresh_stats["result_queue_high_water_mark"] == 0
