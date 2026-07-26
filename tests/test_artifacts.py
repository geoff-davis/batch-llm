"""Versioned JSONL artifacts, privacy defaults, and compatible replay."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from async_batch_llm import (
    ArtifactFormatError,
    ArtifactIdentity,
    ArtifactIOError,
    JsonlArtifactStore,
    LLMWorkItem,
    ProcessorConfig,
    ResumePolicy,
    SqliteArtifactStore,
    WorkItemResult,
    process_prompts,
    process_stream,
)
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy


class _CountingStrategy(LLMCallStrategy[str]):
    def __init__(self, *, failures: set[str] | None = None, delay: float = 0.0) -> None:
        self.failures = failures or set()
        self.delay = delay
        self.calls: list[str] = []

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, dict[str, Any]]:
        self.calls.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        if prompt in self.failures:
            raise ValueError(f"bad prompt: {prompt}")
        return (
            prompt.upper(),
            {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            {"provider_request_id": f"req-{prompt}"},
        )


def _identity(**changes: str) -> ArtifactIdentity:
    values = {
        "provider": "test-provider",
        "model": "test-model",
        "prompt_version": "prompt-v1",
        "parser_version": "parser-v1",
        "application_version": "app-v1",
    }
    values.update(changes)
    return ArtifactIdentity(**values)


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
@pytest.mark.asyncio
async def test_inspection_restores_explicitly_persisted_context(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"stored-context{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [("item", "prompt", {"tenant": "current", "values": [1, 2]})],
        artifact_store=store_type(path, identity=_identity(), include_context=True),
    )

    if store_type is JsonlArtifactStore:
        materialized = store_type.read_results(path)
    else:
        materialized = await store_type.read_results(path)
    assert materialized.results[0].context == {
        "tenant": "current",
        "values": [1, 2],
    }

    reader = store_type(path)
    try:
        streamed = [result async for result in reader.iter_results()]
    finally:
        await reader.close()
    assert streamed[0].context == {"tenant": "current", "values": [1, 2]}


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
@pytest.mark.asyncio
async def test_replay_uses_current_context_without_decoding_historical_context(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"replay-context{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [
            ("dummy", "dummy", {"run": "historical"}),
            ("item", "prompt", {"run": "historical"}),
        ],
        artifact_store=store_type(
            path,
            identity=_identity(),
            include_context=True,
            context_in_identity=False,
        ),
    )

    decoder_calls: list[Any] = []

    def context_decoder(value: Any) -> Any:
        decoder_calls.append(value)
        raise AssertionError("historical context must not be decoded during replay")

    current_context = {"run": "current"}
    replay = _CountingStrategy()
    result = await process_prompts(
        replay,
        [("item", "prompt", current_context)],
        artifact_store=store_type(
            path,
            identity=_identity(),
            context_in_identity=False,
            context_decoder=context_decoder,
        ),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replay.calls == []
    assert decoder_calls == []
    assert result.results[0].context is current_context
    assert result.results[0].submission_index == 0


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
@pytest.mark.asyncio
async def test_inspection_applies_custom_context_decoder(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    class ContextPayload:
        def __init__(self, value: str) -> None:
            self.value = value

    def encoder(value: Any) -> Any:
        if isinstance(value, ContextPayload):
            return {"payload": value.value}
        return value

    decoded: list[Any] = []

    def decoder(value: Any) -> ContextPayload:
        decoded.append(value)
        return ContextPayload(value["payload"])

    path = tmp_path / f"custom-context{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [("item", "prompt", ContextPayload("persisted"))],
        artifact_store=store_type(
            path,
            identity=_identity(),
            include_context=True,
            encoder=encoder,
        ),
    )

    if store_type is JsonlArtifactStore:
        materialized = store_type.read_results(path, context_decoder=decoder)
    else:
        materialized = await store_type.read_results(path, context_decoder=decoder)
    restored = materialized.results[0].context
    assert isinstance(restored, ContextPayload)
    assert restored.value == "persisted"

    reader = store_type(path, context_decoder=decoder)
    try:
        streamed = [result async for result in reader.iter_results()]
    finally:
        await reader.close()
    streamed_context = streamed[0].context
    assert isinstance(streamed_context, ContextPayload)
    assert streamed_context.value == "persisted"
    assert decoded == [{"payload": "persisted"}, {"payload": "persisted"}]


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
@pytest.mark.asyncio
async def test_inspection_does_not_decode_context_that_was_not_persisted(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"omitted-context{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [
            ("omitted", "one", {"private": True}),
            ("none", "two", None),
        ],
        artifact_store=store_type(path, identity=_identity(), include_context=False),
    )

    decoder_calls: list[Any] = []

    def decoder(value: Any) -> Any:
        decoder_calls.append(value)
        return value

    if store_type is JsonlArtifactStore:
        materialized = store_type.read_results(path, context_decoder=decoder)
    else:
        materialized = await store_type.read_results(path, context_decoder=decoder)
    assert [result.context for result in materialized.results] == [None, None]
    assert decoder_calls == []


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
@pytest.mark.asyncio
async def test_original_none_context_does_not_invoke_decoder(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"none-context{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [("none", "prompt", None)],
        artifact_store=store_type(path, identity=_identity(), include_context=True),
    )
    decoder_calls: list[Any] = []

    def decoder(value: Any) -> Any:
        decoder_calls.append(value)
        return value

    if store_type is JsonlArtifactStore:
        materialized = store_type.read_results(path, context_decoder=decoder)
    else:
        materialized = await store_type.read_results(path, context_decoder=decoder)
    assert materialized.results[0].context is None
    assert decoder_calls == []


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
@pytest.mark.asyncio
async def test_context_decoder_failure_uses_artifact_format_error(
    tmp_path: Path, store_type: Any, suffix: str
) -> None:
    path = tmp_path / f"context-decoder-error{suffix}"
    await process_prompts(
        _CountingStrategy(),
        [("item", "prompt", {"stored": True})],
        artifact_store=store_type(path, identity=_identity(), include_context=True),
    )

    def fail_decoder(value: Any) -> Any:
        raise RuntimeError(f"cannot decode {value!r}")

    with pytest.raises(ArtifactFormatError, match="Stored result decoder failed"):
        if store_type is JsonlArtifactStore:
            store_type.read_results(path, context_decoder=fail_decoder)
        else:
            await store_type.read_results(path, context_decoder=fail_decoder)


@pytest.mark.asyncio
async def test_manifest_item_records_privacy_and_optional_cost(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    store = JsonlArtifactStore(
        path,
        identity=_identity(),
        user_metadata={"run": "unit"},
        cost_calculator=lambda result: result.token_usage["total_tokens"] * 0.01,
    )
    result = await process_prompts(
        _CountingStrategy(),
        [("a", "secret prompt", {"customer": "secret context"})],
        artifact_store=store,
    )

    records = _records(path)
    assert records[0]["record_type"] == "manifest"
    assert records[0]["artifact_schema_version"] == 1
    assert records[0]["identity_fingerprint"]
    assert records[0]["user_metadata"] == {"run": "unit"}
    assert len(records) == 2
    item = records[1]
    assert item["record_type"] == "item"
    assert item["prompt_fingerprint"] and item["context_fingerprint"]
    assert item["raw_prompt"] is None
    assert item["raw_context"] is None
    assert item["calculated_cost"] == pytest.approx(0.03)
    # Output/metadata can themselves contain sensitive values (this fake
    # strategy echoes its prompt); privacy defaults specifically omit the raw
    # input fields while retaining caller-requested output for replay.
    assert "secret context" not in path.read_text(encoding="utf-8")
    assert result.results[0].submission_index == 0


@pytest.mark.asyncio
async def test_explicit_prompt_and_context_inclusion(tmp_path: Path) -> None:
    path = tmp_path / "included.jsonl"
    store = JsonlArtifactStore(
        path,
        identity=_identity(),
        include_prompt=True,
        include_context=True,
    )
    await process_prompts(
        _CountingStrategy(),
        [("a", "visible prompt", {"visible": True})],
        artifact_store=store,
    )
    item = _records(path)[1]
    assert item["raw_prompt"] == "visible prompt"
    assert item["raw_context"] == {"visible": True}


@pytest.mark.asyncio
async def test_concurrent_workers_write_complete_non_interleaved_records(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.jsonl"
    store = JsonlArtifactStore(path, identity=_identity())
    prompts = [(str(index), f"p{index}") for index in range(40)]
    result = await process_prompts(
        _CountingStrategy(delay=0.001),
        prompts,
        config=ProcessorConfig(max_workers=10),
        artifact_store=store,
    )

    records = _records(path)
    assert result.total_items == 40
    assert len(records) == 41
    assert all(record["record_type"] == "item" for record in records[1:])
    assert sorted(record["record_sequence"] for record in records[1:]) == list(range(40))


@pytest.mark.asyncio
async def test_cancelled_append_retains_write_lock_until_thread_finishes(tmp_path: Path) -> None:
    path = tmp_path / "cancelled-write.jsonl"
    store = JsonlArtifactStore(path, identity=_identity())
    strategy = _CountingStrategy()
    first_item = LLMWorkItem(item_id="first", strategy=strategy, prompt="one")
    second_item = LLMWorkItem(item_id="second", strategy=strategy, prompt="two")
    first_key = await store.prepare_item(first_item)
    second_key = await store.prepare_item(second_item)

    original_write = store._write_record_sync
    first_entered = threading.Event()
    release_first = threading.Event()
    item_writes = 0

    def blocking_write(record: dict[str, Any]) -> None:
        nonlocal item_writes
        if record.get("record_type") == "item":
            item_writes += 1
            if item_writes == 1:
                first_entered.set()
                assert release_first.wait(timeout=1)
        original_write(record)

    store._write_record_sync = blocking_write  # type: ignore[method-assign]
    first_append = asyncio.create_task(
        store.append(
            first_item,
            first_key,
            WorkItemResult(item_id="first", success=True, output="ONE"),
        )
    )
    assert await asyncio.to_thread(first_entered.wait, 0.5)
    first_append.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_append

    second_append = asyncio.create_task(
        store.append(
            second_item,
            second_key,
            WorkItemResult(item_id="second", success=True, output="TWO"),
        )
    )
    await asyncio.sleep(0.02)
    assert item_writes == 1

    release_first.set()
    await asyncio.wait_for(second_append, timeout=0.5)
    await store.close()

    records = _records(path)
    assert [record.get("item_id") for record in records[1:]] == ["first", "second"]


@pytest.mark.asyncio
async def test_cancelled_append_io_failure_surfaces_on_close(tmp_path: Path) -> None:
    path = tmp_path / "cancelled-write-error.jsonl"
    store = JsonlArtifactStore(path, identity=_identity())
    item = LLMWorkItem(item_id="item", strategy=_CountingStrategy(), prompt="prompt")
    key = await store.prepare_item(item)
    entered = threading.Event()
    release = threading.Event()

    def failing_write(record: dict[str, Any]) -> None:
        entered.set()
        assert release.wait(timeout=1)
        raise OSError("detached disk failure")

    store._write_record_sync = failing_write  # type: ignore[method-assign]
    append = asyncio.create_task(
        store.append(
            item,
            key,
            WorkItemResult(item_id="item", success=True, output="result"),
        )
    )
    assert await asyncio.to_thread(entered.wait, 0.5)
    append.cancel()
    with pytest.raises(asyncio.CancelledError):
        await append
    release.set()

    with pytest.raises(ArtifactIOError, match="detached disk failure"):
        await store.close()


@pytest.mark.asyncio
async def test_sensitive_identity_values_affect_hash_without_being_persisted(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / "context-secret.jsonl"
    await process_prompts(
        _CountingStrategy(),
        [("item", "prompt", {"api_key": "context-old"})],
        artifact_store=JsonlArtifactStore(context_path, identity=_identity()),
    )
    context_replay = _CountingStrategy()
    context_result = await process_prompts(
        context_replay,
        [("item", "prompt", {"api_key": "context-new"})],
        artifact_store=JsonlArtifactStore(context_path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert context_replay.calls == ["prompt"]
    assert not context_result.results[0].replayed_from_artifact

    identity_path = tmp_path / "identity-secret.jsonl"
    old_identity = ArtifactIdentity(provider="test", model="model", extra={"api_key": "old"})
    new_identity = ArtifactIdentity(provider="test", model="model", extra={"api_key": "new"})
    await process_prompts(
        _CountingStrategy(),
        [("item", "prompt")],
        artifact_store=JsonlArtifactStore(identity_path, identity=old_identity),
    )
    identity_replay = _CountingStrategy()
    identity_result = await process_prompts(
        identity_replay,
        [("item", "prompt")],
        artifact_store=JsonlArtifactStore(identity_path, identity=new_identity),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert identity_replay.calls == ["prompt"]
    assert not identity_result.results[0].replayed_from_artifact

    persisted = context_path.read_text(encoding="utf-8") + identity_path.read_text(encoding="utf-8")
    assert "context-old" not in persisted
    assert "context-new" not in persisted
    assert '"api_key":"old"' not in persisted
    assert '"api_key":"new"' not in persisted


@pytest.mark.asyncio
async def test_checkpoint_exists_before_stream_result_is_observed(tmp_path: Path) -> None:
    path = tmp_path / "stream.jsonl"
    store = JsonlArtifactStore(path, identity=_identity())

    async for result in process_stream(_CountingStrategy(), [("a", "one")], artifact_store=store):
        records = _records(path)
        assert records[-1]["item_id"] == result.item_id
        assert records[-1]["result"]["output"] == result.output


@pytest.mark.asyncio
async def test_success_resume_avoids_provider_and_does_not_append_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "resume.jsonl"
    first = _CountingStrategy()
    await process_prompts(
        first,
        [("a", "one"), ("b", "two")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
    )
    lines_before = len(path.read_text(encoding="utf-8").splitlines())

    replay = _CountingStrategy()
    result = await process_prompts(
        replay,
        [("b", "two"), ("a", "one")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
        preserve_order=True,
    )

    assert replay.calls == []
    assert [item.item_id for item in result.results] == ["b", "a"]
    assert [item.submission_index for item in result.results] == [0, 1]
    assert all(item.replayed_from_artifact for item in result.results)
    assert len(path.read_text(encoding="utf-8").splitlines()) == lines_before


@pytest.mark.asyncio
async def test_failure_policy_reruns_or_reuses_prior_failure(tmp_path: Path) -> None:
    rerun_path = tmp_path / "rerun.jsonl"
    await process_prompts(
        _CountingStrategy(failures={"bad"}),
        [("x", "bad")],
        artifact_store=JsonlArtifactStore(rerun_path, identity=_identity()),
    )
    rerun = _CountingStrategy()
    rerun_result = await process_prompts(
        rerun,
        [("x", "bad")],
        artifact_store=JsonlArtifactStore(rerun_path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert rerun.calls == ["bad"]
    assert rerun_result.succeeded == 1

    reuse_path = tmp_path / "reuse-failure.jsonl"
    await process_prompts(
        _CountingStrategy(failures={"bad"}),
        [("x", "bad")],
        artifact_store=JsonlArtifactStore(reuse_path, identity=_identity()),
    )
    reuse = _CountingStrategy()
    reused_result = await process_prompts(
        reuse,
        [("x", "bad")],
        artifact_store=JsonlArtifactStore(reuse_path, identity=_identity()),
        resume=ResumePolicy.REUSE_ALL,
    )
    assert reuse.calls == []
    assert reused_result.failed == 1
    assert reused_result.results[0].replayed_from_artifact


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_entry", "second_entry", "identity_change"),
    [
        (("x", "one", {"v": 1}), ("x", "changed", {"v": 1}), {}),
        (("x", "one", {"v": 1}), ("x", "one", {"v": 2}), {}),
        (("x", "one", {"v": 1}), ("x", "one", {"v": 1}), {"model": "new-model"}),
        (
            ("x", "one", {"v": 1}),
            ("x", "one", {"v": 1}),
            {"provider": "new-provider"},
        ),
        (
            ("x", "one", {"v": 1}),
            ("x", "one", {"v": 1}),
            {"prompt_version": "prompt-v2"},
        ),
        (
            ("x", "one", {"v": 1}),
            ("x", "one", {"v": 1}),
            {"parser_version": "parser-v2"},
        ),
        (
            ("x", "one", {"v": 1}),
            ("x", "one", {"v": 1}),
            {"application_version": "app-v2"},
        ),
    ],
)
async def test_changed_input_or_identity_is_not_reused(
    tmp_path: Path,
    first_entry: tuple[str, str, dict[str, int]],
    second_entry: tuple[str, str, dict[str, int]],
    identity_change: dict[str, str],
) -> None:
    path = tmp_path / "incompatible.jsonl"
    await process_prompts(
        _CountingStrategy(),
        [first_entry],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
    )
    strategy = _CountingStrategy()
    await process_prompts(
        strategy,
        [second_entry],
        artifact_store=JsonlArtifactStore(path, identity=_identity(**identity_change)),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert strategy.calls == [second_entry[1]]


@pytest.mark.asyncio
async def test_newest_compatible_record_wins(tmp_path: Path) -> None:
    path = tmp_path / "newest.jsonl"
    await process_prompts(
        _CountingStrategy(failures={"x"}),
        [("id", "x")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
    )
    await process_prompts(
        _CountingStrategy(),
        [("id", "x")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    strategy = _CountingStrategy(failures={"x"})
    result = await process_prompts(
        strategy,
        [("id", "x")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_ALL,
    )
    assert strategy.calls == []
    assert result.succeeded == 1
    assert result.results[0].output == "X"


@pytest.mark.asyncio
async def test_resume_lookup_uses_prebuilt_compatibility_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "indexed.jsonl"
    prompts = [(str(index), f"p{index}") for index in range(100)]
    await process_prompts(
        _CountingStrategy(),
        prompts,
        config=ProcessorConfig(max_workers=10),
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
    )

    store = JsonlArtifactStore(path, identity=_identity())
    compatibility_checks = 0
    original = store._compatible

    def counted(*args: Any, **kwargs: Any) -> bool:
        nonlocal compatibility_checks
        compatibility_checks += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_compatible", counted)
    replay = _CountingStrategy()
    result = await process_prompts(
        replay,
        [("0", "p0")],
        artifact_store=store,
        resume=ResumePolicy.REUSE_SUCCESSES,
    )

    assert result.results[0].replayed_from_artifact
    assert replay.calls == []
    assert compatibility_checks == 1


def test_truncated_final_line_is_ignored_but_malformed_middle_fails(tmp_path: Path) -> None:
    path = tmp_path / "truncated.jsonl"
    manifest = {
        "record_type": "manifest",
        "artifact_schema_version": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "package_version": "test",
        "identity": {},
        "identity_fingerprint": "x",
        "user_metadata": {},
    }
    path.write_bytes((json.dumps(manifest) + '\n{"record_type":').encode())
    assert JsonlArtifactStore.read_results(path).results == []

    path.write_text(json.dumps(manifest) + "\nnot-json\n{}\n", encoding="utf-8")
    with pytest.raises(ArtifactFormatError, match="non-final line"):
        JsonlArtifactStore.read_results(path)

    future = dict(manifest, artifact_schema_version=999)
    path.write_text(json.dumps(future) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactFormatError, match="future"):
        JsonlArtifactStore.read_results(path)


def test_read_results_rejects_missing_and_empty_artifacts(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ArtifactIOError, match="does not exist"):
        JsonlArtifactStore.read_results(missing)

    empty = tmp_path / "empty.jsonl"
    empty.touch()
    with pytest.raises(ArtifactFormatError, match="empty"):
        JsonlArtifactStore.read_results(empty)


@pytest.mark.asyncio
async def test_audit_only_output_is_not_replayed(tmp_path: Path) -> None:
    path = tmp_path / "audit-only.jsonl"
    await process_prompts(
        _CountingStrategy(),
        [("x", "one")],
        artifact_store=JsonlArtifactStore(
            path,
            identity=_identity(),
            include_output=False,
        ),
    )
    strategy = _CountingStrategy()
    await process_prompts(
        strategy,
        [("x", "one")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert strategy.calls == ["one"]


@pytest.mark.asyncio
async def test_interrupted_stream_can_resume_completed_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.jsonl"
    first = _CountingStrategy(delay=0.005)
    stream = process_stream(
        first,
        [(str(index), f"p{index}") for index in range(6)],
        config=ProcessorConfig(max_workers=1, max_queue_size=1),
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
    )
    async for _ in stream:
        break
    await stream.aclose()

    completed = len(_records(path)) - 1
    second = _CountingStrategy()
    result = await process_prompts(
        second,
        [(str(index), f"p{index}") for index in range(6)],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert result.total_items == 6
    assert len(second.calls) == 6 - completed


@pytest.mark.asyncio
async def test_artifact_failure_surfaces_without_hanging_or_postprocessing(tmp_path: Path) -> None:
    class FailingStore:
        async def prepare_item(self, work_item: Any) -> object:
            return object()

        async def lookup(self, work_item: Any, prepared_item: Any, policy: Any) -> None:
            return None

        async def append(self, work_item: Any, prepared_item: Any, result: Any) -> None:
            raise ArtifactIOError("disk full")

        async def close(self) -> None:
            return None

    postprocessed: list[str] = []

    async def postprocess(result: WorkItemResult) -> None:
        postprocessed.append(result.item_id)

    with pytest.raises(ArtifactIOError, match="disk full"):
        await asyncio.wait_for(
            process_prompts(
                _CountingStrategy(),
                ["one", "two"],
                config=ProcessorConfig(max_workers=1),
                artifact_store=FailingStore(),  # type: ignore[arg-type]
                post_processor=postprocess,
            ),
            timeout=1,
        )
    assert postprocessed == []


def test_read_only_success_iteration_and_no_traceback(tmp_path: Path) -> None:
    path = tmp_path / "read.jsonl"

    async def write() -> None:
        await process_prompts(
            _CountingStrategy(failures={"bad"}),
            [("ok", "good"), ("failed", "bad")],
            artifact_store=JsonlArtifactStore(path, identity=_identity()),
        )

    asyncio.run(write())
    successful = JsonlArtifactStore.read_results(path, successes_only=True)
    assert [result.item_id for result in successful.results] == ["ok"]
    assert "traceback" not in path.read_text(encoding="utf-8").lower()


@pytest.mark.asyncio
async def test_iter_results_streams_without_building_replay_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stream-inspection.jsonl"
    await process_prompts(
        _CountingStrategy(),
        [(str(index), f"p{index}") for index in range(20)],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
    )

    def fail_full_read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("iter_results must not build the replay history")

    from async_batch_llm import artifacts as artifacts_module

    page_calls = 0
    original_page_reader = artifacts_module._read_artifact_page

    def track_page(*args: Any, **kwargs: Any) -> Any:
        nonlocal page_calls
        page_calls += 1
        return original_page_reader(*args, **kwargs)

    monkeypatch.setattr("async_batch_llm.artifacts._read_artifact_records", fail_full_read)
    monkeypatch.setattr("async_batch_llm.artifacts._read_artifact_page", track_page)
    store = JsonlArtifactStore(path)
    results = [result async for result in store.iter_results(successes_only=True)]
    assert [result.item_id for result in results] == [str(index) for index in range(20)]
    assert page_calls == 1
    assert store._records == []
    await store.close()


@pytest.mark.asyncio
async def test_iter_results_uses_finite_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "finite-snapshot.jsonl"
    await process_prompts(
        _CountingStrategy(),
        [("first", "one"), ("second", "two")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
    )

    reader = JsonlArtifactStore(path)
    iterator = reader.iter_results()
    first = await anext(iterator)
    await process_prompts(
        _CountingStrategy(),
        [("later", "three")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
    )
    snapshot = [first, *[result async for result in iterator]]
    assert [result.item_id for result in snapshot] == ["first", "second"]
    assert [result.item_id async for result in reader.iter_results()] == [
        "first",
        "second",
        "later",
    ]
    await reader.close()


@pytest.mark.asyncio
async def test_iter_results_rejects_malformed_middle_and_ignores_truncated_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stream-errors.jsonl"
    await process_prompts(
        _CountingStrategy(),
        [("ok", "good")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
    )
    complete = path.read_bytes()
    path.write_bytes(complete + b'{"record_type":')

    reader = JsonlArtifactStore(path)
    assert [result.item_id async for result in reader.iter_results()] == ["ok"]
    await reader.close()

    path.write_bytes(complete + b"not-json\n")
    reader = JsonlArtifactStore(path)
    iterator = reader.iter_results()
    assert (await anext(iterator)).item_id == "ok"
    with pytest.raises(ArtifactFormatError, match="non-final line"):
        await anext(iterator)
    await reader.close()

    path.write_bytes(b'{"record_type":')
    reader = JsonlArtifactStore(path)
    with pytest.raises(ArtifactFormatError, match="no complete manifest"):
        [result async for result in reader.iter_results()]
    await reader.close()


@pytest.mark.asyncio
async def test_context_free_records_use_null_and_replay_legacy_v020_hashes(
    tmp_path: Path,
) -> None:
    from async_batch_llm._internal.artifact_codec import fingerprint_json

    path = tmp_path / "legacy-null-context.jsonl"
    await process_prompts(
        _CountingStrategy(),
        [("id", "prompt")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
    )
    records = _records(path)
    assert records[1]["context_fingerprint"] is None

    legacy_context = fingerprint_json(None)
    records[1]["context_fingerprint"] = legacy_context
    records[1]["input_fingerprint"] = fingerprint_json(
        {
            "item_id": "id",
            "prompt_fingerprint": records[1]["prompt_fingerprint"],
            "context_fingerprint": legacy_context,
        }
    )
    path.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )

    replay = _CountingStrategy()
    result = await process_prompts(
        replay,
        [("id", "prompt")],
        artifact_store=JsonlArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replay.calls == []
    assert result.results[0].replayed_from_artifact


@pytest.mark.asyncio
async def test_context_free_legacy_replay_uses_custom_v020_fingerprinter(
    tmp_path: Path,
) -> None:
    from async_batch_llm._internal.artifact_codec import fingerprint_json

    def custom_fingerprinter(value: Any) -> str:
        return "custom:none" if value is None else f"custom:{value}"

    path = tmp_path / "legacy-custom-null-context.jsonl"
    await process_prompts(
        _CountingStrategy(),
        [("id", "prompt")],
        artifact_store=JsonlArtifactStore(
            path,
            identity=_identity(),
            context_fingerprinter=custom_fingerprinter,
        ),
    )
    records = _records(path)
    assert records[1]["context_fingerprint"] is None
    records[1]["context_fingerprint"] = "custom:none"
    records[1]["input_fingerprint"] = fingerprint_json(
        {
            "item_id": "id",
            "prompt_fingerprint": records[1]["prompt_fingerprint"],
            "context_fingerprint": "custom:none",
        }
    )
    path.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )

    replay = _CountingStrategy()
    result = await process_prompts(
        replay,
        [("id", "prompt")],
        artifact_store=JsonlArtifactStore(
            path,
            identity=_identity(),
            context_fingerprinter=custom_fingerprinter,
        ),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replay.calls == []
    assert result.results[0].replayed_from_artifact

    def nonnull_only_fingerprinter(value: Any) -> str:
        return f"custom:{value.name}"

    nullable_path = tmp_path / "nullable-custom-context.jsonl"
    result = await process_prompts(
        _CountingStrategy(),
        [("id", "prompt")],
        artifact_store=JsonlArtifactStore(
            nullable_path,
            identity=_identity(),
            context_fingerprinter=nonnull_only_fingerprinter,
        ),
    )
    assert result.results[0].success
    assert _records(nullable_path)[1]["context_fingerprint"] is None
