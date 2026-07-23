"""Indexed SQLite artifact storage, replay, batching, and lifecycle."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import async_batch_llm
from async_batch_llm import (
    ArtifactFormatError,
    ArtifactIdentity,
    ArtifactIOError,
    LLMWorkItem,
    ProcessorConfig,
    ResumePolicy,
    RetryConfig,
    SqliteArtifactStore,
    SqliteDurability,
    WorkItemResult,
    process_prompts,
    process_stream,
)
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.sqlite_artifacts import SQLITE_APPLICATION_ID, SQLITE_SCHEMA_VERSION


class _CountingStrategy(LLMCallStrategy[str]):
    def __init__(self, *, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[str] = []

    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, dict[str, Any]]:
        self.calls.append(prompt)
        if prompt in self.failures:
            raise ValueError(f"bad prompt: {prompt}")
        return (
            prompt.upper(),
            {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            {"request_id": f"req-{prompt}"},
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


def _item(item_id: str, prompt: str, *, context: Any = None) -> LLMWorkItem[Any, str, Any]:
    return LLMWorkItem(
        item_id=item_id,
        strategy=_CountingStrategy(),
        prompt=prompt,
        context=context,
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("commit_batch_size", 0),
        ("commit_batch_size", True),
        ("commit_interval_seconds", -1),
        ("commit_interval_seconds", True),
        ("commit_interval_seconds", float("inf")),
        ("busy_timeout_seconds", -1),
        ("busy_timeout_seconds", float("nan")),
        ("read_batch_size", 0),
        ("read_batch_size", False),
    ],
)
def test_constructor_validation(tmp_path: Path, keyword: str, value: Any) -> None:
    with pytest.raises(ValueError, match=keyword):
        SqliteArtifactStore(tmp_path / "run.sqlite", **{keyword: value})


def test_constructor_rejects_memory_and_unknown_durability(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filesystem path"):
        SqliteArtifactStore(":memory:")
    with pytest.raises(ValueError, match="durability"):
        SqliteArtifactStore(tmp_path / "run.sqlite", durability="unsafe")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="URI"):
        SqliteArtifactStore("file::memory:?cache=shared")


def test_public_exports() -> None:
    assert async_batch_llm.SqliteArtifactStore is SqliteArtifactStore
    assert async_batch_llm.SqliteDurability is SqliteDurability
    assert "SqliteArtifactStore" in async_batch_llm.__all__
    assert "SqliteDurability" in async_batch_llm.__all__


@pytest.mark.asyncio
async def test_creation_schema_manifest_indexes_and_pragmas(tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    store = SqliteArtifactStore(path, identity=_identity(), user_metadata={"run": "unit"})
    await store.prepare_item(_item("one", "prompt"))

    def inspect_database() -> dict[str, Any]:
        connection = store._require_connection()
        return {
            "application_id": connection.execute("PRAGMA application_id").fetchone()[0],
            "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
            "journal_mode": connection.execute("PRAGMA journal_mode").fetchone()[0],
            "synchronous": connection.execute("PRAGMA synchronous").fetchone()[0],
            "auto": connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0],
            "manifest": tuple(connection.execute("SELECT * FROM manifest").fetchone()),
            "tables": {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            },
            "indexes": {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
            },
        }

    details = await store._run_db(inspect_database)
    assert details["application_id"] == SQLITE_APPLICATION_ID
    assert details["user_version"] == SQLITE_SCHEMA_VERSION
    assert details["journal_mode"] == "wal"
    assert details["synchronous"] == 1  # SQLite NORMAL
    assert details["auto"] == 1000
    assert details["manifest"][1:3] == (1, 1)
    assert json.loads(details["manifest"][7]) == {"run": "unit"}
    assert {"manifest", "identities", "item_records"} <= details["tables"]
    assert {
        "idx_item_records_replay_all",
        "idx_item_records_replay_success",
        "idx_item_records_success_sequence",
    } <= details["indexes"]
    await store.close()


@pytest.mark.asyncio
async def test_lifecycle_does_not_depend_on_asyncio_shield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avoid CPython 3.14 completion races for owned store work."""

    def fail_shield(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("SqliteArtifactStore lifecycle must not use asyncio.shield")

    def fail_to_thread(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("SqliteArtifactStore close must not create a default executor")

    monkeypatch.setattr(asyncio, "shield", fail_shield)
    monkeypatch.setattr(asyncio, "to_thread", fail_to_thread)
    store = SqliteArtifactStore(tmp_path / "no-shield.sqlite", identity=_identity())
    prepared = await store.prepare_item(_item("one", "prompt"))
    assert prepared.input_fingerprint
    await store.close()


@pytest.mark.asyncio
async def test_full_durability_uses_full_synchronous(tmp_path: Path) -> None:
    store = SqliteArtifactStore(
        tmp_path / "full.sqlite",
        identity=_identity(),
        durability=SqliteDurability.FULL,
    )
    await store.prepare_item(_item("one", "prompt"))
    synchronous = await store._run_db(
        lambda: store._require_connection().execute("PRAGMA synchronous").fetchone()[0]
    )
    assert synchronous == 2  # SQLite FULL
    await store.close()


@pytest.mark.asyncio
async def test_privacy_logical_fields_cost_and_checkpoint_before_publication(
    tmp_path: Path,
) -> None:
    path = tmp_path / "privacy.sqlite"
    store = SqliteArtifactStore(
        path,
        identity=_identity(),
        cost_calculator=lambda result: result.token_usage["total_tokens"] * 0.01,
    )
    async for result in process_stream(
        _CountingStrategy(),
        [("one", "secret prompt", {"customer": "secret context"})],
        artifact_store=store,
    ):
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT item_id, raw_prompt_json, raw_context_json, calculated_cost, result_json "
                "FROM item_records"
            ).fetchone()
        assert row[:3] == (result.item_id, None, None)
        assert row[3] == pytest.approx(0.03)
        assert json.loads(row[4])["output"] == result.output
    assert "secret context" not in path.read_bytes().decode("utf-8", errors="ignore")


@pytest.mark.asyncio
async def test_success_and_failure_replay_policies_and_live_token_accounting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replay.sqlite"
    config = ProcessorConfig(retry=RetryConfig(max_attempts=1))
    await process_prompts(
        _CountingStrategy(failures={"bad"}),
        [("ok", "good"), ("failed", "bad")],
        config=config,
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
    )

    reuse_success = _CountingStrategy()
    store = SqliteArtifactStore(path, identity=_identity())
    result = await process_prompts(
        reuse_success,
        [("failed", "bad"), ("ok", "good")],
        artifact_store=store,
        resume=ResumePolicy.REUSE_SUCCESSES,
        preserve_order=True,
    )
    assert reuse_success.calls == ["bad"]
    assert not result.results[0].replayed_from_artifact
    assert result.results[1].replayed_from_artifact

    reuse_all = _CountingStrategy(failures={"bad"})
    processor_store = SqliteArtifactStore(path, identity=_identity())
    result = await process_prompts(
        reuse_all,
        [("failed", "bad"), ("ok", "good")],
        artifact_store=processor_store,
        resume=ResumePolicy.REUSE_ALL,
    )
    assert reuse_all.calls == []
    assert all(item.replayed_from_artifact for item in result.results)


@pytest.mark.asyncio
async def test_null_and_non_null_context_replay_use_partial_indexes(tmp_path: Path) -> None:
    path = tmp_path / "nullable.sqlite"
    await process_prompts(
        _CountingStrategy(),
        [("none", "a"), ("context", "b", {"v": 1})],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
    )
    replay = _CountingStrategy()
    result = await process_prompts(
        replay,
        [("none", "a"), ("context", "b", {"v": 1})],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replay.calls == []
    assert all(item.replayed_from_artifact for item in result.results)

    source = inspect.getsource(SqliteArtifactStore._lookup_sync)
    assert "context_fingerprint IS ?" in source
    assert "context_fingerprint = ?" not in source
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT identity_fingerprint, item_id, prompt_fingerprint, "
            "context_fingerprint, input_fingerprint FROM item_records ORDER BY record_sequence"
        ).fetchall()
        assert rows[0][3] is None
        for row in rows:
            for success_clause, expected_index in (
                (" AND success = 1", "idx_item_records_replay_success"),
                ("", "idx_item_records_replay_all"),
            ):
                plan = connection.execute(
                    f"""
                    EXPLAIN QUERY PLAN
                    SELECT record_sequence FROM item_records
                     WHERE identity_fingerprint = ? AND item_id = ?
                       AND prompt_fingerprint = ? AND context_fingerprint IS ?
                       AND input_fingerprint = ? AND replay_eligible = 1{success_clause}
                     ORDER BY record_sequence DESC LIMIT 1
                    """,
                    row,
                ).fetchall()
                assert expected_index in " ".join(str(part) for part in plan[0])


@pytest.mark.asyncio
async def test_changed_identity_and_newest_record_semantics(tmp_path: Path) -> None:
    path = tmp_path / "identities.sqlite"
    config = ProcessorConfig(retry=RetryConfig(max_attempts=1))
    await process_prompts(
        _CountingStrategy(failures={"x"}),
        [("id", "x")],
        config=config,
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
    )
    await process_prompts(
        _CountingStrategy(),
        [("id", "x")],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    replay = _CountingStrategy(failures={"x"})
    result = await process_prompts(
        replay,
        [("id", "x")],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_ALL,
    )
    assert replay.calls == []
    assert result.results[0].success

    changed = _CountingStrategy()
    await process_prompts(
        changed,
        [("id", "x")],
        artifact_store=SqliteArtifactStore(path, identity=_identity(model="other")),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert changed.calls == ["x"]
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM identities").fetchone()[0] == 2


@pytest.mark.asyncio
async def test_audit_only_success_is_not_replayed(tmp_path: Path) -> None:
    path = tmp_path / "audit-only.sqlite"
    await process_prompts(
        _CountingStrategy(),
        [("id", "one")],
        artifact_store=SqliteArtifactStore(path, identity=_identity(), include_output=False),
    )
    replay = _CountingStrategy()
    await process_prompts(
        replay,
        [("id", "one")],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replay.calls == ["one"]


@pytest.mark.asyncio
async def test_batching_and_transaction_rollback(tmp_path: Path) -> None:
    path = tmp_path / "batch.sqlite"
    store = SqliteArtifactStore(
        path,
        identity=_identity(),
        commit_batch_size=10,
        commit_interval_seconds=0.05,
    )
    items = [_item(str(index), f"p{index}") for index in range(10)]
    keys = [await store.prepare_item(item) for item in items]
    await asyncio.gather(
        *(
            store.append(
                item,
                key,
                WorkItemResult(item_id=item.item_id, success=True, output=item.prompt.upper()),
            )
            for item, key in zip(items, keys, strict=True)
        )
    )
    assert store._transaction_count == 1
    await store.close()

    rollback_path = tmp_path / "rollback.sqlite"
    rollback = SqliteArtifactStore(
        rollback_path,
        identity=_identity(),
        commit_batch_size=2,
        commit_interval_seconds=0.05,
    )
    good = _item("good", "good")
    bad = _item("bad", "bad")
    good_key = await rollback.prepare_item(good)
    bad_key = await rollback.prepare_item(bad)

    def install_trigger() -> None:
        rollback._require_connection().execute(
            "CREATE TRIGGER fail_bad BEFORE INSERT ON item_records "
            "WHEN NEW.item_id = 'bad' BEGIN SELECT RAISE(ABORT, 'forced failure'); END"
        )

    await rollback._run_db(install_trigger)
    outcomes = await asyncio.gather(
        rollback.append(good, good_key, WorkItemResult(item_id="good", success=True, output="G")),
        rollback.append(bad, bad_key, WorkItemResult(item_id="bad", success=True, output="B")),
        return_exceptions=True,
    )
    assert all(isinstance(outcome, ArtifactIOError) for outcome in outcomes)
    count = await rollback._run_db(
        lambda: (
            rollback._require_connection()
            .execute("SELECT COUNT(*) FROM item_records")
            .fetchone()[0]
        )
    )
    assert count == 0
    await rollback.close()


@pytest.mark.asyncio
async def test_iteration_is_finite_chunked_and_materialization_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "iteration.sqlite"
    store = SqliteArtifactStore(path, identity=_identity(), read_batch_size=1)
    first = _item("first", "one")
    second = _item("second", "two")
    first_key = await store.prepare_item(first)
    second_key = await store.prepare_item(second)
    await store.append(first, first_key, WorkItemResult(item_id="first", success=True, output="1"))
    await store.append(
        second, second_key, WorkItemResult(item_id="second", success=False, error="failed")
    )

    iterator = store.iter_results()
    observed_first = await anext(iterator)
    later = _item("later", "three")
    later_key = await store.prepare_item(later)
    await store.append(later, later_key, WorkItemResult(item_id="later", success=True, output="3"))
    snapshot = [observed_first, *[result async for result in iterator]]
    assert [result.item_id for result in snapshot] == ["first", "second"]
    assert [result.item_id async for result in store.iter_results(successes_only=True)] == [
        "first",
        "later",
    ]
    await store.close()

    materialized = await SqliteArtifactStore.read_results(path)
    assert [result.item_id for result in materialized.results] == ["first", "second", "later"]


@pytest.mark.asyncio
async def test_reopen_does_not_decode_history_before_lookup(tmp_path: Path) -> None:
    path = tmp_path / "lazy-reopen.sqlite"
    await process_prompts(
        _CountingStrategy(),
        [(str(index), f"p{index}") for index in range(30)],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
    )
    decoded: list[Any] = []
    store = SqliteArtifactStore(
        path,
        identity=_identity(),
        output_decoder=lambda value: decoded.append(value) or value,
    )
    item = _item("0", "p0")
    key = await store.prepare_item(item)
    assert decoded == []
    replayed = await store.lookup(item, key, ResumePolicy.REUSE_SUCCESSES)
    assert replayed is not None
    assert decoded == ["P0"]
    await store.close()


@pytest.mark.asyncio
async def test_schema_and_path_rejections(tmp_path: Path) -> None:
    with pytest.raises(ArtifactIOError, match="does not exist"):
        await SqliteArtifactStore.read_results(tmp_path / "missing.sqlite")

    directory = tmp_path / "directory"
    directory.mkdir()
    directory_store = SqliteArtifactStore(directory, identity=_identity())
    with pytest.raises(ArtifactIOError, match="directory"):
        await directory_store.prepare_item(_item("id", "p"))
    await directory_store.close()

    text_path = tmp_path / "text.sqlite"
    text_path.write_text("not sqlite", encoding="utf-8")
    text_store = SqliteArtifactStore(text_path, identity=_identity())
    with pytest.raises(ArtifactFormatError):
        await text_store.prepare_item(_item("id", "p"))
    await text_store.close()

    foreign_path = tmp_path / "foreign.sqlite"
    with sqlite3.connect(foreign_path) as connection:
        connection.execute("CREATE TABLE foreign_table (id INTEGER)")
    foreign = SqliteArtifactStore(foreign_path, identity=_identity())
    with pytest.raises(ArtifactFormatError, match="not an async-batch-llm artifact"):
        await foreign.prepare_item(_item("id", "p"))
    await foreign.close()

    zero_path = tmp_path / "zero.sqlite"
    zero_path.touch()
    await process_prompts(
        _CountingStrategy(),
        [("id", "p")],
        artifact_store=SqliteArtifactStore(zero_path, identity=_identity()),
    )
    assert zero_path.stat().st_size > 0


@pytest.mark.asyncio
async def test_future_and_malformed_schema_rejected(tmp_path: Path) -> None:
    future_path = tmp_path / "future.sqlite"
    await process_prompts(
        _CountingStrategy(),
        [("id", "p")],
        artifact_store=SqliteArtifactStore(future_path, identity=_identity()),
    )
    with sqlite3.connect(future_path) as connection:
        connection.execute(f"PRAGMA user_version={SQLITE_SCHEMA_VERSION + 1}")
    future = SqliteArtifactStore(future_path, identity=_identity())
    with pytest.raises(ArtifactFormatError, match="future"):
        await future.prepare_item(_item("id", "p"))
    await future.close()

    malformed_path = tmp_path / "malformed.sqlite"
    await process_prompts(
        _CountingStrategy(),
        [("id", "p")],
        artifact_store=SqliteArtifactStore(malformed_path, identity=_identity()),
    )
    with sqlite3.connect(malformed_path) as connection:
        connection.execute("DROP TABLE item_records")
    malformed = SqliteArtifactStore(malformed_path, identity=_identity())
    with pytest.raises(ArtifactFormatError, match="missing columns"):
        await malformed.prepare_item(_item("id", "p"))
    await malformed.close()


@pytest.mark.asyncio
async def test_cancelled_append_failure_is_reported_once_and_close_is_idempotent(
    tmp_path: Path,
) -> None:
    store = SqliteArtifactStore(
        tmp_path / "detached.sqlite",
        identity=_identity(),
        commit_interval_seconds=0,
    )
    item = _item("id", "p")
    key = await store.prepare_item(item)
    entered = threading.Event()
    release = threading.Event()

    def fail_write(records: list[dict[str, Any]]) -> None:
        entered.set()
        assert release.wait(timeout=2)
        raise ArtifactIOError("detached sqlite failure")

    store._insert_batch_sync = fail_write  # type: ignore[method-assign]
    append = asyncio.create_task(
        store.append(item, key, WorkItemResult(item_id="id", success=True, output="P"))
    )
    assert await asyncio.to_thread(entered.wait, 1)
    append.cancel()
    with pytest.raises(asyncio.CancelledError):
        await append
    release.set()
    with pytest.raises(ArtifactIOError, match="detached sqlite failure"):
        await store.close()
    await store.close()
    assert store._executor_shutdown


@pytest.mark.asyncio
async def test_cancelled_append_still_commits_under_store_ownership(tmp_path: Path) -> None:
    store = SqliteArtifactStore(
        tmp_path / "cancelled-success.sqlite",
        identity=_identity(),
        commit_interval_seconds=0,
    )
    item = _item("id", "p")
    key = await store.prepare_item(item)
    entered = threading.Event()
    release = threading.Event()
    original = store._insert_batch_sync

    def blocked_write(records: list[dict[str, Any]]) -> None:
        entered.set()
        assert release.wait(timeout=2)
        original(records)

    store._insert_batch_sync = blocked_write  # type: ignore[method-assign]
    append = asyncio.create_task(
        store.append(item, key, WorkItemResult(item_id="id", success=True, output="P"))
    )
    assert await asyncio.to_thread(entered.wait, 1)
    append.cancel()
    with pytest.raises(asyncio.CancelledError):
        await append
    release.set()
    assert [result.item_id async for result in store.iter_results()] == ["id"]
    await store.close()


@pytest.mark.asyncio
async def test_cancelled_preparation_can_be_closed_without_thread_leak(tmp_path: Path) -> None:
    store = SqliteArtifactStore(tmp_path / "cancel-prepare.sqlite", identity=_identity())
    entered = threading.Event()
    release = threading.Event()
    original = store._open_sync

    def blocked_open() -> None:
        entered.set()
        assert release.wait(timeout=2)
        original()

    store._open_sync = blocked_open  # type: ignore[method-assign]
    prepare = asyncio.create_task(store.prepare_item(_item("id", "p")))
    assert await asyncio.to_thread(entered.wait, 1)
    prepare.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prepare
    release.set()
    await store.close()
    assert store._executor_shutdown


@pytest.mark.asyncio
async def test_close_truncates_wal_and_terminates_executor(tmp_path: Path) -> None:
    path = tmp_path / "wal.sqlite"
    store = SqliteArtifactStore(path, identity=_identity(), commit_interval_seconds=0)
    item = _item("id", "p")
    key = await store.prepare_item(item)
    await store.append(item, key, WorkItemResult(item_id="id", success=True, output="P"))
    await store.close()
    wal = Path(f"{path}-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    assert store._last_checkpoint_busy is False
    assert store._executor_shutdown
    with pytest.raises(ArtifactIOError, match="closed"):
        await store.lookup(item, key, ResumePolicy.REUSE_ALL)


@pytest.mark.asyncio
async def test_busy_timeout_zero_fails_immediately_under_writer_lock(tmp_path: Path) -> None:
    path = tmp_path / "locked.sqlite"
    await process_prompts(
        _CountingStrategy(),
        [("initial", "p")],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
    )
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    store = SqliteArtifactStore(
        path,
        identity=_identity(),
        busy_timeout_seconds=0,
        commit_interval_seconds=0,
    )
    item = _item("locked", "p")
    key = await store.prepare_item(item)
    with pytest.raises(ArtifactIOError, match="locked"):
        await asyncio.wait_for(
            store.append(item, key, WorkItemResult(item_id="locked", success=True, output="P")),
            timeout=0.5,
        )
    blocker.rollback()
    blocker.close()
    await store.close()


@pytest.mark.asyncio
async def test_external_reader_makes_close_checkpoint_busy_without_data_loss(
    tmp_path: Path,
) -> None:
    path = tmp_path / "busy-reader.sqlite"
    await process_prompts(
        _CountingStrategy(),
        [("initial", "p")],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
    )
    reader = sqlite3.connect(path, isolation_level=None)
    reader.execute("BEGIN")
    assert reader.execute("SELECT COUNT(*) FROM item_records").fetchone()[0] == 1

    store = SqliteArtifactStore(
        path,
        identity=_identity(),
        busy_timeout_seconds=0,
        commit_interval_seconds=0,
    )
    item = _item("later", "p2")
    key = await store.prepare_item(item)
    await store.append(item, key, WorkItemResult(item_id="later", success=True, output="P2"))
    await store.close()
    assert store._last_checkpoint_busy is True
    reader.rollback()
    reader.close()
    loaded = await SqliteArtifactStore.read_results(path)
    assert [result.item_id for result in loaded.results] == ["initial", "later"]


@pytest.mark.asyncio
async def test_cancelled_close_finishes_cleanup_under_store_ownership(tmp_path: Path) -> None:
    store = SqliteArtifactStore(tmp_path / "cancel-close.sqlite", identity=_identity())
    await store.prepare_item(_item("id", "p"))
    entered = threading.Event()
    release = threading.Event()
    original = store._checkpoint_and_close_sync

    def blocked_close() -> None:
        entered.set()
        assert release.wait(timeout=2)
        original()

    store._checkpoint_and_close_sync = blocked_close  # type: ignore[method-assign]
    close = asyncio.create_task(store.close())
    assert await asyncio.to_thread(entered.wait, 1)
    close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close
    release.set()
    await store.close()
    assert store._closed
    assert store._executor_shutdown


@pytest.mark.asyncio
async def test_malformed_selected_result_reports_sequence_and_item(tmp_path: Path) -> None:
    path = tmp_path / "malformed-result.sqlite"
    await process_prompts(
        _CountingStrategy(),
        [("id", "p")],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
    )
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE item_records SET result_json = 'not-json'")
    with pytest.raises(ArtifactFormatError, match="sequence.*item 'id'"):
        await SqliteArtifactStore.read_results(path)


@pytest.mark.asyncio
async def test_resume_does_not_append_duplicate_and_uses_current_submission_index(
    tmp_path: Path,
) -> None:
    path = tmp_path / "no-duplicate.sqlite"
    await process_prompts(
        _CountingStrategy(),
        [("a", "one"), ("b", "two")],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
    )
    replay = _CountingStrategy()
    result = await process_prompts(
        replay,
        [("b", "two"), ("a", "one")],
        artifact_store=SqliteArtifactStore(path, identity=_identity()),
        resume=ResumePolicy.REUSE_SUCCESSES,
        preserve_order=True,
    )
    assert replay.calls == []
    assert [item.submission_index for item in result.results] == [0, 1]
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM item_records").fetchone()[0] == 2


@pytest.mark.asyncio
async def test_abrupt_process_after_committed_append_reopens_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "crash.sqlite"
    script = r"""
import asyncio
import os
import sys
from async_batch_llm import ArtifactIdentity, LLMWorkItem, SqliteArtifactStore, WorkItemResult
from async_batch_llm.llm_strategies import LLMCallStrategy

class Strategy(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        return prompt, {}, None

async def main():
    store = SqliteArtifactStore(
        sys.argv[1], identity=ArtifactIdentity(provider="test", model="model")
    )
    item = LLMWorkItem(item_id="committed", strategy=Strategy(), prompt="p")
    key = await store.prepare_item(item)
    await store.append(item, key, WorkItemResult(item_id="committed", success=True, output="P"))
    os._exit(0)

asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, os.fspath(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = await SqliteArtifactStore.read_results(path)
    assert [result.item_id for result in loaded.results] == ["committed"]


@pytest.mark.asyncio
async def test_abrupt_process_before_batch_commit_exposes_no_partial_row(tmp_path: Path) -> None:
    path = tmp_path / "uncommitted-crash.sqlite"
    script = r"""
import asyncio
import os
import sys
from async_batch_llm import ArtifactIdentity, LLMWorkItem, SqliteArtifactStore, WorkItemResult
from async_batch_llm.llm_strategies import LLMCallStrategy

class Strategy(LLMCallStrategy[str]):
    async def execute(self, prompt, attempt, timeout, state=None):
        return prompt, {}, None

async def main():
    store = SqliteArtifactStore(
        sys.argv[1],
        identity=ArtifactIdentity(provider="test", model="model"),
        commit_interval_seconds=60,
    )
    item = LLMWorkItem(item_id="pending", strategy=Strategy(), prompt="p")
    key = await store.prepare_item(item)
    asyncio.create_task(
        store.append(item, key, WorkItemResult(item_id="pending", success=True, output="P"))
    )
    await asyncio.sleep(0.1)
    os._exit(0)

asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, os.fspath(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = await SqliteArtifactStore.read_results(path)
    assert loaded.results == []


@pytest.mark.asyncio
async def test_small_auto_checkpoint_setting_bounds_wal_growth(tmp_path: Path) -> None:
    path = tmp_path / "wal-plateau.sqlite"
    store = SqliteArtifactStore(
        path,
        identity=_identity(),
        commit_batch_size=1,
        commit_interval_seconds=0,
    )
    store._wal_autocheckpoint_pages = 2
    sizes: list[int] = []
    for index in range(80):
        item = _item(str(index), "p")
        key = await store.prepare_item(item)
        await store.append(
            item,
            key,
            WorkItemResult(item_id=item.item_id, success=True, output="P"),
        )
        if index % 10 == 9:
            wal = Path(f"{path}-wal")
            sizes.append(wal.stat().st_size if wal.exists() else 0)
    assert store._effective_wal_autocheckpoint_pages == 2
    assert store._page_size is not None
    assert max(sizes) <= store._page_size * 32
    await store.close()


@pytest.mark.asyncio
async def test_close_checkpoint_error_is_raised_once(tmp_path: Path) -> None:
    store = SqliteArtifactStore(tmp_path / "close-error.sqlite", identity=_identity())
    await store.prepare_item(_item("id", "p"))
    original = store._checkpoint_and_close_sync

    def fail_after_close() -> None:
        original()
        raise ArtifactIOError("checkpoint failed")

    store._checkpoint_and_close_sync = fail_after_close  # type: ignore[method-assign]
    with pytest.raises(ArtifactIOError, match="checkpoint failed"):
        await store.close()
    await store.close()


@pytest.mark.asyncio
async def test_custom_encoder_decoder_and_context_fingerprinter(tmp_path: Path) -> None:
    class Payload:
        def __init__(self, value: str) -> None:
            self.value = value

    class PayloadStrategy(LLMCallStrategy[Payload]):
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, prompt, attempt, timeout, state=None):
            self.calls += 1
            return Payload(prompt.upper()), {}, None

    path = tmp_path / "custom.sqlite"
    strategy = PayloadStrategy()

    def encoder(value: Any) -> Any:
        return {"value": value.value} if isinstance(value, Payload) else value

    def decoder(value: Any) -> Payload:
        return Payload(value["value"])

    await process_prompts(
        strategy,
        [("id", "p", Payload("context"))],
        artifact_store=SqliteArtifactStore(
            path,
            identity=_identity(),
            encoder=encoder,
            output_decoder=decoder,
            context_fingerprinter=lambda value: f"payload:{value.value}",
        ),
    )
    replay = PayloadStrategy()
    current_context = Payload("context")
    result = await process_prompts(
        replay,
        [("id", "p", current_context)],
        artifact_store=SqliteArtifactStore(
            path,
            identity=_identity(),
            encoder=encoder,
            output_decoder=decoder,
            context_fingerprinter=lambda value: f"payload:{value.value}",
        ),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replay.calls == 0
    assert isinstance(result.results[0].output, Payload)
    assert result.results[0].output.value == "P"
    assert result.results[0].context is current_context
