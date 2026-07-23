"""Indexed SQLite artifact storage for large restartable runs."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ._internal.artifact_codec import (
    ContextFingerprinter,
    CostCalculator,
    PreparedArtifactItem,
    build_item_record,
    build_manifest_record,
    canonical_json,
    decode_stored_result,
    fingerprint_identity,
    fingerprint_work_item,
    restore_replayed_result,
)
from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    ArtifactFormatError,
    ArtifactIdentity,
    ArtifactIOError,
    ArtifactSerializationError,
    ResumePolicy,
    infer_artifact_identity,
)
from .base import BatchResult, BatchTermination, LLMWorkItem, WorkItemResult
from .serialization import (
    JSONValue,
    ResultSerializationError,
    ValueDecoder,
    ValueEncoder,
    to_json_value,
)

SQLITE_APPLICATION_ID = 0x41424C21  # "ABL!"
SQLITE_SCHEMA_VERSION = 1
_WAL_AUTOCHECKPOINT_PAGES = 1000


class SqliteDurability(str, Enum):
    """SQLite synchronization policy for committed artifact batches."""

    BALANCED = "balanced"
    FULL = "full"


@dataclass
class _AppendRequest:
    record: dict[str, Any]
    future: asyncio.Future[None]


@dataclass
class _FlushRequest:
    future: asyncio.Future[None]


@dataclass
class _CloseRequest:
    future: asyncio.Future[None]


_WriterRequest = _AppendRequest | _FlushRequest | _CloseRequest


def _package_version() -> str:
    try:
        return version("async-batch-llm")
    except PackageNotFoundError:
        return "0.0.0+dev"


def _validate_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer (got {value!r})")
    return int(value)


def _validate_non_negative_number(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative (got {value!r})")
    return float(value)


async def _await_without_cancelling(future: asyncio.Future[Any]) -> Any:
    """Await owned work without propagating caller cancellation into it.

    ``asyncio.shield()`` and ``asyncio.wait()`` can strand their outer waiter on
    CPython 3.14 when the inner future completes during callback registration.
    A short timer backs up the normal completion callback; cancellation of the
    bridge future still leaves the owned work untouched.
    """
    if future.done():
        return future.result()

    loop = asyncio.get_running_loop()
    waiter = loop.create_future()

    def wake_waiter(_future: asyncio.Future[Any]) -> None:
        if not waiter.done():
            waiter.set_result(None)

    future.add_done_callback(wake_waiter)
    timer = loop.call_later(0.001, wake_waiter, future)
    if future.done():
        wake_waiter(future)
    try:
        while not future.done():
            await waiter
            if not future.done():
                waiter = loop.create_future()
                timer = loop.call_later(0.001, wake_waiter, future)
    finally:
        timer.cancel()
        future.remove_done_callback(wake_waiter)
    return future.result()


class SqliteArtifactStore:
    """Indexed, batched artifact store backed by standard-library SQLite.

    One writable instance owns one connection, one executor thread, and one
    async writer task. It supports sequential reopen but is not a distributed
    work-claiming or exactly-once execution system.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        identity: ArtifactIdentity | None = None,
        user_metadata: Mapping[str, JSONValue] | None = None,
        include_output: bool = True,
        include_metadata: bool = True,
        include_prompt: bool = False,
        include_context: bool = False,
        context_in_identity: bool = True,
        encoder: ValueEncoder | None = None,
        output_decoder: ValueDecoder | None = None,
        context_decoder: ValueDecoder | None = None,
        context_fingerprinter: ContextFingerprinter | None = None,
        cost_calculator: CostCalculator | None = None,
        durability: SqliteDurability = SqliteDurability.BALANCED,
        commit_batch_size: int = 100,
        commit_interval_seconds: float = 0.01,
        busy_timeout_seconds: float = 5.0,
        read_batch_size: int = 1000,
    ) -> None:
        raw_path = os.fspath(path)
        if raw_path == ":memory:":
            raise ValueError(
                "SqliteArtifactStore requires a filesystem path; ':memory:' is unsupported"
            )
        if isinstance(raw_path, str) and raw_path.startswith("file:"):
            raise ValueError("SQLite URI filenames are unsupported; pass a normal filesystem path")
        self.path = Path(path)
        self.identity = identity
        self.user_metadata = user_metadata or {}
        self.include_output = include_output
        self.include_metadata = include_metadata
        self.include_prompt = include_prompt
        self.include_context = include_context
        self.context_in_identity = context_in_identity
        self.encoder = encoder
        self.output_decoder = output_decoder
        self.context_decoder = context_decoder
        self.context_fingerprinter = context_fingerprinter
        self.cost_calculator = cost_calculator
        try:
            self.durability = SqliteDurability(durability)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(value.value for value in SqliteDurability)
            raise ValueError(f"durability must be one of: {choices} (got {durability!r})") from exc
        self.commit_batch_size = _validate_positive_integer("commit_batch_size", commit_batch_size)
        self.commit_interval_seconds = _validate_non_negative_number(
            "commit_interval_seconds", commit_interval_seconds
        )
        self.busy_timeout_seconds = _validate_non_negative_number(
            "busy_timeout_seconds", busy_timeout_seconds
        )
        self.read_batch_size = _validate_positive_integer("read_batch_size", read_batch_size)

        try:
            metadata_value = to_json_value(self.user_metadata, path="$.user_metadata")
        except ResultSerializationError as exc:
            raise ArtifactSerializationError(str(exc)) from exc
        if not isinstance(metadata_value, dict):
            raise ArtifactSerializationError("user_metadata must serialize to an object")
        self._user_metadata_value = metadata_value

        self._identity_value: dict[str, JSONValue] | None = None
        self.identity_fingerprint: str | None = None
        if identity is not None:
            try:
                self._identity_value, self.identity_fingerprint = fingerprint_identity(identity)
            except ResultSerializationError as exc:
                raise ArtifactSerializationError(str(exc)) from exc

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="async-batch-llm-sqlite",
        )
        self._connection: sqlite3.Connection | None = None
        self._prepare_task: asyncio.Task[None] | None = None
        self._prepare_error_observed = False
        self._writer_task: asyncio.Task[None] | None = None
        self._active_writer_batch: list[_AppendRequest] = []
        self._writer_queue: asyncio.Queue[_WriterRequest] = asyncio.Queue()
        self._close_task: asyncio.Task[None] | None = None
        self._close_error_observed = False
        self._closing = False
        self._closed = False
        self._fatal_error: ArtifactError | None = None
        self._retained_error: ArtifactError | None = None
        self._detached_futures: set[asyncio.Future[None]] = set()
        self._close_detached = False
        self._executor_shutdown = False
        self._wal_autocheckpoint_pages = _WAL_AUTOCHECKPOINT_PAGES
        self._effective_wal_autocheckpoint_pages: int | None = None
        self._page_size: int | None = None
        self._transaction_count = 0
        self._last_checkpoint_busy: bool | None = None

    async def prepare_item(self, work_item: LLMWorkItem[Any, Any, Any]) -> PreparedArtifactItem:
        """Open/validate the database and fingerprint input before provider work."""
        await self._before_operation()
        self._resolve_identity_from(work_item.strategy)
        await self._prepare()
        try:
            return fingerprint_work_item(
                work_item,
                context_in_identity=self.context_in_identity,
                encoder=self.encoder,
                context_fingerprinter=self.context_fingerprinter,
            )
        except ResultSerializationError as exc:
            raise ArtifactSerializationError(str(exc)) from exc
        except Exception as exc:
            raise ArtifactSerializationError(
                f"Context fingerprinter failed for item {work_item.item_id!r}: {exc}"
            ) from exc

    def _resolve_identity_from(self, strategy: Any) -> None:
        if self.identity is not None:
            return
        inferred = infer_artifact_identity(strategy)
        try:
            identity_value, identity_fingerprint = fingerprint_identity(inferred)
        except ResultSerializationError as exc:
            raise ArtifactSerializationError(str(exc)) from exc
        self.identity = inferred
        self._identity_value = identity_value
        self.identity_fingerprint = identity_fingerprint

    def _require_resolved_identity(
        self,
    ) -> tuple[ArtifactIdentity, dict[str, JSONValue], str]:
        if (
            self.identity is None
            or self._identity_value is None
            or self.identity_fingerprint is None
        ):
            raise ArtifactError(
                "Artifact identity is not resolved yet. Pass "
                "identity=ArtifactIdentity(...) to SqliteArtifactStore, or run the "
                "store through a processor so prepare_item() can infer the identity."
            )
        return self.identity, self._identity_value, self.identity_fingerprint

    @staticmethod
    def _coerce_prepared(value: Any) -> PreparedArtifactItem:
        if not isinstance(value, PreparedArtifactItem):
            raise ArtifactSerializationError("Artifact item was not prepared by this store")
        return value

    async def lookup(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        policy: ResumePolicy,
    ) -> WorkItemResult[Any, Any] | None:
        """Look up the newest compatible terminal row through a replay index."""
        if policy is ResumePolicy.NONE:
            return None
        await self._before_operation()
        prepared = self._coerce_prepared(prepared_item)
        await self._prepare()
        _, _, identity_fingerprint = self._require_resolved_identity()
        row = await self._run_db(
            self._lookup_sync,
            work_item.item_id,
            prepared,
            identity_fingerprint,
            policy,
        )
        if row is None:
            return None
        try:
            return restore_replayed_result(
                row,
                work_item,
                output_decoder=self.output_decoder,
                context_decoder=self.context_decoder,
            )
        except (KeyError, ResultSerializationError) as exc:
            raise ArtifactFormatError(
                f"Malformed stored result at sequence {row.get('record_sequence')!r} "
                f"for item {work_item.item_id!r}: {exc}"
            ) from exc

    async def append(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        result: WorkItemResult[Any, Any],
    ) -> None:
        """Enqueue one terminal row and return only after its transaction commits."""
        await self._before_operation()
        prepared = self._coerce_prepared(prepared_item)
        await self._prepare()
        identity, identity_value, identity_fingerprint = self._require_resolved_identity()
        try:
            record = build_item_record(
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                work_item=work_item,
                prepared_item=prepared,
                result=result,
                identity_value=identity_value,
                identity_fingerprint=identity_fingerprint,
                identity=identity,
                include_output=self.include_output,
                include_metadata=self.include_metadata,
                include_prompt=self.include_prompt,
                include_context=self.include_context,
                encoder=self.encoder,
                cost_calculator=self.cost_calculator,
            )
        except (ResultSerializationError, TypeError, ValueError) as exc:
            raise ArtifactSerializationError(
                f"Could not serialize artifact result for item {work_item.item_id!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise ArtifactSerializationError(
                f"Cost calculator failed for item {work_item.item_id!r}: {exc}"
            ) from exc

        if self._closing or self._closed:
            raise ArtifactIOError(f"Artifact store is closed: {self.path}")
        if self._fatal_error is not None:
            raise ArtifactIOError(f"Artifact store is unusable after a write failure: {self.path}")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        # Queue ownership commits synchronously: there is no cancellation point
        # between the state check and put_nowait().
        self._writer_queue.put_nowait(_AppendRequest(record=record, future=future))
        await self._await_owned_future(future)

    def iter_results(
        self, *, successes_only: bool = False
    ) -> AsyncIterator[WorkItemResult[Any, Any]]:
        """Return a bounded keyset iterator over a finite committed snapshot."""
        return self._iter_results(successes_only=successes_only)

    async def _iter_results(
        self, *, successes_only: bool
    ) -> AsyncIterator[WorkItemResult[Any, Any]]:
        await self._before_operation()
        await self._prepare()
        await self._flush()
        high_water = await self._run_db(self._max_sequence_sync)
        after = -1
        while after < high_water:
            rows = await self._run_db(
                self._read_page_sync,
                after,
                high_water,
                successes_only,
            )
            if not rows:
                return
            for row in rows:
                after = int(row["record_sequence"])
                try:
                    yield decode_stored_result(
                        row,
                        output_decoder=self.output_decoder,
                        context_decoder=self.context_decoder,
                    )
                except (KeyError, ResultSerializationError) as exc:
                    raise ArtifactFormatError(
                        f"Malformed stored result at sequence {after!r} "
                        f"for item {row.get('item_id')!r}: {exc}"
                    ) from exc

    async def close(self) -> None:
        """Drain writes, checkpoint/truncate WAL, and terminate the executor."""
        await self._settle_detached_futures()
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close_impl())
        task = self._close_task
        try:
            await _await_without_cancelling(task)
        except asyncio.CancelledError:
            self._close_detached = True
            task.add_done_callback(self._finish_detached_close)
            raise
        except ArtifactError:
            if self._close_error_observed:
                return
            if self._retained_error is not None:
                self._close_error_observed = True
                self._raise_retained_error()
                return
            if self._close_detached:
                self._finish_detached_close(task)
                self._close_error_observed = True
                self._raise_retained_error()
                return
            self._close_error_observed = True
            raise
        self._raise_retained_error()

    @classmethod
    async def read_results(
        cls,
        path: str | Path,
        *,
        successes_only: bool = False,
        output_decoder: ValueDecoder | None = None,
        context_decoder: ValueDecoder | None = None,
        read_batch_size: int = 1000,
    ) -> BatchResult[Any, Any]:
        """Materialize stored results without calling a provider."""
        artifact_path = Path(path)
        exists = await asyncio.to_thread(artifact_path.exists)
        if not exists:
            raise ArtifactIOError(f"Artifact does not exist: {artifact_path}")
        store = cls(
            artifact_path,
            output_decoder=output_decoder,
            context_decoder=context_decoder,
            read_batch_size=read_batch_size,
        )
        try:
            results = [result async for result in store.iter_results(successes_only=successes_only)]
        finally:
            await store.close()
        return BatchResult(results=results, termination=BatchTermination())

    async def _before_operation(self) -> None:
        await self._settle_detached_futures()
        self._raise_retained_error()
        if self._closing or self._closed:
            raise ArtifactIOError(f"Artifact store is closed: {self.path}")
        if self._fatal_error is not None:
            raise ArtifactIOError(f"Artifact store is unusable after a write failure: {self.path}")

    async def _prepare(self) -> None:
        if self._prepare_task is None:
            self._prepare_task = asyncio.create_task(self._prepare_impl())
            self._prepare_task.add_done_callback(self._consume_task_exception)
        try:
            await _await_without_cancelling(self._prepare_task)
        except ArtifactError:
            self._prepare_error_observed = True
            raise

    async def _prepare_impl(self) -> None:
        await self._run_db(self._open_sync)
        self._writer_task = asyncio.create_task(self._writer_loop())
        self._writer_task.add_done_callback(self._writer_finished)

    async def _run_db(self, function: Any, /, *args: Any) -> Any:
        if self._executor_shutdown:
            raise ArtifactIOError(f"Artifact store is closed: {self.path}")
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, function, *args)
        return await _await_without_cancelling(future)

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        task.exception()

    def _writer_finished(self, task: asyncio.Task[None]) -> None:
        """Turn an unexpected writer-task crash into failures for every owner."""
        if task.cancelled():
            exception: BaseException | None = asyncio.CancelledError()
        else:
            exception = task.exception()
        if exception is None or self._fatal_error is not None:
            return
        error = ArtifactIOError(f"SQLite artifact writer task failed: {exception}")
        self._fatal_error = error
        for request in self._active_writer_batch:
            if not request.future.done():
                request.future.set_exception(error)
        self._active_writer_batch = []
        self._fail_queued_requests(error)

    async def _await_owned_future(self, future: asyncio.Future[None]) -> None:
        try:
            await _await_without_cancelling(future)
        except asyncio.CancelledError:
            self._detached_futures.add(future)
            future.add_done_callback(self._finish_detached_future)
            raise

    def _finish_detached_future(self, future: asyncio.Future[None]) -> None:
        if future not in self._detached_futures:
            return
        self._detached_futures.discard(future)
        try:
            future.result()
        except asyncio.CancelledError:
            return
        except ArtifactError as exc:
            if self._retained_error is None:
                self._retained_error = exc
        except Exception as exc:  # pragma: no cover - defensive invariant
            if self._retained_error is None:
                self._retained_error = ArtifactIOError(str(exc))

    async def _settle_detached_futures(self) -> None:
        pending = list(self._detached_futures)
        if pending:
            gathered = asyncio.gather(*pending, return_exceptions=True)
            await _await_without_cancelling(gathered)
            for future in pending:
                self._finish_detached_future(future)

    def _raise_retained_error(self) -> None:
        error = self._retained_error
        if error is not None:
            self._retained_error = None
            raise error

    def _finish_detached_close(self, task: asyncio.Task[None]) -> None:
        if not self._close_detached or task.cancelled():
            return
        self._close_detached = False
        try:
            task.result()
        except ArtifactError as exc:
            if self._retained_error is None:
                self._retained_error = exc
        except Exception as exc:  # pragma: no cover - defensive invariant
            if self._retained_error is None:
                self._retained_error = ArtifactIOError(str(exc))

    async def _flush(self) -> None:
        if self._writer_task is None or self._writer_task.done():
            if self._fatal_error is not None:
                raise ArtifactIOError(
                    f"Artifact store is unusable after a write failure: {self.path}"
                )
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._writer_queue.put_nowait(_FlushRequest(future=future))
        await self._await_owned_future(future)

    async def _writer_loop(self) -> None:
        carry: _WriterRequest | None = None
        while True:
            request = carry if carry is not None else await self._writer_queue.get()
            carry = None
            if isinstance(request, _CloseRequest):
                request.future.set_result(None)
                return
            if isinstance(request, _FlushRequest):
                request.future.set_result(None)
                continue

            batch = [request]
            deadline = asyncio.get_running_loop().time() + self.commit_interval_seconds
            while len(batch) < self.commit_batch_size:
                try:
                    next_request = self._writer_queue.get_nowait()
                except asyncio.QueueEmpty:
                    if self.commit_interval_seconds == 0:
                        break
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        next_request = await asyncio.wait_for(
                            self._writer_queue.get(), timeout=remaining
                        )
                    except TimeoutError:
                        break
                if isinstance(next_request, _AppendRequest):
                    batch.append(next_request)
                else:
                    carry = next_request
                    break

            self._active_writer_batch = batch
            try:
                await self._run_db(
                    self._insert_batch_sync,
                    [item.record for item in batch],
                )
            except ArtifactError as exc:
                self._fatal_error = exc
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
                self._fail_queued_requests(exc, carry=carry)
                self._active_writer_batch = []
                return
            except Exception as exc:  # pragma: no cover - executor invariant
                error = ArtifactIOError(f"Could not append SQLite artifact batch: {exc}")
                self._fatal_error = error
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(error)
                self._fail_queued_requests(error, carry=carry)
                self._active_writer_batch = []
                return
            for item in batch:
                if not item.future.done():
                    item.future.set_result(None)
            self._active_writer_batch = []

    def _fail_queued_requests(
        self, error: ArtifactError, *, carry: _WriterRequest | None = None
    ) -> None:
        requests = [carry] if carry is not None else []
        while True:
            try:
                requests.append(self._writer_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        for request in requests:
            if request is not None and not request.future.done():
                request.future.set_exception(error)

    async def _close_impl(self) -> None:
        close_error: ArtifactError | None = None
        try:
            if self._prepare_task is not None:
                try:
                    await _await_without_cancelling(self._prepare_task)
                except ArtifactError as exc:
                    if not self._prepare_error_observed:
                        close_error = exc
            if self._writer_task is not None and not self._writer_task.done():
                future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._writer_queue.put_nowait(_CloseRequest(future=future))
                try:
                    await _await_without_cancelling(future)
                except ArtifactError as exc:
                    close_error = close_error or exc
                await asyncio.gather(self._writer_task, return_exceptions=True)
            if self._connection is not None:
                try:
                    await self._run_db(self._checkpoint_and_close_sync)
                except ArtifactError as exc:
                    close_error = close_error or exc
        finally:
            self._closed = True
            if not self._executor_shutdown:
                # All submitted DB work has drained at this point. Joining the
                # now-idle private executor directly avoids creating a default
                # executor solely for shutdown (and a CPython 3.14 completion
                # race while asyncio.run() later shuts that executor down).
                self._executor.shutdown(wait=True)
                self._executor_shutdown = True
        if close_error is not None:
            raise close_error

    def _open_sync(self) -> None:
        try:
            if self.path.is_dir():
                raise ArtifactIOError(f"Artifact path is a directory: {self.path}")
            existed = self.path.exists()
            size = self.path.stat().st_size if existed else 0
            initialize = not existed or size == 0
            if initialize and self.identity_fingerprint is None:
                raise ArtifactError(
                    f"Cannot create a new artifact {self.path} without a resolved identity. "
                    "Pass identity=ArtifactIdentity(...) to SqliteArtifactStore, or run the "
                    "store through a processor so it can be inferred from the strategy."
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.path,
                timeout=self.busy_timeout_seconds,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            self._connection = connection
            connection.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_seconds * 1000)}")
            if initialize:
                self._initialize_schema_sync(connection)
            else:
                self._validate_schema_sync(connection)
            journal_mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
            if journal_mode != "wal":
                raise ArtifactIOError(
                    f"SQLite could not enable WAL mode for artifact {self.path}: {journal_mode!r}"
                )
            synchronous = "FULL" if self.durability is SqliteDurability.FULL else "NORMAL"
            connection.execute(f"PRAGMA synchronous={synchronous}")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(f"PRAGMA wal_autocheckpoint={self._wal_autocheckpoint_pages}")
            self._effective_wal_autocheckpoint_pages = int(
                connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
            )
            self._page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            if self.identity_fingerprint is not None:
                self._insert_or_verify_identity_sync(connection)
        except ArtifactError:
            self._close_connection_after_open_failure()
            raise
        except sqlite3.OperationalError as exc:
            self._close_connection_after_open_failure()
            if "not a database" in str(exc).lower():
                raise ArtifactFormatError(f"Invalid SQLite artifact {self.path}: {exc}") from exc
            raise ArtifactIOError(f"Could not prepare SQLite artifact {self.path}: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            self._close_connection_after_open_failure()
            raise ArtifactFormatError(f"Invalid SQLite artifact {self.path}: {exc}") from exc
        except OSError as exc:
            self._close_connection_after_open_failure()
            raise ArtifactIOError(f"Could not prepare SQLite artifact {self.path}: {exc}") from exc

    def _close_connection_after_open_failure(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()

    def _initialize_schema_sync(self, connection: sqlite3.Connection) -> None:
        identity, identity_value, identity_fingerprint = self._require_resolved_identity()
        del identity
        manifest = build_manifest_record(
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            package_version=_package_version(),
            identity_value=identity_value,
            identity_fingerprint=identity_fingerprint,
            user_metadata=self._user_metadata_value,
        )
        try:
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE manifest (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    logical_schema_version INTEGER NOT NULL,
                    sqlite_schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    package_version TEXT NOT NULL,
                    initial_identity_json TEXT NOT NULL,
                    initial_identity_fingerprint TEXT NOT NULL,
                    user_metadata_json TEXT NOT NULL
                );
                CREATE TABLE identities (
                    identity_fingerprint TEXT PRIMARY KEY,
                    canonical_identity_json TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    prompt_version TEXT,
                    parser_version TEXT,
                    application_version TEXT,
                    first_seen_at TEXT NOT NULL
                );
                CREATE TABLE item_records (
                    record_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    logical_schema_version INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    submission_index INTEGER,
                    prompt_fingerprint TEXT NOT NULL,
                    context_fingerprint TEXT,
                    input_fingerprint TEXT NOT NULL,
                    identity_fingerprint TEXT NOT NULL REFERENCES identities(identity_fingerprint),
                    strategy_class TEXT NOT NULL,
                    success INTEGER NOT NULL CHECK (success IN (0, 1)),
                    error_category TEXT,
                    replay_eligible INTEGER NOT NULL CHECK (replay_eligible IN (0, 1)),
                    calculated_cost REAL,
                    raw_prompt_json TEXT,
                    raw_context_json TEXT,
                    result_json TEXT NOT NULL
                );
                CREATE INDEX idx_item_records_replay_all
                    ON item_records (
                        identity_fingerprint, item_id, prompt_fingerprint,
                        context_fingerprint, input_fingerprint, record_sequence DESC
                    ) WHERE replay_eligible = 1;
                CREATE INDEX idx_item_records_replay_success
                    ON item_records (
                        identity_fingerprint, item_id, prompt_fingerprint,
                        context_fingerprint, input_fingerprint, record_sequence DESC
                    ) WHERE replay_eligible = 1 AND success = 1;
                CREATE INDEX idx_item_records_success_sequence
                    ON item_records (record_sequence) WHERE success = 1;
                """
            )
            connection.execute(
                """
                INSERT INTO manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    1,
                    ARTIFACT_SCHEMA_VERSION,
                    SQLITE_SCHEMA_VERSION,
                    manifest["created_at"],
                    manifest["package_version"],
                    canonical_json(identity_value),
                    identity_fingerprint,
                    canonical_json(self._user_metadata_value),
                ),
            )
            connection.execute(f"PRAGMA application_id={SQLITE_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={SQLITE_SCHEMA_VERSION}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _validate_schema_sync(self, connection: sqlite3.Connection) -> None:
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        if application_id != SQLITE_APPLICATION_ID:
            raise ArtifactFormatError(
                f"SQLite file is not an async-batch-llm artifact: {self.path}"
            )
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if schema_version > SQLITE_SCHEMA_VERSION:
            raise ArtifactFormatError(
                f"Unsupported future SQLite artifact schema version {schema_version}"
            )
        if schema_version != SQLITE_SCHEMA_VERSION:
            raise ArtifactFormatError(
                f"Unsupported SQLite artifact schema version {schema_version}"
            )
        required = {
            "manifest": {
                "id",
                "logical_schema_version",
                "sqlite_schema_version",
                "created_at",
                "package_version",
                "initial_identity_json",
                "initial_identity_fingerprint",
                "user_metadata_json",
            },
            "identities": {
                "identity_fingerprint",
                "canonical_identity_json",
                "provider",
                "model",
                "prompt_version",
                "parser_version",
                "application_version",
                "first_seen_at",
            },
            "item_records": {
                "record_sequence",
                "logical_schema_version",
                "recorded_at",
                "item_id",
                "submission_index",
                "prompt_fingerprint",
                "context_fingerprint",
                "input_fingerprint",
                "identity_fingerprint",
                "strategy_class",
                "success",
                "error_category",
                "replay_eligible",
                "calculated_cost",
                "raw_prompt_json",
                "raw_context_json",
                "result_json",
            },
        }
        for table, expected_columns in required.items():
            columns = {
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            missing = expected_columns - columns
            if missing:
                raise ArtifactFormatError(
                    f"Malformed SQLite artifact {self.path}: table {table!r} "
                    f"is missing columns {sorted(missing)!r}"
                )
        indexes = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        required_indexes = {
            "idx_item_records_replay_all",
            "idx_item_records_replay_success",
            "idx_item_records_success_sequence",
        }
        missing_indexes = required_indexes - indexes
        if missing_indexes:
            raise ArtifactFormatError(
                f"Malformed SQLite artifact {self.path}: missing indexes "
                f"{sorted(missing_indexes)!r}"
            )
        manifest = connection.execute(
            "SELECT logical_schema_version, sqlite_schema_version FROM manifest WHERE id = 1"
        ).fetchone()
        if manifest is None:
            raise ArtifactFormatError(f"Malformed SQLite artifact {self.path}: manifest is missing")
        logical_version = int(manifest[0])
        if logical_version > ARTIFACT_SCHEMA_VERSION:
            raise ArtifactFormatError(
                f"Unsupported future artifact schema version {logical_version}"
            )
        if logical_version != ARTIFACT_SCHEMA_VERSION or int(manifest[1]) != schema_version:
            raise ArtifactFormatError(f"Malformed SQLite artifact version metadata: {self.path}")

    def _insert_or_verify_identity_sync(self, connection: sqlite3.Connection) -> None:
        identity, identity_value, identity_fingerprint = self._require_resolved_identity()
        canonical = canonical_json(identity_value)
        existing = connection.execute(
            "SELECT canonical_identity_json FROM identities WHERE identity_fingerprint = ?",
            (identity_fingerprint,),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != canonical:
                raise ArtifactFormatError(
                    "Artifact identity fingerprint is associated with different canonical JSON"
                )
            return
        connection.execute(
            """
            INSERT INTO identities (
                identity_fingerprint, canonical_identity_json, provider, model,
                prompt_version, parser_version, application_version, first_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identity_fingerprint,
                canonical,
                identity.provider,
                identity.model,
                identity.prompt_version,
                identity.parser_version,
                identity.application_version,
                build_manifest_record(
                    artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                    package_version=_package_version(),
                    identity_value=identity_value,
                    identity_fingerprint=identity_fingerprint,
                    user_metadata={},
                )["created_at"],
            ),
        )

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ArtifactIOError(f"SQLite artifact is not open: {self.path}")
        return self._connection

    def _insert_batch_sync(self, records: list[dict[str, Any]]) -> None:
        connection = self._require_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            for record in records:
                connection.execute(
                    """
                    INSERT INTO item_records (
                        logical_schema_version, recorded_at, item_id, submission_index,
                        prompt_fingerprint, context_fingerprint, input_fingerprint,
                        identity_fingerprint, strategy_class, success, error_category,
                        replay_eligible, calculated_cost, raw_prompt_json,
                        raw_context_json, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["artifact_schema_version"],
                        record["recorded_at"],
                        record["item_id"],
                        record["submission_index"],
                        record["prompt_fingerprint"],
                        record["context_fingerprint"],
                        record["input_fingerprint"],
                        record["identity_fingerprint"],
                        record["strategy_class"],
                        int(bool(record["success"])),
                        record["error_category"],
                        int(bool(record["replay_eligible"])),
                        record["calculated_cost"],
                        (
                            None
                            if record["raw_prompt"] is None
                            else canonical_json(record["raw_prompt"])
                        ),
                        (
                            None
                            if record["raw_context"] is None
                            else canonical_json(record["raw_context"])
                        ),
                        canonical_json(record["result"]),
                    ),
                )
            connection.commit()
            self._transaction_count += 1
        except sqlite3.DatabaseError as exc:
            connection.rollback()
            item_ids = [record.get("item_id") for record in records[:3]]
            raise ArtifactIOError(
                f"Could not append SQLite artifact batch for items {item_ids!r}: {exc}"
            ) from exc

    def _lookup_sync(
        self,
        item_id: str,
        prepared: PreparedArtifactItem,
        identity_fingerprint: str,
        policy: ResumePolicy,
    ) -> dict[str, Any] | None:
        connection = self._require_connection()
        success_clause = " AND success = 1" if policy is ResumePolicy.REUSE_SUCCESSES else ""
        sql = f"""
            SELECT record_sequence, logical_schema_version, item_id,
                   prompt_fingerprint, context_fingerprint, input_fingerprint,
                   identity_fingerprint, success, result_json
              FROM item_records
             WHERE identity_fingerprint = ?
               AND item_id = ?
               AND prompt_fingerprint = ?
               AND context_fingerprint IS ?
               AND input_fingerprint = ?
               AND replay_eligible = 1{success_clause}
             ORDER BY record_sequence DESC
             LIMIT 1
        """
        try:
            row = connection.execute(
                sql,
                (
                    identity_fingerprint,
                    item_id,
                    prepared.prompt_fingerprint,
                    prepared.context_fingerprint,
                    prepared.input_fingerprint,
                ),
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise ArtifactIOError(f"Could not query SQLite artifact {self.path}: {exc}") from exc
        return None if row is None else self._row_to_record(row)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
        try:
            result = json.loads(row["result_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ArtifactFormatError(
                f"Malformed result_json at sequence {row['record_sequence']!r} "
                f"for item {row['item_id']!r}: {exc}"
            ) from exc
        return {
            "artifact_schema_version": row["logical_schema_version"],
            "record_sequence": row["record_sequence"],
            "item_id": row["item_id"],
            "prompt_fingerprint": row["prompt_fingerprint"],
            "context_fingerprint": row["context_fingerprint"],
            "input_fingerprint": row["input_fingerprint"],
            "identity_fingerprint": row["identity_fingerprint"],
            "success": bool(row["success"]),
            "result": result,
        }

    def _max_sequence_sync(self) -> int:
        connection = self._require_connection()
        try:
            row = connection.execute(
                "SELECT COALESCE(MAX(record_sequence), -1) FROM item_records"
            ).fetchone()
            return int(row[0])
        except sqlite3.DatabaseError as exc:
            raise ArtifactIOError(f"Could not inspect SQLite artifact {self.path}: {exc}") from exc

    def _read_page_sync(
        self,
        after: int,
        high_water: int,
        successes_only: bool,
    ) -> list[dict[str, Any]]:
        connection = self._require_connection()
        success_clause = " AND success = 1" if successes_only else ""
        try:
            cursor = connection.execute(
                f"""
                SELECT record_sequence, logical_schema_version, item_id,
                       prompt_fingerprint, context_fingerprint, input_fingerprint,
                       identity_fingerprint, success, result_json
                  FROM item_records
                 WHERE record_sequence > ? AND record_sequence <= ?{success_clause}
                 ORDER BY record_sequence
                 LIMIT ?
                """,
                (after, high_water, self.read_batch_size),
            )
            rows = cursor.fetchall()
            cursor.close()
            return [self._row_to_record(row) for row in rows]
        except ArtifactError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ArtifactIOError(f"Could not iterate SQLite artifact {self.path}: {exc}") from exc

    def _checkpoint_and_close_sync(self) -> None:
        connection = self._require_connection()
        error: ArtifactError | None = None
        try:
            row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            self._last_checkpoint_busy = bool(row and int(row[0]) != 0)
        except sqlite3.DatabaseError as exc:
            error = ArtifactIOError(f"Could not checkpoint SQLite artifact {self.path}: {exc}")
        try:
            connection.close()
        except sqlite3.DatabaseError as exc:
            error = error or ArtifactIOError(f"Could not close SQLite artifact {self.path}: {exc}")
        finally:
            self._connection = None
        if error is not None:
            raise error


__all__ = [
    "SQLITE_APPLICATION_ID",
    "SQLITE_SCHEMA_VERSION",
    "SqliteArtifactStore",
    "SqliteDurability",
]
