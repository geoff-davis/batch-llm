"""Versioned JSONL audit artifacts and compatible result replay.

The JSONL implementation is concurrency-safe within one store instance and
process. It does not claim cross-process append safety; applications needing
multiple writers must provide an :class:`ArtifactStore` with real file locking
or a transactional backend.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, TextIO, TypeAlias

from ._internal.artifact_codec import (
    ContextFingerprinter,
    CostCalculator,
    PreparedArtifactItem,
    ReplayKey,
    build_item_record,
    build_manifest_record,
    decode_stored_result,
    fingerprint_identity,
    fingerprint_work_item,
    record_is_compatible,
    record_replay_key,
    replay_key,
    restore_replayed_result,
)
from .base import BatchResult, BatchTermination, LLMWorkItem, WorkItemResult
from .serialization import (
    JSONValue,
    ResultSerializationError,
    ValueDecoder,
    ValueEncoder,
    to_json_value,
)

ARTIFACT_SCHEMA_VERSION = 1
_ReplayKey: TypeAlias = ReplayKey


class ArtifactError(RuntimeError):
    """Base class for artifact preparation, format, and persistence failures."""


class ArtifactSerializationError(ArtifactError):
    """An artifact identity/input/result could not be canonically serialized."""


class ArtifactIOError(ArtifactError):
    """An artifact could not be read, written, flushed, or closed."""


class ArtifactFormatError(ArtifactError):
    """An artifact is malformed or uses an unsupported schema version."""


@dataclass(frozen=True)
class ArtifactIdentity:
    """Caller-supplied provenance used to decide whether replay is compatible."""

    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    parser_version: str | None = None
    application_version: str | None = None
    extra: Mapping[str, JSONValue] = field(default_factory=dict)


# Deterministic provider labels for the built-in model classes, used when
# inferring an identity from a strategy (zero-config artifacts, issue #99).
_PROVIDER_BY_MODEL_CLASS = {
    "GeminiModel": "gemini",
    "GeminiCachedModel": "gemini",
    "OpenAIModel": "openai",
    "OpenRouterModel": "openrouter",
    "DeepSeekModel": "deepseek",
}

_UNVERSIONED = "unversioned"
_NO_IDENTITY_HOOK = object()


def infer_artifact_identity(strategy: Any) -> ArtifactIdentity:
    """Derive a deterministic :class:`ArtifactIdentity` from a strategy.

    Used by :class:`JsonlArtifactStore` when no explicit identity is given
    (v0.20.0, issue #99). ``provider`` and ``model`` come from the strategy's
    wrapped model (built-in model classes map to their provider name; other
    models use their class name); the version fields default to
    ``"unversioned"``. The result is deterministic for the same strategy
    setup across processes, so resume keeps working — and changing the model
    changes the identity fingerprint, which invalidates reuse.

    Prompt (and, by default, context) always participate in the per-item
    compatibility fingerprint regardless of identity, so a changed prompt
    never silently replays a stale result even with a defaulted identity.
    """
    explicit_hook = getattr(strategy, "artifact_identity", _NO_IDENTITY_HOOK)
    if explicit_hook is not _NO_IDENTITY_HOOK:
        if explicit_hook is None:
            raise ArtifactError(
                f"{type(strategy).__name__} cannot safely infer artifact identity. "
                "Pass identity=ArtifactIdentity(...) to CallableStrategy or "
                "JsonlArtifactStore."
            )
        if not isinstance(explicit_hook, ArtifactIdentity):
            raise ArtifactError(
                f"{type(strategy).__name__}.artifact_identity must be an ArtifactIdentity or None"
            )
        return explicit_hook

    provider: str | None = None
    model_id: str | None = None

    model_obj = getattr(strategy, "model", None)
    if model_obj is not None:
        raw_model = getattr(model_obj, "_model", None)
        if isinstance(raw_model, str) and raw_model:
            model_id = raw_model
        provider = _PROVIDER_BY_MODEL_CLASS.get(type(model_obj).__name__)
        if provider is None:
            provider = type(model_obj).__name__
    else:
        # PydanticAIStrategy and similar wrappers expose an agent.
        agent = getattr(strategy, "agent", None)
        agent_model = getattr(agent, "model", None) if agent is not None else None
        if isinstance(agent_model, str) and agent_model:
            model_id = agent_model
        elif agent_model is not None:
            name = getattr(agent_model, "model_name", None)
            if isinstance(name, str) and name:
                model_id = name

    if provider is None:
        provider = type(strategy).__name__

    return ArtifactIdentity(
        provider=provider,
        model=model_id or "unknown",
        prompt_version=_UNVERSIONED,
        parser_version=_UNVERSIONED,
        application_version=_UNVERSIONED,
    )


class ResumePolicy(str, Enum):
    """Which compatible terminal artifact records may bypass provider work."""

    NONE = "none"
    REUSE_SUCCESSES = "reuse_successes"
    REUSE_ALL = "reuse_all"


_ItemFingerprint = PreparedArtifactItem


class ArtifactStore(Protocol):
    """Provider-neutral asynchronous checkpoint/replay store."""

    async def prepare_item(self, work_item: LLMWorkItem[Any, Any, Any]) -> Any:
        """Prepare the run and validate/fingerprint an item before execution."""

    async def lookup(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        policy: ResumePolicy,
    ) -> WorkItemResult[Any, Any] | None:
        """Return the newest compatible reusable result, if any."""

    async def append(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        result: WorkItemResult[Any, Any],
    ) -> None:
        """Durably append one newly executed terminal result."""

    def iter_results(self, *, successes_only: bool = False) -> AsyncIterator[WorkItemResult]:
        """Iterate stored results without starting a processor."""

    async def close(self) -> None:
        """Flush and close the store; repeated calls are safe."""


def _package_version() -> str:
    try:
        return version("async-batch-llm")
    except PackageNotFoundError:
        return "0.0.0+dev"


def _identity_mapping(identity: ArtifactIdentity) -> dict[str, JSONValue]:
    """Compatibility wrapper around the shared private identity codec."""
    try:
        value, _ = fingerprint_identity(identity)
    except ResultSerializationError as exc:
        raise ArtifactSerializationError(str(exc)) from exc
    return value


def _read_artifact_records(path: Path, *, allow_create: bool) -> tuple[list[dict[str, Any]], bool]:
    """Read and validate an artifact, optionally treating missing/empty as new."""
    try:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
    except OSError as exc:
        raise ArtifactIOError(f"Could not inspect artifact {path}: {exc}") from exc
    if not exists:
        if allow_create:
            return [], True
        raise ArtifactIOError(f"Artifact does not exist: {path}")
    if size == 0:
        if allow_create:
            return [], True
        raise ArtifactFormatError(f"Artifact is empty: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactIOError(f"Could not read artifact {path}: {exc}") from exc
    segments = raw.split(b"\n")
    has_trailing_newline = raw.endswith(b"\n")
    records: list[dict[str, Any]] = []
    manifest_seen = False
    for index, segment in enumerate(segments):
        if not segment:
            continue
        line_number = index + 1
        try:
            value = json.loads(segment)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_truncated_final = index == len(segments) - 1 and not has_trailing_newline
            if is_truncated_final:
                break
            raise ArtifactFormatError(
                f"Malformed artifact JSON at non-final line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ArtifactFormatError(f"Artifact line {line_number} must be a JSON object")
        schema = value.get("artifact_schema_version")
        if schema != ARTIFACT_SCHEMA_VERSION:
            if isinstance(schema, int) and schema > ARTIFACT_SCHEMA_VERSION:
                raise ArtifactFormatError(
                    f"Unsupported future artifact schema version {schema} at line {line_number}"
                )
            raise ArtifactFormatError(
                f"Unsupported artifact schema version {schema!r} at line {line_number}"
            )
        record_type = value.get("record_type")
        if not manifest_seen:
            if record_type != "manifest":
                raise ArtifactFormatError("The first complete artifact record must be a manifest")
            manifest_seen = True
            continue
        if record_type != "item":
            raise ArtifactFormatError(
                f"Unsupported artifact record_type {record_type!r} at line {line_number}"
            )
        records.append(value)
    if not manifest_seen:
        raise ArtifactFormatError("Artifact has no complete manifest record")
    return records, False


_JSONL_ITER_PAGE_SIZE = 1


def _artifact_snapshot_size(path: Path) -> int:
    """Return a finite byte snapshot for streaming inspection."""
    try:
        if not path.exists():
            raise ArtifactIOError(f"Artifact does not exist: {path}")
        size = path.stat().st_size
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactIOError(f"Could not inspect artifact {path}: {exc}") from exc
    if size == 0:
        raise ArtifactFormatError(f"Artifact is empty: {path}")
    return size


def _read_artifact_page(
    path: Path,
    *,
    offset: int,
    snapshot_size: int,
    line_number: int,
    manifest_seen: bool,
    page_size: int,
) -> tuple[list[dict[str, Any]], int, int, bool, bool]:
    """Read one bounded page from a finite JSONL byte snapshot."""
    records: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            while offset < snapshot_size and len(records) < page_size:
                segment = handle.readline(snapshot_size - offset)
                if not segment:
                    break
                offset += len(segment)
                line_number += 1
                has_trailing_newline = segment.endswith(b"\n")
                if not segment.strip():
                    continue
                try:
                    value = json.loads(segment)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if offset == snapshot_size and not has_trailing_newline:
                        return records, offset, line_number, manifest_seen, True
                    raise ArtifactFormatError(
                        f"Malformed artifact JSON at non-final line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ArtifactFormatError(f"Artifact line {line_number} must be a JSON object")
                schema = value.get("artifact_schema_version")
                if schema != ARTIFACT_SCHEMA_VERSION:
                    if isinstance(schema, int) and schema > ARTIFACT_SCHEMA_VERSION:
                        raise ArtifactFormatError(
                            "Unsupported future artifact schema version "
                            f"{schema} at line {line_number}"
                        )
                    raise ArtifactFormatError(
                        f"Unsupported artifact schema version {schema!r} at line {line_number}"
                    )
                record_type = value.get("record_type")
                if not manifest_seen:
                    if record_type != "manifest":
                        raise ArtifactFormatError(
                            "The first complete artifact record must be a manifest"
                        )
                    manifest_seen = True
                    continue
                if record_type != "item":
                    raise ArtifactFormatError(
                        f"Unsupported artifact record_type {record_type!r} at line {line_number}"
                    )
                records.append(value)
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactIOError(f"Could not read artifact {path}: {exc}") from exc

    done = offset >= snapshot_size
    if done and not manifest_seen:
        raise ArtifactFormatError("Artifact has no complete manifest record")
    return records, offset, line_number, manifest_seen, done


class JsonlArtifactStore:
    """Append-only version-1 JSONL artifact store.

    Prompt and context text are excluded by default; their SHA-256 hashes are
    always recorded for compatibility. Output and metadata are included by
    default because successful replay requires output. Set ``include_output``
    false for audit-only artifacts; those successful records are not replayable.
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
        fsync: bool = False,
    ) -> None:
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
        self.fsync = fsync

        # With no explicit identity, resolution is deferred to the first
        # prepare_item() call, which infers provider/model from the item's
        # strategy. Until then the
        # fingerprint is unset and lookup/append refuse to run.
        self._identity_value: dict[str, JSONValue] | None = None
        self.identity_fingerprint: str | None = None
        if identity is not None:
            try:
                self._identity_value, self.identity_fingerprint = fingerprint_identity(identity)
            except ResultSerializationError as exc:
                raise ArtifactSerializationError(str(exc)) from exc
        try:
            metadata_value = to_json_value(self.user_metadata, path="$.user_metadata")
        except ResultSerializationError as exc:
            raise ArtifactSerializationError(str(exc)) from exc
        if not isinstance(metadata_value, dict):
            raise ArtifactSerializationError("user_metadata must serialize to an object")
        self._user_metadata_value = metadata_value

        self._lock = asyncio.Lock()
        self._prepared = False
        self._closed = False
        self._handle: TextIO | None = None
        self._records: list[dict[str, Any]] = []
        self._latest_replayable: dict[_ReplayKey, dict[str, Any]] = {}
        self._latest_success: dict[_ReplayKey, dict[str, Any]] = {}
        self._next_sequence = 0
        self._detached_io_tasks: set[asyncio.Task[Any]] = set()
        self._detached_io_errors: list[Exception] = []

    async def prepare_item(self, work_item: LLMWorkItem[Any, Any, Any]) -> _ItemFingerprint:
        """Create/validate the artifact and fingerprint input before provider work."""
        self._resolve_identity_from(work_item.strategy)
        await self._prepare()
        return self._fingerprint_item(work_item)

    def _resolve_identity_from(self, strategy: Any) -> None:
        """Infer and pin the identity from the first item's strategy."""
        if self.identity is not None:
            return
        inferred = infer_artifact_identity(strategy)
        self.identity = inferred
        try:
            self._identity_value, self.identity_fingerprint = fingerprint_identity(inferred)
        except ResultSerializationError as exc:
            raise ArtifactSerializationError(str(exc)) from exc

    def _require_resolved_fingerprint(self) -> str:
        if self.identity_fingerprint is None:
            raise ArtifactError(
                "Artifact identity is not resolved yet. Pass "
                "identity=ArtifactIdentity(...) to JsonlArtifactStore, or run the "
                "store through a processor so prepare_item() can infer the "
                "identity from the strategy."
            )
        return self.identity_fingerprint

    def _fingerprint_item(self, work_item: LLMWorkItem[Any, Any, Any]) -> _ItemFingerprint:
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

    async def lookup(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        policy: ResumePolicy,
    ) -> WorkItemResult[Any, Any] | None:
        if policy is ResumePolicy.NONE:
            return None
        fingerprint = self._coerce_fingerprint(prepared_item)
        await self._prepare()
        key = self._replay_key(work_item, fingerprint)
        records = (
            self._latest_success
            if policy is ResumePolicy.REUSE_SUCCESSES
            else self._latest_replayable
        )
        record = records.get(key)
        if record is None or not self._compatible(record, work_item, fingerprint):
            return None
        try:
            result = restore_replayed_result(
                record,
                work_item,
                output_decoder=self.output_decoder,
                context_decoder=self.context_decoder,
            )
        except (KeyError, ResultSerializationError) as exc:
            raise ArtifactFormatError(
                f"Malformed stored result for item {work_item.item_id!r}: {exc}"
            ) from exc
        return result

    def _replay_key(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        fingerprint: _ItemFingerprint,
    ) -> _ReplayKey:
        return replay_key(
            work_item.item_id,
            fingerprint,
            self._require_resolved_fingerprint(),
        )

    @staticmethod
    def _record_replay_key(record: Mapping[str, Any]) -> _ReplayKey | None:
        return record_replay_key(record)

    def _index_record(self, record: dict[str, Any]) -> None:
        if not record.get("replay_eligible", False):
            return
        key = self._record_replay_key(record)
        if key is None:
            return
        self._latest_replayable[key] = record
        if record.get("success") is True:
            self._latest_success[key] = record

    def _rebuild_replay_index(self) -> None:
        self._latest_replayable.clear()
        self._latest_success.clear()
        for record in self._records:
            self._index_record(record)

    def _compatible(
        self,
        record: Mapping[str, Any],
        work_item: LLMWorkItem[Any, Any, Any],
        fingerprint: _ItemFingerprint,
    ) -> bool:
        return record_is_compatible(
            record,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            item_id=work_item.item_id,
            prepared_item=fingerprint,
            identity_fingerprint=self._require_resolved_fingerprint(),
        )

    async def append(
        self,
        work_item: LLMWorkItem[Any, Any, Any],
        prepared_item: Any,
        result: WorkItemResult[Any, Any],
    ) -> None:
        fingerprint = self._coerce_fingerprint(prepared_item)
        await self._prepare()
        try:
            identity_fingerprint = self._require_resolved_fingerprint()
            identity = self.identity
            identity_value = self._identity_value
            assert identity is not None and identity_value is not None
            record = build_item_record(
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                work_item=work_item,
                prepared_item=fingerprint,
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

        await self._append_record(record)

    @staticmethod
    def _coerce_fingerprint(value: Any) -> _ItemFingerprint:
        if not isinstance(value, _ItemFingerprint):
            raise ArtifactSerializationError("Artifact item was not prepared by this store")
        return value

    def _finish_detached_io(self, task: asyncio.Task[Any]) -> None:
        """Retain a cancelled caller's I/O failure for the next store operation."""
        if task not in self._detached_io_tasks:
            return
        self._detached_io_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._detached_io_errors.append(exc)

    def _raise_detached_io_error(self) -> None:
        for task in list(self._detached_io_tasks):
            if task.done():
                self._finish_detached_io(task)
        if self._detached_io_errors:
            raise self._detached_io_errors.pop(0)

    async def _run_lock_owner(self, awaitable: Any) -> Any:
        """Keep the store lock owned until threaded I/O ends after cancellation."""
        task = asyncio.create_task(awaitable)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # asyncio.to_thread() cannot stop an already-running thread. The
            # child task must therefore retain the lock after caller
            # cancellation so another append/close cannot overlap the handle.
            self._detached_io_tasks.add(task)
            task.add_done_callback(self._finish_detached_io)
            raise

    async def _prepare(self) -> None:
        await self._run_lock_owner(self._prepare_locked())

    async def _prepare_locked(self) -> None:
        async with self._lock:
            self._raise_detached_io_error()
            if self._prepared:
                if self._closed:
                    raise ArtifactIOError(f"Artifact store is closed: {self.path}")
                return
            if self._closed:
                raise ArtifactIOError(f"Artifact store is closed: {self.path}")
            try:
                records, needs_manifest = await asyncio.to_thread(
                    _read_artifact_records,
                    self.path,
                    allow_create=True,
                )
                self._records = records
                self._rebuild_replay_index()
                self._next_sequence = (
                    max((int(record.get("record_sequence", -1)) for record in records), default=-1)
                    + 1
                )
                if needs_manifest and self._identity_value is None:
                    raise ArtifactError(
                        f"Cannot create a new artifact {self.path} without a "
                        "resolved identity. Pass identity=ArtifactIdentity(...) "
                        "to JsonlArtifactStore, or run the store through a "
                        "processor so it can be inferred from the strategy."
                    )
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = self.path.open("a", encoding="utf-8", newline="\n")
                if needs_manifest:
                    assert self._identity_value is not None
                    assert self.identity_fingerprint is not None
                    manifest = build_manifest_record(
                        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                        package_version=_package_version(),
                        identity_value=self._identity_value,
                        identity_fingerprint=self.identity_fingerprint,
                        user_metadata=self._user_metadata_value,
                    )
                    await asyncio.to_thread(self._write_record_sync, manifest)
                self._prepared = True
            except ArtifactError:
                raise
            except OSError as exc:
                raise ArtifactIOError(f"Could not prepare artifact {self.path}: {exc}") from exc

    async def _append_record(self, record: dict[str, Any]) -> None:
        await self._run_lock_owner(self._append_record_locked(record))

    async def _append_record_locked(self, record: dict[str, Any]) -> None:
        async with self._lock:
            self._raise_detached_io_error()
            if self._closed or self._handle is None:
                raise ArtifactIOError(f"Artifact store is not writable: {self.path}")
            record["record_sequence"] = self._next_sequence
            try:
                await asyncio.to_thread(self._write_record_sync, record)
            except OSError as exc:
                raise ArtifactIOError(
                    f"Could not append artifact record for item {record.get('item_id')!r}: {exc}"
                ) from exc
            self._records.append(record)
            self._index_record(record)
            self._next_sequence += 1

    def _write_record_sync(self, record: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise OSError("artifact file is not open")
        line = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._handle.write(line + "\n")
        self._handle.flush()
        if self.fsync:
            os.fsync(self._handle.fileno())

    async def iter_results(
        self, *, successes_only: bool = False
    ) -> AsyncIterator[WorkItemResult[Any, Any]]:
        """Stream a finite, bounded-memory snapshot of complete item records."""
        if not self._prepared:
            try:
                exists_and_nonempty = await asyncio.to_thread(
                    lambda: self.path.exists() and self.path.stat().st_size > 0
                )
            except OSError as exc:
                raise ArtifactIOError(f"Could not inspect artifact {self.path}: {exc}") from exc
            if not exists_and_nonempty:
                # Preserve the historical create-on-first-operation behavior
                # for an explicitly identified empty store.
                await self._prepare()

        async def capture_snapshot() -> int:
            async with self._lock:
                self._raise_detached_io_error()
                return await asyncio.to_thread(_artifact_snapshot_size, self.path)

        snapshot_size = await self._run_lock_owner(capture_snapshot())
        offset = 0
        line_number = 0
        manifest_seen = False
        done = False
        while not done:
            page, offset, line_number, manifest_seen, done = await asyncio.to_thread(
                _read_artifact_page,
                self.path,
                offset=offset,
                snapshot_size=snapshot_size,
                line_number=line_number,
                manifest_seen=manifest_seen,
                page_size=_JSONL_ITER_PAGE_SIZE,
            )
            for record in page:
                if successes_only and not record.get("success"):
                    continue
                try:
                    yield decode_stored_result(
                        record,
                        output_decoder=self.output_decoder,
                        context_decoder=self.context_decoder,
                    )
                except (KeyError, ResultSerializationError) as exc:
                    sequence = record.get("record_sequence")
                    item_id = record.get("item_id")
                    raise ArtifactFormatError(
                        f"Malformed stored result at sequence {sequence!r} "
                        f"for item {item_id!r}: {exc}"
                    ) from exc

    async def close(self) -> None:
        await self._run_lock_owner(self._close_locked())

    async def _close_locked(self) -> None:
        async with self._lock:
            if self._closed:
                self._raise_detached_io_error()
                return
            self._closed = True
            handle = self._handle
            self._handle = None
            if handle is None:
                self._raise_detached_io_error()
                return
            try:
                await asyncio.to_thread(self._close_sync, handle)
            except OSError as exc:
                raise ArtifactIOError(f"Could not close artifact {self.path}: {exc}") from exc
            self._raise_detached_io_error()

    def _close_sync(self, handle: TextIO) -> None:
        handle.flush()
        if self.fsync:
            os.fsync(handle.fileno())
        handle.close()

    @classmethod
    def read_results(
        cls,
        path: str | Path,
        *,
        successes_only: bool = False,
        output_decoder: ValueDecoder | None = None,
        context_decoder: ValueDecoder | None = None,
    ) -> BatchResult[Any, Any]:
        """Read stored results without opening a writer or calling a provider."""
        records, _ = _read_artifact_records(Path(path), allow_create=False)
        results: list[WorkItemResult[Any, Any]] = []
        for record in records:
            if successes_only and not record.get("success"):
                continue
            try:
                results.append(
                    decode_stored_result(
                        record,
                        output_decoder=output_decoder,
                        context_decoder=context_decoder,
                    )
                )
            except (KeyError, ResultSerializationError) as exc:
                raise ArtifactFormatError(f"Malformed stored result: {exc}") from exc
        return BatchResult(results=results, termination=BatchTermination())


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactFormatError",
    "ArtifactIOError",
    "ArtifactIdentity",
    "ArtifactSerializationError",
    "ArtifactStore",
    "JsonlArtifactStore",
    "ResumePolicy",
]
