"""Strict, provider-neutral result serialization.

The codec deliberately converts application values to JSON primitives instead
of attempting to recreate arbitrary Python objects. Callers that need typed
rehydration can provide explicit output/context decoders.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias, cast
from uuid import UUID

from .base import (
    AttemptTiming,
    BatchResult,
    BatchTermination,
    TokenUsage,
    WorkItemResult,
    WorkItemTiming,
)

RESULT_SCHEMA_NAME = "async-batch-llm-result"
RESULT_SCHEMA_VERSION = 1

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
ValueEncoder: TypeAlias = Callable[[Any], Any]
ValueDecoder: TypeAlias = Callable[[JSONValue], Any]
_MAX_CUSTOM_ENCODER_DEPTH = 32


class ResultSerializationError(ValueError):
    """A result value or serialized payload is not safely supported."""


_SENSITIVE_KEY = re.compile(
    r"^(?:authorization|proxy-authorization|api[_-]?key|access[_-]?token|secret)$",
    re.IGNORECASE,
)
_SENSITIVE_TEXT = (
    re.compile(
        r"(?i)(\bauthorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+",
    ),
    re.compile(
        r"(?i)(\b(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*)"
        r"(?:['\"]?)[^'\"\s,;]+(?:['\"]?)",
    ),
)


def _redact_sensitive_text(value: str) -> str:
    """Remove labeled credentials from framework-controlled persisted text."""
    for pattern in _SENSITIVE_TEXT:
        value = pattern.sub(r"\1[REDACTED]", value)
    return value


def to_json_value(value: Any, *, encoder: ValueEncoder | None = None, path: str = "$") -> JSONValue:
    """Convert a supported Python value into deterministic JSON primitives.

    Tuples and sets normalize to lists. Dataclasses and Pydantic models
    normalize to mappings. Date/time, UUID, enum, and filesystem path values
    normalize to strings or their JSON-safe enum values.
    """
    return _to_json_value(
        value,
        encoder=encoder,
        path=path,
        encoder_depth=0,
        redact_sensitive_keys=True,
    )


def _to_fingerprint_value(
    value: Any, *, encoder: ValueEncoder | None = None, path: str = "$"
) -> JSONValue:
    """Canonicalize private hash input without dropping sensitive identity data.

    The returned value must only be used as input to a one-way fingerprint. It
    deliberately preserves sensitive mapping values so credential changes
    invalidate replay, while :func:`to_json_value` continues to redact every
    persisted mapping.
    """
    return _to_json_value(
        value,
        encoder=encoder,
        path=path,
        encoder_depth=0,
        redact_sensitive_keys=False,
    )


def _to_json_value(
    value: Any,
    *,
    encoder: ValueEncoder | None,
    path: str,
    encoder_depth: int,
    redact_sensitive_keys: bool,
) -> JSONValue:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResultSerializationError(f"Unsupported non-finite float at {path}: {value!r}")
        return value
    if isinstance(value, Enum):
        return _to_json_value(
            value.value,
            encoder=encoder,
            path=path,
            encoder_depth=encoder_depth,
            redact_sensitive_keys=redact_sensitive_keys,
        )
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, os.PathLike):
        path_value = os.fspath(value)
        if not isinstance(path_value, str):
            raise ResultSerializationError(f"Filesystem path must resolve to text at {path}")
        return path_value

    # Pydantic is already a required dependency, but keep the core codec
    # duck-typed so Pydantic-specific public types do not leak into the API.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except Exception as exc:
            raise ResultSerializationError(
                f"Could not serialize Pydantic value at {path}: {exc}"
            ) from exc
        return _to_json_value(
            dumped,
            encoder=encoder,
            path=path,
            encoder_depth=encoder_depth,
            redact_sensitive_keys=redact_sensitive_keys,
        )

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: (
                "[REDACTED]"
                if redact_sensitive_keys and _SENSITIVE_KEY.match(item.name)
                else _to_json_value(
                    getattr(value, item.name),
                    encoder=encoder,
                    path=f"{path}.{item.name}",
                    encoder_depth=encoder_depth,
                    redact_sensitive_keys=redact_sensitive_keys,
                )
            )
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResultSerializationError(
                    f"JSON object keys must be strings at {path}; got {type(key).__name__}"
                )
            result[key] = (
                "[REDACTED]"
                if redact_sensitive_keys and _SENSITIVE_KEY.match(key)
                else _to_json_value(
                    item,
                    encoder=encoder,
                    path=f"{path}.{key}",
                    encoder_depth=encoder_depth,
                    redact_sensitive_keys=redact_sensitive_keys,
                )
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _to_json_value(
                item,
                encoder=encoder,
                path=f"{path}[{index}]",
                encoder_depth=encoder_depth,
                redact_sensitive_keys=redact_sensitive_keys,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        encoded = [
            _to_json_value(
                item,
                encoder=encoder,
                path=f"{path}[]",
                encoder_depth=encoder_depth,
                redact_sensitive_keys=redact_sensitive_keys,
            )
            for item in value
        ]
        try:
            return sorted(encoded, key=_canonical_json)
        except TypeError as exc:  # pragma: no cover - _canonical_json accepts JSONValue
            raise ResultSerializationError(
                f"Could not deterministically order set at {path}"
            ) from exc

    if encoder is not None:
        if encoder_depth >= _MAX_CUSTOM_ENCODER_DEPTH:
            raise ResultSerializationError(
                f"Custom encoder exceeded {_MAX_CUSTOM_ENCODER_DEPTH} recursive conversions "
                f"at {path}; ensure it eventually returns supported JSON-safe data."
            )
        try:
            converted = encoder(value)
        except Exception as exc:
            raise ResultSerializationError(
                f"Custom encoder failed for {type(value).__name__} at {path}: {exc}"
            ) from exc
        if converted is value:
            raise ResultSerializationError(
                f"Custom encoder returned the original unsupported value at {path}"
            )
        return _to_json_value(
            converted,
            encoder=encoder,
            path=path,
            encoder_depth=encoder_depth + 1,
            redact_sensitive_keys=redact_sensitive_keys,
        )

    raise ResultSerializationError(
        f"Unsupported value of type {type(value).__module__}.{type(value).__qualname__} "
        f"at {path}; pass an encoder that returns JSON-safe data."
    )


def _canonical_json(value: JSONValue) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultSerializationError(f"Expected an object at {path}; got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _check_schema(data: Mapping[str, Any], *, record_type: str) -> None:
    if data.get("schema") != RESULT_SCHEMA_NAME:
        raise ResultSerializationError(
            f"Unsupported or missing result schema: {data.get('schema')!r}"
        )
    version = data.get("schema_version")
    if version != RESULT_SCHEMA_VERSION:
        if isinstance(version, int) and version > RESULT_SCHEMA_VERSION:
            raise ResultSerializationError(
                f"Unsupported future result schema version {version}; "
                f"this library supports version {RESULT_SCHEMA_VERSION}."
            )
        raise ResultSerializationError(f"Unsupported result schema version: {version!r}")
    if data.get("record_type") != record_type:
        raise ResultSerializationError(
            f"Expected record_type={record_type!r}; got {data.get('record_type')!r}"
        )


def _attempt_to_dict(value: AttemptTiming) -> dict[str, JSONValue]:
    return {
        "attempt": value.attempt,
        "try_number": value.try_number,
        "total_seconds": value.total_seconds,
        "admission_wait_seconds": value.admission_wait_seconds,
        "startup_ramp_wait_seconds": value.startup_ramp_wait_seconds,
        "execution_seconds": value.execution_seconds,
        "provider_seconds": value.provider_seconds,
        "cooldown_wait_seconds": value.cooldown_wait_seconds,
        "quota_wait_seconds": value.quota_wait_seconds,
        "estimated_input_tokens": value.estimated_input_tokens,
        "estimated_output_tokens": value.estimated_output_tokens,
        "reserved_tokens": value.reserved_tokens,
        "reported_tokens": value.reported_tokens,
        "reconciliation_delta_tokens": value.reconciliation_delta_tokens,
        "quota_scope_id": value.quota_scope_id,
        "retry_backoff_seconds": value.retry_backoff_seconds,
        "success": value.success,
        "error_type": value.error_type,
        "error_category": value.error_category,
        "timeout_category": value.timeout_category,
    }


def _attempt_from_dict(value: Any, *, path: str) -> AttemptTiming:
    data = _require_mapping(value, path=path)
    try:
        return AttemptTiming(
            attempt=int(data["attempt"]),
            try_number=int(data["try_number"]),
            total_seconds=float(data.get("total_seconds", 0.0)),
            admission_wait_seconds=float(data.get("admission_wait_seconds", 0.0)),
            startup_ramp_wait_seconds=float(data.get("startup_ramp_wait_seconds", 0.0)),
            execution_seconds=float(data.get("execution_seconds", 0.0)),
            provider_seconds=(
                None if data.get("provider_seconds") is None else float(data["provider_seconds"])
            ),
            cooldown_wait_seconds=float(data.get("cooldown_wait_seconds", 0.0)),
            quota_wait_seconds=float(data.get("quota_wait_seconds", 0.0)),
            estimated_input_tokens=_optional_int(
                data.get("estimated_input_tokens"),
                path=f"{path}.estimated_input_tokens",
                non_negative=True,
            ),
            estimated_output_tokens=_optional_int(
                data.get("estimated_output_tokens"),
                path=f"{path}.estimated_output_tokens",
                non_negative=True,
            ),
            reserved_tokens=_required_int(
                data.get("reserved_tokens", 0),
                path=f"{path}.reserved_tokens",
                non_negative=True,
            ),
            reported_tokens=_optional_int(
                data.get("reported_tokens"),
                path=f"{path}.reported_tokens",
                non_negative=True,
            ),
            reconciliation_delta_tokens=_optional_int(
                data.get("reconciliation_delta_tokens"),
                path=f"{path}.reconciliation_delta_tokens",
            ),
            quota_scope_id=_optional_int(
                data.get("quota_scope_id"),
                path=f"{path}.quota_scope_id",
                non_negative=True,
            ),
            retry_backoff_seconds=float(data.get("retry_backoff_seconds", 0.0)),
            success=bool(data.get("success", False)),
            error_type=_optional_str(data.get("error_type")),
            error_category=_optional_str(data.get("error_category")),
            timeout_category=_optional_str(data.get("timeout_category")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultSerializationError(f"Malformed attempt timing at {path}: {exc}") from exc


def _timing_to_dict(value: WorkItemTiming) -> dict[str, JSONValue]:
    return {
        "total_seconds": value.total_seconds,
        "attempts": [_attempt_to_dict(attempt) for attempt in value.attempts],
        "timeout_category": value.timeout_category,
    }


def _timing_from_dict(value: Any) -> WorkItemTiming:
    data = _require_mapping(value, path="$.timing")
    attempts = data.get("attempts", [])
    if not isinstance(attempts, list):
        raise ResultSerializationError("Expected a list at $.timing.attempts")
    try:
        return WorkItemTiming(
            total_seconds=float(data.get("total_seconds", 0.0)),
            attempts=[
                _attempt_from_dict(attempt, path=f"$.timing.attempts[{index}]")
                for index, attempt in enumerate(attempts)
            ],
            timeout_category=_optional_str(data.get("timeout_category")),
        )
    except (TypeError, ValueError) as exc:
        raise ResultSerializationError(f"Malformed work-item timing: {exc}") from exc


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _required_int(value: Any, *, path: str, non_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResultSerializationError(f"Expected an integer at {path}")
    if non_negative and value < 0:
        raise ResultSerializationError(f"Expected a non-negative integer at {path}")
    return int(value)


def _optional_int(value: Any, *, path: str, non_negative: bool = False) -> int | None:
    if value is None:
        return None
    return _required_int(value, path=path, non_negative=non_negative)


def _optional_seconds(value: Any, *, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultSerializationError(f"Expected a number or null at {path}")
    return float(value)


def _exception_descriptor(exception: Exception | None) -> dict[str, JSONValue] | None:
    if exception is None:
        return None
    return {
        "module": type(exception).__module__,
        "class_name": type(exception).__qualname__,
        "message": _redact_sensitive_text(str(exception)),
    }


def work_item_result_to_dict(
    result: WorkItemResult[Any, Any],
    *,
    encoder: ValueEncoder | None = None,
    include_output: bool = True,
    include_context: bool = True,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """Encode one result using result schema version 1."""
    return {
        "schema": RESULT_SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "record_type": "work_item_result",
        "item_id": result.item_id,
        "success": result.success,
        "output": (
            to_json_value(result.output, encoder=encoder, path="$.output")
            if include_output
            else None
        ),
        "output_included": include_output,
        "error": _redact_sensitive_text(result.error) if result.error is not None else None,
        "error_category": result.error_category,
        "context": (
            to_json_value(result.context, encoder=encoder, path="$.context")
            if include_context
            else None
        ),
        "context_included": include_context,
        "token_usage": to_json_value(dict(result.token_usage), path="$.token_usage"),
        "metadata": (
            to_json_value(result.metadata, encoder=encoder, path="$.metadata")
            if include_metadata
            else None
        ),
        "metadata_included": include_metadata,
        "exception": _exception_descriptor(result.exception),
        "admission_wait_seconds": result.admission_wait_seconds,
        "timing": _timing_to_dict(result.timing),
        "submission_index": result.submission_index,
        "replayed_from_artifact": result.replayed_from_artifact,
    }


def work_item_result_from_dict(
    value: Any,
    *,
    output_decoder: ValueDecoder | None = None,
    context_decoder: ValueDecoder | None = None,
) -> WorkItemResult[Any, Any]:
    """Decode one trusted schema mapping without importing arbitrary classes."""
    data = _require_mapping(value, path="$")
    _check_schema(data, record_type="work_item_result")
    try:
        item_id = data["item_id"]
        success = data["success"]
        if not isinstance(item_id, str) or not isinstance(success, bool):
            raise TypeError("item_id must be str and success must be bool")
        output = cast(JSONValue, data.get("output"))
        context = cast(JSONValue, data.get("context"))
        raw_tokens = _require_mapping(data.get("token_usage", {}), path="$.token_usage")
        token_values: dict[str, int] = {}
        for key, item in raw_tokens.items():
            if not isinstance(key, str):
                raise TypeError("token_usage keys must be strings")
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(f"token_usage[{key!r}] must be an integer")
            token_values[key] = item
        tokens = cast(TokenUsage, token_values)
        metadata = data.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be an object or null")
        submission_index = data.get("submission_index")
        if submission_index is not None and (
            isinstance(submission_index, bool) or not isinstance(submission_index, int)
        ):
            raise TypeError("submission_index must be an integer or null")
        return WorkItemResult(
            item_id=item_id,
            success=success,
            output=output_decoder(output) if output_decoder is not None else output,
            error=_optional_str(data.get("error")),
            context=context_decoder(context) if context_decoder is not None else context,
            token_usage=tokens,
            metadata=cast(dict[str, Any] | None, metadata),
            exception=None,  # Never instantiate a class named by untrusted JSON.
            admission_wait_seconds=float(data.get("admission_wait_seconds", 0.0)),
            timing=_timing_from_dict(data.get("timing", {})),
            submission_index=submission_index,
            error_category=_optional_str(data.get("error_category")),
            replayed_from_artifact=bool(data.get("replayed_from_artifact", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ResultSerializationError):
            raise
        raise ResultSerializationError(f"Malformed work-item result: {exc}") from exc


def _termination_to_dict(value: BatchTermination) -> dict[str, JSONValue]:
    return {
        "kind": value.kind,
        "reason": value.reason,
        "error_category": value.error_category,
        "triggering_item_id": value.triggering_item_id,
    }


def _termination_from_dict(value: Any) -> BatchTermination:
    data = _require_mapping(value, path="$.termination")
    kind = data.get("kind", "completed")
    if not isinstance(kind, str):
        raise ResultSerializationError("Batch termination kind must be a string")
    allowed = {"completed", "batch_timeout", "fail_fast", "artifact_error"}
    if kind not in allowed:
        raise ResultSerializationError(f"Unsupported batch termination kind: {kind!r}")
    return BatchTermination(
        kind=cast(Any, kind),
        reason=_optional_str(data.get("reason")),
        error_category=_optional_str(data.get("error_category")),
        triggering_item_id=_optional_str(data.get("triggering_item_id")),
    )


def batch_result_to_dict(
    batch: BatchResult[Any, Any], *, encoder: ValueEncoder | None = None
) -> dict[str, Any]:
    """Encode a batch; aggregate counters are recomputed on decode."""
    return {
        "schema": RESULT_SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "record_type": "batch_result",
        "termination": _termination_to_dict(batch.termination),
        "wall_time_seconds": batch.wall_time_seconds,
        "results": [work_item_result_to_dict(result, encoder=encoder) for result in batch.results],
    }


def batch_result_from_dict(
    value: Any,
    *,
    output_decoder: ValueDecoder | None = None,
    context_decoder: ValueDecoder | None = None,
) -> BatchResult[Any, Any]:
    data = _require_mapping(value, path="$")
    _check_schema(data, record_type="batch_result")
    results = data.get("results")
    if not isinstance(results, list):
        raise ResultSerializationError("Expected a list at $.results")
    return BatchResult(
        results=[
            work_item_result_from_dict(
                result,
                output_decoder=output_decoder,
                context_decoder=context_decoder,
            )
            for result in results
        ],
        termination=_termination_from_dict(data.get("termination", {"kind": "completed"})),
        wall_time_seconds=_optional_seconds(
            data.get("wall_time_seconds"), path="$.wall_time_seconds"
        ),
    )


def batch_result_to_json(
    batch: BatchResult[Any, Any],
    *,
    encoder: ValueEncoder | None = None,
    indent: int | None = 2,
) -> str:
    try:
        return json.dumps(
            batch_result_to_dict(batch, encoder=encoder),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ResultSerializationError):
            raise
        raise ResultSerializationError(f"Could not encode batch JSON: {exc}") from exc


def batch_result_from_json(
    value: str | bytes,
    *,
    output_decoder: ValueDecoder | None = None,
    context_decoder: ValueDecoder | None = None,
) -> BatchResult[Any, Any]:
    try:
        data = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ResultSerializationError(f"Malformed result JSON: {exc}") from exc
    return batch_result_from_dict(
        data,
        output_decoder=output_decoder,
        context_decoder=context_decoder,
    )


def batch_result_to_jsonl(
    batch: BatchResult[Any, Any], path: str | Path, *, encoder: ValueEncoder | None = None
) -> None:
    termination = _termination_to_dict(batch.termination)
    lines = []
    if not batch.results:
        lines.append(
            json.dumps(
                {
                    "schema": RESULT_SCHEMA_NAME,
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "record_type": "batch_metadata",
                    "batch_termination": termination,
                    "batch_wall_time_seconds": batch.wall_time_seconds,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    for result in batch.results:
        record = work_item_result_to_dict(result, encoder=encoder)
        record["batch_termination"] = termination
        record["batch_wall_time_seconds"] = batch.wall_time_seconds
        lines.append(
            json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )
    try:
        Path(path).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    except OSError as exc:
        raise ResultSerializationError(f"Could not write result JSONL {path!s}: {exc}") from exc


def batch_result_from_jsonl(
    path: str | Path,
    *,
    output_decoder: ValueDecoder | None = None,
    context_decoder: ValueDecoder | None = None,
) -> BatchResult[Any, Any]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ResultSerializationError(f"Could not read result JSONL {path!s}: {exc}") from exc
    results: list[WorkItemResult[Any, Any]] = []
    termination: BatchTermination | None = None
    wall_time_seconds: float | None = None
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResultSerializationError(
                f"Malformed result JSONL at line {line_number}: {exc}"
            ) from exc
        data = _require_mapping(record, path=f"line {line_number}")
        if wall_time_seconds is None:
            wall_time_seconds = _optional_seconds(
                data.get("batch_wall_time_seconds"),
                path=f"line {line_number} batch_wall_time_seconds",
            )
        if data.get("record_type") == "batch_metadata":
            _check_schema(data, record_type="batch_metadata")
            if results or termination is not None:
                raise ResultSerializationError(
                    f"Unexpected batch metadata record at JSONL line {line_number}"
                )
            termination = _termination_from_dict(
                data.get("batch_termination", {"kind": "completed"})
            )
            continue
        current = _termination_from_dict(data.get("batch_termination", {"kind": "completed"}))
        if termination is None:
            termination = current
        elif current != termination:
            raise ResultSerializationError(
                f"Inconsistent batch termination metadata at JSONL line {line_number}"
            )
        results.append(
            work_item_result_from_dict(
                data,
                output_decoder=output_decoder,
                context_decoder=context_decoder,
            )
        )
    return BatchResult(
        results=results,
        termination=termination or BatchTermination(),
        wall_time_seconds=wall_time_seconds,
    )


__all__ = [
    "JSONValue",
    "RESULT_SCHEMA_NAME",
    "RESULT_SCHEMA_VERSION",
    "ResultSerializationError",
    "ValueDecoder",
    "ValueEncoder",
    "to_json_value",
]
