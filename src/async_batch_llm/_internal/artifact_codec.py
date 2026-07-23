"""Shared private codec for artifact storage backends.

The logical artifact schema belongs to :mod:`async_batch_llm.artifacts`; this
module only centralizes deterministic encoding, fingerprinting, replay keys,
and safe result restoration so physical stores cannot drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeAlias

from ..base import LLMWorkItem, WorkItemResult
from ..serialization import (
    JSONValue,
    ResultSerializationError,
    ValueDecoder,
    ValueEncoder,
    _to_fingerprint_value,
    to_json_value,
    work_item_result_from_dict,
    work_item_result_to_dict,
)

ReplayKey: TypeAlias = tuple[str, str, str, str | None, str]
CostCalculator: TypeAlias = Callable[[WorkItemResult[Any, Any]], float | None]
ContextFingerprinter: TypeAlias = Callable[[Any], str]


@dataclass(frozen=True)
class PreparedArtifactItem:
    """Canonical compatibility fingerprints for one accepted input."""

    prompt_fingerprint: str
    context_fingerprint: str | None
    input_fingerprint: str
    legacy_context_fingerprint: str | None = None
    legacy_input_fingerprint: str | None = None

    # Private compatibility aliases for the v0.18 JSONL implementation.
    @property
    def prompt(self) -> str:
        return self.prompt_fingerprint

    @property
    def context(self) -> str | None:
        return self.context_fingerprint

    @property
    def combined(self) -> str:
        return self.input_fingerprint


def utc_now() -> str:
    """Return a stable UTC timestamp spelling for persisted records."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: JSONValue) -> str:
    """Encode JSON primitives deterministically for hashing and storage."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint_json(value: JSONValue) -> str:
    """Return the canonical SHA-256 fingerprint of a JSON-safe value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def identity_mapping(identity: Any, *, for_fingerprint: bool = False) -> dict[str, JSONValue]:
    """Normalize the public identity shape for persistence or private hashing."""
    converter = _to_fingerprint_value if for_fingerprint else to_json_value
    extra = converter(identity.extra, path="$.identity.extra")
    if not isinstance(extra, dict):  # Mapping always normalizes to an object.
        raise ResultSerializationError("ArtifactIdentity.extra must serialize to an object")
    return {
        "provider": identity.provider,
        "model": identity.model,
        "prompt_version": identity.prompt_version,
        "parser_version": identity.parser_version,
        "application_version": identity.application_version,
        "extra": extra,
    }


def fingerprint_identity(identity: Any) -> tuple[dict[str, JSONValue], str]:
    """Return the redacted persisted identity and private compatibility hash."""
    persisted = identity_mapping(identity)
    fingerprint = fingerprint_json(identity_mapping(identity, for_fingerprint=True))
    return persisted, fingerprint


def fingerprint_work_item(
    work_item: LLMWorkItem[Any, Any, Any],
    *,
    context_in_identity: bool,
    encoder: ValueEncoder | None,
    context_fingerprinter: ContextFingerprinter | None,
) -> PreparedArtifactItem:
    """Fingerprint one input without retaining its raw prompt or context."""
    prompt_hash = hashlib.sha256(work_item.prompt.encode("utf-8")).hexdigest()
    context_hash: str | None = None
    legacy_context_hash: str | None = None
    legacy_combined_hash: str | None = None
    if context_in_identity and work_item.context is not None:
        if context_fingerprinter is not None:
            context_hash = context_fingerprinter(work_item.context)
            if not isinstance(context_hash, str) or not context_hash:
                raise ResultSerializationError(
                    "context_fingerprinter must return a non-empty string"
                )
        else:
            context_value = _to_fingerprint_value(
                work_item.context,
                encoder=encoder,
                path=f"$.items[{work_item.item_id!r}].context",
            )
            context_hash = fingerprint_json(context_value)
    combined_hash = fingerprint_json(
        {
            "item_id": work_item.item_id,
            "prompt_fingerprint": prompt_hash,
            "context_fingerprint": context_hash,
        }
    )
    # v0.20 hashed a Python ``None`` context instead of representing absence
    # as nullable. Keep its key as a read-only JSONL fallback while all new
    # logical records (and SQLite predicates) use an actual NULL.
    if context_in_identity and work_item.context is None:
        legacy_context_hash = fingerprint_json(None)
        legacy_combined_hash = fingerprint_json(
            {
                "item_id": work_item.item_id,
                "prompt_fingerprint": prompt_hash,
                "context_fingerprint": legacy_context_hash,
            }
        )
    return PreparedArtifactItem(
        prompt_fingerprint=prompt_hash,
        context_fingerprint=context_hash,
        input_fingerprint=combined_hash,
        legacy_context_fingerprint=legacy_context_hash,
        legacy_input_fingerprint=legacy_combined_hash,
    )


def build_manifest_record(
    *,
    artifact_schema_version: int,
    package_version: str,
    identity_value: Mapping[str, JSONValue],
    identity_fingerprint: str,
    user_metadata: Mapping[str, JSONValue],
) -> dict[str, Any]:
    """Build the logical creation-manifest record shared by physical stores."""
    return {
        "record_type": "manifest",
        "artifact_schema_version": artifact_schema_version,
        "created_at": utc_now(),
        "package_version": package_version,
        "identity": dict(identity_value),
        "identity_fingerprint": identity_fingerprint,
        "user_metadata": dict(user_metadata),
    }


def build_item_record(
    *,
    artifact_schema_version: int,
    work_item: LLMWorkItem[Any, Any, Any],
    prepared_item: PreparedArtifactItem,
    result: WorkItemResult[Any, Any],
    identity_value: Mapping[str, JSONValue],
    identity_fingerprint: str,
    identity: Any,
    include_output: bool,
    include_metadata: bool,
    include_prompt: bool,
    include_context: bool,
    encoder: ValueEncoder | None,
    cost_calculator: CostCalculator | None,
) -> dict[str, Any]:
    """Build one backend-neutral logical terminal item record."""
    serialized_result = work_item_result_to_dict(
        result,
        encoder=encoder,
        include_output=include_output,
        include_context=False,
        include_metadata=include_metadata,
    )
    raw_context = (
        to_json_value(work_item.context, encoder=encoder, path="$.raw_context")
        if include_context
        else None
    )
    raw_prompt = to_json_value(work_item.prompt, path="$.raw_prompt") if include_prompt else None
    cost = cost_calculator(result) if cost_calculator is not None else None
    if cost is not None:
        cost = float(cost)
        to_json_value(cost, path="$.calculated_cost")

    strategy_type = type(work_item.strategy)
    return {
        "record_type": "item",
        "artifact_schema_version": artifact_schema_version,
        "recorded_at": utc_now(),
        "record_sequence": None,
        "item_id": work_item.item_id,
        "submission_index": result.submission_index,
        "prompt_fingerprint": prepared_item.prompt_fingerprint,
        "context_fingerprint": prepared_item.context_fingerprint,
        "input_fingerprint": prepared_item.input_fingerprint,
        "identity_fingerprint": identity_fingerprint,
        "strategy_class": f"{strategy_type.__module__}.{strategy_type.__qualname__}",
        "provider": identity.provider,
        "model": identity.model,
        "prompt_version": identity.prompt_version,
        "parser_version": identity.parser_version,
        "application_version": identity.application_version,
        "identity": dict(identity_value),
        "success": result.success,
        "error_category": result.error_category,
        "token_usage": serialized_result["token_usage"],
        "timing": serialized_result["timing"],
        "calculated_cost": cost,
        "replay_eligible": (not result.success) or include_output,
        "raw_prompt": raw_prompt,
        "raw_context": raw_context,
        "result": serialized_result,
    }


def replay_key(
    item_id: str,
    prepared_item: PreparedArtifactItem,
    identity_fingerprint: str,
) -> ReplayKey:
    """Build the complete compatibility key used by every backend."""
    return (
        identity_fingerprint,
        item_id,
        prepared_item.prompt_fingerprint,
        prepared_item.context_fingerprint,
        prepared_item.input_fingerprint,
    )


def record_replay_key(record: Mapping[str, Any]) -> ReplayKey | None:
    """Extract a valid complete replay key from a logical item record."""
    item_id = record.get("item_id")
    prompt = record.get("prompt_fingerprint")
    context = record.get("context_fingerprint")
    combined = record.get("input_fingerprint")
    identity = record.get("identity_fingerprint")
    if (
        not isinstance(item_id, str)
        or not isinstance(prompt, str)
        or (context is not None and not isinstance(context, str))
        or not isinstance(combined, str)
        or not isinstance(identity, str)
    ):
        return None
    return identity, item_id, prompt, context, combined


def record_is_compatible(
    record: Mapping[str, Any],
    *,
    artifact_schema_version: int,
    item_id: str,
    prepared_item: PreparedArtifactItem,
    identity_fingerprint: str,
) -> bool:
    """Check logical replay compatibility independently of physical storage."""
    if record.get("artifact_schema_version") != artifact_schema_version:
        return False
    candidate = record_replay_key(record)
    if candidate == replay_key(item_id, prepared_item, identity_fingerprint):
        return True
    if (
        prepared_item.legacy_context_fingerprint is None
        or prepared_item.legacy_input_fingerprint is None
    ):
        return False
    return candidate == (
        identity_fingerprint,
        item_id,
        prepared_item.prompt_fingerprint,
        prepared_item.legacy_context_fingerprint,
        prepared_item.legacy_input_fingerprint,
    )


def decode_stored_result(
    record: Mapping[str, Any],
    *,
    output_decoder: ValueDecoder | None,
    context_decoder: ValueDecoder | None,
) -> WorkItemResult[Any, Any]:
    """Safely decode the terminal result embedded in a logical item record."""
    return work_item_result_from_dict(
        record["result"],
        output_decoder=output_decoder,
        context_decoder=context_decoder,
    )


def restore_replayed_result(
    record: Mapping[str, Any],
    work_item: LLMWorkItem[Any, Any, Any],
    *,
    output_decoder: ValueDecoder | None,
    context_decoder: ValueDecoder | None,
) -> WorkItemResult[Any, Any]:
    """Decode a record and bind it to the current submission for replay."""
    result = decode_stored_result(
        record,
        output_decoder=output_decoder,
        context_decoder=context_decoder,
    )
    result.context = work_item.context
    result.submission_index = work_item.submission_index
    result.replayed_from_artifact = True
    result.exception = None
    return result
