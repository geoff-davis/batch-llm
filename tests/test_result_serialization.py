"""Stable ordering and strict, versioned result serialization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import BaseModel

from async_batch_llm import (
    AttemptTiming,
    BatchResult,
    BatchTermination,
    ProcessorConfig,
    ResultSerializationError,
    WorkItemResult,
    WorkItemTiming,
    process_prompts,
)
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.llm_strategies import LLMCallStrategy


class _OrderedStrategy(LLMCallStrategy[str]):
    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        if prompt == "slow":
            await asyncio.sleep(0.03)
        if prompt == "fail":
            raise ValueError("invalid input")
        return prompt.upper(), {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}, None


@pytest.mark.asyncio
async def test_process_prompts_default_completion_order_and_opt_in_input_order() -> None:
    prompts = [("dup", "slow"), ("dup", "fast"), ("failure", "fail")]
    config = ProcessorConfig(max_workers=3)

    completion = await process_prompts(_OrderedStrategy(), prompts, config=config)
    ordered = await process_prompts(_OrderedStrategy(), prompts, config=config, preserve_order=True)

    assert [result.output for result in completion.results[:2]] == ["FAST", None]
    assert [result.submission_index for result in ordered.results] == [0, 1, 2]
    assert [result.item_id for result in ordered.results] == ["dup", "dup", "failure"]
    assert [result.success for result in ordered.results] == [True, True, False]


def test_in_input_order_is_non_mutating_and_requires_indexes() -> None:
    later = WorkItemResult(item_id="dup", success=True, submission_index=1)
    earlier = WorkItemResult(item_id="dup", success=False, submission_index=0)
    batch = BatchResult(results=[later, earlier])

    ordered = batch.in_input_order()

    assert ordered is not batch
    assert ordered.results == [earlier, later]
    assert batch.results == [later, earlier]
    with pytest.raises(ValueError, match="lack submission_index"):
        BatchResult(results=[WorkItemResult(item_id="missing", success=True)]).in_input_order()


def test_attempt_timing_preserves_v021_positional_layout() -> None:
    timing = AttemptTiming(
        1,
        2,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        True,
        "Error",
        "category",
        "timeout",
    )

    assert timing.retry_backoff_seconds == 9
    assert timing.success is True
    assert timing.error_type == "Error"
    assert timing.error_category == "category"
    assert timing.timeout_category == "timeout"
    assert timing.quota_wait_seconds == 0


class _Output(BaseModel):
    label: str


@dataclass
class _Context:
    created: datetime
    request_id: UUID
    path: Path


def _rich_result() -> WorkItemResult[Any, Any]:
    timing = WorkItemTiming(
        total_seconds=1.25,
        timeout_category="provider_or_transport_timeout",
        attempts=[
            AttemptTiming(
                attempt=1,
                try_number=1,
                total_seconds=0.5,
                admission_wait_seconds=0.1,
                startup_ramp_wait_seconds=0.02,
                execution_seconds=0.3,
                provider_seconds=0.25,
                cooldown_wait_seconds=0.04,
                quota_wait_seconds=0.03,
                estimated_input_tokens=12,
                estimated_output_tokens=8,
                reserved_tokens=20,
                reported_tokens=0,
                reconciliation_delta_tokens=-20,
                quota_scope_id=2,
                retry_backoff_seconds=0.1,
                success=False,
                error_type="RuntimeError",
                error_category="server_error",
                timeout_category="provider_or_transport_timeout",
            )
        ],
    )
    return WorkItemResult(
        item_id="item-1",
        success=True,
        output=_Output(label="ok"),
        context=_Context(
            created=datetime(2026, 7, 13, 12, 30, tzinfo=timezone.utc),
            request_id=UUID("12345678-1234-5678-1234-567812345678"),
            path=Path("inputs/a.json"),
        ),
        token_usage={
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "cached_input_tokens": 3,
        },
        metadata={"tags": ("a", "b"), "choices": {"x", "y"}},
        admission_wait_seconds=0.1,
        timing=timing,
        submission_index=2,
        error_category=None,
    )


def test_batch_json_round_trip_normalizes_application_types() -> None:
    batch = BatchResult(
        results=[_rich_result()],
        termination=BatchTermination(kind="fail_fast", reason="stopped", error_category="auth"),
    )

    restored = BatchResult.from_json(batch.to_json())

    assert restored.total_items == 1
    assert restored.results[0].output == {"label": "ok"}
    assert restored.results[0].context == {
        "created": "2026-07-13T12:30:00+00:00",
        "request_id": "12345678-1234-5678-1234-567812345678",
        "path": "inputs/a.json",
    }
    assert restored.results[0].metadata == {"choices": ["x", "y"], "tags": ["a", "b"]}
    assert restored.results[0].token_usage["cached_input_tokens"] == 3
    assert restored.results[0].timing == batch.results[0].timing
    assert restored.termination == batch.termination


def test_v021_attempt_timing_defaults_and_unknown_usage_round_trip() -> None:
    data = _rich_result().to_dict()
    attempt = data["timing"]["attempts"][0]
    for key in (
        "quota_wait_seconds",
        "estimated_input_tokens",
        "estimated_output_tokens",
        "reserved_tokens",
        "reported_tokens",
        "reconciliation_delta_tokens",
        "quota_scope_id",
    ):
        attempt.pop(key)

    restored_v021 = WorkItemResult.from_dict(data)
    old_attempt = restored_v021.timing.attempts[0]
    assert old_attempt.quota_wait_seconds == 0
    assert old_attempt.estimated_input_tokens is None
    assert old_attempt.estimated_output_tokens is None
    assert old_attempt.reserved_tokens == 0
    assert old_attempt.reported_tokens is None
    assert old_attempt.reconciliation_delta_tokens is None
    assert old_attempt.quota_scope_id is None

    unknown = _rich_result()
    unknown.timing.attempts[0].reported_tokens = None
    unknown.timing.attempts[0].reconciliation_delta_tokens = None
    restored_unknown = WorkItemResult.from_dict(unknown.to_dict())
    assert restored_unknown.timing.attempts[0].reported_tokens is None
    assert restored_unknown.timing.attempts[0].reconciliation_delta_tokens is None

    restored_zero = WorkItemResult.from_dict(_rich_result().to_dict())
    assert restored_zero.timing.attempts[0].reported_tokens == 0
    assert restored_zero.timing.attempts[0].quota_scope_id == 2


def test_failure_exception_is_a_safe_descriptor_without_traceback() -> None:
    try:
        raise RuntimeError("safe message")
    except RuntimeError as exc:
        result = WorkItemResult(
            item_id="failed",
            success=False,
            error="RuntimeError: safe message",
            exception=exc,
            error_category="unknown",
        )

    data = result.to_dict()
    assert data["exception"] == {
        "module": "builtins",
        "class_name": "RuntimeError",
        "message": "safe message",
    }
    assert "traceback" not in str(data).lower()
    restored = WorkItemResult.from_dict(data)
    assert restored.exception is None
    assert restored.error == result.error


def test_serialization_redacts_labeled_credentials_and_sensitive_keys() -> None:
    result = WorkItemResult(
        item_id="failed",
        success=False,
        output={"api_key": "sk-output-secret", "safe": "visible"},
        error="request failed Authorization: Bearer sk-error-secret",
        exception=RuntimeError("api_key=sk-exception-secret"),
        metadata={"access_token": "metadata-secret"},
    )

    payload = result.to_dict()
    serialized = str(payload)

    assert "sk-output-secret" not in serialized
    assert "sk-error-secret" not in serialized
    assert "sk-exception-secret" not in serialized
    assert "metadata-secret" not in serialized
    assert payload["output"] == {"api_key": "[REDACTED]", "safe": "visible"}


def test_serialization_preserves_user_controlled_string_fidelity() -> None:
    output = "The secret: ingredient is salt; Authorization: Bearer explains the scheme."
    context = "Document api_key=example is educational prose."
    metadata = {"explanation": "access_token: values appear in this specification"}
    result = WorkItemResult(
        item_id="fidelity",
        success=True,
        output=output,
        context=context,
        metadata=metadata,
    )

    restored = WorkItemResult.from_dict(result.to_dict())

    assert restored.output == output
    assert restored.context == context
    assert restored.metadata == metadata


def test_custom_encoder_decoder_and_unsupported_values() -> None:
    class Custom:
        def __init__(self, value: str) -> None:
            self.value = value

    result = WorkItemResult(item_id="custom", success=True, output=Custom("x"))
    with pytest.raises(ResultSerializationError, match="pass an encoder"):
        result.to_dict()

    data = result.to_dict(encoder=lambda value: {"custom": value.value})
    restored = WorkItemResult.from_dict(data, output_decoder=lambda value: Custom(value["custom"]))
    assert restored.output.value == "x"

    with pytest.raises(ResultSerializationError, match="recursive conversions"):
        result.to_dict(encoder=lambda value: Custom("still unsupported"))


@pytest.mark.parametrize("invalid", [1.5, True, "3", None])
def test_token_usage_decode_rejects_non_integer_values(invalid: Any) -> None:
    data = WorkItemResult(item_id="tokens", success=True).to_dict()
    data["token_usage"]["total_tokens"] = invalid

    with pytest.raises(ResultSerializationError, match="must be an integer"):
        WorkItemResult.from_dict(data)


def test_malformed_and_future_json_are_rejected() -> None:
    with pytest.raises(ResultSerializationError, match="Malformed"):
        BatchResult.from_json("{")

    data = BatchResult(results=[]).to_dict()
    data["schema_version"] = 999
    with pytest.raises(ResultSerializationError, match="future"):
        BatchResult.from_dict(data)

    data = BatchResult(results=[]).to_dict()
    data["termination"]["kind"] = "execute_imported_code"
    with pytest.raises(ResultSerializationError, match="termination kind"):
        BatchResult.from_dict(data)


def test_jsonl_round_trip_with_multiple_results(tmp_path: Path) -> None:
    batch = BatchResult(
        results=[
            _rich_result(),
            WorkItemResult(
                item_id="failed",
                success=False,
                error="bad",
                submission_index=3,
                error_category="validation_error",
            ),
        ],
        termination=BatchTermination(kind="completed"),
    )
    path = tmp_path / "results.jsonl"

    batch.to_jsonl(path)
    restored = BatchResult.from_jsonl(path)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert [result.item_id for result in restored.results] == ["item-1", "failed"]
    assert restored.results[1].error_category == "validation_error"


def test_jsonl_round_trip_preserves_empty_batch_termination(tmp_path: Path) -> None:
    batch = BatchResult(
        results=[],
        termination=BatchTermination(
            kind="batch_timeout",
            reason="no work accepted before timeout",
            error_category="batch_deadline_exceeded",
        ),
    )
    path = tmp_path / "empty-results.jsonl"

    batch.to_jsonl(path)
    restored = BatchResult.from_jsonl(path)

    assert restored.results == []
    assert restored.termination == batch.termination
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
