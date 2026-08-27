"""DeepSeek Responses API and strict JSON Schema output (issue #149)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from async_batch_llm import (
    DeepSeekModel,
    DeepSeekStrategy,
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    RetryConfig,
    StructuredOutputSchemaError,
    StructuredOutputValidationError,
)
from async_batch_llm.artifacts import infer_artifact_identity
from async_batch_llm.classifiers.openai import OpenAIErrorClassifier


class Verdict(BaseModel):
    valid: bool
    reason: str


class AlternateVerdict(BaseModel):
    score: int


def _response(
    text: str,
    *,
    request_id: str = "resp_123",
    input_tokens: int = 20,
    output_tokens: int = 8,
    cached_tokens: int = 5,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return SimpleNamespace(
        id=request_id,
        model="deepseek-v4-flash",
        status="completed",
        error=None,
        output_text=text,
        output=[],
        usage=usage,
    )


def _client(*responses: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    if len(responses) == 1:
        client.responses.create = AsyncMock(return_value=responses[0])
    else:
        client.responses.create = AsyncMock(side_effect=list(responses))
    return client


@pytest.mark.asyncio
async def test_responses_schema_request_parses_pydantic_and_preserves_telemetry() -> None:
    response = _response('{"valid":true,"reason":"supported"}')
    client = _client(response)
    model = DeepSeekModel(
        "deepseek-v4-flash",
        client,
        api_surface="responses",
        response_schema=Verdict,
        thinking=False,
        system_instruction="Return the verdict.",
    )
    strategy = DeepSeekStrategy(model, generation_config={"max_tokens": 256})

    output, tokens, metadata = await strategy.execute("Check this", 1, 10.0)

    assert output == Verdict(valid=True, reason="supported")
    assert tokens == {
        "input_tokens": 20,
        "output_tokens": 8,
        "total_tokens": 28,
        "cached_input_tokens": 5,
    }
    assert metadata is not None
    assert metadata["api_surface"] == "responses"
    assert metadata["provider_request_id"] == "resp_123"
    assert metadata["response_schema"] == {
        "name": "Verdict",
        "identity": f"{Verdict.__module__}.{Verdict.__qualname__}",
        "sha256": model.response_schema_hash,
    }

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "deepseek-v4-flash"
    assert kwargs["input"] == "Check this"
    assert kwargs["instructions"] == "Return the verdict."
    assert kwargs["max_output_tokens"] == 256
    assert kwargs["reasoning"] == {"effort": "none"}
    assert kwargs["text"]["format"] == {
        "type": "json_schema",
        "name": "Verdict",
        "schema": Verdict.model_json_schema(),
    }


@pytest.mark.asyncio
async def test_json_schema_mapping_returns_decoded_json() -> None:
    schema = {
        "title": "VerdictPayload",
        "type": "object",
        "properties": {"valid": {"type": "boolean"}},
        "required": ["valid"],
        "additionalProperties": False,
    }
    model = DeepSeekModel(
        "deepseek-v4-flash",
        _client(_response('{"valid":true}')),
        api_surface="responses",
        response_schema=schema,
    )

    output, _, _ = await DeepSeekStrategy(model).execute("Check", 1, 10.0)

    assert output == {"valid": True}


@pytest.mark.asyncio
async def test_provider_schema_rejection_has_distinct_non_retryable_category() -> None:
    class ProviderBadRequest(Exception):
        status_code = 400
        body = {"error": {"message": "Unsupported JSON schema keyword"}}

    client = MagicMock()
    client.responses.create = AsyncMock(side_effect=ProviderBadRequest("invalid json_schema"))
    model = DeepSeekModel(
        "deepseek-v4-flash",
        client,
        api_surface="responses",
        response_schema=Verdict,
    )

    with pytest.raises(StructuredOutputSchemaError) as exc_info:
        await model.generate("Check")

    info = OpenAIErrorClassifier().classify(exc_info.value)
    assert info.error_category == "structured_output_schema_rejected"
    assert info.is_retryable is False


@pytest.mark.asyncio
async def test_malformed_provider_output_is_retryable_and_keeps_failed_usage() -> None:
    model = DeepSeekModel(
        "deepseek-v4-flash",
        _client(_response('{"valid": tru', input_tokens=13, output_tokens=4)),
        api_surface="responses",
        response_schema=Verdict,
    )

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        await DeepSeekStrategy(model).execute("Check", 1, 10.0)

    assert exc_info.value._failed_token_usage == {
        "input_tokens": 13,
        "output_tokens": 4,
        "total_tokens": 17,
        "cached_input_tokens": 5,
    }
    info = OpenAIErrorClassifier().classify(exc_info.value)
    assert info.error_category == "structured_output_validation_error"
    assert info.is_retryable is True


@pytest.mark.asyncio
async def test_malformed_structured_response_retries_through_batch_accounting() -> None:
    client = _client(
        _response("not json", request_id="bad", input_tokens=10, output_tokens=2),
        _response('{"valid":true,"reason":"retry worked"}', request_id="good"),
    )
    strategy = DeepSeekStrategy(
        DeepSeekModel(
            "deepseek-v4-flash",
            client,
            api_surface="responses",
            response_schema=Verdict,
        )
    )
    config = ProcessorConfig(
        max_workers=1,
        retry=RetryConfig(max_attempts=2, initial_wait=0.001, max_wait=0.001, jitter=False),
    )

    async with ParallelBatchProcessor(config=config) as processor:
        await processor.add_work(LLMWorkItem(item_id="one", prompt="Check", strategy=strategy))
        batch = await processor.process_all()

    assert batch.succeeded == 1
    assert client.responses.create.await_count == 2
    assert batch.results[0].output == Verdict(valid=True, reason="retry worked")
    assert len(batch.results[0].timing.attempts) == 2
    assert batch.total_input_tokens == 30
    assert batch.total_output_tokens == 10


def test_artifact_identity_distinguishes_surface_and_schema() -> None:
    chat = DeepSeekStrategy(DeepSeekModel("deepseek-v4-flash", MagicMock(), json_mode=True))
    strict = DeepSeekStrategy(
        DeepSeekModel(
            "deepseek-v4-flash",
            MagicMock(),
            api_surface="responses",
            response_schema=Verdict,
        )
    )
    alternate = DeepSeekStrategy(
        DeepSeekModel(
            "deepseek-v4-flash",
            MagicMock(),
            api_surface="responses",
            response_schema=AlternateVerdict,
        )
    )

    chat_identity = infer_artifact_identity(chat)
    strict_identity = infer_artifact_identity(strict)
    alternate_identity = infer_artifact_identity(alternate)

    assert chat_identity.extra == {
        "api_surface": "chat_completions",
        "output_format": "json_object",
    }
    assert strict_identity.extra["api_surface"] == "responses"
    assert strict_identity.extra["response_schema"]["sha256"]
    assert strict_identity != chat_identity
    assert strict_identity != alternate_identity


def test_responses_capability_fails_closed_without_schema_fallback() -> None:
    with pytest.raises(ValueError, match="currently supports only 'deepseek-v4-flash'"):
        DeepSeekModel(
            "deepseek-v4-pro",
            MagicMock(),
            api_surface="responses",
            response_schema=Verdict,
        )

    with pytest.raises(ValueError, match="requires api_surface='responses'"):
        DeepSeekModel("deepseek-v4-flash", MagicMock(), response_schema=Verdict)


def test_from_api_key_forwards_responses_configuration_to_model() -> None:
    with patch("async_batch_llm.models.AsyncOpenAI"):
        model = DeepSeekModel.from_api_key(
            "deepseek-v4-flash",
            api_key="sk-test",
            api_surface="responses",
            response_schema=Verdict,
            thinking=False,
        )

    assert model.api_surface == "responses"
    assert model.response_schema == Verdict.model_json_schema()
    assert model._default_extra_body is not None
    assert model._default_extra_body["reasoning"] == {"effort": "none"}
