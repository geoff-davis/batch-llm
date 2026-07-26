"""Tests for zero-config checkpointing (issue #99).

JsonlArtifactStore("run.jsonl") with no ArtifactIdentity: provider/model are
inferred from the strategy at run start, remaining identity fields default
to "unversioned", and prompt/context participation in the compatibility
fingerprint still protects against silent reuse across changed inputs.
"""

import pytest

from async_batch_llm import (
    ArtifactError,
    ArtifactIdentity,
    CallableStrategy,
    CallOutcome,
    JsonlArtifactStore,
    LLMCallStrategy,
    LLMWorkItem,
    OpenAIModel,
    OpenAIStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    ResumePolicy,
    SqliteArtifactStore,
    WorkItemResult,
    process_prompts,
)
from async_batch_llm.artifacts import infer_artifact_identity


class FakeModel:
    def __init__(self, model_id: str):
        self._model = model_id


class CountingStrategy(LLMCallStrategy[str]):
    """Deterministic strategy with a model attribute for identity inference."""

    def __init__(self, model_id: str = "fake-model-1"):
        self.model = FakeModel(model_id)
        self.calls = 0

    async def execute(self, prompt, attempt, timeout, state=None):
        self.calls += 1
        tokens = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return f"out:{prompt}", tokens, None


class TestIdentityInference:
    def test_builtin_openai_strategy(self):
        strategy = OpenAIStrategy(OpenAIModel.from_api_key("gpt-4o-mini", api_key="sk-test"))
        identity = infer_artifact_identity(strategy)
        assert identity.provider == "openai"
        assert identity.model == "gpt-4o-mini"
        assert identity.prompt_version == "unversioned"
        assert identity.parser_version == "unversioned"
        assert identity.application_version == "unversioned"

    def test_inference_is_deterministic(self):
        a = infer_artifact_identity(CountingStrategy("m"))
        b = infer_artifact_identity(CountingStrategy("m"))
        assert a == b

    def test_custom_model_class_uses_class_name(self):
        identity = infer_artifact_identity(CountingStrategy("my-model"))
        assert identity.provider == "FakeModel"
        assert identity.model == "my-model"

    def test_strategy_without_model_falls_back_to_strategy_name(self):
        class Bare(LLMCallStrategy[str]):
            async def execute(self, prompt, attempt, timeout, state=None):
                return "x", {}, None

        identity = infer_artifact_identity(Bare())
        assert identity.provider == "Bare"
        assert identity.model == "unknown"


class TestMinimalForm:
    async def test_checkpoints_and_resumes(self, tmp_path):
        path = tmp_path / "run.jsonl"

        first = CountingStrategy()
        batch1 = await process_prompts(
            first,
            ["alpha", "beta"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert batch1.succeeded == 2
        assert first.calls == 2

        second = CountingStrategy()
        batch2 = await process_prompts(
            second,
            ["alpha", "beta"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert batch2.succeeded == 2
        assert second.calls == 0  # everything replayed
        assert all(r.replayed_from_artifact for r in batch2.results)

    async def test_changed_model_invalidates_reuse(self, tmp_path):
        path = tmp_path / "run.jsonl"
        await process_prompts(
            CountingStrategy("model-a"),
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )

        changed = CountingStrategy("model-b")
        batch = await process_prompts(
            changed,
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert changed.calls == 1  # identity changed -> no replay
        assert not batch.results[0].replayed_from_artifact

    async def test_changed_prompt_invalidates_reuse(self, tmp_path):
        path = tmp_path / "run.jsonl"
        await process_prompts(
            CountingStrategy(),
            [("id_1", "original prompt")],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )

        changed = CountingStrategy()
        batch = await process_prompts(
            changed,
            [("id_1", "different prompt")],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert changed.calls == 1
        assert not batch.results[0].replayed_from_artifact

    async def test_original_model_records_still_replay_after_switch(self, tmp_path):
        path = tmp_path / "run.jsonl"
        await process_prompts(
            CountingStrategy("model-a"),
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        await process_prompts(
            CountingStrategy("model-b"),
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )

        back = CountingStrategy("model-a")
        batch = await process_prompts(
            back,
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )
        assert back.calls == 0
        assert batch.results[0].replayed_from_artifact

    async def test_iter_results_reads_zero_config_artifact(self, tmp_path):
        path = tmp_path / "run.jsonl"
        await process_prompts(
            CountingStrategy(),
            ["alpha"],
            artifact_store=JsonlArtifactStore(path),
            resume=ResumePolicy.REUSE_SUCCESSES,
        )

        audit = JsonlArtifactStore(path)
        seen = [result async for result in audit.iter_results()]
        await audit.close()
        assert len(seen) == 1
        assert seen[0].output == "out:alpha"


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_automatic_store_rejects_mixed_model_identities_before_second_prepare(
    tmp_path, store_type, suffix
):
    store = store_type(tmp_path / f"mixed{suffix}")
    first = CountingStrategy("model-a")
    mismatched = CountingStrategy("model-b")
    try:
        await store.prepare_item(
            LLMWorkItem(item_id="first", strategy=first, prompt="first prompt")
        )
        with pytest.raises(ArtifactError, match="Automatic artifact identity changed"):
            await store.prepare_item(
                LLMWorkItem(
                    item_id="mismatched",
                    strategy=mismatched,
                    prompt="mismatched prompt",
                )
            )
    finally:
        await store.close()

    assert first.calls == 0
    assert mismatched.calls == 0


@pytest.mark.parametrize(
    ("first_model", "second_model"),
    [("model-a", "model-b"), ("model-b", "model-a")],
)
@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_automatic_identity_mismatch_is_order_independent(
    tmp_path, store_type, suffix, first_model, second_model
):
    store = store_type(tmp_path / f"order{suffix}")
    try:
        await store.prepare_item(
            LLMWorkItem(
                item_id="first",
                strategy=CountingStrategy(first_model),
                prompt="first",
            )
        )
        with pytest.raises(ArtifactError, match="one inferred execution identity"):
            await store.prepare_item(
                LLMWorkItem(
                    item_id="second",
                    strategy=CountingStrategy(second_model),
                    prompt="second",
                )
            )
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_distinct_strategy_instances_with_same_inferred_identity_are_allowed(
    tmp_path, store_type, suffix
):
    store = store_type(tmp_path / f"same{suffix}")
    try:
        first = await store.prepare_item(
            LLMWorkItem(
                item_id="first",
                strategy=CountingStrategy("shared-model"),
                prompt="first",
            )
        )
        second = await store.prepare_item(
            LLMWorkItem(
                item_id="second",
                strategy=CountingStrategy("shared-model"),
                prompt="second",
            )
        )
    finally:
        await store.close()
    assert first.input_fingerprint != second.input_fingerprint


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_custom_strategy_class_fallbacks_must_match(tmp_path, store_type, suffix):
    class FirstCustomStrategy(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            return prompt, {}, None

    class SecondCustomStrategy(LLMCallStrategy[str]):
        async def execute(self, prompt, attempt, timeout, state=None):
            return prompt, {}, None

    store = store_type(tmp_path / f"custom-classes{suffix}")
    try:
        await store.prepare_item(
            LLMWorkItem(
                item_id="first",
                strategy=FirstCustomStrategy(),
                prompt="first",
            )
        )
        with pytest.raises(ArtifactError, match="Automatic artifact identity changed"):
            await store.prepare_item(
                LLMWorkItem(
                    item_id="second",
                    strategy=SecondCustomStrategy(),
                    prompt="second",
                )
            )
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_callable_strategy_identities_are_checked_without_invocation(
    tmp_path, store_type, suffix
):
    calls = 0

    async def invoke(prompt, *, attempt, timeout, state):
        nonlocal calls
        calls += 1
        return CallOutcome(prompt)

    shared = ArtifactIdentity(provider="app", model="route", application_version="1")
    changed = ArtifactIdentity(provider="app", model="route", application_version="2")
    store = store_type(tmp_path / f"callable{suffix}")
    try:
        await store.prepare_item(
            LLMWorkItem(
                item_id="first",
                strategy=CallableStrategy(invoke, identity=shared),
                prompt="first",
            )
        )
        await store.prepare_item(
            LLMWorkItem(
                item_id="same",
                strategy=CallableStrategy(invoke, identity=shared),
                prompt="same",
            )
        )
        with pytest.raises(ArtifactError, match="Automatic artifact identity changed"):
            await store.prepare_item(
                LLMWorkItem(
                    item_id="changed",
                    strategy=CallableStrategy(invoke, identity=changed),
                    prompt="changed",
                )
            )
    finally:
        await store.close()
    assert calls == 0


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_first_written_item_survives_later_identity_mismatch(tmp_path, store_type, suffix):
    path = tmp_path / f"written-before-mismatch{suffix}"
    store = store_type(path)
    item = LLMWorkItem(
        item_id="written",
        strategy=CountingStrategy("model-a"),
        prompt="written",
    )
    try:
        prepared = await store.prepare_item(item)
        await store.append(
            item,
            prepared,
            WorkItemResult(item_id="written", success=True, output="persisted"),
        )
        with pytest.raises(ArtifactError, match="Automatic artifact identity changed"):
            await store.prepare_item(
                LLMWorkItem(
                    item_id="rejected",
                    strategy=CountingStrategy("model-b"),
                    prompt="rejected",
                )
            )
    finally:
        await store.close()

    if store_type is JsonlArtifactStore:
        loaded = store_type.read_results(path)
    else:
        loaded = await store_type.read_results(path)
    assert [(result.item_id, result.output) for result in loaded.results] == [
        ("written", "persisted")
    ]


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_stale_row_cannot_be_admitted_for_a_mismatched_strategy(tmp_path, store_type, suffix):
    path = tmp_path / f"stale-row{suffix}"
    await process_prompts(
        CountingStrategy("model-a"),
        [("shared", "prompt")],
        artifact_store=store_type(path),
    )

    pinned = CountingStrategy("model-a")
    mismatched = CountingStrategy("model-b")
    async with ParallelBatchProcessor(
        config=ProcessorConfig(max_workers=1),
        artifact_store=store_type(path),
        resume=ResumePolicy.REUSE_SUCCESSES,
    ) as processor:
        await processor.add_work(LLMWorkItem(item_id="pin", strategy=pinned, prompt="pin"))
        with pytest.raises(ArtifactError, match="Automatic artifact identity changed"):
            await processor.add_work(
                LLMWorkItem(
                    item_id="shared",
                    strategy=mismatched,
                    prompt="prompt",
                )
            )

    assert pinned.calls == 0
    assert mismatched.calls == 0


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_identity_mismatch_error_does_not_expose_identity_extra(tmp_path, store_type, suffix):
    async def invoke(prompt, *, attempt, timeout, state):
        return CallOutcome(prompt)

    first_secret = "first-sensitive-extra"
    second_secret = "second-sensitive-extra"
    store = store_type(tmp_path / f"secret-error{suffix}")
    try:
        await store.prepare_item(
            LLMWorkItem(
                item_id="first",
                strategy=CallableStrategy(
                    invoke,
                    identity=ArtifactIdentity(
                        provider="app",
                        model="route",
                        extra={"api_key": first_secret},
                    ),
                ),
                prompt="first",
            )
        )
        with pytest.raises(ArtifactError) as raised:
            await store.prepare_item(
                LLMWorkItem(
                    item_id="second",
                    strategy=CallableStrategy(
                        invoke,
                        identity=ArtifactIdentity(
                            provider="app",
                            model="route",
                            extra={"api_key": second_secret},
                        ),
                    ),
                    prompt="second",
                )
            )
    finally:
        await store.close()
    assert "Automatic artifact identity changed" in str(raised.value)
    assert first_secret not in str(raised.value)
    assert second_secret not in str(raised.value)


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_fresh_store_can_use_another_homogeneous_identity_in_same_artifact(
    tmp_path, store_type, suffix
):
    path = tmp_path / f"sequential-identities{suffix}"
    await process_prompts(
        CountingStrategy("model-a"),
        [("item", "prompt")],
        artifact_store=store_type(path),
    )

    changed = CountingStrategy("model-b")
    changed_result = await process_prompts(
        changed,
        [("item", "prompt")],
        artifact_store=store_type(path),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert changed.calls == 1
    assert not changed_result.results[0].replayed_from_artifact

    replay = CountingStrategy("model-b")
    replay_result = await process_prompts(
        replay,
        [("item", "prompt")],
        artifact_store=store_type(path),
        resume=ResumePolicy.REUSE_SUCCESSES,
    )
    assert replay.calls == 0
    assert replay_result.results[0].replayed_from_artifact


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_concurrent_preparation_pins_exactly_one_automatic_identity(
    tmp_path, store_type, suffix
):
    import asyncio

    store = store_type(tmp_path / f"concurrent{suffix}")
    start = asyncio.Event()

    async def prepare(model_id):
        await start.wait()
        return await store.prepare_item(
            LLMWorkItem(
                item_id=model_id,
                strategy=CountingStrategy(model_id),
                prompt=model_id,
            )
        )

    tasks = [
        asyncio.create_task(prepare("model-a")),
        asyncio.create_task(prepare("model-b")),
    ]
    start.set()
    try:
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await store.close()
    assert sum(not isinstance(outcome, Exception) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, ArtifactError) for outcome in outcomes) == 1


@pytest.mark.parametrize(
    ("store_type", "suffix"),
    [
        pytest.param(JsonlArtifactStore, ".jsonl", id="jsonl"),
        pytest.param(SqliteArtifactStore, ".sqlite", id="sqlite"),
    ],
)
async def test_explicit_store_identity_still_permits_mixed_strategies(tmp_path, store_type, suffix):
    store = store_type(
        tmp_path / f"explicit-mixed{suffix}",
        identity=ArtifactIdentity(provider="routed-app", model="mixed"),
    )
    try:
        first = await store.prepare_item(
            LLMWorkItem(
                item_id="first",
                strategy=CountingStrategy("model-a"),
                prompt="first",
            )
        )
        second = await store.prepare_item(
            LLMWorkItem(
                item_id="second",
                strategy=CountingStrategy("model-b"),
                prompt="second",
            )
        )
    finally:
        await store.close()

    assert first.input_fingerprint != second.input_fingerprint


class TestUnresolvedIdentityGuards:
    async def test_new_artifact_without_identity_or_items_raises(self, tmp_path):
        store = JsonlArtifactStore(tmp_path / "new.jsonl")
        with pytest.raises(ArtifactError, match="without a\\s+resolved identity"):
            _ = [r async for r in store.iter_results()]
        await store.close()
