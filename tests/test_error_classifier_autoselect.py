"""Tests for resolving error classifiers from work-item strategies."""

import pytest

from async_batch_llm import (
    DeepSeekStrategy,
    GeminiStrategy,
    LLMWorkItem,
    OpenAIStrategy,
    OpenRouterStrategy,
    ParallelBatchProcessor,
    ProcessorConfig,
    process_prompts,
)
from async_batch_llm.base import RetryState, TokenUsage
from async_batch_llm.classifiers.gemini import GeminiErrorClassifier
from async_batch_llm.classifiers.openai import OpenAIErrorClassifier
from async_batch_llm.classifiers.openrouter import OpenRouterErrorClassifier
from async_batch_llm.llm_strategies import LLMCallStrategy
from async_batch_llm.strategies.errors import DefaultErrorClassifier


class _NoPreferenceStrategy(LLMCallStrategy[str]):
    async def execute(
        self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
    ) -> tuple[str, TokenUsage, None]:
        return prompt, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, None


class _GeminiRecommendingStrategy(_NoPreferenceStrategy):
    def recommended_error_classifier(self):
        return GeminiErrorClassifier()


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (OpenAIStrategy(object()), OpenAIErrorClassifier),
        (DeepSeekStrategy(object()), OpenAIErrorClassifier),
        (OpenRouterStrategy(object()), OpenRouterErrorClassifier),
        (GeminiStrategy(object()), GeminiErrorClassifier),
        (_NoPreferenceStrategy(), type(None)),
    ],
)
def test_strategy_recommendations(strategy, expected):
    recommended = strategy.recommended_error_classifier()
    assert isinstance(recommended, expected) or (expected is type(None) and recommended is None)


async def _add(processor, *strategies):
    for i, strategy in enumerate(strategies):
        await processor.add_work(LLMWorkItem(item_id=f"i{i}", strategy=strategy, prompt="x"))


@pytest.mark.asyncio
async def test_autoselect_single_provider_per_strategy():
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig())
    strategies = [OpenAIStrategy(object()), OpenAIStrategy(object())]
    await _add(processor, *strategies)
    assert all(
        isinstance(processor._classifier_resolver.resolve(strategy), OpenAIErrorClassifier)
        for strategy in strategies
    )
    await processor.shutdown()


@pytest.mark.asyncio
async def test_abstaining_strategy_gets_default():
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig())
    strategy = _NoPreferenceStrategy()
    await _add(processor, strategy)
    assert isinstance(processor._classifier_resolver.resolve(strategy), DefaultErrorClassifier)
    await processor.shutdown()


@pytest.mark.asyncio
async def test_recommender_and_abstainer_keep_independent_decisions():
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig())
    abstaining = _NoPreferenceStrategy()
    gemini = _GeminiRecommendingStrategy()
    await _add(processor, abstaining, gemini)
    assert isinstance(processor._classifier_resolver.resolve(abstaining), DefaultErrorClassifier)
    assert isinstance(processor._classifier_resolver.resolve(gemini), GeminiErrorClassifier)
    await processor.shutdown()


@pytest.mark.asyncio
async def test_mixed_providers_use_matching_classifiers():
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig())
    openai = OpenAIStrategy(object())
    gemini = GeminiStrategy(object())
    await _add(processor, openai, gemini)
    assert isinstance(processor._classifier_resolver.resolve(openai), OpenAIErrorClassifier)
    assert isinstance(processor._classifier_resolver.resolve(gemini), GeminiErrorClassifier)
    await processor.shutdown()


@pytest.mark.asyncio
async def test_explicit_classifier_is_never_overridden():
    explicit = OpenRouterErrorClassifier()
    processor = ParallelBatchProcessor[str, str, None](
        config=ProcessorConfig(), error_classifier=explicit
    )
    strategy = GeminiStrategy(object())
    await _add(processor, strategy)
    assert processor.error_classifier is explicit
    assert processor._classifier_resolver.resolve(strategy) is explicit
    await processor.shutdown()


@pytest.mark.asyncio
async def test_recommending_strategies_run_in_real_batch():
    processor = ParallelBatchProcessor[str, str, None](config=ProcessorConfig(max_workers=2))
    await _add(processor, _GeminiRecommendingStrategy(), _GeminiRecommendingStrategy())
    result = await processor.process_all()
    assert result.succeeded == 2


@pytest.mark.asyncio
async def test_autoselect_via_process_prompts():
    result = await process_prompts(
        GeminiStrategy(object()),
        ["a", "b"],
        config=ProcessorConfig(max_workers=2, dry_run=True),
    )
    assert result.total_items == 2
