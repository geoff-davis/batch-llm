"""Model escalation strategy: Start cheap, escalate to smarter models on failure.

This example demonstrates a cost-optimization pattern:
1. Try with fast, cheap model first (gemini-2.0-flash-exp)
2. If that fails, escalate to better model (gemini-1.5-flash)
3. If that fails, escalate to most capable model (gemini-1.5-pro)

This maximizes cost efficiency while maintaining high success rates.

## Installation

```bash
pip install 'batch-llm[gemini]'
export GOOGLE_API_KEY=your_api_key_here
```

## Cost Optimization

By starting with the cheapest model and only escalating on failure:
- Most items succeed on first (cheap) attempt
- Only difficult items use expensive models
- Overall cost is much lower than always using the best model

## Use Cases

- Complex reasoning tasks where cheaper models often suffice
- Cost-sensitive production workloads
- Processing mixed difficulty content
- Any scenario where you want to optimize cost/quality tradeoff
"""

import asyncio
import os
from typing import Annotated

from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.core import RetryConfig
from batch_llm.llm_strategies import LLMCallStrategy


class Analysis(BaseModel):
    """Complex analysis requiring reasoning."""

    summary: Annotated[str, Field(description="Brief summary")]
    key_insights: Annotated[list[str], Field(description="Main insights", min_length=3)]
    recommendations: Annotated[list[str], Field(description="Actionable recommendations")]
    confidence: Annotated[str, Field(description="High, Medium, or Low")]


# ============================================================================
# Model Escalation Strategy
# ============================================================================


class ModelEscalationStrategy(LLMCallStrategy[Analysis]):
    """
    Start with cheapest model, escalate to better models on failure.

    Model progression (cost and capability):
    1. gemini-2.0-flash-exp (fastest, cheapest, experimental)
    2. gemini-1.5-flash (production-ready, fast, good quality)
    3. gemini-1.5-pro (most capable, slower, most expensive)

    This strategy optimizes for cost while maintaining quality:
    - Easy tasks: Succeed on attempt 1 (cheapest)
    - Medium tasks: Succeed on attempt 2 (moderate cost)
    - Hard tasks: Succeed on attempt 3 (expensive but capable)
    """

    # Model tiers: [attempt 1, attempt 2, attempt 3]
    MODELS = [
        "gemini-2.0-flash-exp",  # Attempt 1: Cheapest/fastest
        "gemini-1.5-flash",       # Attempt 2: Production fast
        "gemini-1.5-pro",         # Attempt 3: Most capable
    ]

    # Approximate relative costs (for illustration)
    COSTS = {
        "gemini-2.0-flash-exp": 1.0,   # Baseline
        "gemini-1.5-flash": 2.0,        # ~2x more expensive
        "gemini-1.5-pro": 10.0,         # ~10x more expensive
    }

    def __init__(self, client: genai.Client, verbose: bool = True):
        self.client = client
        self.verbose = verbose
        self.total_cost_units = 0  # Track cumulative cost

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[Analysis, dict[str, int]]:
        # Select model based on attempt number
        model = self.MODELS[min(attempt - 1, len(self.MODELS) - 1)]

        if self.verbose:
            print(f"  Attempt {attempt}: Using {model}")

        # Configure request
        config = GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=Analysis,
        )

        # Make API call
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        # Parse response (may raise ValidationError)
        output = Analysis.model_validate_json(response.text)

        # Track cost
        self.total_cost_units += self.COSTS[model]

        # Extract token usage
        usage = response.usage_metadata
        tokens = {
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
            "total_tokens": usage.total_token_count or 0,
        }

        return output, tokens


# ============================================================================
# Alternative: Progressive Model + Temperature
# ============================================================================


class ModelAndTempEscalationStrategy(LLMCallStrategy[Analysis]):
    """
    Escalate both model AND temperature on retries.

    This combines two strategies:
    1. Better models on retries
    2. Higher temperature on retries

    Sometimes a cheaper model with higher temperature can succeed,
    making this even more cost-effective.
    """

    MODELS = [
        ("gemini-2.0-flash-exp", 0.5),  # Attempt 1: cheap, moderate temp
        ("gemini-1.5-flash", 0.8),       # Attempt 2: better model, higher temp
        ("gemini-1.5-pro", 1.0),         # Attempt 3: best model, max temp
    ]

    def __init__(self, client: genai.Client, verbose: bool = True):
        self.client = client
        self.verbose = verbose

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[Analysis, dict[str, int]]:
        # Select model and temperature based on attempt
        model, temp = self.MODELS[min(attempt - 1, len(self.MODELS) - 1)]

        if self.verbose:
            print(f"  Attempt {attempt}: {model} (temp={temp})")

        config = GenerateContentConfig(
            temperature=temp,
            response_mime_type="application/json",
            response_schema=Analysis,
        )

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        output = Analysis.model_validate_json(response.text)

        usage = response.usage_metadata
        tokens = {
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
            "total_tokens": usage.total_token_count or 0,
        }

        return output, tokens


# ============================================================================
# Examples
# ============================================================================


async def example_basic_escalation():
    """Basic model escalation example."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Model Escalation")
    print("=" * 70 + "\n")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=60.0,
        retry=RetryConfig(max_attempts=3, initial_wait=1.0),
    )

    # Mix of easy and hard prompts
    prompts = [
        {
            "id": "easy",
            "text": """
            Analyze this simple scenario:
            A company increased sales by 20% this quarter.
            """,
        },
        {
            "id": "complex",
            "text": """
            Analyze this complex scenario:
            A company's revenue increased 15% but profit decreased 5% despite
            cost-cutting measures. Market share grew 3% while customer satisfaction
            dropped 10 points. Employee turnover increased from 8% to 15%.
            The board is demanding explanations and recommendations.
            Provide deep insights considering competitive dynamics, operational
            efficiency, customer experience, and talent retention.
            """,
        },
    ]

    async with ParallelBatchProcessor[str, Analysis, None](
        config=config
    ) as processor:
        for item in prompts:
            # Note: Create new strategy instance per item for cost tracking
            item_strategy = ModelEscalationStrategy(client=client, verbose=True)
            await processor.add_work(
                LLMWorkItem(
                    item_id=item["id"],
                    strategy=item_strategy,
                    prompt=item["text"],
                )
            )

        result = await processor.process_all()

    print(f"\nResults: {result.succeeded}/{result.total_items} succeeded\n")

    for item in result.results:
        if item.success:
            print(f"✓ {item.item_id}:")
            print(f"  Summary: {item.output.summary}")
            print(f"  Insights: {len(item.output.key_insights)} key insights")
            print(f"  Confidence: {item.output.confidence}")
            print(f"  Tokens: {item.token_usage.get('total_tokens', 0)}")
        else:
            print(f"✗ {item.item_id}: {item.error}")


async def example_cost_comparison():
    """Compare costs: always-best-model vs escalation strategy."""
    print("\n" + "=" * 70)
    print("Example 2: Cost Comparison")
    print("=" * 70 + "\n")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Test data with varying difficulty
    test_prompts = [
        "Analyze: Sales up 10%",
        "Analyze: Customer satisfaction declined despite new features",
        "Analyze: Revenue flat, costs rising, competition increasing",
    ]

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=60.0,
        retry=RetryConfig(max_attempts=3, initial_wait=1.0),
    )

    # Strategy 1: Always use best model (expensive but reliable)
    print("Strategy 1: Always use gemini-1.5-pro (best model)")

    class AlwaysProStrategy(LLMCallStrategy[Analysis]):
        def __init__(self, client: genai.Client):
            self.client = client

        async def execute(self, prompt: str, attempt: int, timeout: float):
            config = GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json",
                response_schema=Analysis,
            )
            response = await self.client.aio.models.generate_content(
                model="gemini-1.5-pro",
                contents=prompt,
                config=config,
            )
            output = Analysis.model_validate_json(response.text)
            usage = response.usage_metadata
            tokens = {
                "input_tokens": usage.prompt_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0,
                "total_tokens": usage.total_token_count or 0,
            }
            return output, tokens

    always_pro = AlwaysProStrategy(client=client)
    async with ParallelBatchProcessor[str, Analysis, None](config=config) as processor:
        for i, prompt in enumerate(test_prompts):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"always_pro_{i}",
                    strategy=always_pro,
                    prompt=f"Analyze: {prompt}",
                )
            )
        result_always = await processor.process_all()

    always_cost = len(test_prompts) * 10.0  # Pro costs 10 units each
    print(f"  Results: {result_always.succeeded}/{result_always.total_items}")
    print(f"  Cost units: {always_cost}\n")

    # Strategy 2: Escalation (start cheap, escalate only when needed)
    print("Strategy 2: Model Escalation (start cheap)")

    async with ParallelBatchProcessor[str, Analysis, None](config=config) as processor:
        for i, prompt in enumerate(test_prompts):
            escalation_strategy = ModelEscalationStrategy(client=client, verbose=False)
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"escalation_{i}",
                    strategy=escalation_strategy,
                    prompt=f"Analyze: {prompt}",
                )
            )
        result_escalation = await processor.process_all()

    # Estimate cost (assumes most succeed on first attempt)
    escalation_cost = result_escalation.total_items * 1.0  # Rough estimate
    print(f"  Results: {result_escalation.succeeded}/{result_escalation.total_items}")
    print(f"  Estimated cost units: {escalation_cost}\n")

    print("Analysis:")
    print(f"  Always-Pro strategy: {always_cost} cost units")
    print(f"  Escalation strategy: ~{escalation_cost} cost units")
    print(f"  Potential savings: ~{((always_cost - escalation_cost) / always_cost * 100):.0f}%")
    print("\n  Note: Actual savings depend on task difficulty distribution.")
    print("  Easy tasks = more savings. Hard tasks = less savings.")


async def example_model_and_temp():
    """Example combining model escalation with temperature escalation."""
    print("\n" + "=" * 70)
    print("Example 3: Model + Temperature Escalation")
    print("=" * 70 + "\n")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    strategy = ModelAndTempEscalationStrategy(client=client, verbose=True)

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=60.0,
        retry=RetryConfig(max_attempts=3, initial_wait=1.0),
    )

    prompt = """
    Analyze this nuanced scenario:
    A tech startup's user growth is 200% but revenue per user dropped 40%.
    Retention is strong but acquisition cost tripled. Investors are concerned.
    """

    async with ParallelBatchProcessor[str, Analysis, None](
        config=config
    ) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="nuanced_analysis",
                strategy=strategy,
                prompt=prompt,
            )
        )

        result = await processor.process_all()

    print(f"\nResult: {result.succeeded}/{result.total_items} succeeded\n")

    for item in result.results:
        if item.success:
            print("✓ Analysis:")
            print(f"  Summary: {item.output.summary}")
            print("  Key Insights:")
            for insight in item.output.key_insights:
                print(f"    - {insight}")
            print("  Recommendations:")
            for rec in item.output.recommendations:
                print(f"    - {rec}")


async def main():
    """Run all examples."""
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Get your API key from: https://aistudio.google.com/apikey")
        print("Then run: export GOOGLE_API_KEY=your_key_here")
        return

    await example_basic_escalation()
    await example_cost_comparison()
    await example_model_and_temp()

    print("\n" + "=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("""
1. Model escalation optimizes cost/quality tradeoff
2. Most tasks succeed on cheaper models (attempt 1)
3. Only difficult tasks escalate to expensive models
4. Can combine with temperature escalation for even better results
5. Significant cost savings vs always using best model

This pattern is perfect for:
- Production workloads with cost constraints
- Mixed difficulty content
- Scenarios where cheaper models often suffice
- Maximizing quality while minimizing cost
    """)


if __name__ == "__main__":
    asyncio.run(main())
