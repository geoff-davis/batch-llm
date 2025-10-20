"""Example demonstrating batch-llm with Anthropic Claude API.

This example shows how to create a custom strategy for Anthropic's API,
including both direct API calls and streaming responses.

Install dependencies:
    pip install 'batch-llm' 'anthropic'
"""

import asyncio
import os

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import LLMCallStrategy


class AnalysisOutput(BaseModel):
    """Example output model for structured analysis."""

    topic: str
    key_insights: list[str]
    complexity_level: str


class AnthropicStrategy(LLMCallStrategy[str]):
    """Strategy for calling Anthropic Claude API with simple text responses."""

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        system_prompt: str | None = None,
    ):
        """
        Initialize Anthropic strategy.

        Args:
            client: Initialized AsyncAnthropic client
            model: Model name (e.g., "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            system_prompt: Optional system prompt
        """
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, dict[str, int]]:
        """Execute Anthropic API call.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        # Build message parameters
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.system_prompt:
            params["system"] = self.system_prompt

        # Make the API call
        response = await self.client.messages.create(**params)

        # Extract output
        output = response.content[0].text if response.content else ""

        # Extract token usage
        usage = response.usage
        tokens = {
            "input_tokens": usage.input_tokens if usage else 0,
            "output_tokens": usage.output_tokens if usage else 0,
            "total_tokens": (usage.input_tokens + usage.output_tokens) if usage else 0,
        }

        return output, tokens


# Example 1: Simple text generation with Claude
async def example_anthropic_text():
    """Example using Anthropic Claude for text generation."""
    print("\n" + "=" * 60)
    print("Example 1: Anthropic Claude Text Generation")
    print("=" * 60 + "\n")

    # Initialize Anthropic client
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Create the strategy
    strategy = AnthropicStrategy(
        client=client,
        model="claude-3-5-haiku-20241022",  # Fast, cost-effective model
        max_tokens=500,
        temperature=1.0,
    )

    # Configure the processor
    config = ProcessorConfig(max_workers=3, timeout_per_item=30.0)

    # Process items
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        questions = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "What are the main benefits of renewable energy?",
        ]

        for i, question in enumerate(questions):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"question_{i}",
                    strategy=strategy,
                    prompt=question,
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print(f"Total tokens used: {result.total_input_tokens + result.total_output_tokens}")
    print("\nResults:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  {item_result.output[:200]}...")  # Truncate for display


# Example 2: Using system prompts for consistent behavior
async def example_anthropic_system_prompt():
    """Example using system prompts to guide Claude's behavior."""
    print("\n" + "=" * 60)
    print("Example 2: Anthropic with System Prompt")
    print("=" * 60 + "\n")

    # Initialize Anthropic client
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Create the strategy with a system prompt
    strategy = AnthropicStrategy(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=1.0,
        system_prompt="You are a helpful assistant that provides concise, factual answers. Keep your responses under 3 sentences.",
    )

    # Configure the processor
    config = ProcessorConfig(max_workers=2, timeout_per_item=30.0)

    # Process items
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        topics = [
            "Explain machine learning",
            "What is blockchain technology?",
            "Describe the water cycle",
        ]

        for i, topic in enumerate(topics):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"topic_{i}",
                    strategy=strategy,
                    prompt=topic,
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print("\nConcise Explanations:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  {item_result.output}")


# Example 3: Different models for different tasks
async def example_anthropic_mixed_models():
    """Example using different Claude models for different task types."""
    print("\n" + "=" * 60)
    print("Example 3: Mixed Claude Models")
    print("=" * 60 + "\n")

    # Initialize Anthropic client
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Create strategies for different use cases
    fast_strategy = AnthropicStrategy(
        client=client,
        model="claude-3-5-haiku-20241022",  # Fast for simple tasks
        max_tokens=200,
        temperature=0.5,
    )

    reasoning_strategy = AnthropicStrategy(
        client=client,
        model="claude-3-5-sonnet-20241022",  # Better for complex reasoning
        max_tokens=1000,
        temperature=1.0,
    )

    # Configure the processor
    config = ProcessorConfig(max_workers=4, timeout_per_item=60.0)

    # Process different task types
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        # Simple factual questions - use Haiku
        await processor.add_work(
            LLMWorkItem(
                item_id="fact_1",
                strategy=fast_strategy,
                prompt="What year did World War II end?",
            )
        )
        await processor.add_work(
            LLMWorkItem(
                item_id="fact_2",
                strategy=fast_strategy,
                prompt="What is the chemical formula for water?",
            )
        )

        # Complex reasoning tasks - use Sonnet
        await processor.add_work(
            LLMWorkItem(
                item_id="reasoning_1",
                strategy=reasoning_strategy,
                prompt="Explain the philosophical implications of artificial consciousness.",
            )
        )
        await processor.add_work(
            LLMWorkItem(
                item_id="reasoning_2",
                strategy=reasoning_strategy,
                prompt="Compare and contrast utilitarianism and deontological ethics.",
            )
        )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print(f"Total tokens: {result.total_input_tokens + result.total_output_tokens}")
    print("\nResults by task type:")

    for item_result in result.results:
        if item_result.success:
            task_type = "FACT" if item_result.item_id.startswith("fact") else "REASONING"
            print(f"\n[{task_type}] {item_result.item_id}:")
            print(f"  {item_result.output[:150]}...")


# Example 4: Progressive temperature for retries
async def example_anthropic_progressive_temperature():
    """Example with custom strategy that adjusts temperature on retries."""
    print("\n" + "=" * 60)
    print("Example 4: Progressive Temperature Strategy")
    print("=" * 60 + "\n")

    class ProgressiveTempAnthropicStrategy(LLMCallStrategy[str]):
        """Anthropic strategy that increases temperature with each retry."""

        def __init__(
            self,
            client: AsyncAnthropic,
            model: str = "claude-3-5-haiku-20241022",
            base_temps: list[float] | None = None,
        ):
            self.client = client
            self.model = model
            self.base_temps = base_temps if base_temps is not None else [0.3, 0.7, 1.0]

        async def execute(
            self, prompt: str, attempt: int, timeout: float
        ) -> tuple[str, dict[str, int]]:
            # Use progressively higher temperature for retries
            temp = self.base_temps[min(attempt - 1, len(self.base_temps) - 1)]

            print(f"  Attempt {attempt} with temperature {temp}")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}],
            )

            output = response.content[0].text if response.content else ""
            usage = response.usage
            tokens = {
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "total_tokens": (usage.input_tokens + usage.output_tokens)
                if usage
                else 0,
            }

            return output, tokens

    # Initialize client and strategy
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    strategy = ProgressiveTempAnthropicStrategy(
        client=client,
        base_temps=[0.3, 0.7, 1.0],
    )

    # Configure with retries
    from batch_llm.core import RetryConfig

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=30.0,
        retry=RetryConfig(max_attempts=3, initial_wait=1.0),
    )

    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="creative_task",
                strategy=strategy,
                prompt="Write a creative one-sentence story about a robot learning to paint.",
            )
        )

        result = await processor.process_all()

    print(f"\nProcessed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    for item_result in result.results:
        if item_result.success:
            print(f"\nResult: {item_result.output}")


async def main():
    """Run all examples."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        return

    # Run examples
    await example_anthropic_text()
    await example_anthropic_system_prompt()
    await example_anthropic_mixed_models()
    await example_anthropic_progressive_temperature()


if __name__ == "__main__":
    asyncio.run(main())
