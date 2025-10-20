"""Example demonstrating batch-llm with OpenAI API.

This example shows how to create a custom strategy for OpenAI's API,
including both direct API calls and structured output with Pydantic models.

Install dependencies:
    pip install 'batch-llm' 'openai'
"""

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import LLMCallStrategy


class SummaryOutput(BaseModel):
    """Example output model for structured summaries."""

    summary: str
    key_points: list[str]
    sentiment: str


class OpenAIStrategy(LLMCallStrategy[str]):
    """Strategy for calling OpenAI API with simple text responses."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize OpenAI strategy.

        Args:
            client: Initialized AsyncOpenAI client
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, dict[str, int]]:
        """Execute OpenAI API call.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        # Make the API call
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Extract output
        output = response.choices[0].message.content or ""

        # Extract token usage
        usage = response.usage
        tokens = {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

        return output, tokens


class OpenAIStructuredStrategy(LLMCallStrategy[SummaryOutput]):
    """Strategy for calling OpenAI API with structured output using Pydantic."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        """
        Initialize OpenAI structured output strategy.

        Args:
            client: Initialized AsyncOpenAI client
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            temperature: Sampling temperature (0.0-2.0)
        """
        self.client = client
        self.model = model
        self.temperature = temperature

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[SummaryOutput, dict[str, int]]:
        """Execute OpenAI API call with structured output.

        Note: timeout parameter is provided for information but timeout enforcement
        is handled by the framework wrapping this call in asyncio.wait_for().
        """
        # Make the API call with structured output
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format=SummaryOutput,
        )

        # Extract parsed output
        output = response.choices[0].message.parsed
        if output is None:
            raise ValueError("OpenAI returned None for parsed output")

        # Extract token usage
        usage = response.usage
        tokens = {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

        return output, tokens


# Example 1: Simple text generation with OpenAI
async def example_openai_text():
    """Example using OpenAI for simple text generation."""
    print("\n" + "=" * 60)
    print("Example 1: OpenAI Text Generation")
    print("=" * 60 + "\n")

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Create the strategy
    strategy = OpenAIStrategy(
        client=client,
        model="gpt-4o-mini",
        temperature=0.7,
    )

    # Configure the processor
    config = ProcessorConfig(max_workers=3, timeout_per_item=30.0)

    # Process items
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        questions = [
            "What is the capital of France?",
            "Explain quantum computing in one sentence.",
            "What is the largest planet in our solar system?",
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
            print(f"  {item_result.output}")


# Example 2: Structured output with Pydantic
async def example_openai_structured():
    """Example using OpenAI with structured output."""
    print("\n" + "=" * 60)
    print("Example 2: OpenAI Structured Output")
    print("=" * 60 + "\n")

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Create the strategy
    strategy = OpenAIStructuredStrategy(
        client=client,
        model="gpt-4o-mini",
        temperature=0.7,
    )

    # Configure the processor
    config = ProcessorConfig(max_workers=2, timeout_per_item=30.0)

    # Process items
    async with ParallelBatchProcessor[None, SummaryOutput, None](
        config=config
    ) as processor:
        documents = [
            """
            Python is a high-level, interpreted programming language known for its
            simplicity and readability. It's widely used in web development, data science,
            machine learning, and automation. Python's extensive standard library and
            active community make it an excellent choice for beginners and experts alike.
            """,
            """
            Climate change refers to long-term shifts in global temperatures and weather
            patterns. While climate change is a natural phenomenon, scientific evidence
            shows that human activities have been the dominant cause of warming since the
            mid-20th century, primarily through greenhouse gas emissions.
            """,
        ]

        for i, doc in enumerate(documents):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"doc_{i}",
                    strategy=strategy,
                    prompt=f"Summarize the following text and extract key points:\n\n{doc}",
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    print("\nSummaries:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  Summary: {item_result.output.summary}")
            print(f"  Key Points: {', '.join(item_result.output.key_points)}")
            print(f"  Sentiment: {item_result.output.sentiment}")


# Example 3: Batch processing with context
async def example_openai_with_context():
    """Example using OpenAI with context data for each item."""
    print("\n" + "=" * 60)
    print("Example 3: OpenAI with Context")
    print("=" * 60 + "\n")

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Create the strategy
    strategy = OpenAIStrategy(
        client=client,
        model="gpt-4o-mini",
        temperature=0.5,
    )

    # Configure the processor
    config = ProcessorConfig(max_workers=3, timeout_per_item=30.0)

    # Define items with context
    items = [
        {
            "id": "review_1",
            "text": "This product is amazing! Highly recommend.",
            "category": "Electronics",
        },
        {
            "id": "review_2",
            "text": "Not worth the money. Very disappointed.",
            "category": "Books",
        },
        {
            "id": "review_3",
            "text": "Good quality but overpriced.",
            "category": "Clothing",
        },
    ]

    # Process items with context
    async with ParallelBatchProcessor[None, str, dict](config=config) as processor:
        for item in items:
            await processor.add_work(
                LLMWorkItem(
                    item_id=item["id"],
                    strategy=strategy,
                    prompt=f"Classify the sentiment of this {item['category']} review as positive, negative, or neutral: \"{item['text']}\"",
                    context=item,  # Pass context through
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print("\nSentiment Analysis:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  Category: {item_result.context['category']}")
            print(f"  Review: {item_result.context['text']}")
            print(f"  Sentiment: {item_result.output.strip()}")


async def main():
    """Run all examples."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return

    # Run examples
    await example_openai_text()
    await example_openai_structured()
    await example_openai_with_context()


if __name__ == "__main__":
    asyncio.run(main())
