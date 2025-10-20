"""Example demonstrating the new LLM call strategy pattern.

This example shows how to use all three built-in strategy classes:
1. PydanticAIStrategy - For PydanticAI agents (structured output)
2. GeminiStrategy - For direct Gemini API calls
3. GeminiCachedStrategy - For Gemini API calls with context caching

The strategy pattern provides a clean, flexible way to configure how LLM calls
are made, including model selection, caching behavior, and response parsing.
"""

import asyncio
import os
from typing import Any

from google import genai
from pydantic import BaseModel
from pydantic_ai import Agent

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import (
    GeminiCachedStrategy,
    GeminiStrategy,
    PydanticAIStrategy,
)


class SummaryOutput(BaseModel):
    """Example output model for structured summaries."""

    summary: str
    key_points: list[str]


# Example 1: Using PydanticAIStrategy
async def example_pydantic_ai_strategy():
    """Example using PydanticAI agent through the strategy pattern."""
    print("\n" + "=" * 60)
    print("Example 1: PydanticAI Strategy")
    print("=" * 60 + "\n")

    # Create a PydanticAI agent
    agent = Agent(
        "gemini-2.0-flash-exp",
        result_type=SummaryOutput,
        system_prompt="You are a helpful assistant that summarizes text.",
    )

    # Wrap the agent in a strategy
    strategy = PydanticAIStrategy(agent=agent)

    # Configure the processor
    config = ProcessorConfig(max_workers=2, timeout_per_item=30.0)

    # Process items using the strategy
    async with ParallelBatchProcessor[None, SummaryOutput, None](
        config=config
    ) as processor:
        documents = [
            "Python is a high-level programming language known for its simplicity and readability.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        ]

        for i, doc in enumerate(documents):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"doc_{i}",
                    strategy=strategy,
                    prompt=f"Summarize this text:\n\n{doc}",
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  Summary: {item_result.output.summary}")
            print(f"  Key Points: {item_result.output.key_points}")


# Example 2: Using GeminiStrategy for direct API calls
async def example_gemini_strategy():
    """Example using direct Gemini API calls through the strategy pattern."""
    print("\n" + "=" * 60)
    print("Example 2: Gemini Strategy (Direct API)")
    print("=" * 60 + "\n")

    # Initialize Gemini client
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Create a response parser
    def parse_response(response: Any) -> str:
        """Extract text from Gemini response."""
        return response.text

    # Create the strategy
    strategy = GeminiStrategy(
        model="gemini-2.0-flash-exp",
        client=client,
        response_parser=parse_response,
        config=genai.types.GenerateContentConfig(temperature=0.7),
    )

    # Configure the processor
    config = ProcessorConfig(max_workers=2, timeout_per_item=30.0)

    # Process items
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        questions = [
            "What is the capital of France?",
            "What is the largest ocean on Earth?",
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
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  Answer: {item_result.output}")


# Example 3: Using GeminiCachedStrategy for RAG-style workflows
async def example_gemini_cached_strategy():
    """Example using Gemini with context caching for RAG-style workflows."""
    print("\n" + "=" * 60)
    print("Example 3: Gemini Cached Strategy (RAG)")
    print("=" * 60 + "\n")

    # Initialize Gemini client
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Large context to cache (e.g., a document, knowledge base, etc.)
    # In a real RAG system, this would be retrieved documents
    large_context = """
    Python Programming Language Overview:

    Python is a high-level, interpreted programming language created by Guido van Rossum.
    It emphasizes code readability with its use of significant indentation.

    Key Features:
    - Dynamic typing
    - Automatic memory management
    - Extensive standard library
    - Multi-paradigm (supports object-oriented, functional, and procedural programming)

    Popular Uses:
    - Web development (Django, Flask)
    - Data science and machine learning (NumPy, Pandas, scikit-learn)
    - Automation and scripting
    - Scientific computing
    """

    # Create cached content (system instruction + context)
    cached_content = [
        genai.types.Content(
            role="user",
            parts=[
                genai.types.Part(
                    text=f"Use the following context to answer questions:\n\n{large_context}"
                )
            ],
        ),
    ]

    # Create response parser
    def parse_response(response: Any) -> str:
        return response.text

    # Create the cached strategy
    strategy = GeminiCachedStrategy(
        model="gemini-2.0-flash-exp",
        client=client,
        response_parser=parse_response,
        cached_content=cached_content,
        cache_ttl_seconds=3600,  # Cache for 1 hour
        cache_refresh_threshold=0.1,  # Refresh if <10% TTL remaining
    )

    # Configure the processor
    config = ProcessorConfig(max_workers=3, timeout_per_item=30.0)

    # Process multiple questions using the same cached context
    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        questions = [
            "What is Python?",
            "What are the key features of Python?",
            "What is Python commonly used for?",
        ]

        for i, question in enumerate(questions):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"rag_question_{i}",
                    strategy=strategy,
                    prompt=question,
                )
            )

        result = await processor.process_all()

    print(f"Processed: {result.total_items} items")
    print(f"Total input tokens: {result.total_input_tokens}")
    print(f"Total output tokens: {result.total_output_tokens}")
    print("\nAnswers:")
    for item_result in result.results:
        if item_result.success:
            print(f"\n{item_result.item_id}:")
            print(f"  {item_result.output}")


# Example 4: Custom strategy
async def example_custom_strategy():
    """Example showing how to create a custom LLM call strategy."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Strategy")
    print("=" * 60 + "\n")

    from batch_llm.llm_strategies import LLMCallStrategy

    class MockLLMStrategy(LLMCallStrategy[str]):
        """
        Custom strategy that simulates an LLM call.

        This demonstrates the strategy lifecycle:
        - prepare() called once before any retries
        - execute() called for each attempt
        - cleanup() called once after all attempts
        """

        def __init__(self):
            self.call_count = 0

        async def prepare(self):
            """Initialize resources."""
            print("  [Strategy] prepare() called - initializing resources")

        async def execute(
            self, prompt: str, attempt: int, timeout: float
        ) -> tuple[str, dict[str, int]]:
            """Execute the LLM call."""
            self.call_count += 1
            print(
                f"  [Strategy] execute() called - attempt {attempt}, prompt: {prompt[:50]}..."
            )

            # Simulate processing
            await asyncio.sleep(0.1)

            return (
                f"Mock response for: {prompt}",
                {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            )

        async def cleanup(self):
            """Clean up resources."""
            print(
                f"  [Strategy] cleanup() called - processed {self.call_count} calls total"
            )

    # Use the custom strategy
    strategy = MockLLMStrategy()
    config = ProcessorConfig(max_workers=1, timeout_per_item=10.0)

    async with ParallelBatchProcessor[None, str, None](config=config) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="custom_1",
                strategy=strategy,
                prompt="Test prompt",
            )
        )

        result = await processor.process_all()

    print(f"\nProcessed: {result.total_items} items")
    print(f"Succeeded: {result.succeeded}")


async def main():
    """Run all examples."""
    # Example 1: PydanticAI Strategy
    if os.environ.get("GEMINI_API_KEY"):
        await example_pydantic_ai_strategy()
        await example_gemini_strategy()
        await example_gemini_cached_strategy()
    else:
        print("Skipping Gemini examples (GEMINI_API_KEY not set)")
        print("Set GEMINI_API_KEY environment variable to run Gemini examples")

    # Example 4: Custom Strategy (doesn't require API key)
    await example_custom_strategy()


if __name__ == "__main__":
    asyncio.run(main())
