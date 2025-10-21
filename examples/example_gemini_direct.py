"""Example of using batch_llm with Google Gemini using GeminiStrategy.

This example shows how to process batches using the built-in GeminiStrategy,
with progressive temperature increases on retries.

## Installation

```bash
pip install 'batch-llm[gemini]'
# or
uv add 'batch-llm[gemini]'
```

## Setup

Set your API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

Get an API key from: https://aistudio.google.com/apikey

## Features Demonstrated

1. Built-in GeminiStrategy for direct Gemini API calls
2. Progressive temperature via custom strategy
3. Structured output with Pydantic response_schema
4. Token usage tracking
5. Async batch processing with concurrency control
6. Error handling and retries

## References

- Gemini API Quickstart: https://ai.google.dev/gemini-api/docs/quickstart
- google-genai package: https://googleapis.github.io/python-genai/
- GeminiStrategy docs: docs/GEMINI_INTEGRATION.md
"""

import asyncio
import os
from typing import Annotated

from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import GeminiStrategy, LLMCallStrategy


class SummaryOutput(BaseModel):
    """Output model for text summarization."""

    summary: Annotated[str, Field(description="A concise summary of the text")]
    key_points: Annotated[list[str], Field(description="Main points from the text")]


class ProgressiveTempGeminiStrategy(LLMCallStrategy[SummaryOutput]):
    """
    Custom Gemini strategy with progressive temperature.

    This demonstrates how to create a custom strategy that adjusts
    temperature based on retry attempt number.
    """

    def __init__(self, client: genai.Client, temps=None):
        """
        Initialize progressive temperature strategy.

        Args:
            client: Initialized Gemini client
            temps: List of temperatures for attempts [attempt1, attempt2, attempt3...]
        """
        self.client = client
        self.temps = temps if temps is not None else [0.0, 0.25, 0.5]

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[SummaryOutput, dict[str, int]]:
        """
        Execute Gemini call with temperature based on attempt.

        Args:
            prompt: The prompt to send
            attempt: Retry attempt number (1, 2, 3, ...)
            timeout: Timeout in seconds (enforced by framework)

        Returns:
            (parsed_output, token_usage_dict)
        """
        # Progressive temperature: 0.0 -> 0.25 -> 0.5
        # Lower temps (0.0) are more deterministic, higher temps (0.5) more creative
        temp = self.temps[min(attempt - 1, len(self.temps) - 1)]

        # Configure the request
        config = GenerateContentConfig(
            temperature=temp,
            response_mime_type="application/json",
            response_schema=SummaryOutput,
        )

        # Make the API call
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )

        # Parse the response
        output = SummaryOutput.model_validate_json(response.text)

        # Extract token usage
        usage_metadata = response.usage_metadata
        token_usage = {
            "input_tokens": usage_metadata.prompt_token_count or 0,
            "output_tokens": usage_metadata.candidates_token_count or 0,
            "total_tokens": usage_metadata.total_token_count or 0,
        }

        return output, token_usage


async def main():
    """
    Run batch processing with Gemini API.

    Demonstrates two approaches:
    1. Built-in GeminiStrategy
    2. Custom ProgressiveTempGeminiStrategy
    """

    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Get your API key from: https://aistudio.google.com/apikey")
        print("Then run: export GOOGLE_API_KEY=your_key_here")
        return

    # Initialize Gemini client
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Sample texts to summarize
    texts = [
        "Artificial intelligence has revolutionized many industries. From healthcare "
        "to finance, AI systems are making processes more efficient and accurate.",
        "Climate change is one of the most pressing challenges of our time. Rising "
        "temperatures and extreme weather events are affecting communities worldwide.",
        "The space industry is experiencing rapid growth. Private companies are now "
        "launching satellites and planning missions to Mars and beyond.",
    ]

    # Example 1: Using built-in GeminiStrategy (simple, no progressive temperature)
    print("\n" + "=" * 60)
    print("Example 1: Built-in GeminiStrategy")
    print("=" * 60 + "\n")

    def parse_response(response) -> SummaryOutput:
        """Parse Gemini response into SummaryOutput."""
        return SummaryOutput.model_validate_json(response.text)

    strategy = GeminiStrategy(
        model="gemini-2.5-flash",
        client=client,
        response_parser=parse_response,
        config=GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=SummaryOutput,
        ),
    )

    config = ProcessorConfig(max_workers=3, timeout_per_item=30.0)

    async with ParallelBatchProcessor[str, SummaryOutput, None](
        config=config
    ) as processor:
        for i, text in enumerate(texts):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"text_{i}",
                    strategy=strategy,
                    prompt=f"Please summarize this text:\n\n{text}",
                )
            )

        result = await processor.process_all()

    print(f"✓ Completed: {result.succeeded}/{result.total_items} successful\n")

    for item_result in result.results:
        if item_result.success and item_result.output:
            print(f"Item {item_result.item_id}:")
            print(f"  Summary: {item_result.output.summary}")
            print(f"  Key points: {', '.join(item_result.output.key_points)}")
            print(f"  Tokens: {item_result.token_usage.get('total_tokens', 0)} total\n")
        else:
            print(f"Item {item_result.item_id}: FAILED - {item_result.error}\n")

    # Example 2: Using custom ProgressiveTempGeminiStrategy
    print("\n" + "=" * 60)
    print("Example 2: Progressive Temperature Strategy")
    print("=" * 60 + "\n")

    progressive_strategy = ProgressiveTempGeminiStrategy(
        client=client,
        temps=[0.0, 0.5, 1.0],  # Increase temp on retries
    )

    async with ParallelBatchProcessor[str, SummaryOutput, None](
        config=config
    ) as processor:
        for i, text in enumerate(texts):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"progressive_{i}",
                    strategy=progressive_strategy,
                    prompt=f"Please summarize this text:\n\n{text}",
                )
            )

        result = await processor.process_all()

    print(f"✓ Completed: {result.succeeded}/{result.total_items} successful\n")

    for item_result in result.results:
        if item_result.success and item_result.output:
            print(f"Item {item_result.item_id}:")
            print(f"  Summary: {item_result.output.summary}")
            print(f"  Tokens: {item_result.token_usage.get('total_tokens', 0)} total\n")


if __name__ == "__main__":
    # Note: Requires GOOGLE_API_KEY environment variable
    # export GOOGLE_API_KEY=your_api_key_here
    asyncio.run(main())
