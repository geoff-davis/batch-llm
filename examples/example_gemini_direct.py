"""Example of using batch_llm with direct Gemini API calls (no PydanticAI required).

This example shows how to process batches using the Gemini API directly,
with progressive temperature increases on retries.

## Installation

```bash
uv add batch-llm[gemini]
# or
pip install batch-llm[gemini]
```

## Setup

Set your API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

Get an API key from: https://aistudio.google.com/apikey

## Features Demonstrated

1. Direct Gemini API calls without PydanticAI
2. Progressive temperature on retries (0.0 → 0.25 → 0.5)
3. Structured output with Pydantic response_schema
4. Token usage tracking
5. Async batch processing with concurrency control
6. Error handling and retries

## References

- Gemini API Quickstart: https://ai.google.dev/gemini-api/docs/quickstart
- google-genai package: https://googleapis.github.io/python-genai/
"""

import asyncio
import os
from typing import Annotated

from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig


class SummaryOutput(BaseModel):
    """Output model for text summarization."""

    summary: Annotated[str, Field(description="A concise summary of the text")]
    key_points: Annotated[list[str], Field(description="Main points from the text")]


async def call_gemini_direct(
    text: str, attempt: int, timeout: float
) -> tuple[SummaryOutput, dict[str, int]]:
    """
    Call Gemini API directly with temperature based on attempt number.

    This demonstrates how to integrate the Gemini API directly with batch_llm
    without using PydanticAI. Features:
    - Progressive temperature: Increases with each retry attempt
    - Structured output: Uses Pydantic schema for response validation
    - Token tracking: Returns usage metadata for monitoring
    - Timeout handling: Wraps async call with timeout

    Args:
        text: Text to summarize
        attempt: Attempt number (1, 2, 3...) - automatically provided by batch_llm
        timeout: Timeout in seconds - automatically provided by batch_llm

    Returns:
        (parsed_output, token_usage)

    Raises:
        TimeoutError: If call exceeds timeout
        ValidationError: If response doesn't match schema
        google.genai.errors.*: Various Gemini API errors
    """
    # Progressive temperature: 0.0 -> 0.25 -> 0.5
    # Lower temps (0.0) are more deterministic, higher temps (0.5) more creative
    temperatures = [0.0, 0.25, 0.5]
    temp = temperatures[min(attempt - 1, len(temperatures) - 1)]

    # Create client - automatically uses GOOGLE_API_KEY environment variable
    # See: https://ai.google.dev/gemini-api/docs/quickstart#set-up-api-key
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Configure the request
    # See: https://ai.google.dev/gemini-api/docs/models/generative-models
    config = GenerateContentConfig(
        temperature=temp,  # Controls randomness (0.0-1.0)
        response_mime_type="application/json",  # JSON output
        response_schema=SummaryOutput,  # Pydantic model for validation
        # Optional: Add other parameters
        # top_p=0.95,  # Nucleus sampling
        # top_k=40,  # Top-k sampling
        # max_output_tokens=1024,  # Limit response length
    )

    # Make the API call with timeout
    # See: https://ai.google.dev/gemini-api/docs/text-generation
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",  # Fast, experimental model
            # model="gemini-1.5-flash",  # Production alternative
            # model="gemini-1.5-pro",  # More capable, slower
            contents=f"Please summarize this text:\n\n{text}",
            config=config,
        ),
        timeout=timeout,
    )

    # Parse the response using Pydantic
    output = SummaryOutput.model_validate_json(response.text)

    # Extract token usage for cost tracking
    # See: https://ai.google.dev/pricing
    usage_metadata = response.usage_metadata
    token_usage = {
        "input_tokens": usage_metadata.prompt_token_count or 0,
        "output_tokens": usage_metadata.candidates_token_count or 0,
        "total_tokens": usage_metadata.total_token_count or 0,
    }

    return output, token_usage


async def main():
    """
    Run batch processing with direct Gemini API calls.

    This demonstrates batch_llm with direct Gemini integration:
    - No PydanticAI dependency required
    - Full control over API parameters
    - Progressive temperature on retries
    - Parallel processing with concurrency limits
    - Automatic retry logic with exponential backoff
    """

    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Get your API key from: https://aistudio.google.com/apikey")
        print("Then run: export GOOGLE_API_KEY=your_key_here")
        return

    # Sample texts to summarize
    texts = [
        "Artificial intelligence has revolutionized many industries. From healthcare "
        "to finance, AI systems are making processes more efficient and accurate.",
        "Climate change is one of the most pressing challenges of our time. Rising "
        "temperatures and extreme weather events are affecting communities worldwide.",
        "The space industry is experiencing rapid growth. Private companies are now "
        "launching satellites and planning missions to Mars and beyond.",
    ]

    # Configure processor with sensible defaults
    config = ProcessorConfig(
        max_workers=3,  # Process 3 texts concurrently (respects rate limits)
        timeout_per_item=30.0,  # 30 seconds per request
        # Retry config with progressive temperature
        # retry=RetryConfig(
        #     max_attempts=3,
        #     initial_wait=1.0,
        #     max_wait=10.0,
        #     exponential_base=2.0,
        # ),
    )

    processor = ParallelBatchProcessor[str, SummaryOutput, None](config=config)

    # Add work items using direct_call mode
    for i, text in enumerate(texts):
        work_item = LLMWorkItem(
            item_id=f"text_{i}",
            direct_call=lambda attempt, timeout, text=text: call_gemini_direct(
                text, attempt, timeout
            ),
            input_data=text,  # Store the text for reference
            context=None,
        )
        await processor.add_work(work_item)

    # Process all items
    print(f"Processing {len(texts)} texts with direct Gemini API calls...\n")
    result = await processor.process_all()

    # Display results
    print(f"✓ Completed: {result.succeeded}/{result.total_items} successful\n")

    for item_result in result.results:
        if item_result.success and item_result.output:
            print(f"Item {item_result.item_id}:")
            print(f"  Summary: {item_result.output.summary}")
            print(f"  Key points: {', '.join(item_result.output.key_points)}")
            print(
                f"  Tokens: {item_result.token_usage.get('total_tokens', 0)} total\n"
            )
        else:
            print(f"Item {item_result.item_id}: FAILED - {item_result.error}\n")

    # Get final stats
    stats = await processor.get_stats()
    print(f"\nFinal stats: {stats}")


if __name__ == "__main__":
    # Note: Requires GOOGLE_API_KEY environment variable
    # export GOOGLE_API_KEY=your_api_key_here
    asyncio.run(main())
