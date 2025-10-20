"""Advanced Gemini strategies with smart retry logic for validation errors.

This example demonstrates two advanced patterns for handling Pydantic validation errors:

1. **Progressive Temperature Retry**: Simple retry with increasing temperature
2. **Smart Partial Retry**: On validation failure, send back which fields succeeded
   and ask the LLM to fix only the failed fields

## Installation

```bash
pip install 'batch-llm[gemini]'
export GOOGLE_API_KEY=your_api_key_here
```

## Use Cases

- Extracting structured data from unstructured text
- Form filling from documents
- Data normalization with strict schemas
- Any scenario where validation errors are common
"""

import asyncio
import os
from typing import Annotated

from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field, ValidationError

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.core import RetryConfig
from batch_llm.llm_strategies import LLMCallStrategy


class PersonData(BaseModel):
    """Strict schema for person information."""

    name: Annotated[str, Field(min_length=2, description="Full name")]
    age: Annotated[int, Field(gt=0, lt=150, description="Age in years")]
    email: Annotated[
        str, Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", description="Valid email address")
    ]
    phone: Annotated[
        str, Field(pattern=r"^\+?1?\d{10,14}$", description="Phone number")
    ]


# ============================================================================
# Variant 1: Simple Progressive Temperature Retry
# ============================================================================


class ProgressiveTempGeminiStrategy(LLMCallStrategy[PersonData]):
    """
    Simple strategy: Just retry with higher temperature on validation errors.

    This is the baseline approach - when validation fails, increase temperature
    and hope the LLM produces better output.
    """

    def __init__(self, client: genai.Client, temps=None):
        self.client = client
        self.temps = temps if temps is not None else [0.0, 0.5, 1.0]

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[PersonData, dict[str, int]]:
        temp = self.temps[min(attempt - 1, len(self.temps) - 1)]

        config = GenerateContentConfig(
            temperature=temp,
            response_mime_type="application/json",
            response_schema=PersonData,
        )

        response = await self.client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=config,
        )

        # This will raise ValidationError if response doesn't match schema
        output = PersonData.model_validate_json(response.text)

        usage = response.usage_metadata
        tokens = {
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
            "total_tokens": usage.total_token_count or 0,
        }

        return output, tokens


# ============================================================================
# Variant 2: Smart Retry with Partial Success Feedback
# ============================================================================


class SmartRetryGeminiStrategy(LLMCallStrategy[PersonData]):
    """
    Smart strategy: On validation failure, tell the LLM which fields succeeded
    and which failed, then ask it to fix only the failed fields.

    This is more efficient than blind retries because:
    1. LLM knows exactly what went wrong
    2. LLM can focus on fixing specific fields
    3. Reduces token usage (shorter focused prompts)
    4. Higher success rate on retries
    """

    def __init__(self, client: genai.Client):
        self.client = client
        self.last_response = None  # Track last response for smart retry

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[PersonData, dict[str, int]]:
        # Adjust prompt based on attempt
        if attempt == 1:
            # First attempt: use original prompt
            final_prompt = prompt
            temperature = 0.0
        else:
            # Retry: create smart prompt with feedback
            final_prompt = self._create_retry_prompt(prompt)
            temperature = 0.5  # Slightly higher temp for retries

        config = GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=PersonData,
        )

        response = await self.client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=final_prompt,
            config=config,
        )

        # Try to parse - this may raise ValidationError
        try:
            output = PersonData.model_validate_json(response.text)
            usage = response.usage_metadata
            tokens = {
                "input_tokens": usage.prompt_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0,
                "total_tokens": usage.total_token_count or 0,
            }
            return output, tokens

        except ValidationError as e:
            # Save the error for retry prompt generation
            self.last_response = response.text
            self.last_error = e
            raise  # Re-raise so framework can retry

    def _create_retry_prompt(self, original_prompt: str) -> str:
        """
        Create a smart retry prompt that tells the LLM:
        1. Which fields succeeded (keep these)
        2. Which fields failed (fix these)
        3. Specific validation errors for each failed field
        """
        if not self.last_response or not self.last_error:
            return original_prompt

        # Parse what we can from the last response
        partial_data = {}
        failed_fields = {}

        for error in self.last_error.errors():
            field_name = error["loc"][0] if error["loc"] else "unknown"
            error_msg = error["msg"]
            failed_fields[field_name] = error_msg

        # Try to extract successfully validated fields
        try:
            import json

            raw_data = json.loads(self.last_response)
            for field_name, _field_info in PersonData.model_fields.items():
                if field_name not in failed_fields and field_name in raw_data:
                    partial_data[field_name] = raw_data[field_name]
        except Exception:
            pass  # If we can't parse, fall back to simple retry

        # Build smart retry prompt
        retry_prompt = f"""The previous extraction had validation errors. Here's what we have:

ORIGINAL REQUEST:
{original_prompt}

PREVIOUS ATTEMPT RESULTS:
"""

        if partial_data:
            retry_prompt += "\nSuccessfully extracted (KEEP THESE):\n"
            for field, value in partial_data.items():
                retry_prompt += f"  - {field}: {value}\n"

        retry_prompt += "\nFailed validation (FIX THESE):\n"
        for field, error in failed_fields.items():
            retry_prompt += f"  - {field}: {error}\n"

        retry_prompt += """
Please provide a COMPLETE JSON response with:
1. All successfully extracted fields (unchanged)
2. Fixed versions of the failed fields that satisfy the validation rules

Remember:
- name: Must be at least 2 characters
- age: Must be between 1 and 149
- email: Must be a valid email format (user@domain.com)
- phone: Must be a valid phone number (10-14 digits, optional + prefix)
"""

        return retry_prompt


# ============================================================================
# Example Usage
# ============================================================================


async def example_progressive_temp():
    """Example using simple progressive temperature retry."""
    print("\n" + "=" * 70)
    print("Example 1: Progressive Temperature Retry")
    print("=" * 70 + "\n")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    strategy = ProgressiveTempGeminiStrategy(client=client, temps=[0.0, 0.5, 1.0])

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=30.0,
        retry=RetryConfig(max_attempts=3, initial_wait=1.0),
    )

    # This text has intentional issues that might cause validation errors
    messy_text = """
    Name: John Smith (but call me Johnny)
    Age: thirty-two years old
    Email: john.smith AT example DOT com
    Phone: call me at 555.123.4567
    """

    async with ParallelBatchProcessor[str, PersonData, None](
        config=config
    ) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="person_1",
                strategy=strategy,
                prompt=f"Extract person information from this text:\n\n{messy_text}",
            )
        )

        result = await processor.process_all()

    print(f"Result: {result.succeeded}/{result.total_items} succeeded\n")
    for item in result.results:
        if item.success:
            print("✓ Extracted successfully:")
            print(f"  Name: {item.output.name}")
            print(f"  Age: {item.output.age}")
            print(f"  Email: {item.output.email}")
            print(f"  Phone: {item.output.phone}")
        else:
            print(f"✗ Failed after retries: {item.error}")


async def example_smart_retry():
    """Example using smart retry with partial success feedback."""
    print("\n" + "=" * 70)
    print("Example 2: Smart Retry with Partial Success Feedback")
    print("=" * 70 + "\n")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    strategy = SmartRetryGeminiStrategy(client=client)

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=30.0,
        retry=RetryConfig(max_attempts=3, initial_wait=1.0),
    )

    # This text has intentional issues
    messy_text = """
    Hi, I'm Jane Doe, I'm 28, you can email me at jane.doe@company.org
    My phone? It's 212-555-0123
    """

    async with ParallelBatchProcessor[str, PersonData, None](
        config=config
    ) as processor:
        await processor.add_work(
            LLMWorkItem(
                item_id="person_2",
                strategy=strategy,
                prompt=f"Extract person information from this text:\n\n{messy_text}",
            )
        )

        result = await processor.process_all()

    print(f"Result: {result.succeeded}/{result.total_items} succeeded\n")
    for item in result.results:
        if item.success:
            print("✓ Extracted successfully with smart retry:")
            print(f"  Name: {item.output.name}")
            print(f"  Age: {item.output.age}")
            print(f"  Email: {item.output.email}")
            print(f"  Phone: {item.output.phone}")
            print(f"\nTokens used: {item.token_usage.get('total_tokens', 0)}")
        else:
            print(f"✗ Failed after smart retries: {item.error}")


async def example_comparison():
    """Compare both strategies on difficult data."""
    print("\n" + "=" * 70)
    print("Example 3: Strategy Comparison")
    print("=" * 70 + "\n")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    # Very messy data that will likely need retries
    difficult_texts = [
        """
        Bob Johnson, age unknown, bob@email, call 555-1234
        """,
        """
        Contact: Alice Williams
        She's in her mid-40s
        alice.williams@company.example.com
        +1-415-555-9876
        """,
    ]

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=30.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.5),
    )

    # Test progressive temperature
    print("Testing Progressive Temperature Strategy:")
    prog_strategy = ProgressiveTempGeminiStrategy(client=client)
    async with ParallelBatchProcessor[str, PersonData, None](
        config=config
    ) as processor:
        for i, text in enumerate(difficult_texts):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"prog_{i}",
                    strategy=prog_strategy,
                    prompt=f"Extract person information:\n\n{text}",
                )
            )
        prog_result = await processor.process_all()

    print(f"  Success rate: {prog_result.succeeded}/{prog_result.total_items}")
    print(f"  Total tokens: {prog_result.total_input_tokens + prog_result.total_output_tokens}\n")

    # Test smart retry
    print("Testing Smart Retry Strategy:")
    async with ParallelBatchProcessor[str, PersonData, None](
        config=config
    ) as processor:
        for i, text in enumerate(difficult_texts):
            # Note: Need new instance per item for stateful strategy
            item_strategy = SmartRetryGeminiStrategy(client=client)
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"smart_{i}",
                    strategy=item_strategy,
                    prompt=f"Extract person information:\n\n{text}",
                )
            )
        smart_result = await processor.process_all()

    print(f"  Success rate: {smart_result.succeeded}/{smart_result.total_items}")
    print(f"  Total tokens: {smart_result.total_input_tokens + smart_result.total_output_tokens}\n")

    print("Analysis:")
    print(
        "  Smart Retry typically uses fewer tokens by being more focused on failed fields."
    )
    print("  Progressive Temp is simpler but may waste tokens on full re-extraction.")


async def main():
    """Run all examples."""
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Get your API key from: https://aistudio.google.com/apikey")
        print("Then run: export GOOGLE_API_KEY=your_key_here")
        return

    await example_progressive_temp()
    await example_smart_retry()
    await example_comparison()


if __name__ == "__main__":
    asyncio.run(main())
