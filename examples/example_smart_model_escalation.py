"""Smart model escalation: Only escalate on validation errors, not network errors.

This example demonstrates using the on_error callback to make intelligent retry
decisions based on error type:

1. Validation error → Escalate to smarter model (LLM output didn't match schema)
2. Network error → Retry same model (transient issue)
3. Rate limit → Wait and retry same model (API quota issue)

This is more cost-efficient than always escalating, since network errors and rate
limits don't benefit from a smarter model.

## Installation

```bash
pip install 'batch-llm[gemini]'
export GOOGLE_API_KEY=your_api_key_here
```

## Cost Optimization

By only escalating on validation errors:
- Network errors retry with same (cheap) model
- Rate limits retry with same model after cooldown
- Only actual quality issues trigger expensive models
- Overall cost is lower than blind escalation

## Use Cases

- Production workloads where network errors are common
- Mixed difficulty content with occasional validation issues
- Cost-sensitive batch processing
- Any scenario where you want smart model selection
"""

import asyncio
import os
from typing import Annotated

from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field, ValidationError

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig, TokenUsage
from batch_llm.core import RetryConfig
from batch_llm.llm_strategies import LLMCallStrategy


class PersonData(BaseModel):
    """Strict schema for person information."""

    name: Annotated[str, Field(min_length=2, description="Full name")]
    age: Annotated[int, Field(gt=0, lt=150, description="Age in years")]
    email: Annotated[
        str,
        Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", description="Valid email address"),
    ]
    phone: Annotated[
        str, Field(pattern=r"^\+?1?\d{10,14}$", description="Phone number")
    ]


# ============================================================================
# Smart Model Escalation Strategy
# ============================================================================


class SmartModelEscalationStrategy(LLMCallStrategy[PersonData]):
    """
    Escalate to smarter models ONLY on validation errors.

    This strategy uses the on_error callback to distinguish:
    - Validation errors: LLM output didn't parse → Try smarter model
    - Network errors: Transient issue → Retry same model
    - Rate limits: API quota → Wait and retry same model

    Model progression (only for validation errors):
    1. gemini-2.5-flash-lite (attempt 1: cheapest)
    2. gemini-2.5-flash (attempt 2 if validation failed)
    3. gemini-2.5-pro (attempt 3 if validation failed again)
    """

    MODELS = [
        "gemini-2.5-flash-lite",  # Cheapest, fastest
        "gemini-2.5-flash",  # Production-ready
        "gemini-2.5-pro",  # Most capable
    ]

    def __init__(self, client: genai.Client, verbose: bool = True):
        self.client = client
        self.verbose = verbose

        # State tracked across retries
        self.last_error: Exception | None = None
        self.validation_failures = 0  # Count of validation errors only
        self.total_attempts = 0

    async def on_error(self, exception: Exception, attempt: int) -> None:
        """
        Track error type to make smart escalation decisions.

        Only validation errors trigger model escalation.
        Network/rate limit errors retry with same model.
        """
        self.last_error = exception
        self.total_attempts = attempt

        # Check if this was a validation error
        if self._is_validation_error(exception):
            self.validation_failures += 1
            if self.verbose:
                print(
                    f"  ⚠️  Validation error on attempt {attempt} "
                    f"(total validation failures: {self.validation_failures})"
                )
        else:
            # Network/rate limit error - don't escalate
            error_type = type(exception).__name__
            if self.verbose:
                print(
                    f"  ⚠️  {error_type} on attempt {attempt} "
                    f"(will retry same model)"
                )

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[PersonData, TokenUsage]:
        """
        Select model based on validation failure count (not total attempts).

        - First attempt OR network error: Use cheapest model
        - After 1 validation failure: Use mid-tier model
        - After 2+ validation failures: Use best model
        """
        # Select model based on validation failures, not total attempts
        # This ensures we only escalate when quality is the issue
        model_index = min(self.validation_failures, len(self.MODELS) - 1)
        model = self.MODELS[model_index]

        if self.verbose:
            if attempt == 1:
                print(f"  Attempt {attempt}: Using {model}")
            elif self.last_error and self._is_validation_error(self.last_error):
                print(
                    f"  Attempt {attempt}: Escalating to {model} "
                    f"(after validation error)"
                )
            else:
                print(
                    f"  Attempt {attempt}: Retrying with {model} "
                    f"(network/rate limit error)"
                )

        # Configure request
        config = GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=PersonData,
        )

        # Make API call
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        # Parse response (may raise ValidationError)
        output = PersonData.model_validate_json(response.text)

        # Extract token usage
        usage = response.usage_metadata
        tokens: TokenUsage = {
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
            "total_tokens": usage.total_token_count or 0,
        }

        return output, tokens

    def _is_validation_error(self, error: Exception) -> bool:
        """Check if error is a validation error (vs network/rate limit)."""
        # Pydantic validation errors
        if isinstance(error, ValidationError):
            return True

        # PydanticAI UnexpectedModelBehavior (wraps validation errors)
        try:
            from pydantic_ai.exceptions import UnexpectedModelBehavior

            if isinstance(error, UnexpectedModelBehavior):
                return True
        except ImportError:
            pass

        # Check error message for validation patterns
        error_str = str(error).lower()
        validation_patterns = ("validation", "parsing", "schema", "invalid")
        return any(pattern in error_str for pattern in validation_patterns)


# ============================================================================
# Comparison Strategy: Blind Escalation (for benchmarking)
# ============================================================================


class BlindEscalationStrategy(LLMCallStrategy[PersonData]):
    """
    Always escalate model on retry (for comparison).

    This strategy escalates on EVERY retry, regardless of error type.
    Less efficient than smart escalation but simpler logic.
    """

    MODELS = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    def __init__(self, client: genai.Client, verbose: bool = True):
        self.client = client
        self.verbose = verbose

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[PersonData, TokenUsage]:
        """Always escalate model on retry."""
        model = self.MODELS[min(attempt - 1, len(self.MODELS) - 1)]

        if self.verbose:
            print(f"  Attempt {attempt}: Using {model}")

        config = GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=PersonData,
        )

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        output = PersonData.model_validate_json(response.text)
        usage = response.usage_metadata
        tokens: TokenUsage = {
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
            "total_tokens": usage.total_token_count or 0,
        }

        return output, tokens


# ============================================================================
# Examples
# ============================================================================


async def example_smart_escalation():
    """Example using smart model escalation (validation errors only)."""
    print("\n" + "=" * 70)
    print("Example 1: Smart Model Escalation (Validation Errors Only)")
    print("=" * 70 + "\n")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=30.0,
        retry=RetryConfig(max_attempts=3, initial_wait=1.0),
    )

    # Messy data that might trigger validation errors
    messy_text = """
    Name: John Smith (but call me Johnny)
    Age: thirty-two years old
    Email: john.smith AT example DOT com
    Phone: call me at 555.123.4567
    """

    async with ParallelBatchProcessor[str, PersonData, None](
        config=config
    ) as processor:
        strategy = SmartModelEscalationStrategy(client=client, verbose=True)
        await processor.add_work(
            LLMWorkItem(
                item_id="person_1",
                strategy=strategy,
                prompt=f"Extract person information from this text:\n\n{messy_text}",
            )
        )

        result = await processor.process_all()

    print(f"\nResult: {result.succeeded}/{result.total_items} succeeded\n")
    for item in result.results:
        if item.success:
            print("✓ Extracted successfully:")
            print(f"  Name: {item.output.name}")
            print(f"  Age: {item.output.age}")
            print(f"  Email: {item.output.email}")
            print(f"  Phone: {item.output.phone}")
            print(f"  Tokens: {item.token_usage.get('total_tokens', 0)}")
        else:
            print(f"✗ Failed: {item.error}")


async def example_comparison():
    """Compare smart escalation vs blind escalation."""
    print("\n" + "=" * 70)
    print("Example 2: Smart vs Blind Escalation Comparison")
    print("=" * 70 + "\n")

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    config = ProcessorConfig(
        max_workers=1,
        timeout_per_item=30.0,
        retry=RetryConfig(max_attempts=3, initial_wait=0.5),
    )

    # Mix of validation issues and potential network simulation
    test_texts = [
        """Bob Johnson, age 32, bob@email.com, 555-1234""",
        """Alice Williams, mid-40s, alice@company.example.com, +1-415-555-9876""",
    ]

    # Test smart escalation
    print("Testing Smart Escalation (validation errors only):")
    async with ParallelBatchProcessor[str, PersonData, None](
        config=config
    ) as processor:
        for i, text in enumerate(test_texts):
            strategy = SmartModelEscalationStrategy(client=client, verbose=False)
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"smart_{i}",
                    strategy=strategy,
                    prompt=f"Extract person information:\n\n{text}",
                )
            )
        smart_result = await processor.process_all()

    print(f"  Success rate: {smart_result.succeeded}/{smart_result.total_items}")
    print(
        f"  Total tokens: {smart_result.total_input_tokens + smart_result.total_output_tokens}\n"
    )

    # Test blind escalation
    print("Testing Blind Escalation (always escalate):")
    async with ParallelBatchProcessor[str, PersonData, None](
        config=config
    ) as processor:
        for i, text in enumerate(test_texts):
            strategy = BlindEscalationStrategy(client=client, verbose=False)
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"blind_{i}",
                    strategy=strategy,
                    prompt=f"Extract person information:\n\n{text}",
                )
            )
        blind_result = await processor.process_all()

    print(f"  Success rate: {blind_result.succeeded}/{blind_result.total_items}")
    print(
        f"  Total tokens: {blind_result.total_input_tokens + blind_result.total_output_tokens}\n"
    )

    print("Analysis:")
    print(
        "  Smart Escalation only uses expensive models when quality issues occur."
    )
    print("  Blind Escalation wastes expensive models on network/rate limit errors.")
    print("  In production with network errors, smart escalation saves significant cost.")


async def main():
    """Run all examples."""
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Get your API key from: https://aistudio.google.com/apikey")
        print("Then run: export GOOGLE_API_KEY=your_key_here")
        return

    await example_smart_escalation()
    await example_comparison()

    print("\n" + "=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print(
        """
1. on_error callback enables intelligent retry decisions
2. Smart escalation distinguishes validation vs network errors
3. Only escalate model when LLM quality is the issue
4. Network/rate limit errors retry with same (cheap) model
5. Significant cost savings in production with transient errors

Best for:
- Production workloads with network variability
- Cost-sensitive batch processing
- Mixed difficulty content
- Any scenario where error type matters for retry strategy
    """
    )


if __name__ == "__main__":
    asyncio.run(main())
