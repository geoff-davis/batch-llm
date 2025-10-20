# Gemini API Integration Guide

Complete guide for using batch_llm with Google's Gemini API directly (without PydanticAI).

## Installation

```bash
# Install batch-llm with Gemini support
uv add batch-llm[gemini]
# or
pip install batch-llm[gemini]
```

This installs:
- `batch-llm` - Core batch processing framework
- `google-genai` - Official Google Gemini SDK
- `pydantic` - For response validation

## Setup

### 1. Get API Key

Get a free API key from Google AI Studio:
https://aistudio.google.com/apikey

### 2. Set Environment Variable

```bash
export GOOGLE_API_KEY=your_api_key_here
```

Or in Python:
```python
import os
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
```

### 3. Verify Setup

```python
from google import genai

client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents="Say hello!"
)
print(response.text)
```

## Basic Usage

### Define Your Output Schema

```python
from pydantic import BaseModel, Field
from typing import Annotated

class SummaryOutput(BaseModel):
    """Structured output for text summarization."""
    summary: Annotated[str, Field(description="A concise summary")]
    key_points: Annotated[list[str], Field(description="Main points")]
    sentiment: Annotated[str, Field(description="Overall sentiment")]
```

### Create Direct Call Function

```python
import asyncio
from google import genai
from google.genai.types import GenerateContentConfig

async def call_gemini(
    text: str,
    attempt: int,
    timeout: float
) -> tuple[SummaryOutput, dict[str, int]]:
    """
    Direct Gemini API call with progressive temperature.

    Args:
        text: Input text to process
        attempt: Retry attempt number (1, 2, 3...)
        timeout: Timeout in seconds

    Returns:
        (parsed_output, token_usage_dict)
    """
    # Progressive temperature based on attempt
    temperatures = [0.0, 0.25, 0.5]
    temp = temperatures[min(attempt - 1, len(temperatures) - 1)]

    # Create client
    client = genai.Client()

    # Configure request
    config = GenerateContentConfig(
        temperature=temp,
        response_mime_type="application/json",
        response_schema=SummaryOutput,
    )

    # Make API call with timeout
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=f"Summarize: {text}",
            config=config,
        ),
        timeout=timeout
    )

    # Parse response
    output = SummaryOutput.model_validate_json(response.text)

    # Extract token usage
    usage = response.usage_metadata
    token_usage = {
        "input_tokens": usage.prompt_token_count or 0,
        "output_tokens": usage.candidates_token_count or 0,
        "total_tokens": usage.total_token_count or 0,
    }

    return output, token_usage
```

### Process Batch

```python
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig

async def main():
    # Configure processor
    config = ProcessorConfig(
        max_workers=5,  # Concurrent requests
        timeout_per_item=30.0,  # 30s timeout
    )

    processor = ParallelBatchProcessor[str, SummaryOutput, None](config=config)

    # Add work items
    texts = ["Text 1...", "Text 2...", "Text 3..."]
    for i, text in enumerate(texts):
        work_item = LLMWorkItem(
            item_id=f"text_{i}",
            direct_call=lambda attempt, timeout, text=text: call_gemini(text, attempt, timeout),
            input_data=text,
        )
        await processor.add_work(work_item)

    # Process all
    result = await processor.process_all()

    # Use results
    for item in result.results:
        if item.success:
            print(f"{item.item_id}: {item.output.summary}")
            print(f"  Tokens: {item.token_usage['total_tokens']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Advanced Features

### Progressive Temperature on Retries

The `attempt` parameter allows you to increase temperature on retries:

```python
def get_temperature(attempt: int) -> float:
    """Progressive temperature: 0.0 → 0.5 → 0.9"""
    temperatures = [0.0, 0.5, 0.9]
    return temperatures[min(attempt - 1, len(temperatures) - 1)]
```

**Why?**
- Attempt 1 (temp=0.0): Deterministic, most likely to succeed
- Attempt 2 (temp=0.5): More creative if first attempt had validation errors
- Attempt 3 (temp=0.9): Maximum creativity as last resort

### Model Selection

```python
# Fast, experimental (free tier)
model="gemini-2.0-flash-exp"

# Production-ready, fast
model="gemini-1.5-flash"

# Most capable, slower
model="gemini-1.5-pro"

# Latest experimental
model="gemini-2.0-flash-thinking-exp"  # Supports extended thinking
```

See: https://ai.google.dev/gemini-api/docs/models/gemini

### Generation Config Options

```python
config = GenerateContentConfig(
    # Temperature: 0.0 (deterministic) to 1.0 (creative)
    temperature=0.7,

    # Nucleus sampling: Consider tokens with cumulative probability top_p
    top_p=0.95,

    # Top-k sampling: Consider only top k tokens
    top_k=40,

    # Maximum tokens in response
    max_output_tokens=2048,

    # Stop sequences
    stop_sequences=["END", "STOP"],

    # Structured output
    response_mime_type="application/json",
    response_schema=YourPydanticModel,

    # Safety settings
    safety_settings=[
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ],
)
```

See: https://ai.google.dev/gemini-api/docs/models/generative-models#model-parameters

### Error Handling

```python
from google.genai import errors

async def call_gemini_with_errors(text: str, attempt: int, timeout: float):
    try:
        # ... make API call
        return output, token_usage

    except errors.ClientError as e:
        # Client errors (400): Usually non-retryable
        if e.code == 400:
            raise ValueError(f"Invalid request: {e}")
        raise

    except errors.ServerError as e:
        # Server errors (500): Retryable
        raise RuntimeError(f"Gemini server error: {e}")

    except asyncio.TimeoutError:
        # Timeout: Retryable
        raise TimeoutError(f"Request timed out after {timeout}s")

    except Exception as e:
        # Unknown error: Retryable
        raise RuntimeError(f"Unexpected error: {e}")
```

batch_llm will automatically retry based on error type and your retry config.

### Custom Error Classifier

Tell batch_llm which Gemini errors are retryable:

```python
from batch_llm.strategies import ErrorClassifier, ErrorInfo
from google.genai import errors

class GeminiErrorClassifier(ErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        if isinstance(exception, errors.ClientError):
            # 400 errors - don't retry
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=(exception.code == 429),
                is_timeout=False,
                error_category="client_error"
            )

        if isinstance(exception, errors.ServerError):
            # 500 errors - retry
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=False,
                error_category="server_error"
            )

        if isinstance(exception, asyncio.TimeoutError):
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="timeout"
            )

        # Unknown - retry by default
        return ErrorInfo(
            is_retryable=True,
            is_rate_limit=False,
            is_timeout=False,
            error_category="unknown"
        )

# Use it
processor = ParallelBatchProcessor(
    config=config,
    error_classifier=GeminiErrorClassifier()
)
```

### Token Usage Tracking

```python
# After processing
result = await processor.process_all()

print(f"Total input tokens: {result.total_input_tokens}")
print(f"Total output tokens: {result.total_output_tokens}")
print(f"Total tokens: {result.total_input_tokens + result.total_output_tokens}")

# Per-item usage
for item in result.results:
    if item.success:
        tokens = item.token_usage
        cost = tokens["input_tokens"] * 0.00001 + tokens["output_tokens"] * 0.00003
        print(f"{item.item_id}: ${cost:.6f}")
```

Pricing: https://ai.google.dev/pricing

### Rate Limit Handling

Gemini has rate limits. batch_llm handles this automatically:

```python
from batch_llm.core import RateLimitConfig

config = ProcessorConfig(
    max_workers=10,  # Concurrent requests
    timeout_per_item=30.0,
    rate_limit=RateLimitConfig(
        cooldown_seconds=60.0,  # Wait 60s after rate limit
        slow_start_items=50,  # Gradually resume over 50 items
        slow_start_initial_delay=2.0,  # 2s between items initially
        slow_start_final_delay=0.1,  # 0.1s between items finally
    )
)
```

When rate limit (429) is detected:
1. All workers pause
2. Wait for cooldown period
3. Resume with slow-start (gradual ramp-up)
4. Automatically retry failed items

### Multimodal Inputs

Process images with text:

```python
from google.genai.types import Part, Content

async def call_gemini_vision(
    image_path: str,
    question: str,
    attempt: int,
    timeout: float
) -> tuple[str, dict]:
    client = genai.Client()

    # Read image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Create multimodal content
    contents = [
        Content(
            parts=[
                Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                Part.from_text(text=question)
            ]
        )
    ]

    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=contents,
        ),
        timeout=timeout
    )

    return response.text, extract_token_usage(response)

# Use in batch
work_item = LLMWorkItem(
    item_id="image_1",
    direct_call=lambda a, t: call_gemini_vision("photo.jpg", "What's in this image?", a, t)
)
```

See: https://ai.google.dev/gemini-api/docs/vision

### System Instructions

Add system instructions to guide model behavior:

```python
async def call_gemini_with_system(text: str, attempt: int, timeout: float):
    client = genai.Client()

    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=text,
            config=GenerateContentConfig(
                system_instruction=(
                    "You are a helpful assistant specialized in technical documentation. "
                    "Always provide concise, accurate answers with code examples."
                ),
                temperature=0.7,
            )
        ),
        timeout=timeout
    )

    return parse_response(response)
```

## Complete Example

See `examples/example_gemini_direct.py` for a complete working example.

Run it:

```bash
export GOOGLE_API_KEY=your_key_here
uv run python examples/example_gemini_direct.py
```

## Comparison: PydanticAI vs Direct

### With PydanticAI (agent mode)

```python
# Requires: batch-llm[pydantic-ai]
from pydantic_ai import Agent

agent = Agent('gemini-2.0-flash-exp', result_type=SummaryOutput)

work_item = LLMWorkItem(
    item_id="item_1",
    agent=agent,
    prompt="Summarize this text..."
)
```

**Pros**: Simpler API, less code
**Cons**: Less control over API parameters, extra dependency

### Direct API (direct_call mode)

```python
# Requires: batch-llm[gemini]
async def call_gemini(text, attempt, timeout):
    # Full control over API call
    ...

work_item = LLMWorkItem(
    item_id="item_1",
    direct_call=call_gemini,
)
```

**Pros**: Full control, no PydanticAI dependency, custom temperature/retries
**Cons**: More code, manual token tracking

## Best Practices

1. **Use structured output**: Set `response_schema` for reliable parsing
2. **Implement progressive temperature**: Start low (0.0), increase on retries
3. **Set reasonable timeouts**: 30s for simple, 120s for complex queries
4. **Handle errors gracefully**: Use custom error classifier for Gemini errors
5. **Monitor token usage**: Track costs using `item.token_usage`
6. **Respect rate limits**: Configure `rate_limit` settings appropriately
7. **Choose right model**: Use flash for speed, pro for quality
8. **Add context in prompts**: Include clear instructions and examples

## Troubleshooting

### API Key Not Found

```
Error: GOOGLE_API_KEY environment variable not set
```

**Fix**: Export your API key before running:
```bash
export GOOGLE_API_KEY=your_key_here
```

### Rate Limit Errors (429)

```
google.genai.errors.ClientError: 429 Resource exhausted
```

**Fix**: Reduce `max_workers` or configure rate limiting:
```python
config = ProcessorConfig(
    max_workers=3,  # Lower concurrency
    rate_limit=RateLimitConfig(cooldown_seconds=60.0)
)
```

### Validation Errors

```
pydantic.ValidationError: response doesn't match schema
```

**Fix**:
1. Check your Pydantic model matches expected output
2. Increase temperature on retries (already done with progressive temp)
3. Add examples in your prompt

### Timeout Errors

```
asyncio.TimeoutError
```

**Fix**: Increase timeout:
```python
config = ProcessorConfig(timeout_per_item=60.0)  # 60 seconds
```

## Resources

- **Gemini API Docs**: https://ai.google.dev/gemini-api/docs
- **Python SDK**: https://googleapis.github.io/python-genai/
- **Pricing**: https://ai.google.dev/pricing
- **Models**: https://ai.google.dev/gemini-api/docs/models/gemini
- **Get API Key**: https://aistudio.google.com/apikey
- **Quickstart**: https://ai.google.dev/gemini-api/docs/quickstart
- **batch_llm Docs**: https://github.com/yourusername/batch-llm

## Support

For issues with:
- **batch_llm**: https://github.com/yourusername/batch-llm/issues
- **Gemini API**: https://developers.google.com/support
- **google-genai SDK**: https://github.com/googleapis/python-genai/issues
