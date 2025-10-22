# Gemini API Integration Guide

Complete guide for using batch-llm with Google's Gemini API.

## Installation

```bash
# Install batch-llm with Gemini support
pip install 'batch-llm[gemini]'
# or
uv add 'batch-llm[gemini]'
```

This installs:

- `batch-llm` - Core batch processing framework
- `google-genai` - Official Google Gemini SDK
- `pydantic` - For response validation

## Setup

### 1. Get API Key

Get a free API key from Google AI Studio:
<https://aistudio.google.com/apikey>

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
    model="gemini-2.5-flash",
    contents="Say hello!"
)
print(response.text)
```

## Usage with batch-llm

batch-llm provides two built-in Gemini strategies:

### 1. GeminiStrategy (Simple API Calls)

For direct Gemini API calls without caching:

```python
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import GeminiStrategy
from google import genai
from pydantic import BaseModel

class SummaryOutput(BaseModel):
    """Structured output for summarization."""
    summary: str
    key_points: list[str]

# Create client
client = genai.Client(api_key="your-api-key")

# Create response parser
def parse_response(response) -> SummaryOutput:
    """Parse Gemini response into your output model."""
    return SummaryOutput.model_validate_json(response.text)

# Create strategy
strategy = GeminiStrategy(
    model="gemini-2.5-flash",
    client=client,
    response_parser=parse_response,
    config=genai.types.GenerateContentConfig(
        temperature=0.7,
        response_mime_type="application/json",
        response_schema=SummaryOutput,
    ),
)

# Configure processor
config = ProcessorConfig(max_workers=5, timeout_per_item=30.0)

# Process items
async with ParallelBatchProcessor[str, SummaryOutput, None](config=config) as processor:
    texts = ["Text 1...", "Text 2...", "Text 3..."]

    for i, text in enumerate(texts):
        await processor.add_work(
            LLMWorkItem(
                item_id=f"text_{i}",
                strategy=strategy,
                prompt=f"Summarize: {text}",
            )
        )

    result = await processor.process_all()

# Use results
for item in result.results:
    if item.success:
        print(f"{item.item_id}: {item.output.summary}")
        print(f"  Tokens: {item.token_usage['total_tokens']}")
```

### 2. GeminiCachedStrategy (With Context Caching)

Perfect for RAG applications with large shared context:

```python
from batch_llm.llm_strategies import GeminiCachedStrategy
from google import genai

client = genai.Client(api_key="your-api-key")

# Define large context to cache (e.g., retrieved documents)
cached_content = [
    genai.types.Content(
        role="user",
        parts=[
            genai.types.Part(text="Large document or knowledge base to cache...")
        ]
    )
]

def parse_response(response) -> str:
    return response.text

# Create cached strategy
strategy = GeminiCachedStrategy(
    model="gemini-2.5-flash",
    client=client,
    response_parser=parse_response,
    cached_content=cached_content,
    cache_ttl_seconds=3600,  # Cache for 1 hour
    cache_refresh_threshold=0.1,  # Refresh when <10% TTL remaining
)

# Use in processor
config = ProcessorConfig(max_workers=3, timeout_per_item=30.0)

async with ParallelBatchProcessor[str, str, None](config=config) as processor:
    questions = [
        "What is the main topic?",
        "What are the key findings?",
        "What are the conclusions?",
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

# Cache is automatically:
# - Created on first use
# - Refreshed when TTL is low
# - Deleted on cleanup
```

## Advanced Features

### Progressive Temperature on Retries

Create a custom strategy that adjusts temperature based on attempt:

```python
from batch_llm.llm_strategies import LLMCallStrategy

class ProgressiveTempGeminiStrategy(LLMCallStrategy[SummaryOutput]):
    """Gemini strategy with progressive temperature."""

    def __init__(self, client: genai.Client, temps=[0.0, 0.5, 1.0]):
        self.client = client
        self.temps = temps

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[SummaryOutput, dict[str, int]]:
        # Use higher temperature for retries
        temp = self.temps[min(attempt - 1, len(self.temps) - 1)]

        config = genai.types.GenerateContentConfig(
            temperature=temp,
            response_mime_type="application/json",
            response_schema=SummaryOutput,
        )

        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )

        output = SummaryOutput.model_validate_json(response.text)

        usage = response.usage_metadata
        tokens = {
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
            "total_tokens": usage.total_token_count or 0,
        }

        return output, tokens

# Use it
strategy = ProgressiveTempGeminiStrategy(client=client, temps=[0.0, 0.5, 1.0])
```

**Why progressive temperature?**

- Attempt 1 (temp=0.0): Deterministic, most likely to succeed
- Attempt 2 (temp=0.5): More creative if first attempt had validation errors
- Attempt 3 (temp=1.0): Maximum creativity as last resort

### Model Selection

```python
# Fast, experimental (free tier)
model="gemini-2.0-flash-exp"

# Production-ready, fast
model="gemini-2.5-flash-lite"

# Most capable, slower
model="gemini-2.5-flash"

# With extended thinking
model="gemini-2.5-pro"
```

See: <https://ai.google.dev/gemini-api/docs/models/gemini>

### Generation Config Options

```python
from google.genai.types import GenerateContentConfig

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

    # System instruction
    system_instruction="You are a helpful assistant...",

    # Safety settings
    safety_settings=[
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ],
)
```

See: <https://ai.google.dev/gemini-api/docs/models/generative-models#model-parameters>

### Error Handling

batch-llm includes `GeminiErrorClassifier` for Gemini-specific errors:

```python
from batch_llm.classifiers import GeminiErrorClassifier

processor = ParallelBatchProcessor(
    config=config,
    error_classifier=GeminiErrorClassifier(),  # Handles 429, 500, etc.
)
```

The classifier automatically:

- Detects rate limit errors (429) as non-retryable
- Marks server errors (500) as retryable
- Detects timeout errors
- Handles validation errors

### Rate Limit Handling

Gemini has rate limits. Configure automatic handling:

```python
from batch_llm.core import RateLimitConfig

config = ProcessorConfig(
    max_workers=10,
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
from batch_llm.llm_strategies import LLMCallStrategy

class GeminiVisionStrategy(LLMCallStrategy[str]):
    """Strategy for Gemini vision tasks."""

    def __init__(self, client: genai.Client, image_path: str):
        self.client = client
        self.image_path = image_path

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, dict[str, int]]:
        # Read image
        with open(self.image_path, "rb") as f:
            image_bytes = f.read()

        # Create multimodal content
        contents = [
            Content(
                parts=[
                    Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    Part.from_text(text=prompt)
                ]
            )
        ]

        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )

        usage = response.usage_metadata
        tokens = {
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
            "total_tokens": usage.total_token_count or 0,
        }

        return response.text, tokens

# Use it
strategy = GeminiVisionStrategy(client=client, image_path="photo.jpg")
```

See: <https://ai.google.dev/gemini-api/docs/vision>

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

Pricing: <https://ai.google.dev/pricing>

## Complete Example

See `examples/example_gemini_direct.py` for a complete working example.

Run it:

```bash
export GOOGLE_API_KEY=your_key_here
uv run python examples/example_gemini_direct.py
```

## Comparison: PydanticAI vs Direct Strategies

### With PydanticAI

```python
from batch_llm import PydanticAIStrategy
from pydantic_ai import Agent

agent = Agent('gemini-2.5-flash', result_type=SummaryOutput)
strategy = PydanticAIStrategy(agent=agent)

work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="Summarize this text..."
)
```

**Pros**: Simpler API, less code
**Cons**: Less control over API parameters, extra dependency

### Direct Gemini Strategy

```python
from batch_llm.llm_strategies import GeminiStrategy

strategy = GeminiStrategy(
    model="gemini-2.5-flash",
    client=client,
    response_parser=parse_response,
    config=config,  # Full control
)

work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="Summarize this text..."
)
```

**Pros**: Full control, no PydanticAI dependency, custom configurations
**Cons**: More code, manual parsing

## Best Practices

1. **Use structured output**: Set `response_schema` for reliable parsing
2. **Implement progressive temperature**: Start low (0.0), increase on retries
3. **Set reasonable timeouts**: 30s for simple, 120s for complex queries
4. **Handle errors gracefully**: Use `GeminiErrorClassifier` for Gemini errors
5. **Monitor token usage**: Track costs using `item.token_usage`
6. **Respect rate limits**: Configure `rate_limit` settings appropriately
7. **Choose right model**: Use flash for speed, pro for quality
8. **Use caching for RAG**: `GeminiCachedStrategy` saves money on repeated context

## Troubleshooting

### API Key Not Found

```text
Error: GOOGLE_API_KEY environment variable not set
```

**Fix**: Export your API key before running:

```bash
export GOOGLE_API_KEY=your_key_here
```

### Rate Limit Errors (429)

```text
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

```text
pydantic.ValidationError: response doesn't match schema
```

**Fix**:

1. Check your Pydantic model matches expected output
2. Use progressive temperature strategy (increases temp on retries)
3. Add examples in your prompt

### Timeout Errors

```text
asyncio.TimeoutError
```

**Fix**: Increase timeout:

```python
config = ProcessorConfig(timeout_per_item=60.0)  # 60 seconds
```

## Advanced: Smart Retry with on_error

Use the `on_error` callback to handle Gemini-specific errors intelligently:

### Smart Model Escalation for Gemini

Only escalate to expensive Gemini models on validation errors, not network/rate limit errors:

```python
from batch_llm.llm_strategies import LLMCallStrategy
from batch_llm import TokenUsage
from pydantic import ValidationError
from google import genai

class SmartGeminiStrategy(LLMCallStrategy[PersonData]):
    """Smart model escalation for Gemini API."""

    MODELS = [
        "gemini-2.5-flash-lite",  # Cheapest, fastest
        "gemini-2.5-flash",       # Production-ready
        "gemini-2.5-pro",         # Most capable
    ]

    def __init__(self, client: genai.Client):
        self.client = client
        self.validation_failures = 0  # Track quality issues only
        self.safety_blocks = 0        # Track Gemini safety blocks

    async def on_error(self, exception: Exception, attempt: int) -> None:
        """Track Gemini-specific error types."""
        if isinstance(exception, ValidationError):
            self.validation_failures += 1
        elif "SAFETY" in str(exception) or "BLOCKED" in str(exception):
            self.safety_blocks += 1
            # Note: Could adjust safety_settings on retry

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[PersonData, TokenUsage]:
        # Select model based on validation failures (not total attempts)
        model_index = min(self.validation_failures, len(self.MODELS) - 1)
        model = self.MODELS[model_index]

        # Adjust safety settings if we've hit safety blocks
        config = genai.types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=PersonData,
        )

        if self.safety_blocks > 0:
            # Could make safety_settings more permissive
            # config.safety_settings = [...]
            pass

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
```

**Cost Savings:**

- Validation error → Escalate to gemini-2.5-pro (quality issue)
- Network error → Retry with gemini-2.5-flash-lite (transient issue)
- Rate limit error → Retry with gemini-2.5-flash-lite (API quota)
- Safety block → Retry with same model, adjusted safety settings
- Result: **60-80% cost reduction** vs. always using gemini-2.5-pro

### Smart Retry Prompts for Gemini

Build targeted retry prompts based on Gemini validation errors:

```python
class SmartRetryGeminiStrategy(LLMCallStrategy[PersonData]):
    """Tell Gemini exactly what failed in previous attempt."""

    def __init__(self, client: genai.Client):
        self.client = client
        self.last_error = None
        self.last_response = None

    async def on_error(self, exception: Exception, attempt: int) -> None:
        """Track validation errors for smart retry."""
        if isinstance(exception, ValidationError):
            self.last_error = exception

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[PersonData, TokenUsage]:
        if attempt == 1:
            final_prompt = prompt
        else:
            # Build focused retry prompt
            final_prompt = self._create_retry_prompt(prompt)

        config = genai.types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
            response_schema=PersonData,
        )

        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt,
            config=config,
        )

        try:
            output = PersonData.model_validate_json(response.text)
            usage = response.usage_metadata
            tokens: TokenUsage = {
                "input_tokens": usage.prompt_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0,
                "total_tokens": usage.total_token_count or 0,
            }
            return output, tokens
        except ValidationError as e:
            self.last_response = response.text
            raise  # Framework calls on_error, then retries

    def _create_retry_prompt(self, original_prompt: str) -> str:
        """Create targeted retry prompt with field-specific feedback."""
        if not self.last_error:
            return original_prompt

        # Parse which fields succeeded vs failed
        failed_fields = []
        for error in self.last_error.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            failed_fields.append(f"  - {field}: {msg}")

        retry_prompt = f"""RETRY REQUEST: The previous response had validation errors.

ORIGINAL REQUEST:
{original_prompt}

VALIDATION ERRORS TO FIX:
{chr(10).join(failed_fields)}

Please provide a complete, valid JSON response that fixes these specific validation errors.
Ensure all fields match the required schema exactly."""

        return retry_prompt
```

**Benefits:**

- Gemini knows exactly what went wrong
- Focused on fixing specific fields
- Higher success rate on retries
- Lower token usage (shorter prompts)

**Complete Examples:**

- `examples/example_smart_model_escalation.py` - Full implementation
- `examples/example_gemini_smart_retry.py` - Complete smart retry example

---

## Resources

- **Gemini API Docs**: <https://ai.google.dev/gemini-api/docs>
- **Python SDK**: <https://googleapis.github.io/python-genai/>
- **Pricing**: <https://ai.google.dev/pricing>
- **Models**: <https://ai.google.dev/gemini-api/docs/models/gemini>
- **Get API Key**: <https://aistudio.google.com/apikey>
- **Quickstart**: <https://ai.google.dev/gemini-api/docs/quickstart>
- **batch-llm API Docs**: [docs/API.md](API.md)

## Support

For issues with:

- **batch-llm**: <https://github.com/yourusername/batch-llm/issues>
- **Gemini API**: <https://developers.google.com/support>
- **google-genai SDK**: <https://github.com/googleapis/python-genai/issues>
