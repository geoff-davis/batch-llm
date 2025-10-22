# batch-llm

**Provider-agnostic parallel LLM processing with automatic retries, rate limiting, and flexible strategies.**

Process thousands of LLM requests efficiently across any provider (OpenAI, Anthropic, Google, LangChain, or custom)
with built-in error handling, retry logic, and observability.

[![PyPI version](https://badge.fury.io/py/batch-llm.svg)](https://badge.fury.io/py/batch-llm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why batch-llm?

- ✅ **Universal**: Works with any LLM provider through a simple strategy interface
- ✅ **Reliable**: Built-in retry logic, timeout handling, and rate limiting
- ✅ **Fast**: Parallel async processing with configurable concurrency
- ✅ **Observable**: Token tracking, metrics, and middleware hooks
- ✅ **Clean**: Strategy pattern separates business logic from API integration
- ✅ **Type-safe**: Full generic type support with Pydantic validation

---

## Table of Contents

- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Basic Example (PydanticAI)](#basic-example-pydanticai)
- [Features](#features)
  - [Strategy Pattern for Any LLM Provider](#-strategy-pattern-for-any-llm-provider)
  - [Automatic Retry Logic](#-automatic-retry-logic)
  - [Rate Limiting](#-rate-limiting)
  - [Middleware & Observers](#-middleware--observers)
  - [Token Tracking](#-token-tracking)
- [Automatic Retry on Validation Errors](#automatic-retry-on-validation-errors)
- [Provider Examples](#provider-examples)
  - [OpenAI](#openai)
  - [Anthropic Claude](#anthropic-claude)
  - [Google Gemini with Caching (RAG)](#google-gemini-with-caching-rag)
  - [LangChain Integration](#langchain-integration)
- [Core Concepts](#core-concepts)
  - [Strategy Pattern](#strategy-pattern)
  - [Work Items](#work-items)
  - [Results](#results)
- [Advanced Usage](#advanced-usage)
  - [Progressive Temperature on Retries](#progressive-temperature-on-retries)
  - [Smart Retry with Validation Feedback](#smart-retry-with-validation-feedback)
  - [Model Escalation for Cost Optimization](#model-escalation-for-cost-optimization)
  - [Context and Post-Processing](#context-and-post-processing)
  - [Error Classification](#error-classification)
- [Configuration Reference](#configuration-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Performance](#performance)
- [FAQ](#faq)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

### Installation

```bash
# Basic installation
pip install batch-llm

# With PydanticAI support (recommended for structured output)
pip install 'batch-llm[pydantic-ai]'

# With Google Gemini support
pip install 'batch-llm[gemini]'

# With everything
pip install 'batch-llm[all]'
```

### Basic Example (PydanticAI)

```python
import asyncio
from batch_llm import (
    ParallelBatchProcessor,
    LLMWorkItem,
    ProcessorConfig,
    PydanticAIStrategy,
)
from pydantic_ai import Agent
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]

async def main():
    # Create agent and wrap in strategy
    agent = Agent("gemini-2.5-flash", result_type=Summary)
    strategy = PydanticAIStrategy(agent=agent)

    # Configure processor
    config = ProcessorConfig(max_workers=5, timeout_per_item=30.0)

    # Process items
    async with ParallelBatchProcessor[str, Summary, None](config=config) as processor:
        # Add work items
        for doc in ["Document 1 text...", "Document 2 text..."]:
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"doc_{hash(doc)}",
                    strategy=strategy,
                    prompt=f"Summarize: {doc}",
                )
            )

        # Process all in parallel
        result = await processor.process_all()

    print(f"Succeeded: {result.succeeded}/{result.total_items}")
    print(f"Tokens used: {result.total_input_tokens + result.total_output_tokens}")

asyncio.run(main())
```

---

## Features

### 🎯 Strategy Pattern for Any LLM Provider

Built-in strategies:

- **`PydanticAIStrategy`** - PydanticAI agents with structured output
- **`GeminiStrategy`** - Direct Google Gemini API calls
- **`GeminiCachedStrategy`** - Gemini with context caching (great for RAG)

Create custom strategies for any provider:

- OpenAI (see `examples/example_openai.py`)
- Anthropic Claude (see `examples/example_anthropic.py`)
- LangChain (see `examples/example_langchain.py`)
- Your own custom API

### 🔄 Automatic Retry Logic

```python
from batch_llm.core import RetryConfig

config = ProcessorConfig(
    max_workers=5,
    timeout_per_item=30.0,
    retry=RetryConfig(
        max_attempts=3,
        initial_wait=1.0,
        exponential_base=2.0,
        jitter=True,
    ),
)
```

### 🚦 Rate Limiting

Automatic rate limit detection with coordinated cooldown across all workers:

```python
from batch_llm.core import RateLimitConfig

config = ProcessorConfig(
    rate_limit=RateLimitConfig(
        cooldown_seconds=60.0,        # Initial cooldown after 429 error
        backoff_multiplier=2.0,       # Double cooldown on repeated limits
        slow_start_items=50,          # Gradual ramp-up over 50 items
        slow_start_initial_delay=2.0, # Start with 2s between items
        slow_start_final_delay=0.1,   # End with 0.1s between items
    ),
)
```

When any worker hits a rate limit (429 error), **all workers pause** during cooldown, then gradually resume with slow start to prevent immediate re-limiting.

### 🔌 Middleware & Observers

Extend functionality with middleware and observers:

**Middleware** - Transform work items before/after processing:

```python
from batch_llm import BaseMiddleware

class LoggingMiddleware(BaseMiddleware):
    async def before_process(self, work_item):
        """Called before processing each item."""
        print(f"Starting {work_item.item_id}")
        return work_item  # Return modified or original item

    async def after_process(self, work_item, result):
        """Called after processing succeeds."""
        print(f"Completed {work_item.item_id}: {result.success}")
        return result

    async def on_error(self, work_item, error):
        """Called when processing fails permanently."""
        print(f"Failed {work_item.item_id}: {error}")
```

**Observers** - Monitor events without modifying data:

```python
from batch_llm import BaseObserver, ProcessingEvent, MetricsObserver

class CustomObserver(BaseObserver):
    async def on_event(self, event: ProcessingEvent, data: dict):
        """Receive all processing events."""
        if event == ProcessingEvent.ITEM_COMPLETED:
            print(f"Item {data['item_id']} completed")
        elif event == ProcessingEvent.RATE_LIMIT_HIT:
            print(f"Rate limit hit at {data['timestamp']}")

# Built-in MetricsObserver collects statistics
metrics = MetricsObserver()

processor = ParallelBatchProcessor(
    config=config,
    middlewares=[LoggingMiddleware()],
    observers=[metrics, CustomObserver()],
)

# After processing, retrieve collected metrics
result = await processor.process_all()
collected_metrics = await metrics.get_metrics()
# Returns: {
#   "total_items": 100,
#   "successful": 95,
#   "failed": 5,
#   "avg_duration": 1.2,
#   ...
# }
```

**Available Events**:
- `PROCESSING_STARTED` - Batch processing begins
- `ITEM_STARTED` - Item processing starts
- `ITEM_COMPLETED` - Item succeeds
- `ITEM_FAILED` - Item fails
- `RATE_LIMIT_HIT` - Rate limit encountered
- `PROCESSING_COMPLETED` - Batch processing ends

**Progress Callbacks** - Monitor progress in real-time:

```python
def progress_callback(completed: int, total: int, current_item: str):
    """Called after each item completes."""
    print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%) - Current: {current_item}")

config = ProcessorConfig(
    progress_callback=progress_callback,
    progress_callback_timeout=5.0,  # Timeout per callback
)
```

### 📊 Token Tracking

```python
result = await processor.process_all()
print(f"Input tokens: {result.total_input_tokens}")
print(f"Output tokens: {result.total_output_tokens}")
print(f"Total cost: ${estimate_cost(result)}")
```

---

## Automatic Retry on Validation Errors

One of the most powerful features is automatic retry when Pydantic validation fails:

```python
from pydantic import BaseModel, Field, ValidationError
from batch_llm import PydanticAIStrategy, LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.core import RetryConfig
from pydantic_ai import Agent

class StructuredData(BaseModel):
    """Strict schema - LLM must follow this exactly."""
    name: str = Field(min_length=1)
    age: int = Field(gt=0, lt=150)
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')

# Agent with structured output
agent = Agent("gemini-2.5-flash", result_type=StructuredData)
strategy = PydanticAIStrategy(agent=agent)

# Configure retries with progressive temperature
config = ProcessorConfig(
    max_workers=5,
    timeout_per_item=30.0,
    retry=RetryConfig(
        max_attempts=3,
        initial_wait=1.0,
        exponential_base=2.0,
    ),
)

async with ParallelBatchProcessor[str, StructuredData, None](config=config) as processor:
    await processor.add_work(
        LLMWorkItem(
            item_id="item_1",
            strategy=strategy,
            prompt="Extract: John Smith, 32 years old, john.smith@example.com",
        )
    )

    result = await processor.process_all()

# If parsing fails:
# - Attempt 1: Fails validation → automatic retry
# - Attempt 2: Fails validation → automatic retry
# - Attempt 3: Success! (or final failure if still invalid)
```

**How it works:**

1. LLM returns malformed JSON → `ValidationError` raised
2. Framework catches error and retries automatically
3. Each retry can use different temperature (via custom strategy)
4. Success or final failure after max attempts

---

## Provider Examples

### OpenAI

```python
from batch_llm.llm_strategies import LLMCallStrategy
from batch_llm import TokenUsage
from openai import AsyncOpenAI

class OpenAIStrategy(LLMCallStrategy[str]):
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, TokenUsage]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        output = response.choices[0].message.content or ""
        tokens: TokenUsage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return output, tokens

# Use it
client = AsyncOpenAI(api_key=API_KEY)
strategy = OpenAIStrategy(client=client)
```

See [`examples/example_openai.py`](examples/example_openai.py) for complete examples including structured output.

### Anthropic Claude

```python
from batch_llm.llm_strategies import LLMCallStrategy
from batch_llm import TokenUsage
from anthropic import AsyncAnthropic

class AnthropicStrategy(LLMCallStrategy[str]):
    def __init__(self, client: AsyncAnthropic, model: str = "claude-3-5-sonnet-20241022"):
        self.client = client
        self.model = model

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, TokenUsage]:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        output = response.content[0].text
        tokens: TokenUsage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return output, tokens
```

See [`examples/example_anthropic.py`](examples/example_anthropic.py) for system prompts and mixed model examples.

### Google Gemini with Caching (RAG)

Perfect for RAG applications where you have large context that stays constant:

```python
from batch_llm import GeminiCachedStrategy
from google import genai

client = genai.Client(api_key="your-api-key")

# Define large context to cache (e.g., retrieved documents)
cached_content = [
    genai.types.Content(
        role="user",
        parts=[genai.types.Part(text="Large document context...")]
    )
]

strategy = GeminiCachedStrategy(
    model="gemini-2.0-flash",
    client=client,
    response_parser=lambda r: r.text,
    cached_content=cached_content,
    cache_ttl_seconds=3600,  # Cache for 1 hour
)

# Strategy automatically:
# - Creates cache on first use
# - Refreshes TTL when needed
# - Deletes cache on cleanup
```

### LangChain Integration

```python
from batch_llm.llm_strategies import LLMCallStrategy
from batch_llm import TokenUsage
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

class LangChainStrategy(LLMCallStrategy[str]):
    def __init__(self, chain: LLMChain):
        self.chain = chain

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, TokenUsage]:
        result = await self.chain.arun(input=prompt)
        # Return result and token usage (if LangChain provides it)
        tokens: TokenUsage = {
            "input_tokens": 0,  # Extract from LangChain if available
            "output_tokens": 0,
            "total_tokens": 0,
        }
        return result, tokens

# Create LangChain components
llm = ChatOpenAI(model="gpt-4o-mini")
chain = LLMChain(llm=llm, prompt=your_prompt_template)

# Use with batch-llm
strategy = LangChainStrategy(chain=chain)
```

See [`examples/example_langchain.py`](examples/example_langchain.py) for RAG pipeline examples.

---

## Core Concepts

### Strategy Pattern

All LLM calls go through a `LLMCallStrategy`:

```python
from batch_llm.llm_strategies import LLMCallStrategy
from batch_llm import TokenUsage

class LLMCallStrategy(ABC):
    async def prepare(self) -> None:
        """Called once before processing starts. Initialize resources here."""
        pass

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, TokenUsage]:
        """Execute the LLM call. Called for each retry attempt.

        Returns:
            Tuple of (output, token_usage) where token_usage is a TokenUsage dict
            with optional keys: input_tokens, output_tokens, total_tokens, cached_input_tokens
        """
        pass

    async def on_error(self, exception: Exception, attempt: int) -> None:
        """Called when execute() raises an exception. Use to track errors and adjust retry behavior."""
        pass

    async def cleanup(self) -> None:
        """Called once after processing. Clean up resources here."""
        pass
```

This design:

- ✅ Decouples framework from LLM providers
- ✅ Enables resource lifecycle management (caches, connections)
- ✅ Supports progressive temperature strategies on retries
- ✅ Enables smart retry logic based on error type (via `on_error`)
- ✅ Makes testing easy with mock strategies

### Work Items

```python
work_item = LLMWorkItem(
    item_id="unique-id",           # Required: unique identifier
    strategy=your_strategy,         # Required: how to call the LLM
    prompt="Your prompt here",      # Optional: prompt string
    context={"key": "value"},       # Optional: context data
)
```

### Results

```python
result = await processor.process_all()

# Aggregate results
print(f"Total: {result.total_items}")
print(f"Succeeded: {result.succeeded}")
print(f"Failed: {result.failed}")

# Individual results
for item_result in result.results:
    if item_result.success:
        print(f"{item_result.item_id}: {item_result.output}")
    else:
        print(f"{item_result.item_id} failed: {item_result.error}")
```

---

## Advanced Usage

### Progressive Temperature on Retries

```python
from batch_llm.llm_strategies import LLMCallStrategy
from batch_llm import TokenUsage

class ProgressiveTempStrategy(LLMCallStrategy[str]):
    """Increase temperature with each retry attempt."""

    def __init__(self, client, temps: list[float] | None = None):
        self.client = client
        self.temps = temps if temps is not None else [0.0, 0.5, 1.0]

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, TokenUsage]:
        # Use higher temperature for retries
        temp = self.temps[min(attempt - 1, len(self.temps) - 1)]

        # Make call with progressive temperature
        response = await self.client.generate(prompt, temperature=temp)

        # Extract token usage from response
        tokens: TokenUsage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return response.text, tokens
```

### Smart Retry with Validation Feedback

When validation fails, use the `on_error` callback to create smarter retry prompts that tell the LLM exactly which fields succeeded and which failed:

```python
from pydantic import ValidationError

class SmartRetryStrategy(LLMCallStrategy[PersonData]):
    """On validation failure, tell LLM which fields succeeded vs failed."""

    def __init__(self, client):
        self.client = client
        self.last_error = None
        self.last_response = None

    async def on_error(self, exception: Exception, attempt: int) -> None:
        """Track validation errors for smart retry prompt generation."""
        if isinstance(exception, ValidationError):
            self.last_error = exception
            # last_response saved in execute() before raising

    async def execute(self, prompt: str, attempt: int, timeout: float):
        if attempt == 1:
            final_prompt = prompt
        else:
            # Use error information to create smart retry prompt
            final_prompt = self._create_retry_prompt(prompt)

        try:
            response = await self.client.generate(final_prompt)
            # Parse and validate
            output = PersonData.model_validate_json(response.text)
            return output, tokens
        except ValidationError as e:
            # Save response for retry prompt generation
            self.last_response = response.text
            raise  # Framework calls on_error, then retries

    def _create_retry_prompt(self, original_prompt: str) -> str:
        """Create targeted retry prompt using self.last_error and self.last_response."""
        # Parse which fields succeeded vs failed from self.last_error
        # Build focused prompt telling LLM what to fix
        return retry_prompt
```

This approach is more efficient than blind retries because:

- ✅ **`on_error` callback captures failure context** before retry
- ✅ LLM knows exactly what went wrong
- ✅ LLM can focus on fixing specific fields
- ✅ Reduces token usage with shorter, focused prompts
- ✅ Higher success rate on retries

See [`examples/example_gemini_smart_retry.py`](examples/example_gemini_smart_retry.py) for complete
implementation with automatic error parsing.

### Model Escalation for Cost Optimization

Start with cheap models and escalate to expensive ones **only on validation errors** (not network/rate limit errors). Use the `on_error` callback to distinguish error types:

```python
from pydantic import ValidationError

class SmartModelEscalationStrategy(LLMCallStrategy[Analysis]):
    """Escalate to smarter models ONLY on validation errors."""

    MODELS = [
        "gemini-2.5-flash-lite",  # Attempt 1: Cheapest/fastest
        "gemini-2.5-flash",       # Attempt 2: Production fast
        "gemini-2.5-pro",         # Attempt 3: Most capable
    ]

    def __init__(self, client):
        self.client = client
        self.validation_failures = 0  # Track validation errors only

    async def on_error(self, exception: Exception, attempt: int) -> None:
        """Only escalate model on validation errors, not network/rate limit errors."""
        if isinstance(exception, ValidationError):
            self.validation_failures += 1
        # Network/rate limit errors don't increment counter

    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Select model based on VALIDATION failures, not total attempts
        model_index = min(self.validation_failures, len(self.MODELS) - 1)
        model = self.MODELS[model_index]

        # Network error on attempt 2? Retry with same (cheap) model
        # Validation error on attempt 2? Escalate to better model
```

**Cost savings example:**

- Validation error → Escalate to smarter model ✅
- Network error → Retry same cheap model ✅
- Rate limit error → Retry same cheap model ✅
- Most tasks succeed on attempt 1 (cheap model)
- Only quality issues trigger expensive models
- Result: **~60-80% cost reduction** vs. always using best model

You can also combine model escalation with temperature escalation for even better results.

See [`examples/example_smart_model_escalation.py`](examples/example_smart_model_escalation.py) for complete implementation with cost comparisons.

### Context and Post-Processing

Pass context through and run custom logic after each success:

```python
from dataclasses import dataclass

@dataclass
class WorkContext:
    user_id: str
    document_id: str

async def save_result(result: WorkItemResult):
    """Save successful results to database."""
    if result.success:
        await db.save(
            user_id=result.context.user_id,
            document_id=result.context.document_id,
            summary=result.output,
        )

config = ProcessorConfig(max_workers=5)
processor = ParallelBatchProcessor(
    config=config,
    post_processor=save_result,
)

# Add work with context
await processor.add_work(
    LLMWorkItem(
        item_id="doc_123",
        strategy=strategy,
        prompt="Summarize...",
        context=WorkContext(user_id="user_1", document_id="doc_123"),
    )
)
```

### Error Classification

Custom error handling per provider:

```python
from batch_llm.classifiers import GeminiErrorClassifier

processor = ParallelBatchProcessor(
    config=config,
    error_classifier=GeminiErrorClassifier(),  # Provider-specific errors
)
```

---

## Configuration Reference

### ProcessorConfig

Complete configuration options for `ProcessorConfig`:

```python
from batch_llm import ProcessorConfig
from batch_llm.core import RetryConfig, RateLimitConfig

config = ProcessorConfig(
    # === Core Settings ===
    max_workers=5,              # Number of parallel workers (default: 5)
    timeout_per_item=120.0,     # Max seconds per item including retries (default: 120.0)

    # === Retry Configuration ===
    retry=RetryConfig(
        max_attempts=3,          # Maximum retry attempts (default: 3)
        initial_wait=1.0,        # Initial retry delay in seconds (default: 1.0)
        max_wait=60.0,           # Maximum retry delay (default: 60.0)
        exponential_base=2.0,    # Backoff multiplier (default: 2.0)
        jitter=True,             # Add random jitter to prevent thundering herd (default: True)
    ),

    # === Rate Limit Configuration ===
    rate_limit=RateLimitConfig(
        cooldown_seconds=300.0,        # Cooldown after rate limit (default: 300.0 / 5 min)
        backoff_multiplier=1.5,        # Increase cooldown on repeated limits (default: 1.5)
        slow_start_items=50,           # Items to ramp up after cooldown (default: 50)
        slow_start_initial_delay=2.0,  # Initial delay during slow start (default: 2.0)
        slow_start_final_delay=0.1,    # Final delay after ramp up (default: 0.1)
    ),

    # === Progress Reporting ===
    progress_interval=10,              # Log progress every N items (default: 10)
    progress_callback_timeout=5.0,     # Timeout for progress callbacks (default: 5.0)

    # === Queue Management ===
    max_queue_size=0,                  # Max items in queue (0 = unlimited, default: 0)

    # === Testing ===
    dry_run=False,                     # Skip API calls, use mock data (default: False)

    # === Logging ===
    enable_detailed_logging=False,     # Enable verbose debug logs (default: False)
)
```

### Key Configuration Tips

**Worker Count:**
- **CPU-bound tasks**: `max_workers = cpu_count()`
- **I/O-bound (LLM calls)**: `max_workers = 5-10`
- **Rate-limited APIs**: Start with `max_workers = 3-5`, increase gradually

**Timeout Settings:**
- Set `timeout_per_item` > sum of all retry delays
- Formula: `timeout_per_item ≥ max_attempts × max_wait × 2`
- Framework will warn you if timeout is too short

**Retry Strategy:**
- `jitter=True` prevents all workers retrying simultaneously
- Exponential backoff: delay = `initial_wait × (exponential_base ^ attempt)`
- Example: attempt 1 = 1s, attempt 2 = 2s, attempt 3 = 4s

**Rate Limiting:**
- When any worker hits 429 error, all workers pause
- `slow_start` gradually resumes processing to avoid immediate re-limiting
- `backoff_multiplier` increases cooldown if limits persist

**Dry Run Mode:**
- Set `dry_run=True` to test configuration without API calls
- Strategies return mock data via `dry_run()` method
- Useful for validating workflow before spending API credits

---

## Best Practices

### Choosing Worker Count

```python
import multiprocessing

# For I/O-bound LLM calls (recommended starting point)
config = ProcessorConfig(max_workers=5)

# For CPU-bound pre/post-processing
config = ProcessorConfig(max_workers=multiprocessing.cpu_count())

# For rate-limited APIs (start conservatively)
config = ProcessorConfig(max_workers=3)
```

### Resource Management

**Always use context managers** for automatic cleanup:

```python
# ✅ GOOD - Automatic cleanup
async with ParallelBatchProcessor(config=config) as processor:
    await processor.add_work(work_item)
    result = await processor.process_all()
# Cleanup happens automatically here

# ❌ BAD - Manual cleanup required
processor = ParallelBatchProcessor(config=config)
result = await processor.process_all()
await processor.cleanup()  # Easy to forget!
```

### Error Handling Strategy

```python
# Use provider-specific error classifiers when available
from batch_llm.classifiers import GeminiErrorClassifier

processor = ParallelBatchProcessor(
    config=config,
    error_classifier=GeminiErrorClassifier(),  # Better error detection
)

# Or implement custom classifier for your provider
from batch_llm import ErrorClassifier, ErrorInfo

class CustomErrorClassifier(ErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        # Classify provider-specific errors
        if "RateLimitError" in str(type(exception)):
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
                suggested_wait=60.0,
            )
        # ... more classifications
```

### Memory Considerations

Each worker holds one work item in memory at a time. For large batches:

```python
# For 10,000 items with 10 workers:
# - In-flight: ~10 items in memory
# - Results: All 10,000 accumulated in results list
# - Peak memory: ~10-50MB depending on output size

# If memory is a concern, use post-processor to save results immediately
async def save_and_clear(result):
    await db.save(result)
    # Result will be GC'd after this function returns

processor = ParallelBatchProcessor(
    config=config,
    post_processor=save_and_clear,
)
```

### Progressive Strategies

Combine multiple strategies for optimal cost/quality:

```python
class SmartEscalationStrategy(LLMCallStrategy[Output]):
    """Escalate both model and temperature on retries."""

    CONFIGS = [
        ("gpt-4o-mini", 0.0),      # Attempt 1: Cheap + deterministic
        ("gpt-4o-mini", 0.7),      # Attempt 2: Cheap + creative
        ("gpt-4o", 0.0),           # Attempt 3: Expensive + deterministic
    ]

    async def execute(self, prompt: str, attempt: int, timeout: float):
        model, temp = self.CONFIGS[min(attempt - 1, len(self.CONFIGS) - 1)]
        # Make call with escalated config...
```

### Testing Before Production

```python
# 1. Test with dry_run mode
config = ProcessorConfig(dry_run=True)
result = await processor.process_all()
# No API calls made, validates workflow

# 2. Test with MockAgent
from batch_llm.testing import MockAgent

mock = MockAgent(
    response_factory=lambda p: YourOutput(...),
    latency=0.01,
)
strategy = PydanticAIStrategy(agent=mock)
# Fast testing without API

# 3. Test with small batch first
test_items = your_items[:10]  # Try 10 items first
# Verify success before processing full batch
```

---

## Troubleshooting

### Common Errors

#### `FrameworkTimeoutError: Framework timeout after X.Xs`

**Cause**: Item exceeded `timeout_per_item` seconds.

**Solutions**:
```python
# 1. Increase timeout
config = ProcessorConfig(timeout_per_item=300.0)  # 5 minutes

# 2. Check if retry delays fit within timeout
# timeout_per_item should be > sum of all retry delays

# 3. Review retry configuration
config = ProcessorConfig(
    timeout_per_item=120.0,
    retry=RetryConfig(
        max_attempts=2,      # Reduce attempts
        max_wait=20.0,       # Reduce max wait
    ),
)
```

#### `Rate Limit Errors (429)`

**Symptoms**: Logs show "Rate limit detected", all workers pause.

**Solutions**:
```python
# 1. Increase cooldown period
config = ProcessorConfig(
    rate_limit=RateLimitConfig(
        cooldown_seconds=600.0,  # 10 minutes
    ),
)

# 2. Reduce worker count
config = ProcessorConfig(max_workers=2)

# 3. Add delays between requests (slow start)
config = ProcessorConfig(
    rate_limit=RateLimitConfig(
        slow_start_items=100,          # Longer ramp up
        slow_start_initial_delay=5.0,  # Slower start
    ),
)
```

#### `Validation Errors Keep Failing`

**Cause**: LLM consistently returns invalid structured output.

**Solutions**:
```python
# 1. Use progressive temperature
class ProgressiveTempStrategy(LLMCallStrategy[Output]):
    def __init__(self, agent):
        self.agent = agent
        self.temps = [0.0, 0.5, 1.0]

    async def execute(self, prompt: str, attempt: int, timeout: float):
        temp = self.temps[min(attempt - 1, len(self.temps) - 1)]
        # Use higher temperature on retries
        ...

# 2. Improve prompt clarity
prompt = f"""Extract data in this EXACT JSON format:
{{
    "name": "string",
    "age": integer between 0-150,
    "email": "valid email address"
}}

Input: {input_text}
"""

# 3. Use smart retry with validation feedback
# See examples/example_gemini_smart_retry.py
```

#### `Workers Hanging / Not Finishing`

**Symptoms**: `process_all()` never completes, workers timeout warning.

**Debug steps**:
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check for:
# - Deadlocks in post-processor
# - Infinite loops in middleware
# - Blocking I/O in callbacks
```

**Solutions**:
```python
# 1. Add timeouts to post-processors
async def safe_post_processor(result):
    try:
        async with asyncio.timeout(10.0):  # 10 second timeout
            await your_post_processor(result)
    except asyncio.TimeoutError:
        logger.warning(f"Post-processor timeout for {result.item_id}")

# 2. Use progress callbacks to monitor
def progress(completed, total, current):
    print(f"Progress: {completed}/{total} - Current: {current}")

processor = ParallelBatchProcessor(
    config=config,
    progress_callback=progress,
)
```

### Performance Issues

#### **Slow Throughput**

```python
# 1. Check worker count
config = ProcessorConfig(max_workers=10)  # Increase if CPU allows

# 2. Profile LLM latency
import time

class TimingStrategy(LLMCallStrategy[Output]):
    async def execute(self, prompt: str, attempt: int, timeout: float):
        start = time.time()
        result = await self.base_strategy.execute(prompt, attempt, timeout)
        logger.info(f"LLM call took {time.time() - start:.2f}s")
        return result

# 3. Check for rate limiting (look for cooldown logs)
```

#### **High Memory Usage**

```python
# Use post-processor to stream results instead of accumulating
async def stream_to_disk(result):
    with open(f"results/{result.item_id}.json", "w") as f:
        json.dump(result.output, f)
    # Don't keep results in memory

processor = ParallelBatchProcessor(
    config=config,
    post_processor=stream_to_disk,
)
```

### Debug Logging

```python
import logging

# Enable detailed framework logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or enable only for batch_llm
logging.getLogger('batch_llm').setLevel(logging.DEBUG)

# Enable detailed config validation warnings
config = ProcessorConfig(
    enable_detailed_logging=True,
    # ... other config
)
```

---

## FAQ

### General Questions

**Q: How is this different from using threading or multiprocessing?**

A: `batch-llm` uses asyncio for concurrency, which is ideal for I/O-bound LLM API calls:
- **Threading**: Limited by GIL, complex synchronization
- **Multiprocessing**: High overhead, can't share data easily
- **Asyncio (batch-llm)**: Lightweight, efficient for I/O, built-in coordination

**Q: Can I use this with synchronous LLM clients?**

A: Yes, but you need to wrap sync calls in `asyncio.to_thread()`:
```python
class SyncClientStrategy(LLMCallStrategy[str]):
    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Run sync call in thread pool
        output = await asyncio.to_thread(
            self.sync_client.generate, prompt
        )
        return output, tokens
```

**Q: What happens if my process crashes mid-batch?**

A: In-flight items are lost. For critical workloads:
```python
# Use post-processor to mark items as completed
async def mark_complete(result):
    if result.success:
        await db.mark_processed(result.item_id)

# On restart, skip already processed items
unprocessed_items = await db.get_unprocessed()
```

**Q: How do I handle API key rotation?**

A: Strategies are instantiated once, so update the client:
```python
class RotatingKeyStrategy(LLMCallStrategy[Output]):
    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Refresh API key if needed
        if self.should_rotate():
            self.client.api_key = self.get_next_key()
        # Make call...
```

### Configuration Questions

**Q: What's the optimal worker count?**

A: Start with 5, adjust based on:
- **Too low**: Underutilized resources, slow throughput
- **Too high**: Rate limits, high memory, diminishing returns
- **Monitor**: If hitting rate limits, reduce workers; if CPU idle, increase

**Q: Why do I see "timeout may be too short" warnings?**

A: Framework validates that `timeout_per_item` can accommodate retry delays:
```python
# Warning triggered if:
# timeout_per_item < sum_of_retry_delays

# Fix by increasing timeout or reducing retry config
config = ProcessorConfig(
    timeout_per_item=180.0,  # Increase
    retry=RetryConfig(max_attempts=2),  # Or reduce
)
```

**Q: What does `dry_run` mode do?**

A: Skips actual API calls, uses mock data:
```python
config = ProcessorConfig(dry_run=True)
# - No API calls made
# - Strategies return mock via dry_run() method
# - Useful for testing workflow logic
```

### Advanced Questions

**Q: Can I mix different strategies in one batch?**

A: Yes! Each work item can have its own strategy:
```python
gpt4_strategy = OpenAIStrategy(client, "gpt-4o")
gpt4_mini_strategy = OpenAIStrategy(client, "gpt-4o-mini")

# Use expensive model for complex items
await processor.add_work(LLMWorkItem(
    item_id="complex_1",
    strategy=gpt4_strategy,
    prompt="Complex analysis...",
))

# Use cheap model for simple items
await processor.add_work(LLMWorkItem(
    item_id="simple_1",
    strategy=gpt4_mini_strategy,
    prompt="Simple summary...",
))
```

**Q: How do I implement caching for identical prompts?**

A: Use middleware or custom strategy:
```python
class CachingStrategy(LLMCallStrategy[Output]):
    def __init__(self, base_strategy, cache):
        self.base_strategy = base_strategy
        self.cache = cache  # e.g., Redis

    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Check cache
        cached = await self.cache.get(prompt)
        if cached:
            return cached, {"cached_input_tokens": 100, ...}

        # Call LLM
        result, tokens = await self.base_strategy.execute(prompt, attempt, timeout)

        # Cache result
        await self.cache.set(prompt, result)
        return result, tokens
```

**Q: Can I pause/resume processing?**

A: Not directly, but you can implement checkpointing:
```python
processed_ids = set()

async def checkpoint(result):
    if result.success:
        processed_ids.add(result.item_id)
        await save_checkpoint(processed_ids)

# On resume
checkpoint_data = await load_checkpoint()
unprocessed = [item for item in all_items if item.id not in checkpoint_data]
```

**Q: How do I implement retries with different prompts?**

A: Use the `on_error` callback to track errors and modify prompts on retries:
```python
class AdaptivePromptStrategy(LLMCallStrategy[Output]):
    def __init__(self):
        self.last_error = None

    async def on_error(self, exception: Exception, attempt: int) -> None:
        """Track error for next retry attempt."""
        self.last_error = str(exception)

    async def execute(self, prompt: str, attempt: int, timeout: float):
        if attempt > 1 and self.last_error:
            # Modify prompt based on previous error
            prompt = f"{prompt}\n\nNote: Previous attempt failed with: {self.last_error}"

        # Make LLM call with modified prompt
        return await self.call_llm(prompt)
```

The `on_error` callback is called automatically by the framework when `execute()` raises an exception, making it cleaner than try/except in `execute()`.

For validation errors specifically, see [`examples/example_gemini_smart_retry.py`](examples/example_gemini_smart_retry.py) for a complete example that parses which fields succeeded vs. failed and creates targeted retry prompts.

---

## Documentation

- **[API Reference](docs/API.md)** - Complete API documentation
- **[Migration Guide](docs/MIGRATION_V3.md)** - Upgrading from v0.0.x
- **[Examples](examples/)** - Working examples for all providers

---

## Testing

### Three Testing Approaches

**1. Dry-Run Mode (Recommended for Quick Tests)**

Test your workflow without making any API calls:

```python
from batch_llm import ParallelBatchProcessor, ProcessorConfig

# Enable dry-run mode
config = ProcessorConfig(
    max_workers=5,
    dry_run=True,  # ← No API calls will be made
)

async with ParallelBatchProcessor(config=config) as processor:
    # Add work items as normal
    await processor.add_work(work_item)

    # Process returns mock data
    result = await processor.process_all()

# Validates workflow logic without spending API credits
# Useful for testing configuration, error handling, post-processors, etc.
```

**2. Mock Strategies (For Unit Tests)**

Use `MockAgent` for fast, controlled testing:

```python
from batch_llm.testing import MockAgent
from batch_llm import PydanticAIStrategy, LLMWorkItem

# Create mock agent with custom responses
mock_agent = MockAgent(
    response_factory=lambda p: Summary(
        title="Test Summary",
        key_points=["Point A", "Point B"]
    ),
    latency=0.01,              # Simulate 10ms latency
    rate_limit_on_call=None,   # Optional: Simulate rate limit on Nth call
    timeout_on_call=None,      # Optional: Simulate timeout on Nth call
)

strategy = PydanticAIStrategy(agent=mock_agent)

# Use in tests - fast, predictable, no API calls
await processor.add_work(LLMWorkItem(
    item_id="test_1",
    strategy=strategy,
    prompt="Test input",
))
```

**3. Integration Tests with Small Batches**

Test with real API but limit scope:

```python
# Start with a tiny batch
test_items = your_full_dataset[:5]  # Just 5 items

config = ProcessorConfig(
    max_workers=2,            # Low concurrency for testing
    timeout_per_item=30.0,
    retry=RetryConfig(max_attempts=2),  # Fewer retries
)

# Verify everything works before processing thousands
result = await process_batch(test_items, config)

if result.succeeded == len(test_items):
    # Good! Now process full batch
    full_result = await process_batch(your_full_dataset, config)
```

### Run Tests

```bash
# Using uv (recommended)
uv run pytest

# Run with coverage
uv run pytest --cov=batch_llm

# Or using pytest directly
pytest
```

---

## Performance

### Parallel Processing

- **Throughput**: ~5-10 items/second per worker (depends on LLM latency)
- **Cost**: Same as sequential (no additional API costs)
- **Latency**: Real-time (seconds per item)
- **Concurrency**: Configurable workers (default: 5)

### Example: 1000 Items

- **Sequential**: ~16 minutes (1 req/sec)
- **5 workers**: ~3 minutes (5 req/sec)
- **10 workers**: ~1.5 minutes (10 req/sec)

---

## Potential Future Directions

These are ideas being considered, not committed features:

- **Cost tracking and budgeting** - Automatic cost calculation per provider/model, budget enforcement, cost estimation before processing
- **Prometheus/StatsD metrics** - Export metrics for monitoring dashboards (throughput, latency, success rates, token usage)
- **CLI tool** - Command-line interface for processing files without writing Python code
- **Streaming support** - Real-time token streaming for long-running tasks (would require significant architectural changes)
- **Batch API integration** - Support for provider batch APIs (50% cost savings, 24hr latency) - this might be better as a separate tool given the fundamentally different async model

Contributions and feedback welcome!

---

## Contributing

Contributions welcome! Areas of interest:

- Additional provider strategies (AWS Bedrock, Azure OpenAI, etc.)
- Improved error classification for specific providers
- Performance optimizations
- Documentation improvements

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Examples

Check out the [`examples/`](examples/) directory:

- [`example_llm_strategies.py`](examples/example_llm_strategies.py) - All built-in strategies
- [`example_openai.py`](examples/example_openai.py) - OpenAI integration
- [`example_anthropic.py`](examples/example_anthropic.py) - Anthropic Claude
- [`example_langchain.py`](examples/example_langchain.py) - LangChain & RAG
- [`example_gemini_direct.py`](examples/example_gemini_direct.py) - Direct Gemini API
- [`example_gemini_smart_retry.py`](examples/example_gemini_smart_retry.py) - Smart retry with validation feedback using `on_error`
- [`example_model_escalation.py`](examples/example_model_escalation.py) - Basic model escalation
- [`example_smart_model_escalation.py`](examples/example_smart_model_escalation.py) - Smart model escalation (validation errors only) using `on_error`
- [`example_context_manager.py`](examples/example_context_manager.py) - Resource management

---

**Questions?** Open an issue on GitHub or check the [API documentation](docs/API.md).
