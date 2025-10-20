# batch-llm

**Provider-agnostic parallel LLM processing with automatic retries, rate limiting, and flexible strategies.**

Process thousands of LLM requests efficiently across any provider (OpenAI, Anthropic, Google, LangChain, or custom) with built-in error handling, retry logic, and observability.

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
    agent = Agent("gemini-2.0-flash-exp", result_type=Summary)
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

```python
from batch_llm.core import RateLimitConfig

config = ProcessorConfig(
    rate_limit=RateLimitConfig(
        requests_per_minute=60,
        strategy="exponential_backoff",  # or "fixed_delay"
    ),
)
```

### 🔌 Middleware & Observers

Extend functionality with middleware and observers:

```python
from batch_llm.middleware import LoggingMiddleware
from batch_llm.observers import MetricsObserver

processor = ParallelBatchProcessor(
    config=config,
    middleware=[LoggingMiddleware()],
    observers=[MetricsObserver()],
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

## Provider Examples

### OpenAI

```python
from batch_llm.llm_strategies import LLMCallStrategy
from openai import AsyncOpenAI

class OpenAIStrategy(LLMCallStrategy[str]):
    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, dict[str, int]]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        output = response.choices[0].message.content or ""
        tokens = {
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
from anthropic import AsyncAnthropic

class AnthropicStrategy(LLMCallStrategy[str]):
    def __init__(self, client: AsyncAnthropic, model: str = "claude-3-5-sonnet-20241022"):
        self.client = client
        self.model = model

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, dict[str, int]]:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        output = response.content[0].text
        tokens = {
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
    model="gemini-2.0-flash-exp",
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
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

class LangChainStrategy(LLMCallStrategy[str]):
    def __init__(self, chain: LLMChain):
        self.chain = chain

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, dict[str, int]]:
        result = await self.chain.arun(input=prompt)
        # Return result and token usage
        return result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

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
class LLMCallStrategy(ABC):
    async def prepare(self) -> None:
        """Called once before processing starts. Initialize resources here."""
        pass

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, dict[str, int]]:
        """Execute the LLM call. Called for each retry attempt."""
        pass

    async def cleanup(self) -> None:
        """Called once after processing. Clean up resources here."""
        pass
```

This design:
- ✅ Decouples framework from LLM providers
- ✅ Enables resource lifecycle management (caches, connections)
- ✅ Supports progressive temperature strategies on retries
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
class ProgressiveTempStrategy(LLMCallStrategy[str]):
    """Increase temperature with each retry attempt."""

    def __init__(self, client, temps=[0.0, 0.5, 1.0]):
        self.client = client
        self.temps = temps

    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Use higher temperature for retries
        temp = self.temps[min(attempt - 1, len(self.temps) - 1)]

        # Make call with progressive temperature
        response = await self.client.generate(prompt, temperature=temp)
        return response.text, response.token_usage
```

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

## Documentation

- **[API Reference](docs/API.md)** - Complete API documentation
- **[Migration Guide](docs/MIGRATION_V3.md)** - Upgrading from v2.x
- **[Examples](examples/)** - Working examples for all providers

---

## Testing

### Mock Strategies

```python
from batch_llm.testing import MockAgent

mock_agent = MockAgent(
    response_factory=lambda p: Summary(title="Test", key_points=["A", "B"]),
    latency=0.01,  # Simulate 10ms latency
)

strategy = PydanticAIStrategy(agent=mock_agent)
# Test your code without API calls!
```

### Run Tests

```bash
# Using uv (recommended)
uv run pytest

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

## Roadmap

- [ ] Batch API support (50% cost reduction, hours latency)
- [ ] Streaming support for long-running tasks
- [ ] Built-in cost tracking and budgeting
- [ ] Prometheus/StatsD metrics export
- [ ] CLI for batch processing from files

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
- [`example_context_manager.py`](examples/example_context_manager.py) - Resource management

---

**Questions?** Open an issue on GitHub or check the [API documentation](docs/API.md).
