# Migration Guide: v2.x → v3.0

This guide helps you migrate from batch-llm v2.x to v3.0, which introduces the new
**LLM call strategy pattern** to provide greater flexibility and cleaner separation of concerns.

## Overview of Changes

**v3.0 Breaking Changes:**

- Replaced `agent=` parameter with `strategy=` in `LLMWorkItem`
- Removed `client=` parameter from `LLMWorkItem`
- Introduced `LLMCallStrategy` abstract base class
- New built-in strategies: `PydanticAIStrategy`, `GeminiStrategy`, `GeminiCachedStrategy`
- Improved timeout enforcement (now framework-level)

**Why the change?**

- **Flexibility**: Support any LLM provider (OpenAI, Anthropic, LangChain, etc.) through custom strategies
- **Clean separation**: Strategy encapsulates all model-specific logic (caching, parsing, retries)
- **Extensibility**: Easy to create custom strategies with prepare/execute/cleanup lifecycle
- **Consistency**: Unified interface regardless of underlying LLM provider

## Quick Migration Patterns

### Pattern 1: PydanticAI Agent (Most Common)

**v2.x Code:**

```python
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from pydantic_ai import Agent

# Old way - passing agent directly
agent = Agent("gemini-2.0-flash-exp", result_type=MyOutput)

work_item = LLMWorkItem(
    item_id="item_1",
    agent=agent,  # ❌ Removed in v3.0
    prompt="Test prompt",
)
```

**v3.0 Code:**

```python
from batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,  # ✅ New import
)
from pydantic_ai import Agent

# New way - wrap agent in strategy
agent = Agent("gemini-2.0-flash-exp", result_type=MyOutput)
strategy = PydanticAIStrategy(agent=agent)  # ✅ Wrap in strategy

work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,  # ✅ Use strategy= instead of agent=
    prompt="Test prompt",
)
```

**Migration steps:**

1. Import `PydanticAIStrategy` from `batch_llm`
2. Wrap your agent: `strategy = PydanticAIStrategy(agent=agent)`
3. Replace `agent=agent` with `strategy=strategy`

---

### Pattern 2: Direct Gemini API Calls

**v2.x Code:**

```python
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from google import genai

client = genai.Client(api_key=API_KEY)

# Old way - passing client directly
work_item = LLMWorkItem(
    item_id="item_1",
    client=client,  # ❌ Removed in v3.0
    prompt="Test prompt",
)
```

**v3.0 Code:**

```python
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import GeminiStrategy  # ✅ New import
from google import genai

client = genai.Client(api_key=API_KEY)

# Create response parser
def parse_response(response) -> str:
    return response.text

# New way - use GeminiStrategy
strategy = GeminiStrategy(
    model="gemini-2.0-flash-exp",
    client=client,
    response_parser=parse_response,
    config=genai.types.GenerateContentConfig(temperature=0.7),
)

work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,  # ✅ Use strategy
    prompt="Test prompt",
)
```

**Migration steps:**

1. Import `GeminiStrategy` from `batch_llm.llm_strategies`
2. Create a response parser function
3. Create strategy with model, client, parser, and optional config
4. Replace `client=client` with `strategy=strategy`

---

### Pattern 3: Gemini with Context Caching

**v2.x Code:**

```python
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from google import genai

client = genai.Client(api_key=API_KEY)

# Old way - caching handled implicitly or manually
work_item = LLMWorkItem(
    item_id="item_1",
    client=client,  # ❌ Removed
    prompt="Question about cached context",
)
```

**v3.0 Code:**

```python
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.llm_strategies import GeminiCachedStrategy  # ✅ New import
from google import genai

client = genai.Client(api_key=API_KEY)

# Define content to cache (e.g., large documents, knowledge base)
cached_content = [
    genai.types.Content(
        role="user",
        parts=[genai.types.Part(text="Large context to cache...")]
    ),
]

def parse_response(response) -> str:
    return response.text

# New way - use GeminiCachedStrategy
strategy = GeminiCachedStrategy(
    model="gemini-2.0-flash-exp",
    client=client,
    response_parser=parse_response,
    cached_content=cached_content,
    cache_ttl_seconds=3600,  # Cache for 1 hour
    cache_refresh_threshold=0.1,  # Refresh at 10% TTL
)

work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,  # ✅ Use cached strategy
    prompt="Question about cached context",
)
```

**Migration steps:**

1. Import `GeminiCachedStrategy` from `batch_llm.llm_strategies`
2. Define your cached content as list of `Content` objects
3. Create cached strategy with TTL and refresh settings
4. Strategy automatically handles cache lifecycle (create, refresh, delete)

---

## Complete Migration Example

Here's a complete example showing before and after:

### v2.x Complete Example

```python
import asyncio
from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from pydantic import BaseModel
from pydantic_ai import Agent

class Output(BaseModel):
    result: str

async def main():
    # Create agent
    agent = Agent("gemini-2.0-flash-exp", result_type=Output)

    # Configure processor
    config = ProcessorConfig(max_workers=5, timeout_per_item=30.0)

    # Process items
    async with ParallelBatchProcessor[str, Output, None](config=config) as processor:
        # Add work items
        for i in range(10):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"item_{i}",
                    agent=agent,  # ❌ Old way
                    prompt=f"Process item {i}",
                )
            )

        # Process all
        result = await processor.process_all()

    print(f"Succeeded: {result.succeeded}/{result.total_items}")

if __name__ == "__main__":
    asyncio.run(main())
```

### v3.0 Complete Example

```python
import asyncio
from batch_llm import (
    LLMWorkItem,
    ParallelBatchProcessor,
    ProcessorConfig,
    PydanticAIStrategy,  # ✅ New import
)
from pydantic import BaseModel
from pydantic_ai import Agent

class Output(BaseModel):
    result: str

async def main():
    # Create agent
    agent = Agent("gemini-2.0-flash-exp", result_type=Output)

    # Wrap in strategy ✅
    strategy = PydanticAIStrategy(agent=agent)

    # Configure processor
    config = ProcessorConfig(max_workers=5, timeout_per_item=30.0)

    # Process items
    async with ParallelBatchProcessor[str, Output, None](config=config) as processor:
        # Add work items
        for i in range(10):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"item_{i}",
                    strategy=strategy,  # ✅ New way
                    prompt=f"Process item {i}",
                )
            )

        # Process all
        result = await processor.process_all()

    print(f"Succeeded: {result.succeeded}/{result.total_items}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Key differences:**

1. Import `PydanticAIStrategy`
2. Create strategy: `strategy = PydanticAIStrategy(agent=agent)`
3. Use `strategy=` instead of `agent=`

---

## Custom Strategies (New in v3.0)

One of the biggest benefits of v3.0 is the ability to create custom strategies for any LLM provider:

### Example: OpenAI Custom Strategy

```python
from batch_llm.llm_strategies import LLMCallStrategy
from openai import AsyncOpenAI

class OpenAIStrategy(LLMCallStrategy[str]):
    """Custom strategy for OpenAI API."""

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, dict[str, int]]:
        # Make API call
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract output and token usage
        output = response.choices[0].message.content or ""
        tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return output, tokens

# Use it
client = AsyncOpenAI(api_key=API_KEY)
strategy = OpenAIStrategy(client=client, model="gpt-4o-mini")

work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="Your prompt here",
)
```

See `examples/example_openai.py`, `examples/example_anthropic.py`, and `examples/example_langchain.py` for more examples.

---

## Strategy Lifecycle (New in v3.0)

Strategies support a lifecycle with three methods:

```python
class LLMCallStrategy(ABC):
    async def prepare(self) -> None:
        """Called once before any retry attempts. Use for initialization."""
        pass

    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, dict[str, int]]:
        """Called for each attempt (including retries). Must be implemented."""
        pass

    async def cleanup(self) -> None:
        """Called once after all attempts. Use for cleanup."""
        pass
```

**Example with lifecycle:**

```python
class CachedStrategy(LLMCallStrategy[str]):
    async def prepare(self):
        # Initialize cache, open connections, etc.
        self.cache = await create_cache()

    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Use the cache
        return await self.cache.query(prompt)

    async def cleanup(self):
        # Clean up resources
        await self.cache.delete()
```

---

## Timeout Enforcement Improvements

**v3.0 changes how timeouts work:**

- **v2.x**: Each strategy was responsible for timeout enforcement
- **v3.0**: Framework enforces timeout with `asyncio.wait_for()` wrapper

**Impact on custom strategies:**

- You no longer need to wrap your strategy execution in `asyncio.wait_for()`
- The `timeout` parameter is still passed to `execute()` for informational purposes
- Framework handles timeout consistently across all strategies

**Example:**

```python
# v2.x - Strategy had to handle timeout
async def execute(self, prompt: str, attempt: int, timeout: float):
    return await asyncio.wait_for(
        self.agent.run(prompt),
        timeout=timeout,  # ❌ No longer needed
    )

# v3.0 - Framework handles timeout
async def execute(self, prompt: str, attempt: int, timeout: float):
    # Framework wraps this in asyncio.wait_for()
    return await self.agent.run(prompt)  # ✅ Simpler
```

---

## Summary Checklist

To migrate from v2.x to v3.0:

- [ ] **For PydanticAI users:**
  - [ ] Import `PydanticAIStrategy` from `batch_llm`
  - [ ] Wrap agents: `strategy = PydanticAIStrategy(agent=agent)`
  - [ ] Replace `agent=` with `strategy=`

- [ ] **For Gemini API users:**
  - [ ] Import `GeminiStrategy` or `GeminiCachedStrategy` from `batch_llm.llm_strategies`
  - [ ] Create response parser function
  - [ ] Create strategy with model, client, parser
  - [ ] Replace `client=` with `strategy=`

- [ ] **For custom implementations:**
  - [ ] Implement `LLMCallStrategy` abstract base class
  - [ ] Define `execute()` method (required)
  - [ ] Optionally define `prepare()` and `cleanup()`
  - [ ] Remove `asyncio.wait_for()` from execute (framework handles it)

- [ ] **Testing:**
  - [ ] Run your test suite
  - [ ] Verify timeout behavior works as expected
  - [ ] Check token usage is tracked correctly

---

## Need Help?

- **API Documentation**: See `docs/API.md` for complete API reference
- **Examples**: See `examples/` directory for working examples:
  - `example_llm_strategies.py` - All built-in strategies
  - `example_openai.py` - OpenAI integration
  - `example_anthropic.py` - Anthropic Claude integration
  - `example_langchain.py` - LangChain integration
- **Issues**: Report bugs at <https://github.com/yourusername/batch-llm/issues>

---

## Benefits of v3.0

**Why upgrade?**

1. **Universal Provider Support**: Use any LLM provider (OpenAI, Anthropic, LangChain, etc.)
2. **Better Caching**: First-class support for context caching (Gemini, etc.)
3. **Cleaner Code**: Separation of concerns between framework and model logic
4. **More Reliable**: Framework-level timeout enforcement
5. **Extensible**: Easy to create custom strategies for new providers
6. **Resource Management**: Proper lifecycle with prepare/cleanup hooks

The migration is straightforward and the benefits are significant. Most codebases can be migrated in under an hour.
