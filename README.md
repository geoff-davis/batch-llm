# Batch LLM

A flexible framework for processing multiple LLM requests efficiently with support for PydanticAI agents, direct API calls, and custom factories.

[![PyPI version](https://badge.fury.io/py/batch-llm.svg)](https://badge.fury.io/py/batch-llm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

Install using uv (recommended):

```bash
uv add batch-llm
```

Or using pip:

```bash
# Minimal (for direct API calls)
pip install batch-llm

# With PydanticAI support
pip install batch-llm[pydantic-ai]

# With Gemini SDK
pip install batch-llm[gemini]

# With everything
pip install batch-llm[all]
```

**Note**: PydanticAI is now optional! See [OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md) for details.

## Quick Start

```python
import asyncio
from batch_llm import ParallelBatchProcessor, LLMWorkItem, ProcessorConfig
from pydantic_ai import Agent
from pydantic import BaseModel

class BookSummary(BaseModel):
    title: str
    summary: str

# Create agent
agent = Agent("gemini-2.0-flash", output_type=BookSummary)

# Create processor
config = ProcessorConfig(max_workers=5, timeout_per_item=120.0)
processor = ParallelBatchProcessor(config=config)

# Add work
work_item = LLMWorkItem(
    item_id="book_1",
    agent=agent,
    prompt="Summarize Pride and Prejudice",
)
await processor.add_work(work_item)

# Process all
result = await processor.process_all()
print(f"Succeeded: {result.succeeded}/{result.total_items}")
```

## Features

This module provides a clean abstraction for bulk LLM processing with support for:
- **Flexible LLM Integration** - Three ways to call LLMs:
  - PydanticAI agents (recommended)
  - Direct API calls with custom temperature control
  - Agent factories for progressive retry strategies
- **Parallel Processing** - Efficient asyncio-based concurrent execution
- **Work Queue Management** - Easy batch job coordination
- **Partial Failure Handling** - Graceful error handling and reporting
- **Post-Processing Hooks** - Run custom logic after each success
- **Progress Tracking** - Built-in metrics and observability
- **Provider-Agnostic** - Error classification works with any LLM provider
- **Middleware Pipeline** - Extensible processing pipeline
- **Rate Limiting** - Built-in rate limit handling with configurable strategies
- **Testing Utilities** - MockAgent for testing without API calls

## Architecture

```
┌─────────────────────────────────────────────┐
│           BatchProcessor (Abstract)         │
│  - Queue management                         │
│  - Worker coordination                      │
│  - Result aggregation                       │
│  - Post-processing hooks                    │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼──────────┐  ┌────────▼────────────┐
│ Parallel         │  │ BatchAPI            │
│ Processor        │  │ Processor (future)  │
│                  │  │                     │
│ • Single items   │  │ • True batch API    │
│ • asyncio        │  │ • 50% discount      │
│ • Fast           │  │ • Hours latency     │
└──────────────────┘  └─────────────────────┘
```

## Core Components

### `LLMWorkItem`

Represents a single work item. You must provide **exactly one** of three processing methods:

#### Option 1: PydanticAI Agent (Recommended)
```python
work_item = LLMWorkItem(
    item_id="book_1",
    agent=agent,  # PydanticAI Agent instance
    prompt="Summarize this book",
    context=optional_context,
)
```

#### Option 2: Direct API Call
For custom temperature control or when you need direct API access:
```python
async def call_gemini(attempt: int, timeout: float) -> tuple[OutputModel, dict[str, int]]:
    """Direct call to Gemini API with custom temperature."""
    # Your custom API call here
    # Returns: (result, token_usage_dict)
    return result, {"input_tokens": 100, "output_tokens": 200}

work_item = LLMWorkItem(
    item_id="book_1",
    direct_call=call_gemini,
    input_data=your_input_data,  # Passed to your callable
    context=optional_context,
)
```

#### Option 3: Agent Factory
For progressive temperature increases on retries:
```python
def create_agent(attempt: int) -> Agent:
    """Create agent with temperature based on attempt number."""
    temperature = 0.7 + (attempt * 0.1)  # Increase temp on retries
    return Agent("gemini-2.0-flash", temperature=temperature)

work_item = LLMWorkItem(
    item_id="book_1",
    agent_factory=create_agent,
    prompt="Summarize this book",
    context=optional_context,
)
```

**Common fields:**
- `item_id`: Unique identifier (required)
- `context`: Optional data passed through to results/post-processor

### `WorkItemResult`
Result of processing a single item:
- `item_id`: Work item ID
- `success`: Boolean success flag
- `output`: Agent output (if successful)
- `error`: Error message (if failed)
- `context`: Context data from work item
- `token_usage`: Token statistics

### `BatchResult`
Aggregate results from a batch:
- `results`: List of individual results
- `total_items`: Count of items
- `succeeded`: Count of successes
- `failed`: Count of failures
- `total_input_tokens`: Sum of input tokens
- `total_output_tokens`: Sum of output tokens

### `BatchProcessor` (Abstract)
Base class defining the interface:
- `add_work()`: Add items to queue
- `process_all()`: Process all items
- `_worker()`: Worker implementation (abstract)
- `_process_item()`: Item processing logic (abstract)

## Implementations

### `ParallelBatchProcessor`

Processes items individually with asyncio concurrency:

```python
from batch_llm import ParallelBatchProcessor, LLMWorkItem
from pydantic_ai import Agent
from pydantic import BaseModel

# Define output model
class BookSummary(BaseModel):
    title: str
    summary: str

# Create agent
agent = Agent("gemini-2.0-flash", output_type=BookSummary)

# Create processor
processor = ParallelBatchProcessor(
    max_workers=5,
    timeout_per_item=120.0,
)

# Add work
work_item = LLMWorkItem(
    item_id="book_1",
    agent=agent,
    prompt="Summarize Pride and Prejudice",
)
await processor.add_work(work_item)

# Process all
result = await processor.process_all()
print(f"Succeeded: {result.succeeded}/{result.total_items}")
```

**Pros:**
- True parallel processing (fast)
- Same cost as sequential
- Simple to use

**Cons:**
- No batch pricing discount
- More API calls (same cost, more overhead)

### `BatchAPIProcessor` (Future)

Will use Google's Batch Prediction API:

**Pros:**
- 50% pricing discount
- Optimized for throughput

**Cons:**
- Hours of latency (not real-time)
- More complex setup

## Using Context and Post-Processing

Pass context data through the pipeline and run post-processing after each success:

```python
from dataclasses import dataclass

@dataclass
class EnrichmentContext:
    work_key: str
    original_title: str

async def save_to_db(result: WorkItemResult):
    """Post-processor that saves results to database."""
    if result.success and result.context:
        # Save result.output to database using result.context
        await db.save(result.context.work_key, result.output)

processor = ParallelBatchProcessor(
    max_workers=5,
    post_processor=save_to_db,  # Called after each success
)

# Add work with context
context = EnrichmentContext(work_key="/works/OL123W", original_title="1984")
work_item = LLMWorkItem(
    item_id="work_1",
    agent=agent,
    prompt="Enrich this book...",
    context=context,
)
await processor.add_work(work_item)
```

## Error Handling

The framework handles errors gracefully:

1. **Timeouts**: Items timing out return `WorkItemResult` with `success=False`
2. **Exceptions**: Caught and recorded in `error` field
3. **Partial failures**: Other items continue processing
4. **Post-processor errors**: Logged but don't fail the item

```python
result = await processor.process_all()

# Check individual results
for item_result in result.results:
    if not item_result.success:
        print(f"Failed: {item_result.item_id} - {item_result.error}")
```

## Integration Example

Here's how this would integrate with the existing enrichment pipeline:

```python
from batch_llm import ParallelBatchProcessor, LLMWorkItem
from ingest.openlibrary.clean.enrich import create_agent, format_work_for_prompt
from ingest.openlibrary.clean.popular_works import get_popular_works

# Create agent
agent = create_agent()

# Create processor with post-processing
async def save_enriched_work(result):
    if result.success:
        enriched = canonical_to_enriched_work(
            result.context.work_key,
            result.output
        )
        async with AsyncSession(async_engine) as session:
            session.add(enriched)
            await session.commit()

processor = ParallelBatchProcessor(
    max_workers=5,
    timeout_per_item=120.0,
    post_processor=save_enriched_work,
)

# Add work items
async for work in get_popular_works(limit=100):
    prompt = format_work_for_prompt(work)
    work_item = LLMWorkItem(
        item_id=work.work_key,
        agent=agent,
        prompt=prompt,
        context=work,  # Pass work through as context
    )
    await processor.add_work(work_item)

# Process all
result = await processor.process_all()
print(f"Enriched {result.succeeded}/{result.total_items} works")
```

## Performance Considerations

### Parallel Strategy
- **Throughput**: ~5-10 items/second with 5 workers (depends on response time)
- **Cost**: Same as sequential ($0.075 per 1M input tokens, $0.30 per 1M output)
- **Latency**: Real-time (seconds per item)

### Future Batch API Strategy
- **Throughput**: Higher (Google's infrastructure optimizes)
- **Cost**: 50% discount ($0.0375 per 1M input, $0.15 per 1M output)
- **Latency**: Hours (asynchronous batch job)

## Future Enhancements

1. **Retry logic**: Add configurable retry for failed items
2. **Dynamic batching**: Adjust batch size based on success rate
3. **Rate limiting**: Built-in rate limit handling
4. **Metrics export**: Prometheus/StatsD integration
5. **Progress callbacks**: Real-time progress updates
6. **BatchAPI implementation**: True batch API support

## Examples

See the `examples/` directory for complete usage examples:

```bash
# Run the example (requires API key)
python examples/example.py
```

## Testing

Run the test suite:

```bash
# Using uv
uv run pytest

# Or using pytest directly
pytest
```

## API Reference

### `LLMWorkItem[TInput, TOutput, TContext]`
- `item_id: str` - Unique identifier
- `agent: Agent[None, TOutput]` - PydanticAI agent
- `prompt: str` - Input prompt
- `context: TContext | None` - Optional context

### `WorkItemResult[TOutput, TContext]`
- `item_id: str`
- `success: bool`
- `output: TOutput | None`
- `error: str | None`
- `context: TContext | None`
- `token_usage: dict[str, int]`

### `BatchResult[TOutput, TContext]`
- `results: list[WorkItemResult]`
- `total_items: int`
- `succeeded: int`
- `failed: int`
- `total_input_tokens: int`
- `total_output_tokens: int`

### `ParallelBatchProcessor[TInput, TOutput, TContext]`
```python
def __init__(
    max_workers: int = 5,
    batch_size: int = 1,  # Ignored
    post_processor: PostProcessorFunc | None = None,
    timeout_per_item: float = 120.0,
)

async def add_work(work_item: LLMWorkItem) -> None
async def process_all() -> BatchResult
```
