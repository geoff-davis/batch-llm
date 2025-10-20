# batch-llm API Reference

Complete API documentation for batch-llm v3.0.

## Table of Contents

- [Core Classes](#core-classes)
  - [LLMWorkItem](#llmworkitem)
  - [WorkItemResult](#workitemresult)
  - [BatchResult](#batchresult)
  - [ParallelBatchProcessor](#parallelbatchprocessor)
- [LLM Strategies](#llm-strategies)
  - [LLMCallStrategy (Abstract)](#llmcallstrategy)
  - [PydanticAIStrategy](#pydanticaistrategy)
  - [GeminiStrategy](#geministrategy)
  - [GeminiCachedStrategy](#geminicachedstrategy)
- [Configuration](#configuration)
  - [ProcessorConfig](#processorconfig)
  - [RetryConfig](#retryconfig)
  - [RateLimitConfig](#ratelimitconfig)
- [Error Handling](#error-handling)
  - [ErrorClassifier](#errorclassifier)
  - [ErrorInfo](#errorinfo)
  - [RateLimitStrategy](#ratelimitstrategy)
- [Middleware & Observers](#middleware--observers)
  - [Middleware](#middleware)
  - [ProcessorObserver](#processorobserver)
  - [MetricsObserver](#metricsobserver)
- [Type Aliases](#type-aliases)

---

## Core Classes

### LLMWorkItem

Represents a single work item to be processed by an LLM strategy.

```python
@dataclass
class LLMWorkItem(Generic[TInput, TOutput, TContext]):
    item_id: str
    strategy: LLMCallStrategy[TOutput]
    prompt: str = ""
    context: TContext | None = None
```

**Type Parameters:**

- `TInput`: Input data type (unused in v3.0, kept for backward compatibility)
- `TOutput`: Expected output type from the LLM
- `TContext`: Optional context data type passed through to results

**Fields:**

- `item_id` (str): Unique identifier for this work item. Must be non-empty.
- `strategy` (LLMCallStrategy[TOutput]): Strategy that encapsulates how to make the LLM call
- `prompt` (str, optional): The prompt/input to pass to the LLM. Default: ""
- `context` (TContext | None, optional): Optional context data passed through to results/post-processor

**Example:**

```python
from batch_llm import LLMWorkItem, PydanticAIStrategy
from pydantic_ai import Agent

agent = Agent("openai:gpt-4", result_type=MyOutput)
strategy = PydanticAIStrategy(agent=agent)

work_item = LLMWorkItem(
    item_id="task_1",
    strategy=strategy,
    prompt="Analyze this text...",
    context={"user_id": 123}
)
```

**Validation:**

- Raises `ValueError` if `item_id` is empty or whitespace-only
- Raises `ValueError` if `item_id` is not a string

---

### WorkItemResult

Result of processing a single work item.

```python
@dataclass
class WorkItemResult(Generic[TOutput, TContext]):
    item_id: str
    success: bool
    output: TOutput | None = None
    error: str | None = None
    context: TContext | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    gemini_safety_ratings: dict[str, str] | None = None
```

**Fields:**

- `item_id` (str): ID of the work item
- `success` (bool): Whether processing succeeded
- `output` (TOutput | None): LLM output if successful, None if failed
- `error` (str | None): Error message if failed, None if successful
- `context` (TContext | None): Context data from the work item
- `token_usage` (dict[str, int]): Token usage statistics
  - Keys: `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens` (Gemini only)
- `gemini_safety_ratings` (dict[str, str] | None): Gemini API safety ratings if available

**Example:**

```python
result = await processor.process_all()
for item_result in result.results:
    if item_result.success:
        print(f"✓ {item_result.item_id}: {item_result.output}")
        print(f"  Tokens: {item_result.token_usage}")
    else:
        print(f"✗ {item_result.item_id}: {item_result.error}")
```

---

### BatchResult

Result of processing a batch of work items.

```python
@dataclass
class BatchResult(Generic[TOutput, TContext]):
    results: list[WorkItemResult[TOutput, TContext]]
    total_items: int = 0
    succeeded: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
```

**Fields:**

- `results` (list[WorkItemResult]): List of individual work item results
- `total_items` (int): Total number of items processed
- `succeeded` (int): Number of successful items
- `failed` (int): Number of failed items
- `total_input_tokens` (int): Sum of input tokens across all items
- `total_output_tokens` (int): Sum of output tokens across all items

**Note:** Summary statistics are calculated automatically in `__post_init__`.

**Example:**

```python
result = await processor.process_all()

print(f"Processed {result.total_items} items")
print(f"Success: {result.succeeded}, Failed: {result.failed}")
print(f"Total tokens: {result.total_input_tokens + result.total_output_tokens}")

# Access individual results
for item_result in result.results:
    if item_result.success:
        process_output(item_result.output)
```

---

### ParallelBatchProcessor

Main processor that executes work items in parallel.

```python
class ParallelBatchProcessor(
    BatchProcessor[TInput, TOutput, TContext],
    Generic[TInput, TOutput, TContext]
):
    def __init__(
        self,
        config: ProcessorConfig,
        post_processor: PostProcessorFunc[TOutput, TContext] | None = None,
        progress_callback: ProgressCallbackFunc | None = None,
        error_classifier: ErrorClassifier | None = None,
        rate_limit_strategy: RateLimitStrategy | None = None,
        middlewares: list[Middleware] | None = None,
        observers: list[ProcessorObserver] | None = None,
    )
```

**Parameters:**

- `config` (ProcessorConfig): Configuration for the processor
- `post_processor` (PostProcessorFunc | None): Optional async function called after each item
- `progress_callback` (ProgressCallbackFunc | None): Optional callback for progress updates
- `error_classifier` (ErrorClassifier | None): Custom error classifier. Default: `DefaultErrorClassifier()`
- `rate_limit_strategy` (RateLimitStrategy | None): Custom rate limit handling. Default: `ExponentialBackoffStrategy()`
- `middlewares` (list[Middleware] | None): List of middleware for pre/post processing
- `observers` (list[ProcessorObserver] | None): List of observers for monitoring events

**Methods:**

#### `async def add_work(work_item: LLMWorkItem) -> None`

Add a work item to the processing queue.

```python
await processor.add_work(work_item)
```

**Note:** If `max_queue_size` is set and queue is full, this will block until space is available.

#### `async def process_all() -> BatchResult`

Process all work items in the queue.

```python
result = await processor.process_all()
```

**Returns:** `BatchResult` containing all results and statistics

**Behavior:**

1. Starts worker tasks (up to `max_workers`)
2. Workers process items from queue with retry logic
3. Waits for all work to complete
4. Returns aggregated results

#### `async def cleanup() -> None`

Clean up resources (cancel pending workers, clear queue).

```python
await processor.cleanup()
```

**Note:** Automatically called when using async context manager.

#### Context Manager Support

```python
async with ParallelBatchProcessor(config=config) as processor:
    await processor.add_work(item)
    result = await processor.process_all()
# Automatic cleanup
```

**Example:**

```python
from batch_llm import ParallelBatchProcessor, ProcessorConfig, LLMWorkItem

config = ProcessorConfig(max_workers=5, timeout_per_item=60.0)

async with ParallelBatchProcessor(config=config) as processor:
    for i in range(100):
        work_item = LLMWorkItem(
            item_id=f"item_{i}",
            strategy=my_strategy,
            prompt=f"Task {i}"
        )
        await processor.add_work(work_item)

    result = await processor.process_all()
    print(f"Completed: {result.succeeded}/{result.total_items}")
```

---

## LLM Strategies

### LLMCallStrategy

Abstract base class for LLM call strategies.

```python
class LLMCallStrategy(ABC, Generic[TOutput]):
    async def prepare(self) -> None: ...

    @abstractmethod
    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[TOutput, dict[str, int]]: ...

    async def cleanup(self) -> None: ...
```

**Lifecycle:**

1. `prepare()` - Called once before any retry attempts
2. `execute()` - Called for each attempt (including retries)
3. `cleanup()` - Called once after all attempts complete

**Methods:**

#### `async def prepare() -> None`

Initialize resources before making LLM calls (e.g., create caches, initialize clients).

**Default:** No-op

#### `async def execute(prompt: str, attempt: int, timeout: float) -> tuple[TOutput, dict[str, int]]`

Execute an LLM call.

**Parameters:**

- `prompt` (str): The prompt to send to the LLM
- `attempt` (int): Which retry attempt this is (1, 2, 3, ...)
- `timeout` (float): Maximum time to wait for response (seconds)
  - Note: Timeout enforcement is handled by the framework wrapping this call

**Returns:** Tuple of `(output, token_usage)`

- `output` (TOutput): The LLM response
- `token_usage` (dict[str, int]): Token usage with keys: `input_tokens`, `output_tokens`, `total_tokens`

**Raises:** Any exception to trigger retry (if retryable) or failure

#### `async def cleanup() -> None`

Clean up resources after all attempts complete (e.g., delete caches, close clients).

**Default:** No-op

**Custom Strategy Example:**

```python
from batch_llm import LLMCallStrategy

class MyCustomStrategy(LLMCallStrategy[str]):
    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, dict[str, int]]:
        # Your custom LLM API call
        response = await my_llm_api.generate(prompt)

        tokens = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        }

        return response.text, tokens
```

---

### PydanticAIStrategy

Strategy for using PydanticAI agents.

```python
class PydanticAIStrategy(LLMCallStrategy[TOutput]):
    def __init__(self, agent: Agent[None, TOutput])
```

**Parameters:**

- `agent` (Agent[None, TOutput]): Configured PydanticAI agent

**Requires:** `pip install 'batch-llm[pydantic-ai]'`

**Example:**

```python
from batch_llm import PydanticAIStrategy, LLMWorkItem
from pydantic_ai import Agent
from pydantic import BaseModel

class BookSummary(BaseModel):
    title: str
    summary: str

agent = Agent("openai:gpt-4", result_type=BookSummary)
strategy = PydanticAIStrategy(agent=agent)

work_item = LLMWorkItem(
    item_id="book_1",
    strategy=strategy,
    prompt="Summarize: The Great Gatsby..."
)
```

---

### GeminiStrategy

Strategy for calling Google Gemini API directly (without caching).

```python
class GeminiStrategy(LLMCallStrategy[TOutput]):
    def __init__(
        self,
        model: str,
        client: genai.Client,
        response_parser: Callable[[Any], TOutput],
        config: GenerateContentConfig | None = None,
    )
```

**Parameters:**

- `model` (str): Model name (e.g., "gemini-2.0-flash-exp")
- `client` (genai.Client): Initialized Gemini client
- `response_parser` (Callable): Function to parse response into TOutput
- `config` (GenerateContentConfig | None): Optional generation config (temperature, etc.)

**Requires:** `pip install 'batch-llm[gemini]'`

**Example:**

```python
from batch_llm import GeminiStrategy, LLMWorkItem
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def parse_response(response) -> str:
    return response.text

strategy = GeminiStrategy(
    model="gemini-2.0-flash-exp",
    client=client,
    response_parser=parse_response,
)

work_item = LLMWorkItem(
    item_id="task_1",
    strategy=strategy,
    prompt="Explain quantum computing"
)
```

---

### GeminiCachedStrategy

Strategy for calling Google Gemini API with context caching.

```python
class GeminiCachedStrategy(LLMCallStrategy[TOutput]):
    def __init__(
        self,
        model: str,
        client: genai.Client,
        response_parser: Callable[[Any], TOutput],
        cached_content: list[Content],
        cache_ttl_seconds: int = 3600,
        cache_refresh_threshold: float = 0.1,
        config: GenerateContentConfig | None = None,
    )
```

**Parameters:**

- `model` (str): Model name
- `client` (genai.Client): Initialized Gemini client
- `response_parser` (Callable): Function to parse response
- `cached_content` (list[Content]): Content to cache (system instructions, documents)
- `cache_ttl_seconds` (int): Cache TTL in seconds. Default: 3600 (1 hour)
- `cache_refresh_threshold` (float): Refresh cache when TTL falls below this fraction. Default: 0.1 (10%)
- `config` (GenerateContentConfig | None): Optional generation config

**Lifecycle:**

- `prepare()`: Creates the Gemini cache
- `execute()`: Uses cache, automatically refreshes TTL if needed
- `cleanup()`: Deletes the cache

**Requires:** `pip install 'batch-llm[gemini]'`

**Example:**

```python
from batch_llm import GeminiCachedStrategy, LLMWorkItem
from google import genai
from google.genai.types import Content

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Large document to cache
cached_content = [
    Content(role="system", parts=[{"text": "You are a helpful assistant."}]),
    Content(role="user", parts=[{"text": large_document}]),
]

strategy = GeminiCachedStrategy(
    model="gemini-2.0-flash-exp",
    client=client,
    response_parser=lambda r: r.text,
    cached_content=cached_content,
    cache_ttl_seconds=3600,
)

# Reuse strategy across multiple work items to benefit from caching
for i in range(100):
    work_item = LLMWorkItem(
        item_id=f"task_{i}",
        strategy=strategy,  # Same strategy, shared cache
        prompt=f"Question {i} about the document"
    )
    await processor.add_work(work_item)
```

---

## Configuration

### ProcessorConfig

Complete configuration for batch processor.

```python
@dataclass
class ProcessorConfig:
    max_workers: int = 5
    timeout_per_item: float = 120.0
    retry: RetryConfig = field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    progress_interval: int = 10
    enable_detailed_logging: bool = False
    max_queue_size: int = 0
    dry_run: bool = False
```

**Fields:**

- `max_workers` (int): Maximum number of concurrent workers. Default: 5
- `timeout_per_item` (float): Timeout per item in seconds. Default: 120.0
- `retry` (RetryConfig): Retry configuration
- `rate_limit` (RateLimitConfig): Rate limit handling configuration
- `progress_interval` (int): Log progress every N items. Default: 10
- `enable_detailed_logging` (bool): Enable detailed debug logging. Default: False
- `max_queue_size` (int): Max queue size (0 = unlimited). Default: 0
- `dry_run` (bool): Skip actual API calls, return mock data. Default: False

**Example:**

```python
from batch_llm import ProcessorConfig, RetryConfig

config = ProcessorConfig(
    max_workers=10,
    timeout_per_item=60.0,
    retry=RetryConfig(max_attempts=5, initial_wait=2.0),
    progress_interval=20,
    max_queue_size=1000,
)
```

---

### RetryConfig

Configuration for retry behavior.

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_wait: float = 1.0
    max_wait: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
```

**Fields:**

- `max_attempts` (int): Maximum retry attempts. Default: 3
- `initial_wait` (float): Initial wait time in seconds. Default: 1.0
- `max_wait` (float): Maximum wait time in seconds. Default: 60.0
- `exponential_base` (float): Exponential backoff base. Default: 2.0
- `jitter` (bool): Add random jitter to wait times. Default: True

**Validation:**

- `max_attempts` must be >= 1
- `initial_wait` must be > 0
- `max_wait` must be >= initial_wait
- `exponential_base` must be >= 1

**Example:**

```python
retry_config = RetryConfig(
    max_attempts=5,
    initial_wait=2.0,
    max_wait=120.0,
    exponential_base=2.0,
    jitter=True,
)
```

---

### RateLimitConfig

Configuration for rate limit handling.

```python
@dataclass
class RateLimitConfig:
    cooldown_seconds: float = 300.0
    slow_start_items: int = 50
    slow_start_initial_delay: float = 2.0
    slow_start_final_delay: float = 0.1
    backoff_multiplier: float = 1.5
```

**Fields:**

- `cooldown_seconds` (float): Cooldown after rate limit. Default: 300.0 (5 minutes)
- `slow_start_items` (int): Number of items for slow start. Default: 50
- `slow_start_initial_delay` (float): Initial delay in slow start. Default: 2.0
- `slow_start_final_delay` (float): Final delay in slow start. Default: 0.1
- `backoff_multiplier` (float): Increase cooldown on repeated rate limits. Default: 1.5

**Validation:**

- `cooldown_seconds` must be >= 0
- `slow_start_items` must be >= 0
- `slow_start_initial_delay` must be >= slow_start_final_delay
- `backoff_multiplier` must be >= 1.0

---

## Error Handling

### ErrorClassifier

Interface for classifying errors as retryable or not.

```python
class ErrorClassifier(ABC):
    @abstractmethod
    def classify(self, exception: Exception) -> ErrorInfo: ...
```

**Built-in Implementations:**

- `DefaultErrorClassifier`: Provider-agnostic classification based on exception types
- `GeminiErrorClassifier`: Specialized for Google Gemini API errors

**Custom Example:**

```python
from batch_llm import ErrorClassifier, ErrorInfo

class MyErrorClassifier(ErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        error_str = str(exception).lower()

        if "rate limit" in error_str:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                category="rate_limit",
            )
        elif "timeout" in error_str:
            return ErrorInfo(is_retryable=True, category="timeout")
        else:
            return ErrorInfo(is_retryable=False, category="unknown")
```

---

### ErrorInfo

Information about a classified error.

```python
@dataclass
class ErrorInfo:
    is_retryable: bool
    is_rate_limit: bool = False
    category: str = "unknown"
    suggested_wait: float | None = None
```

**Fields:**

- `is_retryable` (bool): Whether the error should trigger a retry
- `is_rate_limit` (bool): Whether this is a rate limit error
- `category` (str): Error category for logging/metrics
- `suggested_wait` (float | None): Suggested wait time before retry

---

### RateLimitStrategy

Interface for custom rate limit handling strategies.

```python
class RateLimitStrategy(ABC):
    @abstractmethod
    async def handle_rate_limit(
        self, error_info: ErrorInfo, attempt: int
    ) -> float: ...
```

**Built-in Implementations:**

- `ExponentialBackoffStrategy`: Exponential backoff with configurable parameters
- `FixedDelayStrategy`: Fixed delay between retries

---

## Middleware & Observers

### Middleware

Interface for middleware that can modify work items before/after processing.

```python
class Middleware(ABC):
    async def before_process(
        self, work_item: LLMWorkItem
    ) -> LLMWorkItem | None: ...

    async def after_process(
        self, work_item: LLMWorkItem, result: WorkItemResult
    ) -> WorkItemResult: ...

    async def on_error(
        self, work_item: LLMWorkItem, error: Exception
    ) -> None: ...
```

**Methods:**

- `before_process()`: Modify work item before processing. Return `None` to skip.
- `after_process()`: Modify result after processing
- `on_error()`: Handle errors (doesn't stop processing)

**Example:**

```python
from batch_llm.middleware import BaseMiddleware

class LoggingMiddleware(BaseMiddleware):
    async def before_process(self, work_item):
        print(f"Processing {work_item.item_id}")
        return work_item

    async def after_process(self, work_item, result):
        print(f"Completed {work_item.item_id}: {result.success}")
        return result
```

---

### ProcessorObserver

Interface for observers that monitor processing events.

```python
class ProcessorObserver(ABC):
    @abstractmethod
    async def on_event(
        self, event: ProcessingEvent, data: dict[str, Any]
    ) -> None: ...
```

**Events:**

- `PROCESSING_STARTED`: Batch processing started
- `PROCESSING_COMPLETED`: Batch processing completed
- `ITEM_STARTED`: Work item started
- `ITEM_COMPLETED`: Work item completed
- `ITEM_FAILED`: Work item failed
- `RETRY_SCHEDULED`: Retry scheduled
- `RATE_LIMIT_HIT`: Rate limit encountered
- `WORKER_STARTED`: Worker started
- `WORKER_STOPPED`: Worker stopped

---

### MetricsObserver

Built-in observer for collecting metrics.

```python
class MetricsObserver(BaseObserver):
    async def get_metrics(self) -> dict[str, Any]: ...
    async def export_json(self) -> str: ...
    async def export_prometheus(self) -> str: ...
    async def export_dict(self) -> dict[str, Any]: ...
```

**Methods:**

- `get_metrics()`: Get current metrics as dict
- `export_json()`: Export metrics as JSON string
- `export_prometheus()`: Export in Prometheus text format
- `export_dict()`: Export as dictionary

**Example:**

```python
from batch_llm import MetricsObserver

metrics = MetricsObserver()
processor = ParallelBatchProcessor(config=config, observers=[metrics])

await processor.process_all()

# Get metrics
metrics_data = await metrics.get_metrics()
print(f"Items processed: {metrics_data['items_processed']}")
print(f"Success rate: {metrics_data['success_rate']:.1%}")

# Export for monitoring
prometheus_text = await metrics.export_prometheus()
```

---

## Type Aliases

### PostProcessorFunc

Callback function called after each successful item.

```python
PostProcessorFunc = Callable[
    [WorkItemResult[TOutput, TContext]],
    Awaitable[None] | None
]
```

**Example:**

```python
async def save_result(result: WorkItemResult):
    if result.success:
        await database.save(result.item_id, result.output)

processor = ParallelBatchProcessor(
    config=config,
    post_processor=save_result
)
```

### ProgressCallbackFunc

Callback function for progress updates.

```python
ProgressCallbackFunc = Callable[
    [int, int, str],  # (completed, total, current_item_id)
    Awaitable[None] | None
]
```

**Example:**

```python
async def on_progress(completed: int, total: int, current_item: str):
    print(f"Progress: {completed}/{total} - {current_item}")

processor = ParallelBatchProcessor(
    config=config,
    progress_callback=on_progress
)
```

---

## Complete Example

```python
import asyncio
from batch_llm import (
    ParallelBatchProcessor,
    ProcessorConfig,
    LLMWorkItem,
    PydanticAIStrategy,
    MetricsObserver,
)
from pydantic_ai import Agent
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    summary: str

async def main():
    # Configure processor
    config = ProcessorConfig(
        max_workers=10,
        timeout_per_item=60.0,
        max_queue_size=100,
    )

    # Create strategy
    agent = Agent("openai:gpt-4", result_type=Summary)
    strategy = PydanticAIStrategy(agent=agent)

    # Add metrics
    metrics = MetricsObserver()

    # Create processor with context manager
    async with ParallelBatchProcessor(
        config=config,
        observers=[metrics]
    ) as processor:
        # Add work items
        for i in range(50):
            work_item = LLMWorkItem(
                item_id=f"doc_{i}",
                strategy=strategy,
                prompt=f"Summarize document {i}...",
            )
            await processor.add_work(work_item)

        # Process all
        result = await processor.process_all()

        # Report results
        print(f"Completed: {result.succeeded}/{result.total_items}")
        print(f"Tokens used: {result.total_input_tokens + result.total_output_tokens}")

        # Get metrics
        metrics_data = await metrics.get_metrics()
        print(f"Average processing time: {metrics_data['avg_processing_time']:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## See Also

- [README.md](../README.md) - Getting started guide
- [MIGRATION_V3.md](./MIGRATION_V3.md) - Migration guide from v2.x
- [GEMINI_INTEGRATION.md](./GEMINI_INTEGRATION.md) - Detailed Gemini integration guide
- [CHANGELOG.md](../CHANGELOG.md) - Version history
