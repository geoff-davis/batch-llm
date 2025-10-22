# batch-llm API Reference

Complete API documentation for batch-llm v0.1.

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
- [Core Types](#core-types)
  - [TokenUsage](#tokenusage)
  - [FrameworkTimeoutError](#frameworktimeouterror)
- [Type Aliases](#type-aliases)
  - [PostProcessorFunc](#postprocessorfunc)
  - [ProgressCallbackFunc](#progresscallbackfunc)

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

- `TInput`: Input data type (unused in v0.1, kept for backward compatibility)
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
    token_usage: TokenUsage = field(default_factory=dict)  # type: ignore[assignment]
    gemini_safety_ratings: dict[str, str] | None = None
```

**Fields:**

- `item_id` (str): ID of the work item
- `success` (bool): Whether processing succeeded
- `output` (TOutput | None): LLM output if successful, None if failed
- `error` (str | None): Error message if failed, None if successful
- `context` (TContext | None): Context data from the work item
- `token_usage` ([TokenUsage](#tokenusage)): Token usage statistics with optional fields:
  - `input_tokens` (int): Number of tokens in the input/prompt
  - `output_tokens` (int): Number of tokens in the output/completion
  - `total_tokens` (int): Total tokens used (input + output)
  - `cached_input_tokens` (int): Number of input tokens served from cache (Gemini context caching)
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

> **Post-processing:** The optional `post_processor` runs inline on the worker as soon as an item finishes.
> It should hand off any heavy operations (long DB writes, expensive analytics, etc.) to another system;
> if the function takes too long the worker sits idle until the 75 s timeout triggers, reducing overall throughput.

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
    ) -> tuple[TOutput, TokenUsage]: ...

    async def on_error(self, exception: Exception, attempt: int) -> None: ...

    async def cleanup(self) -> None: ...

    async def dry_run(self, prompt: str) -> tuple[TOutput, TokenUsage]: ...
```

**Lifecycle:**

1. `prepare()` - Called once before any retry attempts
2. For each attempt (including retries):
   - `execute()` is called (or `dry_run()` if `config.dry_run=True`)
   - If `execute()` raises an exception, `on_error()` is called before retry logic
3. `cleanup()` - Called once after all attempts complete

**Methods:**

#### `async def prepare() -> None`

Initialize resources before making LLM calls (e.g., create caches, initialize clients).

**Default:** No-op

#### `async def execute(prompt: str, attempt: int, timeout: float) -> tuple[TOutput, TokenUsage]`

Execute an LLM call.

**Parameters:**

- `prompt` (str): The prompt to send to the LLM
- `attempt` (int): Which retry attempt this is (1, 2, 3, ...)
- `timeout` (float): Maximum time to wait for response (seconds)
  - Note: Timeout enforcement is handled by the framework wrapping this call in `asyncio.wait_for()`

**Returns:** Tuple of `(output, token_usage)`

- `output` (TOutput): The LLM response
- `token_usage` ([TokenUsage](#tokenusage)): Token usage dict with optional keys: `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`

**Raises:** Any exception to trigger retry (if retryable) or failure

#### `async def dry_run(prompt: str) -> tuple[TOutput, TokenUsage]`

Return mock output for dry-run mode (testing without API calls).

Called when `ProcessorConfig(dry_run=True)` is set. Override this method to provide realistic mock data for testing.

**Parameters:**

- `prompt` (str): The prompt that would have been sent to the LLM

**Returns:** Tuple of `(mock_output, mock_token_usage)`

**Default behavior:**

- Returns string `"[DRY-RUN] Mock output for prompt: {prompt[:50]}..."` as output
- Returns mock token usage: 100 input, 50 output, 150 total tokens

**Example override:**

```python
class MyStrategy(LLMCallStrategy[Output]):
    async def dry_run(self, prompt: str) -> tuple[Output, TokenUsage]:
        # Return realistic mock data
        mock_output = Output(result="Test result")
        mock_tokens: TokenUsage = {
            "input_tokens": len(prompt.split()),
            "output_tokens": 50,
            "total_tokens": len(prompt.split()) + 50,
        }
        return mock_output, mock_tokens
```

#### `async def on_error(exception: Exception, attempt: int) -> None`

Handle errors that occur during execute().

Called by the framework when `execute()` raises an exception, before deciding whether to retry. This allows strategies to:

- Inspect the error type to adjust retry behavior
- Store error information for use in the next attempt
- Modify prompts based on validation errors
- Track error patterns across attempts
- Make intelligent decisions (e.g., escalate to smarter model only on validation errors)

**Parameters:**

- `exception` (Exception): The exception that was raised during `execute()`
- `attempt` (int): Which attempt number failed (1, 2, 3, ...)

**Default:** No-op

**Use Cases:**

1. **Smart Model Escalation** - Only escalate to expensive models on validation errors, not network errors:

```python
class SmartModelEscalationStrategy(LLMCallStrategy[Output]):
    def __init__(self):
        self.validation_failures = 0

    async def on_error(self, exception: Exception, attempt: int) -> None:
        if isinstance(exception, ValidationError):
            self.validation_failures += 1

    async def execute(self, prompt: str, attempt: int, timeout: float):
        # Only escalate model on validation errors
        model_index = min(self.validation_failures, len(MODELS) - 1)
        model = MODELS[model_index]
        # Make call with appropriate model...
```

2. **Smart Retry with Partial Parsing** - Build better retry prompts based on what failed:

```python
class SmartRetryStrategy(LLMCallStrategy[Output]):
    def __init__(self):
        self.last_error = None
        self.last_response = None

    async def on_error(self, exception: Exception, attempt: int) -> None:
        if isinstance(exception, ValidationError):
            self.last_error = exception
            # last_response set in execute() before raising

    async def execute(self, prompt: str, attempt: int, timeout: float):
        if attempt > 1 and self.last_error:
            # Build smart retry prompt with partial parsing feedback
            prompt = self._create_retry_prompt_with_partial_data(prompt)
        # Make call with improved prompt...
```

3. **Error Type Tracking** - Distinguish between different error types:

```python
class ErrorTrackingStrategy(LLMCallStrategy[Output]):
    def __init__(self):
        self.validation_errors = 0
        self.network_errors = 0
        self.rate_limit_errors = 0

    async def on_error(self, exception: Exception, attempt: int) -> None:
        if isinstance(exception, ValidationError):
            self.validation_errors += 1
        elif isinstance(exception, ConnectionError):
            self.network_errors += 1
        elif "429" in str(exception):
            self.rate_limit_errors += 1
```

**Important Notes:**

- Exceptions in `on_error()` are caught and logged by the framework - they won't crash processing
- `on_error()` is only called when `execute()` raises an exception, not on success
- The error is still propagated to the framework's retry logic after `on_error()` returns
- For stateful strategies, each work item should use a separate strategy instance

**See Also:**

- [examples/example_smart_model_escalation.py](../examples/example_smart_model_escalation.py) - Complete smart model escalation example
- [examples/example_gemini_smart_retry.py](../examples/example_gemini_smart_retry.py) - Smart retry with partial parsing

#### `async def cleanup() -> None`

Clean up resources after all attempts complete (e.g., delete caches, close clients).

**Default:** No-op

**Custom Strategy Example:**

```python
from batch_llm import LLMCallStrategy, TokenUsage

class MyCustomStrategy(LLMCallStrategy[str]):
    async def execute(
        self, prompt: str, attempt: int, timeout: float
    ) -> tuple[str, TokenUsage]:
        # Your custom LLM API call
        response = await my_llm_api.generate(prompt)

        tokens: TokenUsage = {
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

- `model` (str): Model name (e.g., "gemini-2.5-flash")
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
    model="gemini-2.5-flash",
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
    model="gemini-2.5-flash",
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
    progress_callback_timeout: float | None = 5.0
    enable_detailed_logging: bool = False
    max_queue_size: int = 0
    dry_run: bool = False
```

**Fields:**

- `max_workers` (int): Maximum number of concurrent workers. Default: 5
- `timeout_per_item` (float): Timeout per item in seconds (includes retries). Default: 120.0
- `retry` ([RetryConfig](#retryconfig)): Retry configuration
- `rate_limit` ([RateLimitConfig](#ratelimitconfig)): Rate limit handling configuration
- `progress_interval` (int): Log progress every N items. Default: 10
- `progress_callback_timeout` (float | None): Max seconds to wait for progress callback. Default: 5.0. Set to `None` for no timeout.
- `enable_detailed_logging` (bool): Enable detailed debug logging. Default: False
- `max_queue_size` (int): Max queue size (0 = unlimited). Default: 0
- `dry_run` (bool): Skip actual API calls, use mock data from `strategy.dry_run()`. Default: False

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
    is_rate_limit: bool
    is_timeout: bool
    error_category: str
    suggested_wait: float | None = None
```

**Fields:**

- `is_retryable` (bool): Whether the error should trigger a retry
- `is_rate_limit` (bool): Whether this is a rate limit error (429, resource_exhausted, etc.)
- `is_timeout` (bool): Whether this is a timeout error (framework or API timeout)
- `error_category` (str): Error category for logging/metrics. Common values:
  - `"framework_timeout"` - Framework timeout (exceeded `timeout_per_item`)
  - `"api_timeout"` - API-level timeout
  - `"rate_limit"` - Rate limit error
  - `"validation_error"` - Pydantic validation error
  - `"client_error"` - 4xx client error
  - `"server_error"` - 5xx server error
  - `"connection_error"` - Network connection error
  - `"unknown"` - Unclassified error
- `suggested_wait` (float | None): Suggested wait time before retry (seconds). Used for rate limits.

**Example:**

```python
from batch_llm import ErrorInfo

# Rate limit error
rate_limit_info = ErrorInfo(
    is_retryable=False,  # Don't retry via exponential backoff
    is_rate_limit=True,  # Trigger rate limit cooldown
    is_timeout=False,
    error_category="rate_limit",
    suggested_wait=300.0,  # 5 minute cooldown
)

# Framework timeout (retryable, might succeed if faster)
timeout_info = ErrorInfo(
    is_retryable=True,
    is_rate_limit=False,
    is_timeout=True,
    error_category="framework_timeout",
)
```

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

## Core Types

### TokenUsage

TypedDict for token usage statistics from LLM API calls.

```python
class TokenUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
```

**Fields (all optional):**

- `input_tokens` (int): Number of tokens in the input/prompt
- `output_tokens` (int): Number of tokens in the output/completion
- `total_tokens` (int): Total tokens used (input + output)
- `cached_input_tokens` (int): Number of input tokens served from cache (Gemini context caching)

**Notes:**

- All fields are optional to accommodate different provider APIs
- Different providers may return different subsets of these fields
- Use `.get()` method for safe access: `tokens.get("input_tokens", 0)`

**Example:**

```python
from batch_llm import TokenUsage

tokens: TokenUsage = {
    "input_tokens": 150,
    "output_tokens": 75,
    "total_tokens": 225,
}

# Safe access
input_tokens = tokens.get("input_tokens", 0)

# Gemini with caching
gemini_tokens: TokenUsage = {
    "input_tokens": 50,  # New tokens only
    "output_tokens": 75,
    "total_tokens": 125,
    "cached_input_tokens": 1000,  # Tokens served from cache
}
```

---

### FrameworkTimeoutError

Exception raised when framework-level timeout is exceeded.

```python
class FrameworkTimeoutError(TimeoutError):
    """
    Timeout enforced by the batch-llm framework (asyncio.wait_for).

    This distinguishes framework-level timeouts from API-level timeouts.
    Framework timeouts indicate the configured timeout_per_item was exceeded,
    whereas API timeouts indicate the LLM provider returned a timeout error.
    """
```

**Purpose:**

Differentiates between:

- **Framework timeout**: `asyncio.wait_for()` timed out (exceeded `timeout_per_item`)
- **API timeout**: LLM provider returned timeout error (network issue, slow response)

**Error Classification:**

- `is_retryable`: `True` (might succeed if LLM is faster on retry)
- `is_timeout`: `True`
- `error_category`: `"framework_timeout"`

**When to increase `timeout_per_item`:**

If you see frequent `FrameworkTimeoutError`, it indicates:

1. LLM calls are taking longer than configured timeout
2. Retry delays don't fit within timeout window
3. Solution: Increase `timeout_per_item` or reduce retry configuration

**Example:**

```python
from batch_llm import FrameworkTimeoutError

try:
    result = await processor.process_all()
except FrameworkTimeoutError as e:
    print(f"Framework timeout: {e}")
    print("Consider increasing timeout_per_item in config")

# Or check in results
for item_result in result.results:
    if not item_result.success and "FrameworkTimeoutError" in item_result.error:
        print(f"{item_result.item_id} exceeded timeout")
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
- [MIGRATION_V3.md](./MIGRATION_V3.md) - Migration guide from v0.0.x
- [GEMINI_INTEGRATION.md](./GEMINI_INTEGRATION.md) - Detailed Gemini integration guide
- [CHANGELOG.md](../CHANGELOG.md) - Version history
