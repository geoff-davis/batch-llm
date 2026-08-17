# async-batch-llm API Reference

Complete API documentation for async-batch-llm v0.17.0.

## Table of Contents

- [Core Classes](#core-classes)
  - [LLMWorkItem](#llmworkitem)
  - [WorkItemResult](#workitemresult)
  - [BatchResult](#batchresult)
  - [ParallelBatchProcessor](#parallelbatchprocessor)
- [LLM Strategies](#llm-strategies)
  - [llm() factory](#llm-factory)
  - [LLMCallStrategy (Abstract)](#llmcallstrategy)
  - [PydanticAIStrategy](#pydanticaistrategy)
  - [GeminiStrategy](#geministrategy)
  - [GeminiCachedModel](#geminicachedmodel)
  - [Structured JSON Parsing](#structured-json-parsing)
- [Configuration](#configuration)
  - [ProcessorConfig](#processorconfig)
  - [RetryConfig](#retryconfig)
  - [RateLimitConfig](#ratelimitconfig)
- [Error Handling](#error-handling)
  - [ErrorClassifier](#errorclassifier)
  - [ErrorInfo](#errorinfo)
  - [RateLimitStrategy](#ratelimitstrategy)
- [Middleware and Observers](#middleware-and-observers)
  - [Middleware](#middleware)
  - [ProcessorObserver](#processorobserver)
  - [MetricsObserver](#metricsobserver)
- [Core Types](#core-types)
  - [TokenUsage](#tokenusage)
  - [FrameworkTimeoutError](#frameworktimeouterror)
  - [TokenTrackingError](#tokentrackingerror)
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
from async_batch_llm import LLMWorkItem, PydanticAIStrategy
from pydantic_ai import Agent

agent = Agent("openai:gpt-4", output_type=MyOutput)
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
    token_usage: TokenUsage = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    metadata: dict[str, Any] | None = None
    exception: Exception | None = None  # original exception on failure
    admission_wait_seconds: float = 0.0
    timing: WorkItemTiming = field(default_factory=WorkItemTiming)
    gemini_safety_ratings: dict[str, str] | None = None  # deprecated — warns on read
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
- `metadata` (dict[str, Any] | None): Provider metadata (provider name,
  finish reason, safety ratings, ...) forwarded from the strategy
- `exception` (Exception | None): The original exception when the item failed
  (what `call()` / `LLMCallPool.submit()` re-raise); None on success
- `admission_wait_seconds` (float): Cumulative provider-capacity wait across all
  attempts. This wait occurs before `attempt_timeout` starts.
- `quota_wait_seconds` (float, property): Cumulative RPM/TPM wait across all
  physical attempts. Kept separate from provider-capacity admission.
- `timing` (WorkItemTiming): Total wall time and typed per-try timing for
  admission, startup ramp, execution, provider calls where available, cooldown,
  retry backoff, error classification, and timeout category. Each
  `AttemptTiming` also carries `quota_wait_seconds`, estimated input/output,
  `reserved_tokens`, `reported_tokens` (`None` means unknown; `0` is known
  zero), `reconciliation_delta_tokens`, and run-local `quota_scope_id`.
- `structured_output_recovered` (bool): Typed metadata view indicating that
  conservative JSON recovery succeeded.
- `structured_output_recovery_reason` (str | None): Recovery reason, currently
  `trailing_markdown_fence`, or None.
- `structured_output_retries_avoided` (int): Estimated validation retries
  avoided by recovery; 0 when no recovery occurred.
- `gemini_safety_ratings` (dict[str, str] | None): **Deprecated.** Reading
  it emits a `DeprecationWarning`; use `result.metadata["safety_ratings"]`
  instead

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
    # Derived summary fields (init=False) — computed from `results` in
    # __post_init__; they cannot be passed to the constructor.
    total_items: int
    succeeded: int
    failed: int
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    termination: BatchTermination
    wall_time_seconds: float | None
```

**Fields:**

- `results` (list[WorkItemResult]): List of individual work item results
- `total_items` (int): Total number of items processed
- `succeeded` (int): Number of successful items
- `failed` (int): Number of failed items
- `total_input_tokens` (int): Sum of input tokens across all items
- `total_output_tokens` (int): Sum of output tokens across all items
- `total_cached_tokens` (int): Sum of cached input tokens from Gemini context caching
- `termination` (BatchTermination): Why the batch stopped (`completed`,
  `batch_timeout`, `fail_fast`, `artifact_error`)
- `wall_time_seconds` (float | None): Wall-clock duration of the run, stamped
  by `process_all()` / `process_prompts()`. `None` for hand-assembled batches
  and records serialized before v0.20. (v0.20.0)

**Note:** Only `results`, `termination`, and `wall_time_seconds` are
constructor arguments. The summary fields are `init=False` and calculated
automatically in `__post_init__` — construct with `BatchResult(results=[...])`.

**Accessors:**

- `outputs(with_ids=False)` — iterator over successful outputs, or
  `(item_id, output)` pairs with `with_ids=True` (v0.20.0)
- `successes` / `failures` (properties) — lists of successful / failed results
- `summary()` — printable plain-text post-run report (v0.20.0)
- `by_id()`, `in_input_order()`, `cache_hit_rate`,
  `effective_input_tokens()`, `estimated_cost()` — see below
- `to_dict()`/`from_dict()`, `to_json()`/`from_json()`,
  `to_jsonl()`/`from_jsonl()` — versioned round-trips; `summary()` works
  identically on restored batches

**Example:**

```python
batch = await processor.process_all()

print(batch.summary())
for item_id, output in batch.outputs(with_ids=True):
    save(item_id, output)
for failure in batch.failures:
    log_failure(failure.item_id, failure.error, failure.error_category)
```

`print(batch.summary())` produces a complete post-run report:

```text
Batch summary
=============
Items:     1000 total — 993 succeeded, 7 failed (250 replayed from artifact)
Stopped:   completed
Retries:   12 extra attempt(s) across 9 item(s)
Tokens:    in 812,440 (cached 96,510) · out 145,220
Replayed:  in 268,000 (cached 0) · out 47,800 (prior run; excluded above)
Wall time: 184s
  admission wait  p50 0.02s  p95 0.41s  p99 1.20s
  execution       p50 1.10s  p95 3.60s  p99 7.90s
Failures by category:
  rate_limit      4 — item_112, item_363, item_364...
  validation      3 — item_87, item_450, item_612
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
        max_workers: int | None = None,          # deprecated, use config
        post_processor: PostProcessorFunc[TOutput, TContext] | None = None,
        attempt_timeout: float | None = None,   # deprecated, use config
        rate_limit_cooldown: float | None = None,  # deprecated, use config
        config: ProcessorConfig | None = None,
        error_classifier: ErrorClassifier | None = None,
        rate_limit_strategy: RateLimitStrategy | None = None,
        middlewares: list[Middleware] | None = None,
        observers: list[ProcessorObserver] | None = None,
        progress_callback: ProgressCallbackFunc | None = None,
    )
```

Pass everything by keyword — the first positional parameter is the deprecated
`max_workers`, not `config`.

**Parameters:**

- `config` (ProcessorConfig | None): Configuration for the processor (recommended)
- `post_processor` (PostProcessorFunc | None): Optional sync or async function called after each item;
  synchronous callbacks run in a worker thread
- `progress_callback` (ProgressCallbackFunc | None): Optional callback for progress updates
- `error_classifier` (ErrorClassifier | None): Custom error classifier. Default: auto-selected
  from the work items' strategies (e.g. `GeminiStrategy` → `GeminiErrorClassifier`), falling
  back to `DefaultErrorClassifier()` when there is no recommendation or providers conflict
- `rate_limit_strategy` (RateLimitStrategy | None): Custom rate limit handling. Default: `ExponentialBackoffStrategy()`
- `middlewares` (list[Middleware] | None): List of middleware for pre/post processing
- `observers` (list[ProcessorObserver] | None): List of observers for monitoring events
- `max_workers`, `attempt_timeout`, `rate_limit_cooldown`: deprecated loose parameters;
  set them on `ProcessorConfig` instead

> **Post-processing:** By default the optional `post_processor` runs inline with the worker's item lifecycle
> as soon as an item finishes. Synchronous callbacks are offloaded to a thread so they do not block the event loop.
> It should hand off any heavy operations (long DB writes, expensive analytics, etc.) to another system;
> if the function takes too long the worker sits idle until the timeout triggers
> (`ProcessorConfig.post_processor_timeout`, default 90 s), reducing overall throughput. Timing out a synchronous
> callback stops waiting for its thread; Python cannot forcibly stop code already running in that thread.

**Methods:**

#### `async def add_work(work_item: LLMWorkItem) -> None`

Add a work item to the processing queue.

```python
await processor.add_work(work_item)
```

**Note:** If `max_queue_size` is set and the queue is full, `add_work` raises `ValueError`
in batch mode. Only streaming mode (`start()`/`results()`/`finish()`) applies backpressure
by blocking until space is available.

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
from async_batch_llm import ParallelBatchProcessor, ProcessorConfig, LLMWorkItem

config = ProcessorConfig(max_workers=5, attempt_timeout=60.0)

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

### llm() factory

Build a ready-to-use built-in strategy from a `"provider:model"` string.
Added in v0.20.0.

```python
def llm(
    spec: str,
    *,
    response_parser: Callable[[LLMResponse], TOutput] | None = None,
    temperature: float | None = 0.0,
    generation_config: dict[str, Any] | None = None,
    **model_kwargs: Any,
) -> ModelStrategy[TOutput]: ...
```

**Parameters:**

- `spec` (str): `"provider:model"` with one of the prefixes `gemini:`,
  `openai:`, `openrouter:`, `deepseek:`. Everything after the first colon is
  the provider's model id (which may itself contain colons).
- `response_parser`, `temperature`, `generation_config`: Forwarded to the
  strategy (see [GeminiStrategy](#geministrategy) for their semantics).
- `**model_kwargs`: Forwarded to the model constructor — `api_key`,
  `system_instruction`, `max_connections` / `json_mode` / `extra_headers` /
  `extra_body` (OpenAI-compatible providers), `thinking` (DeepSeek),
  `safety_settings` (Gemini).

**Returns** the same objects the two-object form builds (`GeminiStrategy`,
`OpenAIStrategy`, `OpenRouterStrategy`, or `DeepSeekStrategy` wrapping the
matching model), so error-classifier auto-selection and lifecycle management
work unchanged. Unknown prefixes raise `ValueError` listing valid prefixes;
a missing optional dependency raises `ImportError` naming the install extra.

**Example:**

```python
from async_batch_llm import llm

strategy = llm("openai:gpt-4o-mini")            # reads OPENAI_API_KEY
strategy = llm("gemini:gemini-2.5-flash")       # reads GOOGLE_API_KEY
strategy = llm("deepseek:deepseek-v4-flash", thinking=False, max_connections=150)
```

Use the explicit two-object form (e.g.
`OpenAIStrategy(OpenAIModel.from_api_key(...))`) for custom clients, Gemini
cached models, or custom strategies.

---

### LLMCallStrategy

Abstract base class for LLM call strategies.

```python
class LLMCallStrategy(ABC, Generic[TOutput]):
    async def prepare(self) -> None: ...

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        attempt: int,
        timeout: float,
        state: RetryState | None = None,
    ) -> tuple[TOutput, TokenUsage]: ...

    async def on_error(
        self,
        exception: Exception,
        attempt: int,
        state: RetryState | None = None,
    ) -> None: ...

    async def cleanup(self) -> None: ...

    async def dry_run(self, prompt: str) -> tuple[TOutput, TokenUsage]: ...

    def estimate_tokens(
        self, prompt: str, attempt: int, state: RetryState | None
    ) -> TokenEstimate | Awaitable[TokenEstimate] | None: ...

    @property
    def concurrency_scope(self) -> object: ...

    @property
    def quota_scope(self) -> object: ...
```

**Lifecycle:**

1. `prepare()` - Called once before any retry attempts
2. For each attempt (including retries):
   - `execute()` is called (or `dry_run()` if `config.dry_run=True`)
   - If `execute()` raises an exception, `on_error()` is called before retry logic
3. `cleanup()` - Called once after all attempts complete

**Methods:**

#### `estimate_tokens(prompt, attempt, state)`

Optional per-strategy TPM estimator. It runs after middleware and coordinated
cooldown but before the atomic quota reservation and provider-capacity wait.
It may be synchronous or asynchronous and receives retry state so model
escalation can change the expected output. `ProcessorConfig.token_estimator`
takes precedence. The default returns `None`.

`quota_scope` identifies strategies sharing RPM, TPM, and cooldown by object
identity. It defaults to `concurrency_scope`, which identifies shared provider
capacity. Override them independently when account quota and client capacity
have different ownership.

#### `async def prepare() -> None`

Initialize resources before making LLM calls (e.g., create caches, initialize clients).

**Default:** No-op

#### `async def execute(prompt: str, attempt: int, timeout: float, state: RetryState | None = None) -> tuple[TOutput, TokenUsage]`

Execute an LLM call.

**Parameters:**

- `prompt` (str): The prompt to send to the LLM
- `attempt` (int): Which retry attempt this is (1, 2, 3, ...)
- `timeout` (float): Maximum time to wait for response (seconds)
  - Note: Timeout enforcement is handled by the framework wrapping this call in `asyncio.wait_for()`
- `state` (RetryState | None): Mutable per-work-item state provided by the framework
  so strategies can track partial progress across retries

**Returns:** Tuple of `(output, token_usage)`

- `output` (TOutput): The LLM response
- `token_usage` ([TokenUsage](#tokenusage)): Token usage dict with optional keys: `input_tokens`,
  `output_tokens`, `total_tokens`, `cached_input_tokens`

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

#### `async def on_error(exception: Exception, attempt: int, state: RetryState | None = None) -> None`

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
- `state` (RetryState | None): Retry state that persists across attempts (v0.3.0)

**Default:** No-op

**Use Cases:**

1. **Smart Model Escalation** - Only escalate to expensive models on validation errors, not
   network errors:

   ```python
   class SmartModelEscalationStrategy(LLMCallStrategy[Output]):
       async def on_error(self, exception: Exception, attempt: int, state=None) -> None:
           if state is not None and isinstance(exception, ValidationError):
               state.set("validation_failures", state.get("validation_failures", 0) + 1)

       async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
           # Only escalate model on validation errors
           failures = state.get("validation_failures", 0) if state is not None else 0
           model_index = min(failures, len(MODELS) - 1)
           model = MODELS[model_index]
           # Make call with appropriate model...
   ```

1. **Smart Retry with Partial Parsing** - Build better retry prompts based on what failed:

   ```python
   class SmartRetryStrategy(LLMCallStrategy[Output]):
       async def on_error(self, exception: Exception, attempt: int, state=None) -> None:
           if state is not None and isinstance(exception, ValidationError):
               state.set("last_validation_error", exception)
               # last_response set in execute() before raising

       async def execute(self, prompt: str, attempt: int, timeout: float, state=None):
           if attempt > 1 and state and state.get("last_validation_error"):
               # Build smart retry prompt with partial parsing feedback
               prompt = self._create_retry_prompt_with_partial_data(prompt, state)
           # Make call with improved prompt...
   ```

1. **Error Type Tracking** - Distinguish between different error types:

   ```python
   class ErrorTrackingStrategy(LLMCallStrategy[Output]):
       async def on_error(self, exception: Exception, attempt: int, state=None) -> None:
           if state is None:
               return
           if isinstance(exception, ValidationError):
               key = "validation_errors"
           elif isinstance(exception, ConnectionError):
               key = "network_errors"
           elif "429" in str(exception):
               key = "rate_limit_errors"
           else:
               key = "other_errors"
           state.set(key, state.get(key, 0) + 1)
   ```

**Important Notes:**

- Exceptions in `on_error()` are caught and logged by the framework - they won't crash processing
- `on_error()` is only called when `execute()` raises an exception, not on success
- The error is still propagated to the framework's retry logic after `on_error()` returns
- Share strategy/client instances across work items, and keep all item-specific mutation in
  the supplied `RetryState`. Use concurrency-safe metrics or observers for batch-wide counters.

**See Also:**

- [examples/example_smart_model_escalation.py][ex-escalation] - Complete
  smart model escalation example
- [examples/example_gemini_smart_retry.py][ex-smart-retry] - Smart retry with
  partial parsing

[ex-escalation]: https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_smart_model_escalation.py
[ex-smart-retry]: https://github.com/geoff-davis/async-batch-llm/blob/main/examples/example_gemini_smart_retry.py

#### `async def cleanup() -> None`

Clean up resources after all attempts complete (e.g., delete caches, close clients).

**Default:** No-op

**Custom Strategy Example:**

```python
from async_batch_llm import LLMCallStrategy, TokenUsage

class MyCustomStrategy(LLMCallStrategy[str]):
    async def execute(
        self, prompt: str, attempt: int, timeout: float, state=None
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

**Requires:** `pip install 'async-batch-llm[pydantic-ai]'`

**Example:**

```python
from async_batch_llm import PydanticAIStrategy, LLMWorkItem
from pydantic_ai import Agent
from pydantic import BaseModel

class BookSummary(BaseModel):
    title: str
    summary: str

agent = Agent("openai:gpt-4", output_type=BookSummary)
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
        model: LLMModel,
        response_parser: Callable[[LLMResponse], TOutput] | None = None,
        *,
        temperature: float | None = 0.0,
        generation_config: dict[str, Any] | None = None,
    )
```

**Parameters:**

- `model` (LLMModel): Model wrapper such as `GeminiModel` or `GeminiCachedModel`
- `response_parser` (Callable | None): Function to parse `LLMResponse` into `TOutput`.
  Defaults to returning `response.text`.
- `temperature` (float | None): Sampling temperature. Default: 0.0. Pass `None` to
  omit the parameter and use the provider default.
- `generation_config` (dict | None): Extra provider config merged into each
  `generate()` call (e.g. tools, `response_mime_type`, logprobs). Also available
  on `OpenAIStrategy`/`OpenRouterStrategy`/`DeepSeekStrategy` (shared
  `ModelStrategy` base).

**Requires:** `pip install 'async-batch-llm[gemini]'`

> **API key:** Set `GOOGLE_API_KEY` (preferred) or the legacy `GEMINI_API_KEY` environment
> variable before running this example.

**Example:**

```python
from async_batch_llm import GeminiModel, GeminiStrategy, LLMWorkItem
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
model = GeminiModel("gemini-2.5-flash", client)

strategy = GeminiStrategy(
    model=model,
)

work_item = LLMWorkItem(
    item_id="task_1",
    strategy=strategy,
    prompt="Explain quantum computing"
)
```

---

### GeminiCachedModel

Model for calling Google Gemini with context caching. Since v0.6.0, caching is
a property of the **model**, not the strategy — wrap a `GeminiCachedModel` in
the ordinary `GeminiStrategy`. (The old `GeminiCachedStrategy` was removed; the
model now owns the cache find/create/renew/delete lifecycle.)

```python
class GeminiCachedModel:
    def __init__(
        self,
        model: str,
        client: genai.Client,
        cached_content: list[Content],
        *,
        cache_ttl_seconds: int = 3600,
        cache_renewal_buffer_seconds: int = 300,
        auto_renew: bool = True,
        cache_tags: dict[str, str] | None = None,
        safety_settings: list[dict[str, Any]] | None = None,
        metadata_extractors: list[MetadataExtractor] | None = None,
    )
```

**Parameters:**

- `model` (str): Model name
- `client` (genai.Client): Initialized Gemini client
- `cached_content` (list[Content]): Content to cache (system instructions, documents)
- `cache_ttl_seconds` (int): Cache TTL in seconds. Default: 3600 (1 hour)
- `cache_renewal_buffer_seconds` (int): Renew caches this many seconds before expiry (default 300)
- `auto_renew` (bool): Automatically renew caches when they near expiry. Default: True
- `cache_tags` (dict[str, str] | None): Optional metadata for precise cache matching/versioning
  (encoded into the cache's `display_name`)
- `safety_settings` (list[dict] | None): Optional default safety settings for all calls

**Lifecycle** (driven by the framework via `GeminiStrategy`):

- `prepare()`: Finds or creates the Gemini cache (once per shared instance)
- `generate()`: Uses the cache and auto-renews when enabled
- `cleanup()`: Runs once when the processor exits; by default caches are left alive so
  future batches can reuse them (call `delete_cache()` to remove immediately)

**Requires:** `pip install 'async-batch-llm[gemini]'`

> **API key:** Same as above – `GOOGLE_API_KEY` is preferred, `GEMINI_API_KEY` also works.
>
> **Share one instance.** Create a single `GeminiCachedModel` and reuse it
> across every work item that should share the cached context. Constructing a
> new instance per item defeats caching entirely and can cost ~10x more.

**Example:**

```python
from async_batch_llm import GeminiCachedModel, GeminiStrategy, LLMWorkItem
from google import genai
from google.genai.types import Content

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

# Large document to cache
cached_content = [
    Content(role="system", parts=[{"text": "You are a helpful assistant."}]),
    Content(role="user", parts=[{"text": large_document}]),
]

cached_model = GeminiCachedModel(
    "gemini-2.5-flash",
    client,
    cached_content=cached_content,
    cache_ttl_seconds=3600,
)
strategy = GeminiStrategy(cached_model, response_parser=lambda r: r.text)

# Reuse the same strategy/model across work items to benefit from caching
for i in range(100):
    work_item = LLMWorkItem(
        item_id=f"task_{i}",
        strategy=strategy,  # Same strategy, shared cache
        prompt=f"Question {i} about the document"
    )
    await processor.add_work(work_item)
```

---

### Structured JSON Parsing

`pydantic_json_parser(Model)` strips a normal outer Markdown fence and performs
strict Pydantic JSON/schema validation. Recovery is disabled by default.

```python
parser = pydantic_json_parser(
    Classification,
    recover_trailing_markdown=True,
)
strategy = OpenAIStrategy(model, parser)
```

With recovery enabled, strict parsing still runs first. On failure, the parser
uses a real JSON decoder to read exactly one complete top-level object or array,
accepts only the explicitly supported trailing closing-fence artifacts (three
backticks, with or without the observed trailing underscore), and then runs
normal Pydantic schema validation. It does not repair malformed JSON or accept
scalars, multiple JSON values, arbitrary prose, or schema-invalid data. Those
failures continue through the configured retry policy.

A recovered `WorkItemResult` exposes typed recovery properties backed by
metadata. Processor stats and `MetricsObserver` include
`structured_output_recoveries`, `structured_output_retries_avoided`, and counts
by `structured_output_recovery_reasons`.

---

## Configuration

### ProcessorConfig

Complete configuration for batch processor.

```python
@dataclass
class ProcessorConfig:
    max_workers: int = 5
    attempt_timeout: float = 120.0
    post_processor_timeout: float = 90.0
    concurrent_post_processing: bool = False
    retry: RetryConfig = field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    progress_interval: int = 10
    progress_callback_timeout: float | None = 5.0
    enable_detailed_logging: bool = False
    max_queue_size: int = 0
    max_requests_per_minute: float | None = None
    max_tokens_per_minute: int | None = None
    token_estimator: TokenEstimator | None = None
    dry_run: bool = False
    max_provider_concurrency: int | None = None
    startup_ramp: StartupRampConfig | None = None
    concurrency: int | None = None
    max_result_queue_size: int = 0
    progress_refresh_interval_seconds: float = 0.1
```

**Fields:**

- `concurrency` (int | None): **The single concurrency knob** (v0.20.0). When
  set, coherently sizes every alignment-sensitive limit that is not
  explicitly configured: `max_workers`, `max_provider_concurrency`, and the
  httpx connection pool on built-in OpenAI-compatible models built via
  `llm()` / `from_api_key` without an explicit `max_connections` (resized by
  the processor before the first request). Explicit values for any individual
  knob override the derived ones; the capacity warning fires only on a real
  contradiction (an explicit client capacity smaller than the requested
  concurrency), never on an override. Also available as shorthand:
  `process_prompts(..., concurrency=N)` / `process_stream(..., concurrency=N)`.
- `max_workers` (int): Maximum number of concurrent workers. Default: 5
  (or `concurrency` when that is set and `max_workers` is not).
- `max_provider_concurrency` (int | None): Optional provider/client concurrency
  limit applied before `strategy.execute()` and outside `attempt_timeout`.
  When the strategy also advertises `max_concurrency`, the lower limit applies.
  Defaults to `concurrency` when that is set.
- `startup_ramp` (StartupRampConfig | None): Optional initial concurrency ramp.
  Ramp wait is admission time outside `attempt_timeout`. Default: None.
- `attempt_timeout` (float): Timeout applied to each `execute()` attempt in seconds (per-attempt, not a
  total budget across retries). Default: 120.0
- `post_processor_timeout` (float): Max seconds to wait for the sync or async `post_processor`
  callback per item. Default: 90.0
- `concurrent_post_processing` (bool): Run post-processors as tracked background tasks, bounded by
  `max_workers`. Default: False
- `retry` ([RetryConfig](#retryconfig)): Retry configuration
- `rate_limit` ([RateLimitConfig](#ratelimitconfig)): Rate limit handling configuration
- `progress_interval` (int): Log progress every N items. Default: 10
- `progress_callback_timeout` (float | None): Max seconds to wait for progress callback. Default: 5.0.
  Set to `None` for no timeout.
- `progress_refresh_interval_seconds` (float): Minimum seconds between renders
  for the bundled `progress=True` reporter. Custom callbacks remain per-item.
  Default: 0.1.
- `enable_detailed_logging` (bool): Enable detailed debug logging. Default: False
- `max_queue_size` (int): Max queue size (0 = unlimited). Default: 0
- `max_result_queue_size` (int): Completed results waiting for a streaming
  consumer (0 = unlimited). Default: 0
- `max_requests_per_minute` (float | None): Optional proactive rate limiter that throttles
  physical attempts per quota scope before hitting provider limits.
- `max_tokens_per_minute` (int | None): Optional proactive token budget per
  quota scope. Positive integers only. Enabling it requires an estimator for
  every live attempt.
- `token_estimator` (`TokenEstimator | None`): Run-level sync or async estimator;
  takes precedence over the effective strategy's `estimate_tokens()` hook.
  Estimation runs after middleware and cooldown and before quota/provider
  capacity. Default: None.
- `dry_run` (bool): Skip actual API calls, use mock data from `strategy.dry_run()`. Default: False

**Example:**

```python
from async_batch_llm import ProcessorConfig, RetryConfig

config = ProcessorConfig(
    max_workers=10,
    attempt_timeout=60.0,
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
    max_rate_limit_retries: int = 20
```

**Fields:**

- `max_attempts` (int): Maximum attempts for *content/transport* failures
  (validation, timeout, connection, 5xx). Default: 3
- `initial_wait` (float): Initial wait time in seconds. Default: 1.0
- `max_wait` (float): Maximum wait time in seconds. Default: 60.0
- `exponential_base` (float): Exponential backoff base. Default: 2.0
- `jitter` (bool): Add random jitter to wait times. Default: True
- `max_rate_limit_retries` (int): Maximum times an item may be retried after a
  rate-limit/cooldown **without** consuming its `max_attempts` budget. Rate
  limits are retried at the same logical attempt number; exceeding this fails
  the item with a `RateLimitRetriesExceeded` error. `0` makes rate limits fail
  immediately. Default: 20

**Validation:**

- `max_attempts` must be >= 1
- `initial_wait` must be > 0
- `max_wait` must be >= initial_wait
- `exponential_base` must be >= 1
- `max_rate_limit_retries` must be >= 0

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
    max_cooldown_seconds: float = 600.0
```

**Fields:**

- `cooldown_seconds` (float): Cooldown after rate limit. Default: 300.0 (5 minutes)
- `slow_start_items` (int): Number of items for slow start. Default: 50
- `slow_start_initial_delay` (float): Initial delay in slow start. Default: 2.0
- `slow_start_final_delay` (float): Final delay in slow start. Default: 0.1
- `backoff_multiplier` (float): Increase cooldown on repeated rate limits. Default: 1.5
- `max_cooldown_seconds` (float): Cap on the escalated cooldown (v0.16). Default: 600.0

**Validation:**

- `cooldown_seconds` must be >= 0
- `slow_start_items` must be >= 0
- `slow_start_initial_delay` must be >= slow_start_final_delay
- `backoff_multiplier` must be >= 1.0

---

### StartupRampConfig

Optional cold-start concurrency ramp. This is distinct from
`RateLimitConfig.slow_start_*`, which applies only after a rate-limit cooldown.

```python
@dataclass
class StartupRampConfig:
    initial_concurrency: int = 1
    concurrency_step: int = 1
    ramp_interval_seconds: float = 1.0
    max_concurrency: int | None = None
    jitter_seconds: float = 0.0
```

The allowed concurrency begins at `initial_concurrency` and adds
`concurrency_step` after each interval. The effective maximum is the lowest of
the ramp maximum, explicit `max_provider_concurrency`, advertised model capacity,
and host worker limit. Optional jitter spreads cold-start admissions. All ramp
wait occurs before `attempt_timeout`.

---

## Error Handling

### Token estimation errors

Token admission uses three public, framework-owned, non-retryable errors:

- `TokenEstimationError` (`token_estimation_error`): an estimator raised,
  returned an invalid value, or otherwise could not produce a safe estimate.
  The underlying exception is redacted from item-facing text.
- `TokenEstimatorRequired` (`token_estimator_required`): TPM is enabled but no
  config or strategy estimator resolved.
- `TokenEstimateExceedsLimit` (`token_estimate_exceeds_limit`): one estimate is
  larger than the configured TPM bucket and cannot become admissible by
  waiting.

These errors occur before provider start, bypass strategy/middleware error
recovery, and consume no quota or provider capacity.

---

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
from async_batch_llm import ErrorClassifier, ErrorInfo

class MyErrorClassifier(ErrorClassifier):
    def classify(self, exception: Exception) -> ErrorInfo:
        error_str = str(exception).lower()

        if "rate limit" in error_str:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=True,
                is_timeout=False,
                error_category="rate_limit",
            )
        elif "timeout" in error_str:
            return ErrorInfo(
                is_retryable=True,
                is_rate_limit=False,
                is_timeout=True,
                error_category="api_timeout",
            )
        else:
            return ErrorInfo(
                is_retryable=False,
                is_rate_limit=False,
                is_timeout=False,
                error_category="unknown",
            )
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
    hint: str | None = None
```

**Fields:**

- `is_retryable` (bool): Whether the error should trigger a retry
- `is_rate_limit` (bool): Whether this is a rate limit error (429, resource_exhausted, etc.)
- `is_timeout` (bool): Whether this is a timeout error (framework or API timeout)
- `error_category` (str): Error category for logging/metrics. Common values:
  - `"framework_timeout"` - Framework timeout (exceeded `attempt_timeout`)
  - `"api_timeout"` - API-level timeout
  - `"rate_limit"` - Rate limit error
  - `"validation_error"` - Pydantic validation error
  - `"insufficient_balance"` - 402 Payment Required / balance exhausted (non-retryable)
  - `"client_error"` - 4xx client error
  - `"server_error"` - 5xx server error
  - `"connection_error"` - Network connection error
  - `"unknown"` - Unclassified error
- `suggested_wait` (float | None): Suggested wait time before retry (seconds). Used for rate limits.
- `hint` (str | None): Optional operator-facing remediation hint, surfaced in
  the logs at WARNING when a non-retryable error gives up (e.g. the 402
  "top up your prepaid balance" guidance). `None` means no extra guidance.

**Example:**

```python
from async_batch_llm import ErrorInfo

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
    async def on_rate_limit(
        self, worker_id: int, consecutive_limit_count: int
    ) -> float:
        """Called when a rate limit is detected. Returns the cooldown in seconds."""

    @abstractmethod
    def should_apply_slow_start(
        self, items_since_resume: int
    ) -> tuple[bool, float]:
        """Whether to delay the next item after a cooldown, and by how much."""
```

**Built-in Implementations:**

- `ExponentialBackoffStrategy`: Exponential backoff with configurable parameters
- `FixedDelayStrategy`: Fixed delay between retries

---

## Middleware and Observers

### Middleware

Interface for middleware that can modify work items before/after processing.

```python
class Middleware(ABC):
    async def before_process(
        self, work_item: LLMWorkItem
    ) -> LLMWorkItem | None: ...

    async def after_process(
        self, result: WorkItemResult
    ) -> WorkItemResult: ...

    async def on_error(
        self, work_item: LLMWorkItem, error: Exception
    ) -> WorkItemResult | None: ...
```

**Methods:**

- `before_process()`: Modify work item before processing. Return `None` to skip
  the item (it is recorded as failed).
- `after_process()`: Modify the result after processing (takes only the result).
- `on_error()`: Handle errors. Return a `WorkItemResult` to substitute it for the
  error (the first middleware returning non-None wins), or `None` for default
  error handling.

All three are abstract — subclass `BaseMiddleware` for no-op defaults so you
only override the hooks you need.

**Example:**

```python
from async_batch_llm.middleware import BaseMiddleware

class LoggingMiddleware(BaseMiddleware):
    async def before_process(self, work_item):
        print(f"Processing {work_item.item_id}")
        return work_item

    async def after_process(self, result):
        print(f"Completed {result.item_id}: {result.success}")
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

- `BATCH_STARTED`: `{total, max_workers, start_time}`
- `BATCH_COMPLETED`: `{processed, succeeded, failed, total, total_tokens,
  cached_input_tokens, total_admission_wait_seconds,
  max_admission_wait_seconds, total_quota_wait_seconds,
  max_quota_wait_seconds, quota_wait_p50/p95/p99_seconds,
  estimated_input_tokens, estimated_output_tokens, reserved_tokens,
  reported_reconciliation_tokens, refunded_tokens, underestimated_tokens,
  unknown_usage_attempts, known_zero_usage_attempts,
  token_estimation_failures, quota_scope_count,
  admission_wait_p50/p95/p99_seconds,
  execution_p50/p95/p99_seconds, structured_output_recoveries,
  structured_output_retries_avoided, structured_output_recovery_reasons,
  duration}`
- `WORKER_STARTED` / `WORKER_STOPPED`: `{worker_id}`
- `ITEM_STARTED`: `{item_id, worker_id}`
- `ITEM_ADMITTED`: `{item_id, worker_id, attempt, wait_seconds, capacity,
  startup_ramp_wait_seconds}`
- `QUOTA_ADMITTED`: `{item_id, worker_id, attempt, try_number,
  quota_scope_id, wait_seconds, request_reserved, estimated_input_tokens,
  estimated_output_tokens, estimated_total_tokens, reserved_tokens,
  rpm_configured, tpm_configured, limited_by}`
- `QUOTA_RECONCILED`: `{item_id, worker_id, attempt, try_number,
  quota_scope_id, reserved_tokens, reported_tokens, known_usage,
  delta_tokens, disposition}`
- `ITEM_COMPLETED`: `{item_id, duration, tokens, admission_wait_seconds,
  structured_output_recovered, structured_output_recovery_reason,
  structured_output_retries_avoided}`
- `ITEM_FAILED`: `{item_id, error_type}`
- `RATE_LIMIT_HIT`: `{item_id, worker_id}`
- `COOLDOWN_STARTED`: `{worker_id, duration, consecutive}`
- `COOLDOWN_ENDED`: `{duration, error?}`

**Cleanup note:**

- Preferred: wrap `ParallelBatchProcessor` in `async with` so strategy cleanup runs automatically.
- If you do not use a context manager, call `await processor.shutdown()` after `process_all()` to flush
  observers, stop workers, and run strategy cleanups.

---

### MetricsObserver

Built-in observer for collecting metrics.

In addition to item, error, cooldown, and processing-time metrics, it reports
`admission_wait_count`, `admission_wait_seconds_sum`,
`admission_wait_seconds_max`, `avg_admission_wait_seconds`,
`structured_output_recoveries`, `structured_output_retries_avoided`, and
`structured_output_recovery_reasons`, `quota_admitted_attempts`,
`quota_wait_seconds_sum`, `quota_wait_seconds_max`, `reserved_tokens`,
`reported_reconciliation_tokens`, `refunded_tokens`,
`underestimated_tokens`, `unknown_usage_attempts`, and `quota_scope_count`.
Scope ordinals and item IDs are never metric labels.

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
from async_batch_llm import MetricsObserver

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

### TokenEstimate and TokenEstimator

`TokenEstimate` is an immutable estimate for one physical attempt:

```python
@dataclass(frozen=True)
class TokenEstimate:
    input_tokens: int
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int: ...
```

Both fields are non-negative integers (booleans are rejected). A
`TokenEstimator` is a sync or async callable with signature
`(prompt, *, strategy, attempt, state) -> TokenEstimate`. The explicit
`CharacterTokenEstimator` implements
`max(minimum_input_tokens, ceil(len(prompt) / characters_per_token))` plus a
fixed expected output. It is approximate and never auto-enabled.

See [Token-Aware Admission](token-aware-admission.md) for full semantics.

---

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
from async_batch_llm import TokenUsage

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

### RetryState

Mutable per-work-item state that persists across retries. The framework creates a `RetryState`
instance for each `LLMWorkItem` and passes it to both `strategy.execute(...)` and
`strategy.on_error(...)` via the `state` parameter.

```python
from dataclasses import dataclass, field

@dataclass
class RetryState:
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def delete(self, key: str, raise_if_missing: bool = False) -> None: ...
    def clear(self) -> None: ...
```

**Typical uses:**

- Track validation failures to escalate models only when schema validation fails
- Store partial results so retries request only the missing fields
- Record which advanced retry prompt should be used next

**Example:**

```python
async def execute(
    self, prompt: str, attempt: int, timeout: float, state: RetryState | None = None
):
    state = state or RetryState()
    missing = state.get("missing_fields", ["name", "email"])
    response = await self.client.generate(prompt, focus=missing)
    result = parse(response)

    missing = [f for f in ALL_FIELDS if f not in result]
    if missing:
        state.set("missing_fields", missing)
        raise ValidationError("Still missing fields", result)
    state.delete("missing_fields", raise_if_missing=False)
    return result, extract_tokens(response)
```

Because the same `RetryState` instance is reused across attempts, each retry can build on the previous
attempt’s context without relying on global variables.

---

### Response metadata (`WorkItemResult.metadata`)

Provider metadata (Gemini safety ratings and finish reason, OpenRouter
provider/routed model, etc.) flows into `WorkItemResult.metadata` — a plain
`dict[str, Any] | None`. The parsed output stays in `WorkItemResult.output`;
you no longer wrap it in a separate response object.

Conservative structured-output recovery uses three reserved metadata keys:
`structured_output_recovered`, `structured_output_recovery_reason`, and
`structured_output_retries_avoided`. Read them through the corresponding typed
properties on `LLMResponse` or `WorkItemResult`.

> **Removed:** the old `GeminiResponse` wrapper and the `include_metadata`
> opt-in were removed in v0.6.0. Read metadata off `result.metadata` instead.
> For Gemini safety ratings specifically, `result.metadata["safety_ratings"]`
> carries them (the deprecated `result.gemini_safety_ratings` field is still
> backfilled for compatibility).

**Usage:**

```python
result = await processor.process_all()
first = result.results[0]
ratings = (first.metadata or {}).get("safety_ratings")
if ratings and ratings.get("HARM_CATEGORY_HATE_SPEECH") == "HIGH":
    log_flagged_content(first.output)
```

---

### Typed auxiliary output (grounding, reasoning, tool calls, logprobs)

> **Experimental.** This surface is new (v0.16.0) and hasn't seen much
> real-world use yet — the reserved-key dict shapes and the typed views may
> change in a future minor release while they stabilize. The `metadata`
> dict channel itself is stable; if you persist these shapes, read them
> back defensively.

Provider-specific structured output travels through `metadata` under four
**reserved keys** with documented plain-dict shapes (JSON-serializable, so
persisting `metadata` as-is works):

| Key | Shape | Emitted by |
| --- | ----- | ---------- |
| `grounding` | `{"sources": [{"uri", "title"}], "queries": [str], "supports": [dict]}` | Gemini models, when the response carries `google_search` grounding |
| `reasoning` | `str` — the model's reasoning/thinking trace | OpenAI-compatible models (`reasoning_content`, e.g. DeepSeek, falling back to `reasoning`, e.g. OpenRouter) |
| `tool_calls` | `[{"id": str\|None, "name": str, "arguments": str}]` — `arguments` is the raw JSON string | OpenAI-compatible models |
| `logprobs` | provider-shaped `dict`/`list` (via `model_dump()`) | OpenAI-compatible models, when requested |

Both `LLMResponse` and `WorkItemResult` expose **lazy read-only typed
views** over these keys — parsed from `metadata` on each access, never
cached, never stored twice:

```python
result = await processor.process_all()
item = result.results[0]

if item.grounding:                       # Grounding | None
    for source in item.grounding.sources:  # list[GroundingSource]
        print(source.uri, source.title)
    print(item.grounding.queries)          # list[str]

print(item.reasoning)                    # str | None
for call in item.tool_calls or []:       # list[ToolCall] | None
    print(call.name, call.arguments)     # arguments: raw JSON string
print(item.logprobs)                     # Any | None (provider-shaped)
```

The view classes (`Grounding`, `GroundingSource`, `ToolCall`) are exported
at the top level. Parsing is lenient: malformed metadata yields `None` (or
drops the bad entry) rather than raising — which also means a future shape
change degrades to `None` views rather than errors.

**Boundaries:**

- `tool_calls` is **visibility only** — the framework never executes tools.
  Feed them to your own dispatch loop (or use an agent framework via
  `PydanticAIStrategy`). Covered for OpenAI-compatible providers only this
  phase (Gemini function-call parts are not extracted yet).
- A response whose `content` is `null` (e.g. a pure tool-call turn) still
  raises `EmptyResponseError` before any result exists, so `tool_calls`
  surfaces only when the model returned text alongside the calls.
- Auxiliary output does not survive empty/safety-blocked responses — the
  call fails first.

---

### FrameworkTimeoutError

Exception raised when framework-level timeout is exceeded.

```python
class FrameworkTimeoutError(TimeoutError):
    """
    Timeout enforced by the async-batch-llm framework (asyncio.wait_for).

    This distinguishes framework-level timeouts from API-level timeouts.
    Framework timeouts indicate the configured attempt_timeout was exceeded,
    whereas API timeouts indicate the LLM provider returned a timeout error.
    """
```

**Purpose:**

Differentiates between:

- **Framework timeout**: `asyncio.wait_for()` timed out (exceeded `attempt_timeout`)
- **API timeout**: LLM provider returned timeout error (network issue, slow response)

**Error Classification:**

- `is_retryable`: `True` (might succeed if LLM is faster on retry)
- `is_timeout`: `True`
- `error_category`: `"framework_timeout"`

**When to increase `attempt_timeout`:**

If you see frequent `FrameworkTimeoutError`, it indicates:

1. LLM calls are taking longer than configured timeout
2. Retry delays don't fit within timeout window
3. Solution: Increase `attempt_timeout` or reduce retry configuration

**Example:**

```python
from async_batch_llm import FrameworkTimeoutError

try:
    result = await processor.process_all()
except FrameworkTimeoutError as e:
    print(f"Framework timeout: {e}")
    print("Consider increasing attempt_timeout in config")

# Or check in results
for item_result in result.results:
    if not item_result.success and "FrameworkTimeoutError" in item_result.error:
        print(f"{item_result.item_id} exceeded timeout")
```

---

### TokenTrackingError

Exception wrapper that preserves token usage from failed LLM calls.

```python
class TokenTrackingError(Exception):
    """
    Wrapper exception that preserves token usage from failed LLM calls.

    When an LLM call fails (e.g., validation error), we still want to track
    the tokens that were consumed. This wrapper attaches token usage to
    exceptions that don't natively support it.
    """

    def __init__(self, message: str, *, token_usage: dict[str, int] | None = None):
        super().__init__(message)
        self._failed_token_usage = token_usage or {}
```

**Purpose:**

When an LLM call succeeds in getting a response but fails during parsing/validation,
the tokens were still consumed and should be tracked for accurate cost accounting.
This wrapper preserves that token usage even when the original exception doesn't
have a `__dict__` (like built-in exceptions).

**Usage:**

Strategies use this internally to wrap exceptions that don't support attribute assignment:

```python
try:
    output = parse_response(response)
except Exception as e:
    if not hasattr(e, "__dict__"):
        wrapped = TokenTrackingError(str(e), token_usage=tokens)
        wrapped.__cause__ = e
        raise wrapped from e
    else:
        e.__dict__["_failed_token_usage"] = tokens
        raise
```

**Catching TokenTrackingError:**

```python
from async_batch_llm import TokenTrackingError

for item_result in result.results:
    if not item_result.success:
        # Token usage is preserved even for failed items
        print(f"Failed: {item_result.item_id}")
        print(f"Tokens consumed: {item_result.token_usage}")
```

---

## Type Aliases

### PostProcessorFunc

Callback function called after each item (both successes and failures).

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
from async_batch_llm import (
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
        attempt_timeout=60.0,
        max_queue_size=100,
    )

    # Create strategy
    agent = Agent("openai:gpt-4", output_type=Summary)
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

- [README.md](https://github.com/geoff-davis/async-batch-llm/blob/main/README.md) - Getting started guide
- [MIGRATION_V0_1.md](archive/MIGRATION_V0_1.md) - Migration guide from v0.0.x (strategy pattern)
- [MIGRATION_V0_4.md](./MIGRATION_V0_4.md) - Migration guide to v0.4.0 (context managers)
- [GEMINI_INTEGRATION.md](./GEMINI_INTEGRATION.md) - Detailed Gemini integration guide
- [CHANGELOG.md](https://github.com/geoff-davis/async-batch-llm/blob/main/CHANGELOG.md) - Version history
