# Project Knowledge for Claude

This document contains important information about the `batch-llm` project for future AI assistants working on this codebase.

---

## Project Overview

**batch-llm** is a Python package for processing multiple LLM requests efficiently using a **strategy pattern** (v3.0+).

**Current Version: v3.0** - Uses `LLMCallStrategy` for provider-agnostic LLM integration

**Key Features:**
- Parallel asyncio processing with configurable concurrency
- Built-in rate limiting and exponential backoff retry logic
- Thread-safe concurrent operations
- Provider-agnostic through strategy pattern
- Middleware and observer patterns for extensibility
- MockAgent for testing without API calls
- Support for ANY LLM provider (OpenAI, Anthropic, Google, LangChain, custom)

---

## Architecture (v3.0 Strategy Pattern)

### Core Components

**`LLMCallStrategy[TOutput]`** - Abstract base for all LLM integrations:
- `async def prepare()` - Initialize resources (caches, connections)
- `async def execute(prompt, attempt, timeout)` - Make LLM call
- `async def cleanup()` - Clean up resources (delete caches)

**Built-in Strategies:**
- `PydanticAIStrategy` - Wraps PydanticAI agents
- `GeminiStrategy` - Direct Google Gemini API calls
- `GeminiCachedStrategy` - Gemini with context caching (great for RAG)

**`LLMWorkItem[TInput, TOutput, TContext]`** - Work unit:
- `item_id: str` - Unique identifier
- `strategy: LLMCallStrategy[TOutput]` - How to call the LLM
- `prompt: str` - The prompt/input
- `context: TContext | None` - Optional context passed through

**`ParallelBatchProcessor[TInput, TOutput, TContext]`** - Main processing engine:
- Manages worker pool (default 5 workers)
- Coordinates rate limiting across all workers
- Handles retries with exponential backoff
- Framework-level timeout enforcement with `asyncio.wait_for()`
- Collects results and metrics

**Thread Safety:**
- Uses `asyncio.Lock` for all shared mutable state
- Three locks: `_rate_limit_lock`, `_stats_lock`, `_results_lock`
- All locks are independent (no nesting = no deadlocks)

---

## Critical Design Decisions

### 1. Strategy Pattern (v3.0)

**Why:** Decouple framework from specific LLM providers.

**Benefits:**
- Support any LLM provider (OpenAI, Anthropic, Google, LangChain, custom)
- Each strategy encapsulates provider-specific logic
- Framework handles retry, timeout, rate limiting uniformly
- Easy to test with mock strategies
- Resource lifecycle management (prepare/cleanup)

**Migration from v2.x:**
```python
# v2.x (removed)
work_item = LLMWorkItem(item_id="1", agent=agent, prompt="...")

# v3.0 (current)
strategy = PydanticAIStrategy(agent=agent)
work_item = LLMWorkItem(item_id="1", strategy=strategy, prompt="...")
```

See `docs/MIGRATION_V3.md` for complete migration guide.

### 2. Rate Limiting Strategy

**Problem:** Multiple workers hitting rate limits simultaneously.

**Solution:**
- One worker triggers cooldown via atomic check-and-set
- All workers pause using `asyncio.Event`
- Slow-start ramp after cooldown (progressive delays)
- Consecutive rate limits trigger exponential backoff

**Implementation:**
```python
async with self._rate_limit_lock:
    if self._in_cooldown:
        return  # Another worker handling it
    self._in_cooldown = True
    self._rate_limit_event.clear()  # Pause all
```

### 3. Progressive Temperature Retry (v3.0)

**Why:** LLMs may fail validation at low temperature but succeed at higher temps.

**Implementation in v3.0:**
```python
class ProgressiveTempStrategy(LLMCallStrategy[Output]):
    def __init__(self, client, temps=None):
        self.client = client
        self.temps = temps if temps is not None else [0.0, 0.5, 1.0]

    async def execute(self, prompt: str, attempt: int, timeout: float):
        temp = self.temps[min(attempt - 1, len(self.temps) - 1)]
        # Make call with progressive temperature
        response = await self.client.generate(prompt, temperature=temp)
        return parsed_output, token_usage

strategy = ProgressiveTempStrategy(client=client)
work_item = LLMWorkItem(item_id="1", strategy=strategy, prompt="...")
```

See `examples/example_gemini_direct.py` for complete example.

### 4. Token Usage Tracking

**Challenge:** Track tokens even for failed attempts.

**Solution:**
- Extract usage from exception chain via `__cause__`
- Accumulate across all retry attempts
- Attach to final exception via `__dict__['_failed_token_usage']`
- Include in `WorkItemResult.token_usage`

---

## Common Patterns (v3.0)

### Pattern 1: Using PydanticAI Strategy

```python
from batch_llm import PydanticAIStrategy, LLMWorkItem
from pydantic_ai import Agent

agent = Agent("gemini-2.0-flash", result_type=Output)
strategy = PydanticAIStrategy(agent=agent)

work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="Your prompt here"
)
```

### Pattern 2: Custom Strategy for Any Provider

```python
from batch_llm.llm_strategies import LLMCallStrategy

class OpenAIStrategy(LLMCallStrategy[str]):
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def execute(self, prompt: str, attempt: int, timeout: float):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content
        tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        return output, tokens

strategy = OpenAIStrategy(client=openai_client, model="gpt-4o-mini")
work_item = LLMWorkItem(item_id="1", strategy=strategy, prompt="...")
```

### Pattern 3: Post-Processing Results

```python
async def save_result(result: WorkItemResult):
    if result.success and result.context:
        await db.save(result.context.id, result.output)

processor = ParallelBatchProcessor(
    config=config,
    post_processor=save_result
)
```

### Pattern 4: Observing Metrics

```python
metrics = MetricsObserver()
processor = ParallelBatchProcessor(
    config=config,
    observers=[metrics]
)

result = await processor.process_all()
collected_metrics = await metrics.get_metrics()
```

---

## Race Conditions Fixed (v2.0.1)

### Issue History

**Original Problem:** Five critical race conditions in concurrent code:

1. **Rate limit cooldown** - Multiple workers triggering multiple cooldowns
2. **Stats dictionary** - Concurrent updates causing undercounting
3. **Results list** - Potential result loss during concurrent appends
4. **MetricsObserver** - Incorrect metrics under concurrent load
5. **Slow-start counter** - Undercounting items processed

**Fix:** Added three `asyncio.Lock` instances:
- `_rate_limit_lock` - Rate limit coordination
- `_stats_lock` - Stats updates
- `_results_lock` - Results list appends

**Impact:** <1% performance overhead, guaranteed correctness

**Breaking Changes:**
- `get_stats()` is now async
- `MetricsObserver.get_metrics()` is now async

---

## Testing Strategy

### Test Files

1. **`tests/test_basic.py`** - Basic functionality
   - Simple processing
   - Context passing
   - Post-processors
   - Metrics collection
   - Timeout handling

2. **`tests/test_concurrency.py`** - Thread safety
   - Concurrent stats updates (100 items, 10 workers)
   - Rate limit handling
   - Metrics observer accuracy
   - No result loss (200 items, 20 workers)
   - Post-processor calls
   - Stats snapshot consistency
   - Slow-start counter accuracy

### MockAgent Usage

**Purpose:** Test without making real API calls.

```python
mock_agent = MockAgent(
    response_factory=lambda p: YourOutput(...),
    latency=0.01,  # Simulate processing time
    rate_limit_on_call=5,  # Trigger rate limit on 5th call
)
```

---

## Development Workflow

### Running Tests

```bash
# Install dependencies
uv sync --all-extras

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_basic.py -v

# Run concurrency tests
uv run pytest tests/test_concurrency.py -v

# With coverage
uv run pytest --cov=batch_llm --cov-report=html
```

### Code Quality

**IMPORTANT:** Always run ruff after making significant code changes!

```bash
# Format code
uv run ruff format src/ tests/

# Lint and auto-fix issues
uv run ruff check src/ tests/ --fix

# Verify all linting passes
uv run ruff check src/ tests/

# Type check
uv run mypy src/batch_llm/ --ignore-missing-imports
```

**Workflow:** After any code changes, run `uv run ruff check src/ --fix` before committing.

### Building and Publishing

```bash
# Build package
uv build

# Publish to TestPyPI
export UV_PUBLISH_TOKEN="your-testpypi-token"
uv publish --index-url https://test.pypi.org/legacy/

# Publish to PyPI
export UV_PUBLISH_TOKEN="your-pypi-token"
uv publish
```

---

## Important Files

### Package Structure
```
src/batch_llm/
├── __init__.py           # Public API exports
├── base.py               # Core data models
├── parallel.py           # Main processor (1000+ lines)
├── core/
│   ├── config.py         # Configuration classes
│   └── protocols.py      # Type protocols
├── strategies/
│   ├── errors.py         # Error classification
│   └── rate_limit.py     # Rate limit strategies
├── observers/
│   ├── base.py           # Observer protocol
│   └── metrics.py        # Metrics collection
├── middleware/
│   └── base.py           # Middleware protocol
├── classifiers/
│   └── gemini.py         # Gemini error classifier
└── testing/
    └── mocks.py          # MockAgent for testing
```

### Documentation
- `README.md` - User documentation
- `RACE_CONDITION_ANALYSIS.md` - Original analysis
- `RACE_CONDITION_FIXES.md` - Implementation details
- `CONTRIBUTING.md` - Developer guide
- `PUBLISHING.md` - Publishing instructions
- `CHANGELOG.md` - Version history
- `CLAUDE.md` - This file

---

## Known Limitations

1. **No multi-process support** - Designed for single-process asyncio only
2. **No true batch API** - Parallel individual calls, not batched API requests
3. **Provider-specific classifiers** - Only Gemini classifier implemented
4. **No persistent queue** - In-memory queue only (lost on crash)

---

## Future Enhancements

1. **Distributed locks** - Support multi-process scenarios
2. **Batch API support** - True batch API for 50% cost savings
3. **More classifiers** - OpenAI, Anthropic, etc.
4. **Persistent queue** - Redis/DB-backed queue
5. **Prometheus metrics** - Built-in metrics export
6. **Progress callbacks** - Real-time progress updates
7. **Dynamic worker scaling** - Adjust workers based on load

---

## Common Pitfalls (v3.0)

### ❌ Don't: Forget to await async methods

```python
# WRONG
stats = processor.get_stats()

# RIGHT
stats = await processor.get_stats()
```

### ❌ Don't: Use old v2.x API

```python
# WRONG - v2.x API (removed)
work_item = LLMWorkItem(
    item_id="item_1",
    agent=agent,  # No longer supported
    prompt="..."
)

# RIGHT - v3.0 API (current)
strategy = PydanticAIStrategy(agent=agent)
work_item = LLMWorkItem(
    item_id="item_1",
    strategy=strategy,
    prompt="..."
)
```

### ❌ Don't: Forget to wrap agents in strategies

```python
# WRONG - agent is not a strategy
work_item = LLMWorkItem(item_id="1", strategy=agent, prompt="...")

# RIGHT - wrap agent in PydanticAIStrategy
strategy = PydanticAIStrategy(agent=agent)
work_item = LLMWorkItem(item_id="1", strategy=strategy, prompt="...")
```

### ❌ Don't: Mutate results in post-processor

```python
# WRONG - modifying shared state
async def bad_post_processor(result):
    result.output.field = "modified"  # Don't do this!

# RIGHT - read-only or create new objects
async def good_post_processor(result):
    await db.save(result.output)  # Read and save
```

### ✅ Do: Use context for passing data through

```python
work_item = LLMWorkItem(
    item_id="item_1",
    agent=agent,
    prompt="Your prompt",
    context={"user_id": 123, "original_data": data}
)

# Access in post-processor
async def post_process(result):
    if result.success:
        user_id = result.context["user_id"]
        await save_for_user(user_id, result.output)
```

---

## Performance Notes

### Optimal Worker Count

- **CPU-bound:** Use `max_workers = cpu_count()`
- **I/O-bound (LLM calls):** Use `max_workers = 5-10`
- **Rate-limited:** Start with 3-5, increase gradually

### Memory Usage

- Each worker holds 1 item in memory
- Results accumulate in memory (all items)
- For 1000 items: ~10-50MB depending on output size

### Throughput Estimates

- **Gemini 2.0 Flash:** ~5-10 items/sec with 5 workers
- Limited by API latency (~200-500ms per call)
- Rate limits: ~10 requests/minute initially

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check for Rate Limits

```python
result = await processor.process_all()
stats = await processor.get_stats()

if stats["rate_limit_count"] > 0:
    print(f"Hit {stats['rate_limit_count']} rate limits")
    print(f"Error breakdown: {stats['error_counts']}")
```

### Verify Stats Accuracy

```python
result = await processor.process_all()
assert result.total_items == result.succeeded + result.failed
assert len(result.results) == result.total_items
```

---

## Version History

- **v2.0.0** - Initial PyPI release with full feature set
- **v2.0.1** - Fixed 5 critical race conditions
- **v3.0.0** - Strategy pattern refactor (current)
  - Breaking: Replaced `agent=`, `agent_factory=`, `direct_call=` with `strategy=`
  - Added `LLMCallStrategy` abstract base class
  - Framework-level timeout enforcement
  - Built-in strategies: `PydanticAIStrategy`, `GeminiStrategy`, `GeminiCachedStrategy`

---

## Advanced Retry Patterns (v3.0)

### Smart Retry with Validation Feedback

When Pydantic validation fails, create targeted retry prompts:

```python
class SmartRetryStrategy(LLMCallStrategy[PersonData]):
    """On validation failure, tell LLM which fields succeeded vs failed."""

    def _create_retry_prompt(self, original_prompt: str) -> str:
        # Parse validation errors
        # Create focused prompt with:
        # - Fields that succeeded (keep these)
        # - Fields that failed (fix these with specific error messages)
        return retry_prompt

    async def execute(self, prompt: str, attempt: int, timeout: float):
        if attempt == 1:
            final_prompt = prompt
        else:
            final_prompt = self._create_retry_prompt(prompt)
        # Make LLM call with focused prompt
```

See `examples/example_gemini_smart_retry.py` for complete implementation.

### Model Escalation for Cost Optimization

Start with cheap models, escalate only when needed:

```python
class ModelEscalationStrategy(LLMCallStrategy[Analysis]):
    """Start with cheapest model, escalate on failure."""

    MODELS = [
        "gemini-2.5-flash-lite",  # Attempt 1: Cheapest
        "gemini-2.5-flash",       # Attempt 2: Moderate
        "gemini-2.5-pro",         # Attempt 3: Most capable
    ]

    async def execute(self, prompt: str, attempt: int, timeout: float):
        model = self.MODELS[min(attempt - 1, len(self.MODELS) - 1)]
        # Make call with escalated model
```

**Cost savings:** ~60-80% vs always using best model, since most tasks succeed on first (cheap) attempt.

See `examples/example_model_escalation.py` for complete implementation with cost comparisons.

---

## Contact & Support

- GitHub: https://github.com/yourusername/batch-llm
- Issues: https://github.com/yourusername/batch-llm/issues
- PyPI: https://pypi.org/project/batch-llm/

---

## Lessons Learned (For Future Claude Sessions)

### Code Quality and Linting

1. **Always run linters after code changes:**
   - Python: `uv run ruff check src/ tests/ --fix`
   - Markdown: `markdownlint-cli2 "README.md" "docs/*.md" --fix`
   - Both should pass with 0 errors before committing

2. **Common ruff issues to watch for:**
   - Mutable default arguments: Use `None` and initialize in function body
   - Unused imports: Remove them
   - Unnecessary f-strings: Remove `f` prefix if no placeholders
   - Import sorting: Let ruff auto-fix this

3. **Markdown linting config:**
   - Created `.markdownlint.json` with line-length: 120
   - Code blocks need language specifiers (use `text` for error messages)
   - Blank lines required around lists and code fences

### Documentation Best Practices

1. **Example files are critical:**
   - Users learn from examples more than docs
   - Every new pattern needs a complete working example
   - Examples should be runnable (handle missing API keys gracefully)

2. **Keep docs in sync with code:**
   - When refactoring API (like v2.x → v3.0), update ALL docs
   - Search for old patterns: `agent=`, `agent_factory=`, `direct_call=`
   - Update: README, docs/, examples/, CLAUDE.md

3. **Migration guides are essential:**
   - Breaking changes need migration docs
   - Show before/after code examples
   - Explain WHY the change was made

### Testing Strategy

1. **Test coverage is comprehensive:**
   - 61 tests covering core functionality
   - Concurrency tests with high worker counts (10-20)
   - Edge cases (empty queues, timeouts, validation errors)

2. **MockAgent is powerful:**
   - Test without API calls
   - Simulate rate limits, errors, latency
   - Much faster than integration tests

3. **Run tests in parallel when possible:**
   - Multiple test files can run concurrently
   - Use background bash shells for long-running tests

### Development Workflow Insights

1. **Strategy pattern benefits:**
   - Decouples framework from providers
   - Easy to add new providers (OpenAI, Anthropic examples)
   - Resource lifecycle (prepare/cleanup) is powerful for caching

2. **Advanced retry patterns are valuable:**
   - Smart retry (field-specific feedback) reduces token usage
   - Model escalation (cheap → expensive) saves 60-80% cost
   - Progressive temperature helps with validation errors

3. **Optional dependencies strategy:**
   - Keep minimal: only `[pydantic-ai]` and `[gemini]`
   - Don't add `[openai]`, `[anthropic]`, etc.
   - Let users control their provider SDK versions
   - Examples serve as documentation, not product features

### Common Pitfalls to Avoid

1. **Don't use bash for file operations:**
   - Use Read, Edit, Write, Glob, Grep tools instead
   - Bash is only for terminal operations (git, npm, pytest, etc.)

2. **Always read files before editing:**
   - Edit tool requires prior Read
   - Prevents editing non-existent files

3. **Watch for mutable defaults in Python:**
   - `def __init__(self, temps=[0.0, 0.5, 1.0])` is wrong
   - Use `temps=None` and initialize: `self.temps = temps if temps is not None else [0.0, 0.5, 1.0]`

4. **Markdown line length:**
   - 120 chars is reasonable (not 80)
   - Long URLs and code examples need flexibility
   - Use config file to relax rules

### Project-Specific Knowledge

1. **File structure:**
   - Main code: `src/batch_llm/`
   - Tests: `tests/`
   - Examples: `examples/`
   - Docs: `docs/` + `README.md`

2. **Key files to update together:**
   - API changes → Update: `README.md`, `docs/API.md`, `examples/*.py`
   - New patterns → Update: `README.md` Advanced Usage, `examples/`, `CLAUDE.md`

3. **Examples to reference:**
   - `example_gemini_smart_retry.py` - Smart retry patterns
   - `example_model_escalation.py` - Cost optimization
   - `example_gemini_direct.py` - Direct Gemini API usage
   - `example_openai.py`, `example_anthropic.py` - Other providers

4. **Versioning:**
   - Major version (v3.0) for breaking API changes
   - Document breaking changes in `docs/MIGRATION_V3.md`
   - Update version history in `CLAUDE.md`

---

## Quick Reference

### Minimal Example (v3.0)

```python
from batch_llm import ParallelBatchProcessor, LLMWorkItem, ProcessorConfig, PydanticAIStrategy
from pydantic_ai import Agent
from pydantic import BaseModel

class Output(BaseModel):
    result: str

agent = Agent("gemini-2.5-flash", result_type=Output)
strategy = PydanticAIStrategy(agent=agent)
config = ProcessorConfig(max_workers=5, timeout_per_item=120.0)

async with ParallelBatchProcessor[str, Output, None](config=config) as processor:
    await processor.add_work(LLMWorkItem(
        item_id="1",
        strategy=strategy,
        prompt="Hello"
    ))

    result = await processor.process_all()
    print(f"Success: {result.succeeded}/{result.total_items}")
```

### Full-Featured Example

See `examples/example.py` for complete examples including:
- Context passing
- Post-processors
- Middleware
- Observers
- Error handling
- MockAgent testing

---

## License

MIT License - See LICENSE file
