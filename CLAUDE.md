# Project Knowledge for Claude

This document contains important information about the `batch-llm` project for future AI assistants working on this codebase.

---

## Project Overview

**batch-llm** is a Python package for processing multiple LLM requests efficiently with three integration modes:

1. **PydanticAI Agents** - Standard agent-based processing
2. **Direct API Calls** - Custom temperature control and direct LLM access
3. **Agent Factories** - Progressive temperature increases for retries

**Key Features:**
- Parallel asyncio processing
- Built-in rate limiting and retry logic
- Thread-safe concurrent operations
- Provider-agnostic error classification
- Middleware and observer patterns
- MockAgent for testing without API calls

---

## Architecture

### Core Components

**`LLMWorkItem`** - Three processing modes (exactly one required):
- `agent`: PydanticAI Agent instance
- `agent_factory`: Callable that creates agents per attempt
- `direct_call`: Async callable for direct LLM API calls

**`ParallelBatchProcessor`** - Main processing engine:
- Manages worker pool (default 5 workers)
- Coordinates rate limiting across all workers
- Handles retries with exponential backoff
- Collects results and metrics

**Thread Safety:**
- Uses `asyncio.Lock` for all shared mutable state
- Three locks: `_rate_limit_lock`, `_stats_lock`, `_results_lock`
- All locks are independent (no nesting = no deadlocks)

---

## Critical Design Decisions

### 1. Rate Limiting Strategy

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

### 2. Progressive Temperature Retry

**Why:** LLMs may fail validation at low temperature but succeed at higher temps.

**Implementation:**
- Attempt 1: temperature = 0.0 (deterministic)
- Attempt 2: temperature = 0.25
- Attempt 3: temperature = 0.5

**Usage:**
```python
def create_agent(attempt: int) -> Agent:
    temp = [0.0, 0.25, 0.5][attempt - 1]
    return Agent("gemini-2.0-flash", temperature=temp)

work_item = LLMWorkItem(
    item_id="item_1",
    agent_factory=create_agent,
    prompt="Your prompt"
)
```

### 3. Token Usage Tracking

**Challenge:** Track tokens even for failed attempts.

**Solution:**
- Extract usage from exception chain via `__cause__`
- Accumulate across all retry attempts
- Attach to final exception via `__dict__['_failed_token_usage']`
- Include in `WorkItemResult.token_usage`

---

## Common Patterns

### Pattern 1: Using Direct Calls

```python
async def call_gemini(attempt: int, timeout: float) -> tuple[Output, dict]:
    """Direct LLM call with custom temperature."""
    temp = [0.0, 0.25, 0.5][attempt - 1]
    # Your API call here
    return result, {
        "input_tokens": 100,
        "output_tokens": 200,
        "total_tokens": 300
    }

work_item = LLMWorkItem(
    item_id="item_1",
    direct_call=call_gemini,
    input_data=your_data
)
```

### Pattern 2: Post-Processing Results

```python
async def save_result(result: WorkItemResult):
    if result.success and result.context:
        await db.save(result.context.id, result.output)

processor = ParallelBatchProcessor(
    config=config,
    post_processor=save_result
)
```

### Pattern 3: Observing Metrics

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

## Common Pitfalls

### ❌ Don't: Forget to await async methods

```python
# WRONG
stats = processor.get_stats()

# RIGHT
stats = await processor.get_stats()
```

### ❌ Don't: Provide multiple processing methods

```python
# WRONG - will raise ValueError
work_item = LLMWorkItem(
    item_id="item_1",
    agent=agent,
    direct_call=my_func  # Can't have both!
)

# RIGHT - exactly one
work_item = LLMWorkItem(
    item_id="item_1",
    agent=agent
)
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
- **v2.0.1** - Fixed 5 critical race conditions (current)

---

## Contact & Support

- GitHub: https://github.com/yourusername/batch-llm
- Issues: https://github.com/yourusername/batch-llm/issues
- PyPI: https://pypi.org/project/batch-llm/

---

## Quick Reference

### Minimal Example

```python
from batch_llm import ParallelBatchProcessor, LLMWorkItem, ProcessorConfig
from pydantic_ai import Agent
from pydantic import BaseModel

class Output(BaseModel):
    result: str

agent = Agent("gemini-2.0-flash", output_type=Output)
config = ProcessorConfig(max_workers=5, timeout_per_item=120.0)
processor = ParallelBatchProcessor(config=config)

await processor.add_work(LLMWorkItem(
    item_id="1",
    agent=agent,
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
