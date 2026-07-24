# Production checklist

Tuning knobs that matter when you move from a 10-item test to a real bulk run.
Each item links to the deeper reference where one exists.

When a run is already failing or appears stuck, use the
[Troubleshooting and FAQ](troubleshooting.md) alongside this preflight list.

## 1. Worker count (`max_workers`)

LLM calls are **I/O-bound**, so `max_workers` is "how many calls in flight at
once", not a CPU count. Do **not** use `os.cpu_count()`.

| Situation | Starting point |
| --- | --- |
| General I/O-bound (most providers) | `5`–`10` |
| Rate-limited / low-quota endpoint | `3`–`5`, lean on the coordinated cooldown |
| High-concurrency provider (e.g. DeepSeek) | `50`–`250`+ — but size the connection pool to match (below) |

Throughput from added workers flattens out well before you exhaust sockets/fds,
so raising `max_workers` past the point where you're latency-bound just adds
contention. Measure with `examples/benchmark_worker_overhead.py` (no network).

## 2. Connection pool (`max_connections`) vs `max_workers`

For the OpenAI-compatible models (`OpenAIModel` / `OpenRouterModel` /
`DeepSeekModel`), the SDK uses httpx's **default ~100-connection pool**. If
`max_workers` exceeds that without an ABL capacity signal, the extra workers
block inside httpx — no extra throughput. **Set `max_connections` explicitly:**

```python
model = DeepSeekModel.from_api_key(
    "deepseek-v4-flash",
    max_connections=200,   # match your ProcessorConfig(max_workers=...)
)
```

Models built with `from_api_key(max_connections=N)` advertise that capacity to
their strategy. `ParallelBatchProcessor` and `LLMCallPool` emit a `UserWarning`
when `max_workers > N`; the shared executor holds excess attempts in ABL
admission before `strategy.execute()` and before `attempt_timeout` starts.
Matching values avoids unnecessary admission wait:

```python
model = DeepSeekModel.from_api_key("deepseek-v4-flash", max_connections=32)
strategy = DeepSeekStrategy(model)
config = ProcessorConfig(max_workers=32)
```

ABL cannot reliably inspect a caller-supplied `AsyncOpenAI`/httpx transport, so
models constructed directly with `OpenAIModel(model, client)` advertise unknown
capacity and do not warn. Set `max_provider_concurrency` to protect a known
custom-client limit:

```python
config = ProcessorConfig(
    max_workers=100,
    max_provider_concurrency=32,  # outside attempt_timeout
)
```

When both the config and strategy advertise a capacity, the lower value wins.
Limits are scoped to the underlying model, so multiple `ModelStrategy` instances
sharing one model also share one capacity semaphore.

See [OpenAI integration → connection-pool sizing](OPENAI_INTEGRATION.md#connection-pool-sizing-max_connections).

## 3. Open-file limit (`RLIMIT_NOFILE`)

Each in-flight request holds a socket (a file descriptor). A high `max_workers`
plus the connection pool plus your app's own fds can hit the OS open-file limit
— `OSError: [Errno 24] Too many open files`. This bites hardest on **macOS**
(default soft limit ~256). The processor emits a `UserWarning` at construction
when `max_workers` is close to the soft limit; it does not raise the limit for
you. Fix by raising it (`ulimit -n 8192`, or `resource.setrlimit` early in the
process) or lowering `max_workers`. Full guidance:
[Troubleshooting → throughput and connection pools](troubleshooting.md#throughput-stops-below-the-worker-count).

## 4. Timeout and concurrency semantics

`attempt_timeout` is **per attempt**, enforced via `asyncio.wait_for` around
each `execute()` — it is **not** a total budget across retries. With
`retry.max_attempts=3`, a single item can spend up to ~`3 × attempt_timeout`
in calls, plus backoff waits.

The timeout boundary is deliberately narrow:

| Phase | Counts against `attempt_timeout`? |
| --- | --- |
| Batch queue wait / streaming backpressure | No |
| Worker or shared-call semaphore admission | No |
| Provider-capacity admission | No |
| Coordinated cooldown / post-cooldown slow-start | No |
| Proactive request-rate limiter wait | No |
| `strategy.execute()` | **Yes** |
| httpx pool wait occurring inside `strategy.execute()` | **Yes** |
| Retry backoff between attempts | No |

The transport-pool row is the common trap: ABL cannot distinguish provider time
from a lower-level wait once `strategy.execute()` begins. Advertised or explicit
capacity prevents that hidden wait by gating attempts first:

```python
# Safe: 100 workers may do middleware/post-processing, but only 32 attempts
# enter strategy.execute() at once.
model = DeepSeekModel.from_api_key("deepseek-v4-flash", max_connections=32)
config = ProcessorConfig(max_workers=100, attempt_timeout=30)
```

Aligning the worker count avoids admission queues when the extra workers provide
no other benefit:

```python
model = DeepSeekModel.from_api_key("deepseek-v4-flash", max_connections=32)
config = ProcessorConfig(max_workers=32, attempt_timeout=30)
```

Rate limits are **exempt** from `max_attempts` (a 429 is "wait and retry", not a
failed attempt — see below), so a throttled item can sit through many cooldowns
without consuming its attempt budget. Bound that separately with
`retry.max_rate_limit_retries` (default `20`). Net: size `attempt_timeout` for
one slow call, and use the two retry budgets to bound total effort.

```python
ProcessorConfig(
    attempt_timeout=60.0,          # per attempt
    retry=RetryConfig(
        max_attempts=3,              # content/transport failures
        max_rate_limit_retries=20,   # throttling retries (separate budget)
    ),
)
```

For `LLMCallPool`, semaphore wait is outside `attempt_timeout`, while
`submit_timeout` wraps the full caller path: both admission waits, cooldown, all
attempts, and backoff. Use `submit_timeout` for an end-to-end request latency
budget and `attempt_timeout` for one provider attempt.

Each `WorkItemResult.admission_wait_seconds` reports cumulative provider-capacity
wait across attempts. `get_stats()` exposes total/max item wait, and
`MetricsObserver` exposes attempt-level count, sum, max, and average wait.
`WorkItemResult.timing` further separates execution, built-in provider-call,
cooldown, retry-backoff, and startup-ramp time for every physical try.
Processor stats retain the latest 10,000 attempt samples and report admission
and execution p50, p95, and p99 values.

## 5. Rate-limit configuration

When one worker hits a 429/quota/overload, the framework runs a **coordinated
cooldown** — all workers pause, then slow-start back up — instead of each worker
hammering a throttled endpoint. Tune via `RateLimitConfig`:

| Field | What it does |
| --- | --- |
| `cooldown_seconds` | Base pause after a rate limit (a server `Retry-After` raises it as a floor) |
| `backoff_multiplier` | Grows the cooldown on consecutive rate limits |
| `slow_start_items` / `slow_start_initial_delay` / `slow_start_final_delay` | Ramp delays as workers resume after a cooldown |

Pair with proactive limiting (`ProcessorConfig(max_requests_per_minute=...)`) to
stay under quota before you trip a 429 at all.

For cold-start burst protection, configure `StartupRampConfig` separately. It
limits initial concurrency and raises it by a fixed step on each interval; unlike
the fields above, it applies before the first provider call and does not require
a preceding rate limit. Ramp wait remains outside `attempt_timeout`.

See the [OpenAI-compatible high-throughput guide](openai-high-throughput.md) for
owned/custom client recipes and troubleshooting.

## 6. Bounded streaming for large inputs

For a very large (or unbounded) input, don't buffer all the work up front. Use
**streaming mode** with bounded `max_queue_size` and
`max_result_queue_size`: workers run while you feed, so full queues apply
**backpressure** instead of deadlocking or accumulating work in proportion to
input size.

```python
from async_batch_llm import ProcessorConfig, process_stream

config = ProcessorConfig(
    max_workers=50,
    max_queue_size=200,
    max_result_queue_size=100,
)

async for result in process_stream(strategy, huge_prompt_source, config=config):
    if result.success:
        await save(result.item_id, result.output)   # completion order
```

`huge_prompt_source` can be any sync or async iterable (e.g. a generator reading
a file lazily). The low-level equivalent is
`processor.start()` / `add_work()` / `finish()` / `results()`.

`process_prompts()` retains every result in its returned `BatchResult`, and
`process_all()` requires work to be added before workers start. Neither is a
bounded-memory result path for an unbounded workload. `process_stream()` avoids
retaining results itself; set `max_result_queue_size` so a slow consumer applies
backpressure to provider workers instead of accumulating completed items. See
[Bounded Work and Backpressure](bounded-work.md) for incremental database input,
low-level streaming, and separate input/output/provider/shared-call limits.

## 7. Single calls and the shared call pool (request paths)

For a web service's request path — where work arrives one call at a time, not as
a batch — use [`LLMCallPool`](api/single-gateway.md) instead of standing up a
processor per request:

- **One long-lived pool per app.** Create it once at startup (e.g. a FastAPI
  lifespan handler) and share it across all request handlers. A single pool
  means one shared rate-limit cooldown — when one caller hits a 429, all callers
  briefly pause and then slow-start, instead of a thundering herd.
- **Set `max_pending` and `submit_timeout` for web paths.** `max_pending` caps
  in-flight requests (running + waiting) so an overload sheds load instantly
  (rejecting with a failed result) rather than growing an unbounded waiter list;
  `submit_timeout` bounds per-caller latency so a request stuck behind a cooldown
  returns instead of hanging the handler. Both are off by default.
- **Do not create unbounded outer tasks.** `max_pending` bounds pool
  admission, but one large `asyncio.gather()` still materializes every caller
  task. Use the [bounded batch pattern](bounded-work.md#recommended-large-batch-pattern)
  for ingestion jobs or an explicitly bounded task window.
- **Shutdown drains admitted requests.** `aclose()` (the `async with` exit) stops
  accepting new work, then waits for already-admitted requests to finish before
  cleaning up the shared strategy, so in-flight calls aren't cut off mid-flight.
  Set `submit_timeout` to bound how long shutdown waits for that drain; with no
  timeout it waits as long as the admitted work takes.

For a single ad-hoc call, [`call()` / `call_result()`](api/single-gateway.md)
run one prompt through the same resilience pipeline with no pool at all.

## 8. Cleanup

Use the processor as an `async with` context manager so workers, caches, and
HTTP clients are released. If you can't, call `await processor.shutdown()` when
done.

## 9. Artifact storage (checkpoint/resume runs)

- **Backend choice.** JSONL for portable, human-inspectable audit logs;
  `SqliteArtifactStore` for 100k+ restartable runs needing indexed replay.
  See the [comparison table](results-and-artifacts.md#indexed-sqlite-artifacts-for-large-restartable-runs).
- **Disk space and sidecars.** Budget from a measured bytes-per-record on
  your payloads. SQLite runs also create `-wal`/`-shm` sidecar files next to
  the database; the WAL plateaus during healthy writes and is truncated on
  clean close (an `-shm` file may harmlessly remain).
- **Durability mode.** `SqliteDurability.BALANCED` (default) survives process
  crashes; a power failure can lose recently committed transactions. Choose
  `FULL` when power-loss durability matters more than write throughput.
  Neither mode encrypts the file — use filesystem permissions and volume
  encryption for sensitive outputs.
- **Commit batch latency.** Results are published only after their
  transaction commits; `commit_interval_seconds` (default 0.01 s) is added
  result latency. Lower it for latency-sensitive streaming, raise the batch
  size for maximum append throughput.
- **Write permissions.** The store creates parent directories and needs
  write access for the database and its sidecars.
- **Single-writer deployment.** One writable store instance per artifact
  file, in one process. Sequential reopen is supported; concurrent external
  readers obey `busy_timeout_seconds`. There is no distributed work claiming
  or cross-process exactly-once execution.
- **Before million-item claims.** Run `make scale-100k` (and `scale-1m` for
  seven digits) on a representative environment and review the report —
  don't extrapolate from profile existence. See
  [Benchmarks](benchmarks.md#scale-soak-harness).
