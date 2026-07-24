# Bounded Work and Backpressure

Provider concurrency and pending-work memory are different limits. A provider
gate can keep only 32 calls active while an application still creates a million
queued items or asyncio tasks. Bound both layers deliberately.

## Limits at a Glance

| Control | Bounds | Behavior at the limit |
| --- | --- | --- |
| `ProcessorConfig.max_workers` | Active processor workers | Work remains queued |
| Model `max_concurrency` / `max_provider_concurrency` | Attempts inside `strategy.execute()` | Workers wait before the execution timeout |
| `ProcessorConfig.max_queue_size` in streaming mode | Work items waiting for a worker | Producer awaits queue space (backpressure) |
| `ProcessorConfig.max_result_queue_size` | Completed results waiting for the stream consumer | Provider workers await result capacity |
| `LLMCallPool.max_pending` | Shared calls running or waiting | New calls are rejected immediately |
| `LLMCallPool.submit_timeout` | One caller's total shared-call wall time | The call returns a timeout failure |

`max_queue_size` is the batch equivalent of a bounded pending-work buffer. It
does not change provider concurrency. Set provider capacity independently.

## Recommended Large-Batch Pattern

Use `process_stream` with a positive `max_queue_size`, feed it a lazy sync or
async iterable, and persist each result as it arrives:

```python
from collections.abc import AsyncIterator

from async_batch_llm import ProcessorConfig, process_stream


async def prompts_from_database() -> AsyncIterator[tuple[str, str, int]]:
    async for row in database.iter_documents(fetch_size=500):
        yield row.id, f"Classify:\n{row.text}", row.version


config = ProcessorConfig(
    max_workers=32,
    max_queue_size=128,
    max_result_queue_size=64,
    max_provider_concurrency=32,
)

async for result in process_stream(strategy, prompts_from_database(), config=config):
    await save_result(result.item_id, result.output, result.context)
```

At most 128 items wait in the processor queue, up to 32 workers process items,
at most 32 attempts enter the provider call, and at most 64 completed results
wait in the handoff queue. When the input queue fills, `process_stream` pauses
`prompts_from_database()` until a worker frees space. When the output handoff
fills, provider workers pause after terminal bookkeeping until the consumer
dequeues results. The database iterator and result production therefore cannot
run ahead indefinitely.

When the feed finishes, fails, is cancelled, or stops under a batch guardrail,
the high-level streaming API calls `aclose()` on async input iterators that
provide it. This releases database cursors and generator `finally` blocks
promptly. Treat a supplied async iterator as owned by that one stream; do not
expect to resume or share it afterward.

Results arrive in completion order. Persist or aggregate them incrementally if
the result set itself is large.

The output limit bounds completed results **queued for the consumer**, not every
result object alive in the process. A result currently being handled by the
consumer is outside the queue, and each active worker may temporarily hold one
completed result while waiting to publish. The meaningful result-object bound
is therefore approximately:

```text
max_result_queue_size + active worker count + current consumer item
```

It is independent of total run size, but not independent of `concurrency` or
data retained by application code.

Artifact persistence, terminal statistics, observers, and progress bookkeeping
occur before a worker reserves result capacity. A newly executed checkpoint is
therefore durable before the corresponding result can be observed, and slow
artifact I/O does not occupy an output slot. Worker-crash and end-of-stream
signals use a separate control path and remain deliverable when result capacity
is exhausted.

## APIs That Collect Work or Results

- `process_stream(...)` bounds pending input when its input is lazy and
  `max_queue_size` is positive. A positive `max_result_queue_size` separately
  bounds completed results waiting to be consumed; zero preserves the
  historical unbounded handoff.
- `process_prompts(...)` uses streaming execution internally but returns one
  `BatchResult`, so it retains every completed result. It is convenient for
  bounded jobs, not bounded-memory output handling.
- `ParallelBatchProcessor.process_all()` starts workers only after work has been
  added. Adding an entire input first materializes that input. A bounded queue
  cannot apply backpressure in this mode because no worker is draining it;
  `add_work()` raises when the queue fills.

Do not use `max_queue_size=0` for an unbounded or poorly bounded source. Zero
means the processor queue is unlimited.

## Exact Queue High-Water Marks

`get_stats()` (and the `batch_completed` observer payload) reports exactly how
deep each queue actually got (v0.21):

```python
stats = await processor.get_stats()
stats["input_queue_high_water_mark"]   # peak accepted-but-unstarted items
stats["result_queue_high_water_mark"]  # peak completed results awaiting the consumer
```

Both counters are exact, observed at queue-ownership points rather than
sampled, and count only data items — end-of-stream and worker-crash control
messages never consume result capacity or inflate the mark. They reset per
run and report observed depth for unbounded configurations too, which makes
them the right evidence for choosing a bound: run once unbounded, read the
high-water marks, then set bounds with the headroom you actually need.

Distinguish the five places work can live when reading these numbers: the
accepted input queue (bounded by `max_queue_size`), active workers (bounded
by `concurrency`), provider admission, the completed-result queue (bounded by
`max_result_queue_size`), and the result your consumer currently holds —
plus, for collection APIs like `process_prompts()`, the deliberately retained
full result list, which no queue bound limits.

## Low-Level Streaming Lifecycle

Use the processor directly when items need different strategies or custom
construction. Start workers before feeding the bounded queue:

```python
async with ParallelBatchProcessor(config=config) as processor:
    processor.start()

    async def produce() -> None:
        try:
            async for row in database.iter_documents(fetch_size=500):
                await processor.add_work(
                    LLMWorkItem(
                        item_id=row.id,
                        strategy=strategy_for(row),
                        prompt=format_prompt(row),
                        context=row.version,
                    )
                )
        finally:
            await processor.finish()

    producer = asyncio.create_task(produce())
    async for result in processor.results():
        await save_result(result.item_id, result.output, result.context)
    await producer
```

`add_work()` is the backpressure point. Always call `finish()` after the
producer reaches end-of-input so `results()` can terminate.

## Shared Call Task Counts

`LLMCallPool` bounds calls inside the in-process shared call pool, but this still
creates one asyncio task per input item:

```python
# Avoid for an unbounded or very large source.
await asyncio.gather(*(pool.submit(prompt) for prompt in prompts))
```

`max_pending` rejects overload; it does not prevent the outer `gather()` from
materializing every coroutine. Use `process_stream` for batch ingestion. When a
shared call pool is required, keep the outer task window bounded as well:

```python
async def submit_windowed(pool, prompts, *, window: int = 100):
    pending: set[asyncio.Task] = set()
    async for prompt in prompts:
        pending.add(asyncio.create_task(pool.submit_result(prompt)))
        if len(pending) >= window:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                yield task.result()

    for task in asyncio.as_completed(pending):
        yield await task
```

For request-serving applications, also set `max_pending` and `submit_timeout`
so server traffic cannot grow an unlimited shared-call waiter list.
