# Large Runs (100k–1M items)

How to run six or seven digits of work items with bounded memory, restart
safety, and evidence instead of hope. This page assumes the concepts in
[Bounded Work and Backpressure](bounded-work.md) and
[Results, Artifacts, and Resume](results-and-artifacts.md).

## Stream, don't collect

`process_prompts()` retains every terminal result in its returned
`BatchResult` — at large scale that is deliberate collection, not bounded
memory. For large runs use `process_stream()` with a lazy source and consume
each result as it arrives:

```python
from async_batch_llm import (
    ProcessorConfig,
    ResumePolicy,
    SqliteArtifactStore,
    process_stream,
)

async def source():
    async for row in database.cursor("SELECT id, prompt FROM work"):
        yield row.id, row.prompt          # never materialize the full input

config = ProcessorConfig(
    concurrency=64,
    max_queue_size=512,          # input backpressure: producer blocks when full
    max_result_queue_size=512,   # result backpressure: workers pause for a slow sink
)

store = SqliteArtifactStore("runs/big-run.sqlite")
async for item in process_stream(
    strategy,
    source(),
    config=config,
    artifact_store=store,
    resume=ResumePolicy.REUSE_SUCCESSES,
):
    await sink(item)                      # keep aggregates, not the item list
```

An async sink that writes downstream (database, queue, file) should hold
aggregates and counters, not accumulate `WorkItemResult` objects.

## Sizing the knobs

- **`concurrency`** sizes workers, provider admission, and built-in
  connection pools together. Start at 32–64 for fake-fast or high-quota
  endpoints, 5–10 for ordinary rate-limited APIs, and let
  [Choosing Your Limits](choosing-your-limits.md) guide refinement.
- **`max_queue_size`** bounds accepted-but-unstarted items. A few multiples
  of `concurrency` (e.g. 512 for 64 workers) keeps workers fed while capping
  producer read-ahead.
- **`max_result_queue_size`** bounds completed results awaiting your sink.
  Size it for your sink's burst tolerance; workers pause when it is full.

Bounded queues cap the *queued* objects. Active workers each hold one item,
and your sink holds the result it is processing — the true in-flight ceiling
is roughly `max_queue_size + concurrency + max_result_queue_size + 1`.

Watch the exact high-water marks in
[`get_stats()`](bounded-work.md#exact-queue-high-water-marks) to see how much
of each bound a real run actually used.

## Choose SQLite for restartable scale

For 100k+ restartable work, `SqliteArtifactStore` gives indexed replay
lookup and reopening without decoding history; JSONL remains right for
portable, human-readable audit trails. The full comparison and durability
semantics live in
[Results, Artifacts, and Resume](results-and-artifacts.md#indexed-sqlite-artifacts-for-large-restartable-runs).

Checkpoint batching trades result-publication latency for throughput: a
result is yielded only after its transaction commits, so each item pays up to
`commit_interval_seconds` (default 0.01 s) of batching delay on top of the
commit itself.

## Restart workflow

1. Run with an artifact store and `resume=ResumePolicy.REUSE_SUCCESSES`.
2. On crash or controlled stop (deadline, fail-fast), fix the cause.
3. Re-run the **same source** with a fresh store instance on the same path.
   Compatible successes replay without provider calls; failures and
   never-accepted items execute normally.
4. Replayed items carry `replayed_from_artifact=True`; live token stats
   exclude their historical usage.

## Resource limits

- **File descriptors:** each worker's HTTP connection, the artifact database,
  its WAL/SHM sidecars, and your sink's outputs all consume descriptors —
  check `RLIMIT_NOFILE` per the
  [production checklist](production-checklist.md).
- **Disk:** the artifact grows with record count and payload size (the WAL
  stays at a bounded plateau during healthy writes). Budget from a measured
  bytes-per-record on your own payloads, not from anyone else's numbers.
- **Memory:** framework memory stays bounded under the pattern above;
  the on-disk database intentionally grows, and anything your sink retains
  is yours.

## Verify with the scale harness

The repository ships a deterministic, credential-free harness
(`benchmarks/scale_soak`) that drives these exact code paths — bounded
streaming, slow consumers, 429 waves, retry mixes, stop/resume, the SQLite
backend, and mixed RPM+TPM reservation/reconciliation — with exact item
accounting and resource assertions:

```bash
make scale-smoke     # reduced CI-sized regression, ~15 s
make scale-100k      # reference profile, minutes
make scale-1m        # large-scenario subset, long
```

See [Benchmarks](benchmarks.md#scale-soak-harness) for what each report
proves and — just as important — what it does not: fake-provider throughput is
not live-provider performance, local TPM smoothing is not GPU/KV-cache
scheduling, and no million-item claim is made without a completed 1m report.
The [v0.22 dated evidence](v0.22-scale-evidence.md) records exactly which 100k
and 1m scenarios were completed on the release-candidate behavior.
