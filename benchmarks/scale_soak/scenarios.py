"""The eight v0.21 scale-soak scenarios (plan §8.1).

Every scenario drives the real ``process_stream()`` execution surface (or the
public artifact contract for the microbenchmark) with the seeded fake
provider. Sources never materialize all items; sinks keep only counters,
digests, and bounded samples.
"""

from __future__ import annotations

import asyncio
import gc
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from async_batch_llm import (
    LLMWorkItem,
    ProcessorConfig,
    RateLimitConfig,
    ResumePolicy,
    RetryConfig,
    SqliteArtifactStore,
    WorkItemResult,
    process_stream,
)
from async_batch_llm.artifacts import ArtifactIdentity, JsonlArtifactStore
from async_batch_llm.core.config import AbortMode, GuardrailConfig
from async_batch_llm.observers import BaseObserver, ProcessingEvent
from async_batch_llm.streaming import _ProgressReporter

from .config import ScenarioSettings
from .fake_provider import (
    TOKENS_FAILED_ATTEMPT,
    TOKENS_OK,
    FakeProviderConfig,
    FakeProviderStrategy,
    ScaleSoakClassifier,
    behavior_hash,
    item_id_for,
)
from .monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    loop_default_executor_thread_count,
    peak_rss_bytes,
    sqlite_executor_thread_count,
)
from .report import RollingDigest, ScenarioResult

TRANSPORT_MODULUS = 17
VALIDATION_MODULUS = 13


# ── Shared plumbing ─────────────────────────────────────────────────────


class StatsCapture(BaseObserver):
    """Counts processor events; only bounded aggregates are retained."""

    def __init__(self) -> None:
        self.batch_completed: dict[str, Any] | None = None
        self.cooldowns_started = 0
        self.cooldowns_ended = 0
        self.rate_limit_hits = 0
        self.replayed = 0
        self.terminal_failure_categories: dict[str, int] = {}

    async def on_event(self, event: ProcessingEvent, data: dict[str, Any]) -> None:
        if event is ProcessingEvent.BATCH_COMPLETED:
            self.batch_completed = data
        elif event is ProcessingEvent.COOLDOWN_STARTED:
            self.cooldowns_started += 1
        elif event is ProcessingEvent.COOLDOWN_ENDED:
            self.cooldowns_ended += 1
        elif event is ProcessingEvent.RATE_LIMIT_HIT:
            self.rate_limit_hits += 1
        elif event is ProcessingEvent.ITEM_REPLAYED:
            self.replayed += 1
        elif event is ProcessingEvent.ITEM_FAILED:
            category = str(data.get("error_category"))
            self.terminal_failure_categories[category] = (
                self.terminal_failure_categories.get(category, 0) + 1
            )


def harness_processor_config(
    settings: ScenarioSettings,
    *,
    guardrails: GuardrailConfig | None = None,
) -> ProcessorConfig:
    """Millisecond-scale retry/cooldown waits; everything else per settings."""
    return ProcessorConfig(
        concurrency=settings.concurrency,
        max_queue_size=settings.max_queue_size,
        max_result_queue_size=settings.max_result_queue_size,
        attempt_timeout=30.0,
        retry=RetryConfig(
            max_attempts=3,
            initial_wait=0.002,
            max_wait=0.02,
            jitter=False,
        ),
        rate_limit=RateLimitConfig(
            cooldown_seconds=0.02,
            slow_start_items=4,
            slow_start_initial_delay=0.004,
            slow_start_final_delay=0.001,
            backoff_multiplier=1.5,
        ),
        guardrails=guardrails or GuardrailConfig(),
    )


def make_store(
    settings: ScenarioSettings, stem: str
) -> SqliteArtifactStore | JsonlArtifactStore | None:
    if settings.store == "none":
        return None
    if settings.store == "jsonl":
        return JsonlArtifactStore(settings.work_dir / f"{stem}.jsonl")
    return SqliteArtifactStore(
        settings.work_dir / f"{stem}.sqlite",
        commit_batch_size=settings.commit_batch_size,
        commit_interval_seconds=settings.commit_interval_seconds,
        durability=settings.durability,
    )


async def source(item_count: int) -> AsyncIterator[tuple[str, str]]:
    """Lazy source: ids/prompts derived, never materialized."""
    for index in range(item_count):
        item_id = item_id_for(index)
        yield item_id, f"p:{item_id}"


def expected_digest(
    item_count: int, success_for: Callable[[int], bool] | None = None
) -> RollingDigest:
    digest = RollingDigest()
    for index in range(item_count):
        success = True if success_for is None else success_for(index)
        digest.add(item_id_for(index), index, success)
    return digest


def count_matching(item_count: int, seed: int, modulus: int, residue: int) -> int:
    return sum(1 for i in range(item_count) if behavior_hash(seed, i) % modulus == residue)


def record_common(
    result: ScenarioResult,
    settings: ScenarioSettings,
    strategy: FakeProviderStrategy,
    capture: StatsCapture,
    wall_seconds: float,
) -> dict[str, Any]:
    stats = capture.batch_completed or {}
    result.configuration = {
        "items": settings.items,
        "concurrency": settings.concurrency,
        "max_queue_size": settings.max_queue_size,
        "max_result_queue_size": settings.max_result_queue_size,
        "store": settings.store,
        "durability": settings.durability,
        "seed": settings.seed,
        "provider_latency_ms": settings.provider_latency_s * 1000,
        "sink_latency_ms": settings.sink_latency_s * 1000,
        "commit_batch_size": settings.commit_batch_size,
        "commit_interval_seconds": settings.commit_interval_seconds,
        "retry": {"max_attempts": 3, "initial_wait": 0.002},
        "rate_limit": {"cooldown_seconds": 0.02, "slow_start_items": 4},
    }
    result.counts = {
        "accepted": stats.get("total"),
        "processed": stats.get("processed"),
        "succeeded": stats.get("succeeded"),
        "failed": stats.get("failed"),
        "replayed": capture.replayed,
        "rate_limit_hits": capture.rate_limit_hits,
        "terminal_failure_categories": capture.terminal_failure_categories,
        "total_tokens": stats.get("total_tokens"),
    }
    result.throughput = {
        "wall_seconds": round(wall_seconds, 3),
        "items_per_second": (round(settings.items / wall_seconds, 1) if wall_seconds > 0 else None),
    }
    result.provider = {
        "calls": strategy.calls,
        "peak_concurrent_calls": strategy.peak_concurrent_calls,
        "rate_limit_failures": strategy.rate_limit_failures,
        "transport_failures": strategy.transport_failures,
        "validation_failures": strategy.validation_failures,
        "cooldowns_started": capture.cooldowns_started,
    }
    result.queues = {
        "input_bound": settings.max_queue_size,
        "result_bound": settings.max_result_queue_size,
        "input_high_water": stats.get("input_queue_high_water_mark"),
        "result_high_water": stats.get("result_queue_high_water_mark"),
    }
    return stats


def check_queue_bounds(
    result: ScenarioResult, settings: ScenarioSettings, stats: dict[str, Any]
) -> None:
    input_high = stats.get("input_queue_high_water_mark")
    result_high = stats.get("result_queue_high_water_mark")
    if settings.max_queue_size > 0:
        result.check(
            "input_high_water_within_bound",
            isinstance(input_high, int) and input_high <= settings.max_queue_size,
            f"high water {input_high} bound {settings.max_queue_size}",
        )
    if settings.max_result_queue_size > 0:
        result.check(
            "result_high_water_within_bound",
            isinstance(result_high, int) and result_high <= settings.max_result_queue_size,
            f"high water {result_high} bound {settings.max_result_queue_size}",
        )


async def wait_for_store_threads_gone(timeout_s: float = 5.0) -> int:
    """Store executor threads terminate asynchronously; wait briefly."""
    deadline = time.monotonic() + timeout_s
    while sqlite_executor_thread_count() > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    return sqlite_executor_thread_count()


async def check_cleanup(
    result: ScenarioResult,
    settings: ScenarioSettings,
    before: ResourceSnapshot,
    monitor: ResourceMonitor,
) -> None:
    gc.collect()
    store_threads = await wait_for_store_threads_gone()
    default_executor_threads = loop_default_executor_thread_count()
    after = ResourceSnapshot.capture()
    result.resources = {
        "baseline": before.to_json(),
        "after_cleanup": after.to_json(),
        "store_executor_threads_after": store_threads,
        "loop_default_executor_threads": default_executor_threads,
        "peak_rss_bytes": peak_rss_bytes(),
        "monitor": monitor.to_json(),
    }
    result.check(
        "store_executor_threads_terminated",
        store_threads == 0,
        f"named store executor threads remaining: {store_threads}",
    )
    result.check(
        "threads_return_to_baseline",
        after.thread_count - default_executor_threads <= before.thread_count,
        f"before {before.thread_count} after {after.thread_count} "
        f"(excluding {default_executor_threads} loop default-executor workers, "
        "which asyncio.run() reclaims at loop shutdown)",
    )
    if before.task_count is not None and after.task_count is not None:
        result.check(
            "tasks_return_to_baseline",
            after.task_count <= before.task_count + settings.max_task_leak,
            f"before {before.task_count} after {after.task_count}",
        )
    else:
        result.caveats.append("task counts unavailable; task-leak check skipped")
    if before.fd_count is not None and after.fd_count is not None:
        result.check(
            "fds_return_to_baseline",
            after.fd_count <= before.fd_count + settings.max_fd_leak,
            f"before {before.fd_count} after {after.fd_count} tolerance {settings.max_fd_leak}",
        )
    else:
        result.caveats.append("fd counts unavailable; fd-leak check skipped")
    growth = monitor.post_warmup_growth_bytes()
    if settings.max_post_warmup_rss_growth_mib is not None:
        limit = settings.max_post_warmup_rss_growth_mib * 1024 * 1024
        result.check(
            "post_warmup_rss_growth_within_limit",
            growth is not None and growth <= limit,
            f"growth {growth} limit {int(limit)} bytes",
        )


def check_sqlite_wal_after_close(
    result: ScenarioResult, settings: ScenarioSettings, db_path: Path
) -> None:
    if settings.store != "sqlite":
        return
    wal = Path(f"{db_path}-wal")
    size = wal.stat().st_size if wal.exists() else 0
    result.check(
        "wal_truncated_after_close",
        size == 0,
        f"wal bytes after close: {size}",
    )


async def drain(
    settings: ScenarioSettings,
    strategy: FakeProviderStrategy,
    *,
    store: SqliteArtifactStore | JsonlArtifactStore | None,
    resume: ResumePolicy = ResumePolicy.NONE,
    guardrails: GuardrailConfig | None = None,
    per_result: Callable[[WorkItemResult[Any, Any], int], Awaitable[None]] | None = None,
    items: int | None = None,
    progress: Any = False,
) -> tuple[RollingDigest, StatsCapture, float, int, int]:
    """Run the stream, folding results into a digest without retaining them."""
    capture = StatsCapture()
    digest = RollingDigest()
    succeeded = 0
    failed = 0
    item_count = items if items is not None else settings.items
    started = time.monotonic()
    seen = 0
    async for result in process_stream(
        strategy,
        source(item_count),
        config=harness_processor_config(settings, guardrails=guardrails),
        artifact_store=store,
        resume=resume,
        error_classifier=ScaleSoakClassifier(),
        observers=[capture],
        progress=progress,
    ):
        index = int(result.item_id[1:])
        digest.add(result.item_id, index, result.success)
        if result.success:
            succeeded += 1
        else:
            failed += 1
        if per_result is not None:
            await per_result(result, seen)
        seen += 1
    wall = time.monotonic() - started
    return digest, capture, wall, succeeded, failed


# ── Scenario A: healthy fixed-latency throughput ─────────────────────────


async def run_healthy(settings: ScenarioSettings) -> ScenarioResult:
    result = ScenarioResult("healthy")
    before = ResourceSnapshot.capture()
    db_path = settings.work_dir / "healthy.sqlite"
    monitor = ResourceMonitor(
        interval_s=settings.monitor_interval_s,
        wal_path=Path(f"{db_path}-wal") if settings.store == "sqlite" else None,
    )
    monitor.start()
    strategy = FakeProviderStrategy(
        FakeProviderConfig(seed=settings.seed, latency_s=settings.provider_latency_s)
    )
    store = make_store(settings, "healthy")
    monitor.mark_warmup()
    digest, capture, wall, succeeded, failed = await drain(settings, strategy, store=store)
    await monitor.stop()

    stats = record_common(result, settings, strategy, capture, wall)
    expected = expected_digest(settings.items)
    result.check("digest_matches", digest == expected, str(digest.to_json()))
    result.check("all_succeeded", succeeded == settings.items and failed == 0)
    result.check(
        "provider_calls_exact",
        strategy.calls == settings.items,
        f"calls {strategy.calls} items {settings.items}",
    )
    result.check("strategy_lifecycle", strategy.prepared == 1 and strategy.cleaned_up == 1)
    check_queue_bounds(result, settings, stats)
    check_sqlite_wal_after_close(result, settings, db_path)
    await check_cleanup(result, settings, before, monitor)
    return result


# ── Scenario B: slow result consumer ─────────────────────────────────────


async def run_slow_consumer(settings: ScenarioSettings) -> ScenarioResult:
    import dataclasses

    result = ScenarioResult("slow_consumer")
    before = ResourceSnapshot.capture()
    result_bound = min(settings.max_result_queue_size or 8, 8)
    settings = dataclasses.replace(settings, max_result_queue_size=result_bound)
    sink_latency = max(settings.sink_latency_s, settings.provider_latency_s * 4, 0.002)
    slow_results = min(settings.items, max(result_bound * 8, 64))

    monitor = ResourceMonitor(interval_s=settings.monitor_interval_s)
    monitor.start()
    strategy = FakeProviderStrategy(
        FakeProviderConfig(seed=settings.seed, latency_s=settings.provider_latency_s)
    )
    store = make_store(settings, "slow_consumer")

    async def slow_sink(_result: WorkItemResult[Any, Any], seen: int) -> None:
        if seen < slow_results:
            await asyncio.sleep(sink_latency)

    monitor.mark_warmup()
    digest, capture, wall, succeeded, failed = await drain(
        settings, strategy, store=store, per_result=slow_sink
    )
    await monitor.stop()

    stats = record_common(result, settings, strategy, capture, wall)
    result.configuration["slow_results"] = slow_results
    result.configuration["effective_sink_latency_ms"] = sink_latency * 1000
    result.check("digest_matches", digest == expected_digest(settings.items))
    result.check("all_succeeded", succeeded == settings.items and failed == 0)
    result_high = stats.get("result_queue_high_water_mark")
    result.check(
        "result_backpressure_reached_bound",
        result_high == result_bound,
        f"high water {result_high} bound {result_bound}",
    )
    check_queue_bounds(result, settings, stats)
    result.check(
        "no_result_lost_or_duplicated",
        digest.count == settings.items,
        f"consumed {digest.count}",
    )
    await check_cleanup(result, settings, before, monitor)
    return result


# ── Scenario C: coordinated 429/overload wave ────────────────────────────


async def run_rate_limit_wave(settings: ScenarioSettings) -> ScenarioResult:
    result = ScenarioResult("rate_limit_wave")
    before = ResourceSnapshot.capture()
    wave_size = max(min(settings.items // 50, 512), min(settings.concurrency * 2, settings.items))
    wave_start = settings.items // 4
    wave = (wave_start, min(wave_start + wave_size, settings.items))
    wave_size = wave[1] - wave[0]

    monitor = ResourceMonitor(interval_s=settings.monitor_interval_s)
    monitor.start()
    strategy = FakeProviderStrategy(
        FakeProviderConfig(
            seed=settings.seed,
            latency_s=settings.provider_latency_s,
            rate_limit_wave=wave,
        )
    )
    store = make_store(settings, "rate_limit_wave")
    monitor.mark_warmup()
    digest, capture, wall, succeeded, failed = await drain(settings, strategy, store=store)
    await monitor.stop()

    stats = record_common(result, settings, strategy, capture, wall)
    result.configuration["wave"] = list(wave)
    result.check("digest_matches", digest == expected_digest(settings.items))
    result.check("all_succeeded", succeeded == settings.items and failed == 0)
    result.check(
        "no_retry_storm_exact_attempts",
        strategy.calls == settings.items + wave_size,
        f"calls {strategy.calls} expected {settings.items + wave_size}",
    )
    result.check(
        "coordinated_cooldown_engaged",
        capture.cooldowns_started >= 1 and capture.rate_limit_hits >= 1,
        f"cooldowns {capture.cooldowns_started} rate_limit_hits {capture.rate_limit_hits}",
    )
    result.check(
        "cooldowns_completed",
        capture.cooldowns_ended == capture.cooldowns_started,
        f"started {capture.cooldowns_started} ended {capture.cooldowns_ended}",
    )
    check_queue_bounds(result, settings, stats)
    await check_cleanup(result, settings, before, monitor)
    return result


# ── Scenario D: network/5xx retry mix ────────────────────────────────────


async def run_transport_retry(settings: ScenarioSettings) -> ScenarioResult:
    result = ScenarioResult("transport_retry")
    before = ResourceSnapshot.capture()
    monitor = ResourceMonitor(interval_s=settings.monitor_interval_s)
    monitor.start()
    strategy = FakeProviderStrategy(
        FakeProviderConfig(
            seed=settings.seed,
            latency_s=settings.provider_latency_s,
            transport_modulus=TRANSPORT_MODULUS,
        )
    )
    store = make_store(settings, "transport_retry")
    monitor.mark_warmup()
    digest, capture, wall, succeeded, failed = await drain(settings, strategy, store=store)
    await monitor.stop()

    expected_failures = count_matching(settings.items, settings.seed, TRANSPORT_MODULUS, 0)
    stats = record_common(result, settings, strategy, capture, wall)
    result.configuration["transport_modulus"] = TRANSPORT_MODULUS
    result.check("digest_matches", digest == expected_digest(settings.items))
    result.check("all_succeeded", succeeded == settings.items and failed == 0)
    result.check(
        "physical_retries_exact",
        strategy.calls == settings.items + expected_failures,
        f"calls {strategy.calls} expected {settings.items + expected_failures}",
    )
    result.check(
        "no_terminal_failures",
        not capture.terminal_failure_categories,
        str(capture.terminal_failure_categories),
    )
    result.check(
        "token_totals_exclude_transport_failures",
        stats.get("total_tokens") == settings.items * TOKENS_OK["total_tokens"],
        f"total_tokens {stats.get('total_tokens')}",
    )
    check_queue_bounds(result, settings, stats)
    await check_cleanup(result, settings, before, monitor)
    return result


# ── Scenario E: stateful validation recovery ─────────────────────────────


async def run_validation_recovery(settings: ScenarioSettings) -> ScenarioResult:
    result = ScenarioResult("validation_recovery")
    before = ResourceSnapshot.capture()
    monitor = ResourceMonitor(interval_s=settings.monitor_interval_s)
    monitor.start()
    strategy = FakeProviderStrategy(
        FakeProviderConfig(
            seed=settings.seed,
            latency_s=settings.provider_latency_s,
            validation_modulus=VALIDATION_MODULUS,
        )
    )
    store = make_store(settings, "validation_recovery")
    monitor.mark_warmup()
    digest, capture, wall, succeeded, failed = await drain(settings, strategy, store=store)
    await monitor.stop()

    recovered = count_matching(settings.items, settings.seed, VALIDATION_MODULUS, 1)
    stats = record_common(result, settings, strategy, capture, wall)
    result.configuration["validation_modulus"] = VALIDATION_MODULUS
    result.check("digest_matches", digest == expected_digest(settings.items))
    result.check("all_succeeded", succeeded == settings.items and failed == 0)
    result.check(
        "recovery_attempts_exact",
        strategy.calls == settings.items + recovered,
        f"calls {strategy.calls} expected {settings.items + recovered}",
    )
    result.check(
        "no_cross_item_state_leak",
        not strategy.isolation_violations,
        "; ".join(strategy.isolation_violations[:5]) or "no violations",
    )
    expected_total = (
        settings.items * TOKENS_OK["total_tokens"]
        + recovered * TOKENS_FAILED_ATTEMPT["total_tokens"]
    )
    result.check(
        "failed_attempt_tokens_retained",
        stats.get("total_tokens") == expected_total,
        f"total_tokens {stats.get('total_tokens')} expected {expected_total}",
    )
    check_queue_bounds(result, settings, stats)
    await check_cleanup(result, settings, before, monitor)
    return result


# ── Scenario F: controlled stop and resume ───────────────────────────────


def _sqlite_row_count(db_path: Path) -> int | None:
    try:
        with sqlite3.connect(db_path) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM item_records").fetchone()[0])
    except sqlite3.DatabaseError:
        return None


async def run_stop_resume(settings: ScenarioSettings) -> ScenarioResult:
    result = ScenarioResult("stop_resume")
    before = ResourceSnapshot.capture()
    stem = "stop_resume"
    db_path = settings.work_dir / f"{stem}.sqlite"
    monitor = ResourceMonitor(
        interval_s=settings.monitor_interval_s,
        wal_path=Path(f"{db_path}-wal") if settings.store == "sqlite" else None,
    )
    monitor.start()
    poison_index = settings.items // 3

    # Run 1: deterministic fail-fast on the poison category.
    strategy_one = FakeProviderStrategy(
        FakeProviderConfig(
            seed=settings.seed,
            latency_s=settings.provider_latency_s,
            poison_index=poison_index,
        )
    )
    store_one = make_store(settings, stem)
    guardrails = GuardrailConfig(
        abort_on_error_categories=frozenset({"poison_pill"}),
        abort_mode=AbortMode.DRAIN_ACTIVE,
    )
    monitor.mark_warmup()
    digest_one, capture_one, wall_one, succeeded_one, failed_one = await drain(
        settings, strategy_one, store=store_one, guardrails=guardrails
    )
    rows_after_run_one = _sqlite_row_count(db_path) if settings.store == "sqlite" else None

    # Run 2: fresh store on the same database, healthy provider, full source.
    strategy_two = FakeProviderStrategy(
        FakeProviderConfig(seed=settings.seed, latency_s=settings.provider_latency_s)
    )
    store_two = make_store(settings, stem)
    digest_two, capture_two, wall_two, succeeded_two, failed_two = await drain(
        settings, strategy_two, store=store_two, resume=ResumePolicy.REUSE_SUCCESSES
    )
    await monitor.stop()

    stats_two = record_common(result, settings, strategy_two, capture_two, wall_two)
    result.configuration["poison_index"] = poison_index
    result.counts["run_one"] = {
        "succeeded": succeeded_one,
        "failed": failed_one,
        "terminal_failure_categories": capture_one.terminal_failure_categories,
        "wall_seconds": round(wall_one, 3),
    }
    result.check(
        "run_one_stopped_early",
        succeeded_one < settings.items and failed_one >= 1,
        f"succeeded {succeeded_one} failed {failed_one} of {settings.items}",
    )
    result.check(
        "resume_completes_all_items",
        succeeded_two == settings.items and failed_two == 0,
        f"succeeded {succeeded_two} failed {failed_two}",
    )
    result.check("digest_matches_full_source", digest_two == expected_digest(settings.items))
    result.check(
        "replayed_equals_first_run_successes",
        capture_two.replayed == succeeded_one,
        f"replayed {capture_two.replayed} first-run successes {succeeded_one}",
    )
    result.check(
        "no_provider_calls_for_replayed",
        strategy_two.calls == settings.items - succeeded_one,
        f"calls {strategy_two.calls} expected {settings.items - succeeded_one}",
    )
    result.check(
        "live_tokens_exclude_replayed",
        stats_two.get("total_tokens")
        == (settings.items - succeeded_one) * TOKENS_OK["total_tokens"],
        f"total_tokens {stats_two.get('total_tokens')}",
    )
    if settings.store == "sqlite" and rows_after_run_one is not None:
        rows_after_run_two = _sqlite_row_count(db_path)
        expected_new_rows = settings.items - succeeded_one
        result.check(
            "no_duplicate_rows_for_replays",
            rows_after_run_two == rows_after_run_one + expected_new_rows,
            f"rows {rows_after_run_one} -> {rows_after_run_two}, expected +{expected_new_rows}",
        )
    check_sqlite_wal_after_close(result, settings, db_path)
    await check_cleanup(result, settings, before, monitor)
    return result


# ── Scenario G: artifact backend microbenchmark ──────────────────────────


def _bench_identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        provider="scale-soak",
        model="fake-1",
        prompt_version="v1",
        parser_version="v1",
        application_version="bench",
    )


def _bench_work_item(strategy: FakeProviderStrategy, index: int) -> LLMWorkItem[Any, str, Any]:
    item_id = item_id_for(index)
    return LLMWorkItem(item_id=item_id, strategy=strategy, prompt=f"p:{item_id}")


def _bench_result(index: int) -> WorkItemResult[str, Any]:
    item_id = item_id_for(index)
    return WorkItemResult(
        item_id=item_id, success=True, output=f"ok:{item_id}", token_usage=dict(TOKENS_OK)
    )


async def _append_range(
    store: SqliteArtifactStore | JsonlArtifactStore,
    strategy: FakeProviderStrategy,
    start: int,
    stop: int,
    *,
    window: int = 256,
) -> None:
    async def append_one(index: int) -> None:
        item = _bench_work_item(strategy, index)
        prepared = await store.prepare_item(item)
        await store.append(item, prepared, _bench_result(index))

    pending: set[asyncio.Task[None]] = set()
    for index in range(start, stop):
        if len(pending) >= window:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        pending.add(asyncio.create_task(append_one(index)))
    if pending:
        done, _ = await asyncio.wait(pending)
        for task in done:
            task.result()


async def run_artifact_bench(settings: ScenarioSettings) -> ScenarioResult:
    result = ScenarioResult("artifact_bench")
    before = ResourceSnapshot.capture()
    strategy = FakeProviderStrategy(FakeProviderConfig(seed=settings.seed))
    db_path = settings.work_dir / "artifact_bench.sqlite"
    wal_path = Path(f"{db_path}-wal")
    monitor = ResourceMonitor(interval_s=settings.monitor_interval_s, wal_path=wal_path)
    monitor.start()

    store = SqliteArtifactStore(
        db_path,
        identity=_bench_identity(),
        commit_batch_size=settings.commit_batch_size,
        commit_interval_seconds=settings.commit_interval_seconds,
        durability=settings.durability,
    )
    monitor.mark_warmup()

    started = time.monotonic()
    await _append_range(store, strategy, 0, settings.items)
    append_wall = time.monotonic() - started
    transactions = store._transaction_count
    page_size = store._page_size
    autocheckpoint = store._effective_wal_autocheckpoint_pages

    close_started = time.monotonic()
    await store.close()
    close_wall = time.monotonic() - close_started
    checkpoint_busy = store._last_checkpoint_busy
    wal_after_close = wal_path.stat().st_size if wal_path.exists() else 0

    reopen_started = time.monotonic()
    reopened = SqliteArtifactStore(db_path, identity=_bench_identity())
    probe_item = _bench_work_item(strategy, 0)
    prepared = await reopened.prepare_item(probe_item)
    reopen_wall = time.monotonic() - reopen_started

    lookups = min(settings.items, 2_000)
    stride = max(settings.items // lookups, 1)
    lookup_started = time.monotonic()
    found = 0
    for index in range(0, lookups * stride, stride):
        item = _bench_work_item(strategy, index)
        prepared = await reopened.prepare_item(item)
        row = await reopened.lookup(item, prepared, ResumePolicy.REUSE_SUCCESSES)
        if row is not None:
            found += 1
    lookup_wall = time.monotonic() - lookup_started

    iter_started = time.monotonic()
    iterated = 0
    async for _stored in reopened.iter_results():
        iterated += 1
    iter_wall = time.monotonic() - iter_started
    await reopened.close()

    # Concurrent inspection: slow reader pages while a writer appends.
    inspect_items = min(settings.items, 10_000)
    writer_store = SqliteArtifactStore(
        db_path,
        identity=_bench_identity(),
        commit_batch_size=settings.commit_batch_size,
        commit_interval_seconds=settings.commit_interval_seconds,
        read_batch_size=200,
    )
    inspect_seen = 0
    peak_wal_during_inspect = 0

    async def slow_inspect() -> None:
        nonlocal inspect_seen, peak_wal_during_inspect
        async for _stored in writer_store.iter_results():
            inspect_seen += 1
            if inspect_seen % 200 == 0:
                await asyncio.sleep(0.005)
                if wal_path.exists():
                    peak_wal_during_inspect = max(peak_wal_during_inspect, wal_path.stat().st_size)

    writer_task = asyncio.create_task(
        _append_range(writer_store, strategy, settings.items, settings.items + inspect_items)
    )
    inspect_task = asyncio.create_task(slow_inspect())
    await asyncio.gather(writer_task, inspect_task)
    await writer_store.close()
    await monitor.stop()

    db_bytes = db_path.stat().st_size
    result.configuration = {
        "items": settings.items,
        "durability": settings.durability,
        "commit_batch_size": settings.commit_batch_size,
        "commit_interval_seconds": settings.commit_interval_seconds,
        "inspect_items": inspect_items,
        "lookups": lookups,
    }
    result.artifact = {
        "append_wall_seconds": round(append_wall, 3),
        "append_records_per_second": round(settings.items / append_wall, 1),
        "transactions": transactions,
        "records_per_transaction": round(settings.items / max(transactions, 1), 1),
        "database_bytes": db_bytes,
        "bytes_per_record": round(db_bytes / settings.items, 1),
        "sqlite_page_size": page_size,
        "wal_autocheckpoint_pages": autocheckpoint,
        "peak_wal_bytes": monitor.peak_wal_bytes,
        "wal_bytes_after_close": wal_after_close,
        "close_wall_seconds": round(close_wall, 4),
        "close_checkpoint_busy": checkpoint_busy,
        "reopen_seconds": round(reopen_wall, 4),
        "lookup_count": lookups,
        "lookups_per_second": round(lookups / lookup_wall, 1) if lookup_wall > 0 else None,
        "iteration_records_per_second": (round(iterated / iter_wall, 1) if iter_wall > 0 else None),
        "concurrent_inspect_seen": inspect_seen,
        "peak_wal_bytes_during_concurrent_inspect": peak_wal_during_inspect,
    }
    result.check("all_appends_committed", transactions >= 1, f"transactions {transactions}")
    result.check(
        "iteration_complete",
        iterated == settings.items,
        f"iterated {iterated} expected {settings.items}",
    )
    result.check("all_lookups_replayable", found == lookups, f"found {found} of {lookups}")
    result.check("wal_truncated_after_close", wal_after_close == 0)
    if page_size is not None and autocheckpoint is not None:
        plateau_limit = page_size * autocheckpoint * 8
        result.check(
            "wal_plateau_bounded",
            (monitor.peak_wal_bytes or 0) <= plateau_limit,
            f"peak wal {monitor.peak_wal_bytes} limit {plateau_limit}",
        )
        result.check(
            "concurrent_inspect_wal_bounded",
            peak_wal_during_inspect <= plateau_limit,
            f"peak wal during inspect {peak_wal_during_inspect} limit {plateau_limit}",
        )
    result.check(
        "writer_progress_during_slow_inspection",
        inspect_seen > 0,
        f"reader saw {inspect_seen} rows while writer appended {inspect_items}",
    )

    # JSONL comparison at a safe size (explicit override required for large).
    jsonl_items = settings.items
    if settings.items > 20_000 and not settings.allow_large_jsonl:
        jsonl_items = 20_000
        result.caveats.append(
            f"JSONL comparison reduced to {jsonl_items} records; "
            "pass --allow-large-jsonl to compare at full size"
        )
    jsonl_path = settings.work_dir / "artifact_bench.jsonl"
    jsonl_store = JsonlArtifactStore(jsonl_path, identity=_bench_identity())
    jsonl_started = time.monotonic()
    await _append_range(jsonl_store, strategy, 0, jsonl_items, window=64)
    jsonl_wall = time.monotonic() - jsonl_started
    await jsonl_store.close()
    result.artifact["jsonl_comparison"] = {
        "records": jsonl_items,
        "append_wall_seconds": round(jsonl_wall, 3),
        "append_records_per_second": round(jsonl_items / jsonl_wall, 1),
        "file_bytes": jsonl_path.stat().st_size,
    }

    await check_cleanup(result, settings, before, monitor)
    return result


# ── Scenario H: progress overhead ────────────────────────────────────────


class _CountingBar:
    """Non-terminal stand-in for tqdm; counts refreshes only."""

    def __init__(self, total: int | None = None, **_: Any) -> None:
        self.total = total
        self.n = 0
        self.refreshes = 0
        self.closed = False

    def refresh(self) -> None:
        self.refreshes += 1

    def close(self) -> None:
        self.closed = True


async def run_progress_overhead(settings: ScenarioSettings) -> ScenarioResult:
    result = ScenarioResult("progress_overhead")
    before = ResourceSnapshot.capture()
    monitor = ResourceMonitor(interval_s=settings.monitor_interval_s)
    monitor.start()
    refresh_interval = 0.05

    async def timed_run(progress: Any) -> tuple[float, RollingDigest]:
        strategy = FakeProviderStrategy(
            FakeProviderConfig(seed=settings.seed, latency_s=settings.provider_latency_s)
        )
        digest, _capture, wall, _s, _f = await drain(
            settings, strategy, store=None, progress=progress
        )
        return wall, digest

    monitor.mark_warmup()
    disabled_wall, disabled_digest = await timed_run(False)

    bars: list[_CountingBar] = []

    def bar_factory(**kwargs: Any) -> _CountingBar:
        bar = _CountingBar(**kwargs)
        bars.append(bar)
        return bar

    reporter = _ProgressReporter(refresh_interval, bar_factory=bar_factory)
    bundled_wall, bundled_digest = await timed_run(reporter)
    await reporter.aclose()
    renders = sum(bar.refreshes for bar in bars)

    callback_count = 0

    async def user_callback(completed: int, total: int, current_item_id: str) -> None:
        nonlocal callback_count
        callback_count += 1

    user_wall, user_digest = await timed_run(user_callback)
    await monitor.stop()

    expected = expected_digest(settings.items)
    result.configuration = {
        "items": settings.items,
        "concurrency": settings.concurrency,
        "refresh_interval_seconds": refresh_interval,
    }
    result.throughput = {
        "disabled_wall_seconds": round(disabled_wall, 3),
        "bundled_wall_seconds": round(bundled_wall, 3),
        "user_callback_wall_seconds": round(user_wall, 3),
        "bundled_renders": renders,
        "user_callback_invocations": callback_count,
    }
    result.check(
        "digests_match_in_all_modes",
        disabled_digest == expected and bundled_digest == expected and user_digest == expected,
    )
    render_budget = int(bundled_wall / refresh_interval) + 3
    result.check(
        "bundled_renders_bounded_by_time_not_items",
        renders <= render_budget,
        f"renders {renders} budget {render_budget} items {settings.items}",
    )
    result.check(
        "user_callback_sees_every_item",
        callback_count == settings.items,
        f"invocations {callback_count}",
    )
    result.caveats.append(
        "bundled mode drives the private _ProgressReporter with an injected "
        "non-terminal bar; user-facing behavior is progress=True"
    )
    await check_cleanup(result, settings, before, monitor)
    return result


SCENARIO_RUNNERS: dict[str, Callable[[ScenarioSettings], Awaitable[ScenarioResult]]] = {
    "healthy": run_healthy,
    "slow_consumer": run_slow_consumer,
    "rate_limit_wave": run_rate_limit_wave,
    "transport_retry": run_transport_retry,
    "validation_recovery": run_validation_recovery,
    "stop_resume": run_stop_resume,
    "artifact_bench": run_artifact_bench,
    "progress_overhead": run_progress_overhead,
}
