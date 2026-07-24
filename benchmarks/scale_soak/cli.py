"""Argparse CLI for the scale-soak harness."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

from .config import (
    DURABILITIES,
    PROFILES,
    PROGRESS_MODES,
    SCENARIOS,
    STORES,
    HarnessConfig,
    resolve_settings,
    selected_scenarios,
)
from .report import ScenarioResult, build_report, validate_report, write_report
from .scenarios import SCENARIO_RUNNERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.scale_soak",
        description=(
            "Deterministic, credential-free scale/restart soak harness for "
            "async-batch-llm. Emits a versioned JSON report."
        ),
    )
    parser.add_argument("--profile", choices=(*PROFILES, "custom"), default="ci")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=(*SCENARIOS, "all"),
        help="repeatable; default all (the 1m profile defaults to its large subset)",
    )
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-queue-size", type=int, default=None)
    parser.add_argument("--max-result-queue-size", type=int, default=None)
    parser.add_argument("--store", choices=STORES, default="sqlite")
    parser.add_argument("--durability", choices=DURABILITIES, default="balanced")
    parser.add_argument("--seed", type=int, default=HarnessConfig().seed)
    parser.add_argument("--provider-latency-ms", type=float, default=1.0)
    parser.add_argument("--sink-latency-ms", type=float, default=0.0)
    parser.add_argument("--progress", choices=PROGRESS_MODES, default="compare")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/scale-soak.json"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="artifact scratch directory (default: a fresh temporary directory)",
    )
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--max-post-warmup-rss-growth-mib", type=float, default=None)
    parser.add_argument("--max-task-leak", type=int, default=0)
    parser.add_argument("--max-fd-leak", type=int, default=4)
    parser.add_argument(
        "--allow-large-jsonl",
        action="store_true",
        help="run the JSONL comparison at full item count (may be slow/large)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="keep framework retry/abort logging (noisy: scenarios inject "
        "thousands of deliberate failures)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> HarnessConfig:
    scenario_args = args.scenario or ["all"]
    scenarios = SCENARIOS if "all" in scenario_args else tuple(dict.fromkeys(scenario_args))
    config = HarnessConfig(
        profile=args.profile,
        scenarios=scenarios,
        items=args.items,
        concurrency=args.concurrency,
        max_queue_size=args.max_queue_size,
        max_result_queue_size=args.max_result_queue_size,
        store=args.store,
        durability=args.durability,
        seed=args.seed,
        provider_latency_ms=args.provider_latency_ms,
        sink_latency_ms=args.sink_latency_ms,
        progress=args.progress,
        output=args.output,
        work_dir=args.work_dir,
        keep_artifacts=args.keep_artifacts,
        max_post_warmup_rss_growth_mib=args.max_post_warmup_rss_growth_mib,
        max_task_leak=args.max_task_leak,
        max_fd_leak=args.max_fd_leak,
        allow_large_jsonl=args.allow_large_jsonl,
    )
    config.validate()
    return config


def _clear_work_dir(work_dir: Path) -> None:
    for path in work_dir.iterdir():
        try:
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def run_config(config: HarnessConfig) -> tuple[dict, int]:
    """Run every selected scenario; returns (report, exit_code)."""
    owns_work_dir = config.work_dir is None
    work_dir = config.work_dir or Path(tempfile.mkdtemp(prefix="abl-scale-soak-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[ScenarioResult] = []
    try:
        for name in selected_scenarios(config):
            settings = resolve_settings(config, name, work_dir)
            if config.progress == "disabled" and name == "progress_overhead":
                continue
            print(f"[scale-soak] {name}: {settings.items} items ...", flush=True)
            started = time.monotonic()
            # One fresh event loop per scenario keeps task/thread baselines
            # honest and prevents cross-scenario loop-state bleed.
            try:
                scenario_result = asyncio.run(SCENARIO_RUNNERS[name](settings))
            except Exception:
                scenario_result = ScenarioResult(name)
                scenario_result.error = traceback.format_exc(limit=20)
            results.append(scenario_result)
            print(
                f"[scale-soak] {name}: {scenario_result.status} "
                f"in {time.monotonic() - started:.1f}s",
                flush=True,
            )
            if not config.keep_artifacts:
                # A scenario's databases matter only within that scenario.
                # Dropping them immediately keeps the whole-profile footprint
                # at one scenario's artifacts (a 1m sqlite file is ~2 GB;
                # accumulating four of them can exhaust a tmpfs work dir).
                _clear_work_dir(work_dir)
    finally:
        if owns_work_dir and not config.keep_artifacts:
            shutil.rmtree(work_dir, ignore_errors=True)

    report = build_report(config.profile, results)
    problems = validate_report(report)
    for problem in problems:
        print(f"[scale-soak] report schema problem: {problem}", file=sys.stderr)
    failed = [result.name for result in results if result.status != "passed"]
    exit_code = 0 if not failed and not problems else 1
    if failed:
        print(f"[scale-soak] FAILED scenarios: {', '.join(failed)}", file=sys.stderr)
    return report, exit_code


def main(argv: list[str] | None = None) -> int:
    import logging

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.verbose:
        # The scenarios deliberately inject thousands of failures; the
        # framework's per-failure logging would swamp the run output.
        logging.getLogger("async_batch_llm").setLevel(logging.CRITICAL)
    report, exit_code = run_config(config)
    write_report(report, config.output)
    print(f"[scale-soak] report written to {config.output}")
    return exit_code
