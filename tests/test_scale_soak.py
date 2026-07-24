"""Unit and reduced end-to-end tests for the scale-soak harness (plan §8.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.scale_soak import monitor as monitor_module
from benchmarks.scale_soak.cli import build_parser, config_from_args, run_config
from benchmarks.scale_soak.config import (
    PROFILES,
    SCENARIOS,
    HarnessConfig,
    resolve_settings,
    selected_scenarios,
)
from benchmarks.scale_soak.fake_provider import (
    FakeProviderConfig,
    FakeProviderStrategy,
    behavior_hash,
)
from benchmarks.scale_soak.monitor import MAX_SAMPLES, ResourceMonitor
from benchmarks.scale_soak.report import (
    REPORT_SCHEMA_NAME,
    REPORT_SCHEMA_VERSION,
    RollingDigest,
    ScenarioResult,
    build_report,
    validate_report,
)
from benchmarks.scale_soak.scenarios import SCENARIO_RUNNERS

pytestmark = pytest.mark.timeout(120)


def _settings(scenario: str, tmp_path: Path, items: int = 80, **overrides):
    config = HarnessConfig(profile="ci", items=items, concurrency=8)
    for key, value in overrides.items():
        setattr(config, key, value)
    config.validate()
    return resolve_settings(config, scenario, tmp_path)


# ── CLI parsing and validation ───────────────────────────────────────────


def test_cli_defaults_select_all_scenarios() -> None:
    args = build_parser().parse_args([])
    config = config_from_args(args)
    assert config.profile == "ci"
    assert config.scenarios == SCENARIOS


def test_cli_rejects_custom_profile_without_items() -> None:
    args = build_parser().parse_args(["--profile", "custom"])
    with pytest.raises(ValueError, match="requires --items"):
        config_from_args(args)


def test_cli_rejects_store_none_with_store_scenarios() -> None:
    args = build_parser().parse_args(["--store", "none"])
    with pytest.raises(ValueError, match="store none"):
        config_from_args(args)


def test_cli_scenario_subset_and_dedup() -> None:
    args = build_parser().parse_args(
        ["--scenario", "healthy", "--scenario", "healthy", "--scenario", "slow_consumer"]
    )
    config = config_from_args(args)
    assert config.scenarios == ("healthy", "slow_consumer")


def test_one_million_profile_defaults_to_large_subset() -> None:
    config = HarnessConfig(profile="1m")
    assert selected_scenarios(config) == (
        "healthy",
        "slow_consumer",
        "stop_resume",
        "artifact_bench",
    )


def test_profiles_cover_ci_100k_1m() -> None:
    assert set(PROFILES) == {"ci", "100k", "1m"}
    assert PROFILES["100k"].items == 100_000
    assert PROFILES["1m"].items == 1_000_000


# ── Digest ───────────────────────────────────────────────────────────────


def test_digest_is_order_independent() -> None:
    forward = RollingDigest()
    backward = RollingDigest()
    for index in range(500):
        forward.add(f"i{index}", index, True)
    for index in reversed(range(500)):
        backward.add(f"i{index}", index, True)
    assert forward == backward


def test_digest_detects_loss_duplication_and_status_change() -> None:
    base = RollingDigest()
    lossy = RollingDigest()
    duplicated = RollingDigest()
    flipped = RollingDigest()
    for index in range(100):
        base.add(f"i{index}", index, True)
        if index != 50:
            lossy.add(f"i{index}", index, True)
        duplicated.add(f"i{index}", index, True)
        flipped.add(f"i{index}", index, index != 50)
    duplicated.add("i50", 50, True)
    assert base != lossy
    assert base != duplicated
    assert base != flipped


# ── Deterministic behavior ───────────────────────────────────────────────


def test_behavior_hash_is_seed_stable() -> None:
    assert behavior_hash(1, 42) == behavior_hash(1, 42)
    assert behavior_hash(1, 42) != behavior_hash(2, 42)


async def test_fake_provider_is_deterministic_for_a_seed() -> None:
    async def run_once() -> tuple[str, dict]:
        strategy = FakeProviderStrategy(FakeProviderConfig(seed=7))
        output, tokens, _ = await strategy.execute("p:i3", 1, 30.0)
        return output, tokens

    assert await run_once() == await run_once()


# ── Resource probes and bounded retention ────────────────────────────────


def test_resource_probes_degrade_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monitor_module, "_PROC_STATUS", Path("/nonexistent/status"))
    monkeypatch.setattr(monitor_module, "_PROC_FD", Path("/nonexistent/fd"))
    assert monitor_module.current_rss_bytes() is None
    assert monitor_module.fd_count() is None


async def test_monitor_sample_buffer_is_bounded() -> None:
    monitor = ResourceMonitor(interval_s=10.0)
    monitor.start()
    for _ in range(MAX_SAMPLES * 2):
        monitor.samples.append(monitor._sample())
    await monitor.stop()
    assert len(monitor.samples) == MAX_SAMPLES


# ── Report schema ────────────────────────────────────────────────────────


def test_report_schema_round_trip() -> None:
    scenario = ScenarioResult("healthy")
    scenario.check("something", True, "ok")
    report = build_report("ci", [scenario])
    assert report["schema_name"] == REPORT_SCHEMA_NAME
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert validate_report(report) == []
    json.dumps(report, allow_nan=False)


def test_report_validation_flags_missing_fields() -> None:
    report = build_report("ci", [ScenarioResult("healthy")])
    del report["scenarios"][0]["assertions"]
    problems = validate_report(report)
    assert any("assertions" in problem for problem in problems)


def test_environment_contains_no_identifying_paths() -> None:
    import os

    report = build_report("ci", [])
    text = json.dumps(report)
    assert os.path.expanduser("~") not in text
    assert os.uname().nodename not in text


# ── Reduced scenario runs ────────────────────────────────────────────────


@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_each_scenario_reduced_form_passes(scenario: str, tmp_path: Path) -> None:
    settings = _settings(scenario, tmp_path, items=120)
    result = await SCENARIO_RUNNERS[scenario](settings)
    failed = [check.to_json() for check in result.assertions if not check.passed]
    assert result.status == "passed", failed


async def test_reduced_jsonl_store_scenarios_pass(tmp_path: Path) -> None:
    for scenario in ("healthy", "stop_resume"):
        settings = _settings(scenario, tmp_path / scenario, items=60, store="jsonl")
        result = await SCENARIO_RUNNERS[scenario](settings)
        assert result.status == "passed", [
            check.to_json() for check in result.assertions if not check.passed
        ]


async def test_failure_reports_contain_actionable_detail(tmp_path: Path) -> None:
    settings = _settings("healthy", tmp_path, items=40)
    result = await SCENARIO_RUNNERS["healthy"](settings)
    # Simulate a violated invariant and verify the report carries evidence.
    result.check("simulated_violation", False, "high water 9 bound 8")
    assert result.status == "failed"
    payload = result.to_json()
    failing = [check for check in payload["assertions"] if not check["passed"]]
    assert failing and failing[0]["detail"] == "high water 9 bound 8"


async def test_no_raw_prompts_or_outputs_leak_into_report(tmp_path: Path) -> None:
    settings = _settings("healthy", tmp_path, items=40)
    result = await SCENARIO_RUNNERS["healthy"](settings)
    text = json.dumps(build_report("ci", [result]))
    assert "p:i" not in text  # prompt payload marker
    assert "ok:i" not in text  # output payload marker


def test_cleanup_after_scenario_exception(tmp_path: Path) -> None:
    config = HarnessConfig(
        profile="custom",
        items=30,
        concurrency=4,
        scenarios=("healthy",),
        output=tmp_path / "report.json",
        work_dir=tmp_path / "work",
    )
    config.validate()

    async def boom(_settings) -> ScenarioResult:
        raise RuntimeError("deliberate scenario crash")

    original = SCENARIO_RUNNERS["healthy"]
    SCENARIO_RUNNERS["healthy"] = boom
    try:
        report, exit_code = run_config(config)
    finally:
        SCENARIO_RUNNERS["healthy"] = original
    assert exit_code == 1
    assert report["scenarios"][0]["status"] == "error"
    assert "deliberate scenario crash" in report["scenarios"][0]["error"]
    # The report itself still validates and serializes.
    assert validate_report(report) == []


# ── Reduced end-to-end through the CLI ───────────────────────────────────


def test_cli_end_to_end_reduced_run(tmp_path: Path) -> None:
    from benchmarks.scale_soak.cli import main

    output = tmp_path / "report.json"
    exit_code = main(
        [
            "--profile",
            "custom",
            "--items",
            "60",
            "--concurrency",
            "6",
            "--scenario",
            "healthy",
            "--scenario",
            "stop_resume",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    report = json.loads(output.read_text())
    assert validate_report(report) == []
    assert [scenario["name"] for scenario in report["scenarios"]] == [
        "healthy",
        "stop_resume",
    ]
    assert all(scenario["status"] == "passed" for scenario in report["scenarios"])
