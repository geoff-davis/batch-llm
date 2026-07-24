"""Profiles, scenario settings, and validation for the scale-soak harness."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

SCENARIOS = (
    "healthy",
    "slow_consumer",
    "rate_limit_wave",
    "transport_retry",
    "validation_recovery",
    "stop_resume",
    "artifact_bench",
    "progress_overhead",
)

# The 1m profile deliberately runs only the large-scenario subset; the 100k
# profile covers every error mix (plan §8.4).
LARGE_PROFILE_SCENARIOS = ("healthy", "slow_consumer", "stop_resume", "artifact_bench")

STORES = ("sqlite", "jsonl", "none")
DURABILITIES = ("balanced", "full")
PROGRESS_MODES = ("enabled", "disabled", "compare")


@dataclass(frozen=True)
class ScenarioSettings:
    """Fully resolved settings for one scenario run."""

    name: str
    items: int
    concurrency: int
    max_queue_size: int
    max_result_queue_size: int
    store: str
    durability: str
    seed: int
    provider_latency_s: float
    sink_latency_s: float
    commit_batch_size: int
    commit_interval_seconds: float
    work_dir: Path
    keep_artifacts: bool
    progress_mode: str
    max_post_warmup_rss_growth_mib: float | None
    max_task_leak: int
    max_fd_leak: int
    allow_large_jsonl: bool
    monitor_interval_s: float


@dataclass(frozen=True)
class Profile:
    """Item counts and bounds that define a named profile."""

    items: int
    concurrency: int
    max_queue_size: int
    max_result_queue_size: int
    commit_batch_size: int
    commit_interval_seconds: float
    monitor_interval_s: float
    # progress_overhead and artifact lookup sampling get smaller counts on the
    # large profiles so the harness measures the framework, not the monitor.
    progress_items: int


PROFILES: dict[str, Profile] = {
    "ci": Profile(
        items=2_000,
        concurrency=16,
        max_queue_size=64,
        max_result_queue_size=32,
        commit_batch_size=64,
        commit_interval_seconds=0.005,
        monitor_interval_s=0.2,
        progress_items=2_000,
    ),
    "100k": Profile(
        items=100_000,
        concurrency=64,
        max_queue_size=512,
        max_result_queue_size=512,
        commit_batch_size=256,
        commit_interval_seconds=0.005,
        monitor_interval_s=1.0,
        progress_items=20_000,
    ),
    "1m": Profile(
        items=1_000_000,
        concurrency=64,
        max_queue_size=512,
        max_result_queue_size=512,
        commit_batch_size=256,
        commit_interval_seconds=0.005,
        monitor_interval_s=5.0,
        progress_items=20_000,
    ),
}

DEFAULT_SEED = 20260721


@dataclass
class HarnessConfig:
    """Parsed CLI options before per-scenario resolution."""

    profile: str = "ci"
    scenarios: tuple[str, ...] = SCENARIOS
    items: int | None = None
    concurrency: int | None = None
    max_queue_size: int | None = None
    max_result_queue_size: int | None = None
    store: str = "sqlite"
    durability: str = "balanced"
    seed: int = DEFAULT_SEED
    provider_latency_ms: float = 1.0
    sink_latency_ms: float = 0.0
    progress: str = "compare"
    output: Path = field(default_factory=lambda: Path("benchmark-results/scale-soak.json"))
    work_dir: Path | None = None
    keep_artifacts: bool = False
    max_post_warmup_rss_growth_mib: float | None = None
    max_task_leak: int = 0
    max_fd_leak: int = 4
    allow_large_jsonl: bool = False

    def validate(self) -> None:
        if self.profile not in (*PROFILES, "custom"):
            raise ValueError(f"unknown profile {self.profile!r}")
        unknown = [name for name in self.scenarios if name not in SCENARIOS]
        if unknown:
            raise ValueError(f"unknown scenario(s): {', '.join(unknown)}")
        if self.store not in STORES:
            raise ValueError(f"store must be one of {STORES} (got {self.store!r})")
        if self.durability not in DURABILITIES:
            raise ValueError(f"durability must be one of {DURABILITIES}")
        if self.progress not in PROGRESS_MODES:
            raise ValueError(f"progress must be one of {PROGRESS_MODES}")
        if self.profile == "custom" and (self.items is None or self.items < 1):
            raise ValueError("--profile custom requires --items >= 1")
        if self.items is not None and self.items < 1:
            raise ValueError("--items must be >= 1")
        if self.concurrency is not None and self.concurrency < 1:
            raise ValueError("--concurrency must be >= 1")
        for name in ("max_queue_size", "max_result_queue_size"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
        if self.store == "none" and any(
            name in self.scenarios for name in ("stop_resume", "artifact_bench")
        ):
            raise ValueError(
                "--store none cannot run the stop_resume or artifact_bench "
                "scenarios; select other scenarios or a persistent store"
            )
        if self.provider_latency_ms < 0 or self.sink_latency_ms < 0:
            raise ValueError("latencies must be >= 0")
        if (
            self.max_post_warmup_rss_growth_mib is not None
            and self.max_post_warmup_rss_growth_mib <= 0
        ):
            raise ValueError("--max-post-warmup-rss-growth-mib must be > 0")
        if self.max_task_leak < 0 or self.max_fd_leak < 0:
            raise ValueError("leak tolerances must be >= 0")


def selected_scenarios(config: HarnessConfig) -> tuple[str, ...]:
    """Scenario list after applying the 1m large-subset default."""
    if config.profile == "1m" and config.scenarios == SCENARIOS:
        return LARGE_PROFILE_SCENARIOS
    return config.scenarios


def resolve_settings(config: HarnessConfig, scenario: str, work_dir: Path) -> ScenarioSettings:
    """Merge profile defaults and explicit overrides for one scenario."""
    profile = PROFILES.get(config.profile, PROFILES["ci"])
    items = config.items if config.items is not None else profile.items
    if scenario == "progress_overhead" and config.items is None:
        items = profile.progress_items
    return ScenarioSettings(
        name=scenario,
        items=items,
        concurrency=(config.concurrency if config.concurrency is not None else profile.concurrency),
        max_queue_size=(
            config.max_queue_size if config.max_queue_size is not None else profile.max_queue_size
        ),
        max_result_queue_size=(
            config.max_result_queue_size
            if config.max_result_queue_size is not None
            else profile.max_result_queue_size
        ),
        store=config.store,
        durability=config.durability,
        seed=config.seed,
        provider_latency_s=config.provider_latency_ms / 1000.0,
        sink_latency_s=config.sink_latency_ms / 1000.0,
        commit_batch_size=profile.commit_batch_size,
        commit_interval_seconds=profile.commit_interval_seconds,
        work_dir=work_dir,
        keep_artifacts=config.keep_artifacts,
        progress_mode=config.progress,
        max_post_warmup_rss_growth_mib=config.max_post_warmup_rss_growth_mib,
        max_task_leak=config.max_task_leak,
        max_fd_leak=config.max_fd_leak,
        allow_large_jsonl=config.allow_large_jsonl,
        monitor_interval_s=profile.monitor_interval_s,
    )


def with_items(settings: ScenarioSettings, items: int) -> ScenarioSettings:
    """Settings copy with a different item count (used by sub-measurements)."""
    return replace(settings, items=items)
