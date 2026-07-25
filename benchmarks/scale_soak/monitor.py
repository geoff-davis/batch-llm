"""Bounded resource probing for the harness.

Every probe degrades to ``None`` when a platform source is unavailable —
the report records the reason instead of fabricating a zero (plan §8.3).
"""

from __future__ import annotations

import asyncio
import resource
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROC_STATUS = Path("/proc/self/status")
_PROC_FD = Path("/proc/self/fd")

MAX_SAMPLES = 300


def current_rss_bytes() -> int | None:
    """Current RSS from /proc; None off Linux."""
    try:
        for line in _PROC_STATUS.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def peak_rss_bytes() -> int | None:
    """Process peak RSS via getrusage (kilobytes on Linux, bytes on macOS)."""
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ValueError, OSError):
        return None
    return peak * 1024 if sys.platform != "darwin" else peak


def fd_count() -> int | None:
    try:
        return sum(1 for _ in _PROC_FD.iterdir())
    except OSError:
        return None


def thread_count() -> int:
    return threading.active_count()


def sqlite_executor_thread_count() -> int:
    """Threads owned by SqliteArtifactStore executors (by name prefix)."""
    return sum(
        1 for thread in threading.enumerate() if thread.name.startswith("async-batch-llm-sqlite")
    )


def loop_default_executor_thread_count() -> int:
    """asyncio's lazily-spawned to_thread workers; live until loop close."""
    return sum(1 for thread in threading.enumerate() if thread.name.startswith("asyncio_"))


def task_count() -> int | None:
    try:
        return len(asyncio.all_tasks())
    except RuntimeError:
        return None


def resource_methods() -> dict[str, str]:
    return {
        "rss_current": "/proc/self/status VmRSS",
        "rss_peak": "resource.getrusage ru_maxrss",
        "fd_count": "/proc/self/fd",
        "threads": "threading.active_count",
        "tasks": "asyncio.all_tasks",
    }


@dataclass
class ResourceSnapshot:
    rss_bytes: int | None
    fd_count: int | None
    thread_count: int
    task_count: int | None

    @staticmethod
    def capture() -> ResourceSnapshot:
        return ResourceSnapshot(
            rss_bytes=current_rss_bytes(),
            fd_count=fd_count(),
            thread_count=thread_count(),
            task_count=task_count(),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "rss_bytes": self.rss_bytes,
            "fd_count": self.fd_count,
            "thread_count": self.thread_count,
            "task_count": self.task_count,
        }


@dataclass
class ResourceMonitor:
    """Periodic sampler with a bounded buffer and optional WAL-size tracking."""

    interval_s: float = 0.5
    wal_path: Path | None = None
    samples: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))
    warmup_rss_bytes: int | None = None
    peak_post_warmup_rss_bytes: int | None = None
    peak_wal_bytes: int | None = None
    _task: asyncio.Task[None] | None = None
    _started_at: float = 0.0

    def _sample(self) -> dict[str, Any]:
        wal_bytes: int | None = None
        if self.wal_path is not None:
            try:
                wal_bytes = self.wal_path.stat().st_size
            except OSError:
                wal_bytes = 0
            if wal_bytes is not None:
                self.peak_wal_bytes = max(self.peak_wal_bytes or 0, wal_bytes)
        rss = current_rss_bytes()
        if rss is not None and self.warmup_rss_bytes is not None:
            self.peak_post_warmup_rss_bytes = max(self.peak_post_warmup_rss_bytes or 0, rss)
        return {
            "t": round(time.monotonic() - self._started_at, 3),
            "rss_bytes": rss,
            "wal_bytes": wal_bytes,
        }

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            self.samples.append(self._sample())

    def start(self) -> None:
        self._started_at = time.monotonic()
        self.samples.append(self._sample())
        self._task = asyncio.get_running_loop().create_task(self._run())

    def mark_warmup(self) -> None:
        self.warmup_rss_bytes = current_rss_bytes()
        self.peak_post_warmup_rss_bytes = self.warmup_rss_bytes

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.samples.append(self._sample())

    def post_warmup_growth_bytes(self) -> int | None:
        """Growth from warm-up to the *highest* sampled RSS since warm-up.

        Comparing against the sampled maximum (refreshed once more at stop())
        means a large transient allocation cannot pass the configured ceiling
        just because it was freed before cleanup ran.
        """
        if self.peak_post_warmup_rss_bytes is None or self.warmup_rss_bytes is None:
            return None
        return self.peak_post_warmup_rss_bytes - self.warmup_rss_bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "sampling_interval_seconds": self.interval_s,
            "sample_count": len(self.samples),
            "samples_bounded_to": MAX_SAMPLES,
            "warmup_rss_bytes": self.warmup_rss_bytes,
            "peak_post_warmup_rss_bytes": self.peak_post_warmup_rss_bytes,
            "post_warmup_rss_growth_bytes": self.post_warmup_growth_bytes(),
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_wal_bytes": self.peak_wal_bytes,
            "samples": list(self.samples),
        }
