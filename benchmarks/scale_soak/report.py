"""Order-independent item accounting and the versioned JSON report."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .monitor import resource_methods

REPORT_SCHEMA_NAME = "async-batch-llm-scale-soak"
REPORT_SCHEMA_VERSION = 1

DEFERRED_FEATURES = [
    "token-aware admission for mixed prompt/output sizes (#122)",
    "adaptive concurrency (#89)",
    "distributed writers",
    "provider-native batch execution",
    "real provider quotas or pricing",
]


@dataclass
class RollingDigest:
    """Order-independent proof of exact item accounting.

    XOR of per-item SHA-256 digests is order independent and collision
    resistant for accidental loss/duplication; count and modular index sum are
    independent cheap cross-checks. No item IDs are retained.
    """

    count: int = 0
    index_sum: int = 0
    xor_digest: int = 0

    _MOD = 2**64

    def add(self, item_id: str, index: int, success: bool) -> None:
        self.count += 1
        self.index_sum = (self.index_sum + index) % self._MOD
        entry = hashlib.sha256(f"{item_id}|{int(success)}".encode()).digest()
        self.xor_digest ^= int.from_bytes(entry, "big")

    def to_json(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "index_sum_mod_2_64": self.index_sum,
            "xor_sha256": f"{self.xor_digest:064x}",
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RollingDigest):
            return NotImplemented
        return (
            self.count == other.count
            and self.index_sum == other.index_sum
            and self.xor_digest == other.xor_digest
        )


@dataclass
class Check:
    """One named assertion with its observed evidence."""

    name: str
    passed: bool
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class ScenarioResult:
    name: str
    configuration: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    throughput: dict[str, Any] = field(default_factory=dict)
    provider: dict[str, Any] = field(default_factory=dict)
    queues: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    artifact: dict[str, Any] = field(default_factory=dict)
    assertions: list[Check] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    error: str | None = None

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.assertions.append(Check(name, bool(passed), detail))

    @property
    def status(self) -> str:
        if self.error is not None:
            return "error"
        return "passed" if all(check.passed for check in self.assertions) else "failed"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "configuration": self.configuration,
            "counts": self.counts,
            "throughput": self.throughput,
            "provider": self.provider,
            "queues": self.queues,
            "resources": self.resources,
            "artifact": self.artifact,
            "assertions": [check.to_json() for check in self.assertions],
            "caveats": self.caveats,
        }


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("async-batch-llm")
    except Exception:
        return "unknown"


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def environment() -> dict[str, Any]:
    """Report environment without hostname, username, or home paths."""
    import os

    return {
        "package_version": _package_version(),
        "git_revision": _git_revision(),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "cpu_count": os.cpu_count(),
        "resource_methods": resource_methods(),
    }


def build_report(profile: str, scenarios: list[ScenarioResult]) -> dict[str, Any]:
    return {
        "schema_name": REPORT_SCHEMA_NAME,
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "environment": environment(),
        "profile": profile,
        "scenarios": [scenario.to_json() for scenario in scenarios],
        "deferred_features": DEFERRED_FEATURES,
    }


def validate_report(report: dict[str, Any]) -> list[str]:
    """Structural validation; returns a list of problems (empty when valid)."""
    problems: list[str] = []
    if report.get("schema_name") != REPORT_SCHEMA_NAME:
        problems.append("schema_name mismatch")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        problems.append("schema_version mismatch")
    for key in ("generated_at", "environment", "profile", "scenarios", "deferred_features"):
        if key not in report:
            problems.append(f"missing key {key!r}")
    for index, scenario in enumerate(report.get("scenarios", [])):
        for key in (
            "name",
            "status",
            "configuration",
            "counts",
            "throughput",
            "provider",
            "queues",
            "resources",
            "artifact",
            "assertions",
            "caveats",
        ):
            if key not in scenario:
                problems.append(f"scenario[{index}] missing key {key!r}")
        for check_index, check in enumerate(scenario.get("assertions", [])):
            if not isinstance(check.get("passed"), bool):
                problems.append(f"scenario[{index}].assertions[{check_index}] missing passed")
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as exc:
        problems.append(f"not JSON serializable: {exc}")
    return problems


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
