"""Deterministic, credential-free scale and restart soak harness (v0.21).

Drives the real ``process_stream()`` / ``ParallelBatchProcessor`` execution
surfaces with a seeded fake provider — no network, no credentials — and emits
a versioned machine-readable JSON report. See ``benchmarks/README.md``.
"""

from .config import PROFILES, HarnessConfig, ScenarioSettings, resolve_settings
from .report import REPORT_SCHEMA_NAME, REPORT_SCHEMA_VERSION, RollingDigest

__all__ = [
    "PROFILES",
    "HarnessConfig",
    "ScenarioSettings",
    "resolve_settings",
    "REPORT_SCHEMA_NAME",
    "REPORT_SCHEMA_VERSION",
    "RollingDigest",
]
