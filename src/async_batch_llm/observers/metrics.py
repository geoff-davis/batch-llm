"""Metrics collection observer."""

import asyncio
import json
from typing import Any

from .base import BaseObserver, ProcessingEvent


def _initial_metrics() -> dict[str, Any]:
    return {
        "items_processed": 0,
        "items_succeeded": 0,
        "items_failed": 0,
        "rate_limits_hit": 0,
        "total_cooldown_time": 0.0,
        "error_counts": {},
        "processing_times_count": 0,
        "processing_times_sum": 0.0,
        "admission_wait_count": 0,
        "admission_wait_seconds_sum": 0.0,
        "admission_wait_seconds_max": 0.0,
        "quota_admitted_attempts": 0,
        "quota_wait_seconds_sum": 0.0,
        "quota_wait_seconds_max": 0.0,
        "estimated_input_tokens": 0,
        "estimated_output_tokens": 0,
        "reserved_tokens": 0,
        "reported_reconciliation_tokens": 0,
        "refunded_tokens": 0,
        "underestimated_tokens": 0,
        "unknown_usage_attempts": 0,
        "known_zero_usage_attempts": 0,
        "token_estimation_failures": 0,
        "quota_scope_count": 0,
        "structured_output_recoveries": 0,
        "structured_output_retries_avoided": 0,
        "structured_output_recovery_reasons": {},
        "items_replayed": 0,
        "items_aborted": 0,
        "batches_aborted": 0,
    }


class MetricsObserver(BaseObserver):
    """Collect metrics for monitoring (thread-safe)."""

    # Lock-guarded in-memory counter updates only — fast and non-blocking, so
    # it opts out of the per-event wait_for wrapper. See ProcessorObserver.
    _abl_fast_observer: bool = True

    def __init__(self, *, max_processing_samples: int = 100):
        """Initialize metrics collector."""
        if max_processing_samples <= 0:
            raise ValueError("max_processing_samples must be positive")
        self.metrics: dict[str, Any] = _initial_metrics()
        self._processing_times: list[float] = []
        self._quota_scopes: set[int] = set()
        self._max_processing_samples = max_processing_samples
        self._lock = asyncio.Lock()

    async def on_event(
        self,
        event: ProcessingEvent,
        data: dict[str, Any],
    ) -> None:
        """Collect metrics from events (thread-safe)."""
        async with self._lock:
            if event == ProcessingEvent.ITEM_COMPLETED:
                self.metrics["items_processed"] += 1
                self.metrics["items_succeeded"] += 1
                if "duration" in data:
                    duration = float(data["duration"])
                    self.metrics["processing_times_sum"] += duration
                    self.metrics["processing_times_count"] += 1
                    self._processing_times.append(duration)
                    if len(self._processing_times) > self._max_processing_samples:
                        self._processing_times.pop(0)
                if data.get("structured_output_recovered") is True:
                    self.metrics["structured_output_recoveries"] += 1
                    retries_avoided = data.get("structured_output_retries_avoided", 0)
                    if (
                        not isinstance(retries_avoided, bool)
                        and isinstance(retries_avoided, int)
                        and retries_avoided > 0
                    ):
                        self.metrics["structured_output_retries_avoided"] += retries_avoided
                    reason = data.get("structured_output_recovery_reason")
                    if isinstance(reason, str) and reason:
                        reasons = self.metrics["structured_output_recovery_reasons"]
                        reasons[reason] = reasons.get(reason, 0) + 1

            elif event == ProcessingEvent.ITEM_ADMITTED:
                wait_seconds = float(data.get("wait_seconds", 0.0))
                self.metrics["admission_wait_count"] += 1
                self.metrics["admission_wait_seconds_sum"] += wait_seconds
                self.metrics["admission_wait_seconds_max"] = max(
                    self.metrics["admission_wait_seconds_max"], wait_seconds
                )

            elif event == ProcessingEvent.QUOTA_ADMITTED:
                wait_seconds = float(data.get("wait_seconds", 0.0))
                self.metrics["quota_admitted_attempts"] += 1
                self.metrics["quota_wait_seconds_sum"] += wait_seconds
                self.metrics["quota_wait_seconds_max"] = max(
                    self.metrics["quota_wait_seconds_max"], wait_seconds
                )
                for key in (
                    "estimated_input_tokens",
                    "estimated_output_tokens",
                    "reserved_tokens",
                ):
                    value = data.get(key, 0)
                    if not isinstance(value, bool) and isinstance(value, int):
                        self.metrics[key] += value
                scope_id = data.get("quota_scope_id")
                if not isinstance(scope_id, bool) and isinstance(scope_id, int):
                    self._quota_scopes.add(scope_id)
                    self.metrics["quota_scope_count"] = len(self._quota_scopes)

            elif event == ProcessingEvent.QUOTA_RECONCILED:
                reported = data.get("reported_tokens")
                delta = data.get("delta_tokens")
                reserved = data.get("reserved_tokens", 0)
                if not isinstance(reported, bool) and isinstance(reported, int):
                    self.metrics["reported_reconciliation_tokens"] += reported
                    if reported == 0 and reserved:
                        self.metrics["known_zero_usage_attempts"] += 1
                elif reserved:
                    self.metrics["unknown_usage_attempts"] += 1
                if not isinstance(delta, bool) and isinstance(delta, int):
                    if delta < 0:
                        self.metrics["refunded_tokens"] += -delta
                    elif delta > 0:
                        self.metrics["underestimated_tokens"] += delta

            elif event == ProcessingEvent.ITEM_FAILED:
                self.metrics["items_processed"] += 1
                self.metrics["items_failed"] += 1
                if data.get("error_category") in {"batch_aborted", "batch_deadline_exceeded"}:
                    self.metrics["items_aborted"] += 1
                if "error_type" in data:
                    error_type = data["error_type"]
                    self.metrics["error_counts"][error_type] = (
                        self.metrics["error_counts"].get(error_type, 0) + 1
                    )
                if data.get("error_category") in {
                    "token_estimation_error",
                    "token_estimator_required",
                    "token_estimate_exceeds_limit",
                }:
                    self.metrics["token_estimation_failures"] += 1

            elif event == ProcessingEvent.ITEM_REPLAYED:
                self.metrics["items_processed"] += 1
                self.metrics["items_replayed"] += 1
                if data.get("success") is True:
                    self.metrics["items_succeeded"] += 1
                else:
                    self.metrics["items_failed"] += 1

            elif event == ProcessingEvent.BATCH_ABORTED:
                self.metrics["batches_aborted"] += 1

            elif event == ProcessingEvent.RATE_LIMIT_HIT:
                self.metrics["rate_limits_hit"] += 1

            elif event == ProcessingEvent.COOLDOWN_ENDED:
                if "duration" in data:
                    self.metrics["total_cooldown_time"] += data["duration"]

    async def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics with computed statistics (thread-safe)."""
        async with self._lock:
            return {
                **self.metrics,
                "processing_times": list(self._processing_times),
                "avg_processing_time": (
                    self.metrics["processing_times_sum"] / self.metrics["processing_times_count"]
                    if self.metrics["processing_times_count"] > 0
                    else 0
                ),
                "avg_admission_wait_seconds": (
                    self.metrics["admission_wait_seconds_sum"]
                    / self.metrics["admission_wait_count"]
                    if self.metrics["admission_wait_count"] > 0
                    else 0
                ),
                "avg_quota_wait_seconds": (
                    self.metrics["quota_wait_seconds_sum"] / self.metrics["quota_admitted_attempts"]
                    if self.metrics["quota_admitted_attempts"] > 0
                    else 0
                ),
                "success_rate": (
                    self.metrics["items_succeeded"] / self.metrics["items_processed"]
                    if self.metrics["items_processed"] > 0
                    else 0
                ),
            }

    async def reset(self) -> None:
        """Reset all metrics (thread-safe).

        Async since v0.16: the reset acquires the same lock as
        ``on_event``, so counts from an in-flight event can no longer land
        in the discarded pre-reset dict.
        """
        async with self._lock:
            self.metrics = _initial_metrics()
            self._processing_times = []
            self._quota_scopes = set()

    async def export_json(self) -> str:
        """Export metrics as JSON string.

        Returns:
            JSON string containing all metrics and computed statistics

        Example:
            >>> observer = MetricsObserver()
            >>> # ... process items ...
            >>> json_str = await observer.export_json()
            >>> print(json_str)
        """
        metrics = await self.get_metrics()
        # Convert processing_times list to just count for cleaner export
        export_data = {
            **{k: v for k, v in metrics.items() if k != "processing_times"},
            "processing_times_count": metrics.get("processing_times_count", 0),
        }
        return json.dumps(export_data, indent=2)

    async def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string

        Example:
            >>> observer = MetricsObserver()
            >>> # ... process items ...
            >>> prom_text = await observer.export_prometheus()
            >>> print(prom_text)
            # HELP async_batch_llm_items_processed Total items processed
            # TYPE async_batch_llm_items_processed counter
            async_batch_llm_items_processed 100
            ...
        """
        metrics = await self.get_metrics()

        lines = []

        # Counter metrics
        counters = [
            ("items_processed", "Total items processed"),
            ("items_succeeded", "Total items succeeded"),
            ("items_failed", "Total items failed"),
            ("rate_limits_hit", "Total rate limits encountered"),
            ("structured_output_recoveries", "Total structured outputs recovered"),
            ("items_replayed", "Total terminal items replayed from artifacts"),
            ("items_aborted", "Total collateral items aborted by guardrails"),
            ("batches_aborted", "Total batches stopped by guardrails"),
            ("quota_admitted_attempts", "Total live quota reservations admitted"),
            ("reserved_tokens", "Total tokens reserved for TPM admission"),
            ("reported_reconciliation_tokens", "Total known tokens used for reconciliation"),
            ("refunded_tokens", "Total unused reserved tokens refunded"),
            ("underestimated_tokens", "Total token underestimation charged as debt"),
            ("unknown_usage_attempts", "Started quota attempts with unknown usage"),
            ("known_zero_usage_attempts", "Started quota attempts reporting zero usage"),
            ("token_estimation_failures", "Total framework token-estimation failures"),
            (
                "structured_output_retries_avoided",
                "Total validation retries avoided by structured-output recovery",
            ),
        ]

        for metric_name, help_text in counters:
            lines.append(f"# HELP async_batch_llm_{metric_name} {help_text}")
            lines.append(f"# TYPE async_batch_llm_{metric_name} counter")
            lines.append(f"async_batch_llm_{metric_name} {metrics.get(metric_name, 0)}")
            lines.append("")

        # Gauge metrics
        gauges = [
            ("avg_processing_time", "Average processing time in seconds"),
            ("success_rate", "Success rate (0.0 to 1.0)"),
            ("total_cooldown_time", "Total time spent in rate limit cooldown (seconds)"),
            ("processing_times_count", "Number of recorded processing time samples"),
            ("admission_wait_count", "Number of provider-capacity admissions"),
            ("admission_wait_seconds_sum", "Total provider-capacity wait time in seconds"),
            ("admission_wait_seconds_max", "Maximum provider-capacity wait time in seconds"),
            ("avg_admission_wait_seconds", "Average provider-capacity wait time in seconds"),
            ("quota_wait_seconds_sum", "Total RPM/TPM quota wait time in seconds"),
            ("quota_wait_seconds_max", "Maximum RPM/TPM quota wait time in seconds"),
            ("avg_quota_wait_seconds", "Average RPM/TPM quota wait time in seconds"),
            ("estimated_input_tokens", "Total estimated input tokens"),
            ("estimated_output_tokens", "Total estimated output tokens"),
            ("quota_scope_count", "Number of run-local quota scopes observed"),
        ]

        for metric_name, help_text in gauges:
            lines.append(f"# HELP async_batch_llm_{metric_name} {help_text}")
            lines.append(f"# TYPE async_batch_llm_{metric_name} gauge")
            lines.append(f"async_batch_llm_{metric_name} {metrics.get(metric_name, 0)}")
            lines.append("")

        # Error counts as labeled counter
        error_counts = metrics.get("error_counts", {})
        if error_counts:
            lines.append("# HELP async_batch_llm_errors_total Total errors by type")
            lines.append("# TYPE async_batch_llm_errors_total counter")
            for error_type, count in error_counts.items():
                # Sanitize error type for Prometheus label
                safe_type = error_type.replace('"', '\\"')
                lines.append(f'async_batch_llm_errors_total{{error_type="{safe_type}"}} {count}')
            lines.append("")

        recovery_reasons = metrics.get("structured_output_recovery_reasons", {})
        if recovery_reasons:
            lines.append(
                "# HELP async_batch_llm_structured_output_recoveries_by_reason "
                "Structured-output recoveries by reason"
            )
            lines.append("# TYPE async_batch_llm_structured_output_recoveries_by_reason counter")
            for reason, count in recovery_reasons.items():
                safe_reason = reason.replace('"', '\\"')
                lines.append(
                    "async_batch_llm_structured_output_recoveries_by_reason"
                    f'{{reason="{safe_reason}"}} {count}'
                )
            lines.append("")

        return "\n".join(lines)

    async def export_dict(self) -> dict[str, Any]:
        """Export metrics as a dictionary.

        Returns:
            Dictionary containing all metrics and computed statistics

        Example:
            >>> observer = MetricsObserver()
            >>> # ... process items ...
            >>> data = await observer.export_dict()
            >>> print(data["success_rate"])
        """
        return await self.get_metrics()
