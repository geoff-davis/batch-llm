"""Metrics collection observer."""

import asyncio
import json
from typing import Any

from .base import BaseObserver, ProcessingEvent


class MetricsObserver(BaseObserver):
    """Collect metrics for monitoring (thread-safe)."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {
            "items_processed": 0,
            "items_succeeded": 0,
            "items_failed": 0,
            "rate_limits_hit": 0,
            "total_cooldown_time": 0.0,
            "processing_times": [],
            "error_counts": {},
        }
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
                    self.metrics["processing_times"].append(data["duration"])

            elif event == ProcessingEvent.ITEM_FAILED:
                self.metrics["items_processed"] += 1
                self.metrics["items_failed"] += 1
                if "error_type" in data:
                    error_type = data["error_type"]
                    self.metrics["error_counts"][error_type] = (
                        self.metrics["error_counts"].get(error_type, 0) + 1
                    )

            elif event == ProcessingEvent.RATE_LIMIT_HIT:
                self.metrics["rate_limits_hit"] += 1

            elif event == ProcessingEvent.COOLDOWN_ENDED:
                if "duration" in data:
                    self.metrics["total_cooldown_time"] += data["duration"]

    async def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics with computed statistics (thread-safe)."""
        async with self._lock:
            processing_times = self.metrics["processing_times"]
            return {
                **self.metrics,
                "avg_processing_time": (
                    sum(processing_times) / len(processing_times) if processing_times else 0
                ),
                "success_rate": (
                    self.metrics["items_succeeded"] / self.metrics["items_processed"]
                    if self.metrics["items_processed"] > 0
                    else 0
                ),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        self.__init__()

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
            **metrics,
            "processing_times_count": len(metrics.get("processing_times", [])),
        }
        # Remove the full list to keep export clean
        export_data.pop("processing_times", None)
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
            # HELP batch_llm_items_processed Total items processed
            # TYPE batch_llm_items_processed counter
            batch_llm_items_processed 100
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
        ]

        for metric_name, help_text in counters:
            lines.append(f"# HELP batch_llm_{metric_name} {help_text}")
            lines.append(f"# TYPE batch_llm_{metric_name} counter")
            lines.append(f"batch_llm_{metric_name} {metrics.get(metric_name, 0)}")
            lines.append("")

        # Gauge metrics
        gauges = [
            ("avg_processing_time", "Average processing time in seconds"),
            ("success_rate", "Success rate (0.0 to 1.0)"),
            ("total_cooldown_time", "Total time spent in rate limit cooldown (seconds)"),
        ]

        for metric_name, help_text in gauges:
            lines.append(f"# HELP batch_llm_{metric_name} {help_text}")
            lines.append(f"# TYPE batch_llm_{metric_name} gauge")
            lines.append(f"batch_llm_{metric_name} {metrics.get(metric_name, 0)}")
            lines.append("")

        # Error counts as labeled counter
        error_counts = metrics.get("error_counts", {})
        if error_counts:
            lines.append("# HELP batch_llm_errors_total Total errors by type")
            lines.append("# TYPE batch_llm_errors_total counter")
            for error_type, count in error_counts.items():
                # Sanitize error type for Prometheus label
                safe_type = error_type.replace('"', '\\"')
                lines.append(f'batch_llm_errors_total{{error_type="{safe_type}"}} {count}')
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
