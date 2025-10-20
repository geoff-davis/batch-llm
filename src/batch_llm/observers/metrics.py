"""Metrics collection observer."""

import asyncio
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
