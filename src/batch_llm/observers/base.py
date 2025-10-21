"""Observer system for processor events."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class ProcessingEvent(Enum):
    """Events that can be observed during processing."""

    ITEM_STARTED = "item_started"
    ITEM_COMPLETED = "item_completed"
    ITEM_FAILED = "item_failed"
    RATE_LIMIT_HIT = "rate_limit_hit"
    COOLDOWN_STARTED = "cooldown_started"
    COOLDOWN_ENDED = "cooldown_ended"
    WORKER_STARTED = "worker_started"
    WORKER_STOPPED = "worker_stopped"
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"


class ProcessorObserver(ABC):
    """Abstract base class for processor event observers."""

    @abstractmethod
    async def on_event(
        self,
        event: ProcessingEvent,
        data: dict[str, Any],
    ) -> None:
        """
        Handle processor event.

        Args:
            event: The event type
            data: Event-specific data
        """
        pass


class BaseObserver(ProcessorObserver):
    """Base observer with no-op implementation."""

    async def on_event(
        self,
        event: ProcessingEvent,
        data: dict[str, Any],
    ) -> None:
        """Default: do nothing."""
        pass
