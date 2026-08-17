"""Observer system for processor events."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class ProcessingEvent(Enum):
    """Events that can be observed during processing."""

    ITEM_STARTED = "item_started"
    ITEM_ADMITTED = "item_admitted"
    QUOTA_ADMITTED = "quota_admitted"
    QUOTA_RECONCILED = "quota_reconciled"
    ITEM_COMPLETED = "item_completed"
    ITEM_FAILED = "item_failed"
    ITEM_REPLAYED = "item_replayed"
    ITEM_DEADLINE_EXCEEDED = "item_deadline_exceeded"
    RATE_LIMIT_HIT = "rate_limit_hit"
    COOLDOWN_STARTED = "cooldown_started"
    COOLDOWN_ENDED = "cooldown_ended"
    WORKER_STARTED = "worker_started"
    WORKER_STOPPED = "worker_stopped"
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"
    BATCH_ABORTED = "batch_aborted"


class ProcessorObserver(ABC):
    """Abstract base class for processor event observers."""

    # Trusted built-in observers set this True to opt out of the per-event
    # asyncio.wait_for timeout wrapper in EventDispatcher.emit (its task+timer
    # setup is measurable hot-path overhead). User observers default to False
    # so a slow/hanging on_event() can never block the worker loop.
    _abl_fast_observer: bool = False

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
