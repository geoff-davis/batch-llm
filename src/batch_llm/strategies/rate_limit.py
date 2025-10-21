"""Rate limit handling strategies."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RateLimitStrategy(ABC):
    """Strategy for handling rate limits."""

    @abstractmethod
    async def on_rate_limit(
        self,
        worker_id: int,
        consecutive_limit_count: int,
    ) -> float:
        """
        Called when rate limit detected.

        Args:
            worker_id: ID of the worker that hit the rate limit
            consecutive_limit_count: Number of consecutive rate limits

        Returns:
            Cooldown duration in seconds
        """
        ...

    @abstractmethod
    def should_apply_slow_start(self, items_since_resume: int) -> tuple[bool, float]:
        """
        Check if slow-start delay should be applied.

        Args:
            items_since_resume: Number of items processed since last rate limit

        Returns:
            (should_delay, delay_seconds)
        """
        ...


class ExponentialBackoffStrategy(RateLimitStrategy):
    """Exponential backoff with progressive slow-start."""

    def __init__(
        self,
        initial_cooldown: float = 60.0,
        max_cooldown: float = 600.0,
        backoff_multiplier: float = 2.0,
        slow_start_items: int = 50,
        slow_start_initial_delay: float = 2.0,
        slow_start_final_delay: float = 0.1,
    ):
        """
        Initialize exponential backoff strategy.

        Args:
            initial_cooldown: Initial cooldown duration in seconds
            max_cooldown: Maximum cooldown duration in seconds
            backoff_multiplier: Multiplier for exponential backoff
            slow_start_items: Number of items for slow start after cooldown
            slow_start_initial_delay: Initial delay between items during slow start
            slow_start_final_delay: Final delay between items during slow start
        """
        self.initial_cooldown = initial_cooldown
        self.max_cooldown = max_cooldown
        self.backoff_multiplier = backoff_multiplier
        self.slow_start_items = slow_start_items
        self.slow_start_initial_delay = slow_start_initial_delay
        self.slow_start_final_delay = slow_start_final_delay

    async def on_rate_limit(self, worker_id: int, consecutive_count: int) -> float:
        """Calculate exponential backoff cooldown."""
        cooldown = min(
            self.initial_cooldown * (self.backoff_multiplier ** (consecutive_count - 1)),
            self.max_cooldown,
        )
        logger.info(
            f"Worker {worker_id} hit rate limit #{consecutive_count}, "
            f"cooling down for {cooldown:.1f}s"
        )
        return cooldown

    def should_apply_slow_start(self, items_since_resume: int) -> tuple[bool, float]:
        """Apply progressive slow-start after rate limit."""
        if items_since_resume >= self.slow_start_items:
            return (False, 0.0)

        # Progressive delay from initial to final
        progress = items_since_resume / self.slow_start_items
        delay = max(
            self.slow_start_final_delay,
            self.slow_start_initial_delay
            - (self.slow_start_initial_delay - self.slow_start_final_delay) * progress,
        )
        return (True, delay)


class FixedDelayStrategy(RateLimitStrategy):
    """Simple fixed delay between requests after rate limit."""

    def __init__(self, cooldown: float = 300.0, delay_between_requests: float = 1.0):
        """
        Initialize fixed delay strategy.

        Args:
            cooldown: Fixed cooldown duration in seconds
            delay_between_requests: Fixed delay between each request
        """
        self.cooldown = cooldown
        self.delay = delay_between_requests

    async def on_rate_limit(self, worker_id: int, consecutive_count: int) -> float:
        """Return fixed cooldown duration."""
        logger.info(f"Worker {worker_id} hit rate limit, cooling down for {self.cooldown:.1f}s")
        return self.cooldown

    def should_apply_slow_start(self, items_since_resume: int) -> tuple[bool, float]:
        """Always apply fixed delay."""
        return (True, self.delay)
