"""Configuration management for batch processor."""

from dataclasses import dataclass, field


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_wait: float = 1.0
    max_wait: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    def validate(self) -> None:
        """Validate retry configuration."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_wait <= 0:
            raise ValueError("initial_wait must be > 0")
        if self.max_wait < self.initial_wait:
            raise ValueError("max_wait must be >= initial_wait")
        if self.exponential_base < 1:
            raise ValueError("exponential_base must be >= 1")


@dataclass
class RateLimitConfig:
    """Configuration for rate limit handling."""

    cooldown_seconds: float = 300.0
    slow_start_items: int = 50
    slow_start_initial_delay: float = 2.0
    slow_start_final_delay: float = 0.1
    backoff_multiplier: float = 1.5  # Increase cooldown on repeated rate limits

    def validate(self) -> None:
        """Validate rate limit configuration."""
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        if self.slow_start_items < 0:
            raise ValueError("slow_start_items must be >= 0")
        if self.slow_start_initial_delay < self.slow_start_final_delay:
            raise ValueError(
                "slow_start_initial_delay must be >= slow_start_final_delay"
            )
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")


@dataclass
class ProcessorConfig:
    """Complete configuration for batch processor."""

    max_workers: int = 5
    timeout_per_item: float = 120.0

    retry: RetryConfig = field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Progress reporting
    progress_interval: int = 10  # Log every N items

    # Observability
    enable_detailed_logging: bool = False

    # Queue management
    max_queue_size: int = 0  # 0 = unlimited, >0 = max items in queue

    def validate(self) -> None:
        """Validate complete configuration."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.timeout_per_item <= 0:
            raise ValueError("timeout_per_item must be > 0")
        if self.progress_interval < 1:
            raise ValueError("progress_interval must be >= 1")
        if self.max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0 (0 = unlimited)")

        self.retry.validate()
        self.rate_limit.validate()
