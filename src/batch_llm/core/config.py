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
            raise ValueError(
                f"max_attempts must be >= 1 (got {self.max_attempts}). "
                f"Set retry.max_attempts to a positive integer."
            )
        if self.initial_wait <= 0:
            raise ValueError(
                f"initial_wait must be > 0 (got {self.initial_wait}). "
                f"Set retry.initial_wait to a positive number in seconds."
            )
        if self.max_wait < self.initial_wait:
            raise ValueError(
                f"max_wait must be >= initial_wait (got max_wait={self.max_wait}, initial_wait={self.initial_wait}). "
                f"Set retry.max_wait to be at least as large as retry.initial_wait."
            )
        if self.exponential_base < 1:
            raise ValueError(
                f"exponential_base must be >= 1 (got {self.exponential_base}). "
                f"Set retry.exponential_base to 1.0 or higher (typical values: 2.0-3.0)."
            )


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
            raise ValueError(
                f"cooldown_seconds must be >= 0 (got {self.cooldown_seconds}). "
                f"Set rate_limit.cooldown_seconds to a non-negative number."
            )
        if self.slow_start_items < 0:
            raise ValueError(
                f"slow_start_items must be >= 0 (got {self.slow_start_items}). "
                f"Set rate_limit.slow_start_items to 0 to disable or a positive number."
            )
        if self.slow_start_initial_delay < self.slow_start_final_delay:
            raise ValueError(
                f"slow_start_initial_delay must be >= slow_start_final_delay "
                f"(got initial={self.slow_start_initial_delay}, final={self.slow_start_final_delay}). "
                f"The delay should decrease during slow start, not increase."
            )
        if self.backoff_multiplier < 1.0:
            raise ValueError(
                f"backoff_multiplier must be >= 1.0 (got {self.backoff_multiplier}). "
                f"Set rate_limit.backoff_multiplier to 1.0 (no increase) or higher (typical: 1.5-2.0)."
            )


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

    # Dry-run mode (for testing configuration without making API calls)
    dry_run: bool = False

    def validate(self) -> None:
        """Validate complete configuration."""
        if self.max_workers < 1:
            raise ValueError(
                f"max_workers must be >= 1 (got {self.max_workers}). "
                f"Set config.max_workers to a positive integer (typical: 5-20)."
            )
        if self.timeout_per_item <= 0:
            raise ValueError(
                f"timeout_per_item must be > 0 (got {self.timeout_per_item}). "
                f"Set config.timeout_per_item to a positive number in seconds (typical: 60-300)."
            )
        if self.progress_interval < 1:
            raise ValueError(
                f"progress_interval must be >= 1 (got {self.progress_interval}). "
                f"Set config.progress_interval to a positive integer."
            )
        if self.max_queue_size < 0:
            raise ValueError(
                f"max_queue_size must be >= 0 (got {self.max_queue_size}). "
                f"Set config.max_queue_size to 0 for unlimited, or a positive number to limit queue size."
            )

        self.retry.validate()
        self.rate_limit.validate()
