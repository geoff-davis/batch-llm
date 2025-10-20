"""Base classes and interfaces for batch LLM processing."""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

# Conditional imports for type checking
if TYPE_CHECKING:
    from .llm_strategies import LLMCallStrategy

# Type variables for generic typing
TInput = TypeVar("TInput")  # Input data type
TOutput = TypeVar("TOutput")  # Agent output type
TContext = TypeVar("TContext")  # Optional context passed through


@dataclass
class LLMWorkItem(Generic[TInput, TOutput, TContext]):
    """
    Represents a single work item to be processed by an LLM strategy.

    Attributes:
        item_id: Unique identifier for this work item
        strategy: LLM call strategy that encapsulates how to make the LLM call
        prompt: The prompt/input to pass to the LLM
        context: Optional context data passed through to results/post-processor
    """

    item_id: str
    strategy: "LLMCallStrategy[TOutput]"
    prompt: str = ""
    context: TContext | None = None


@dataclass
class WorkItemResult(Generic[TOutput, TContext]):
    """
    Result of processing a single work item.

    Attributes:
        item_id: ID of the work item
        success: Whether processing succeeded
        output: Agent output if successful, None if failed
        error: Error message if failed, None if successful
        context: Context data from the work item
        token_usage: Token usage stats (input_tokens, output_tokens, total_tokens)
        gemini_safety_ratings: Gemini API safety ratings if available
    """

    item_id: str
    success: bool
    output: TOutput | None = None
    error: str | None = None
    context: TContext | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    gemini_safety_ratings: dict[str, str] | None = None


@dataclass
class BatchResult(Generic[TOutput, TContext]):
    """
    Result of processing a batch of work items.

    Attributes:
        results: List of individual work item results
        total_items: Total number of items in the batch
        succeeded: Number of successful items
        failed: Number of failed items
        total_input_tokens: Sum of input tokens across all items
        total_output_tokens: Sum of output tokens across all items
    """

    results: list[WorkItemResult[TOutput, TContext]]
    total_items: int = 0
    succeeded: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def __post_init__(self):
        """Calculate summary statistics from results."""
        self.total_items = len(self.results)
        self.succeeded = sum(1 for r in self.results if r.success)
        self.failed = sum(1 for r in self.results if not r.success)
        self.total_input_tokens = sum(
            r.token_usage.get("input_tokens", 0) for r in self.results
        )
        self.total_output_tokens = sum(
            r.token_usage.get("output_tokens", 0) for r in self.results
        )


# Type alias for post-processor function
PostProcessorFunc = Callable[
    [WorkItemResult[TOutput, TContext]], Awaitable[None] | None
]


class BatchProcessor(ABC, Generic[TInput, TOutput, TContext]):
    """
    Abstract base class for batch LLM processing strategies.

    Subclasses implement different strategies for processing batches:
    - ParallelBatchProcessor: Process items in parallel as individual requests
    - BatchAPIProcessor: Use Google's true batch API (future)
    """

    def __init__(
        self,
        max_workers: int = 5,
        post_processor: PostProcessorFunc[TOutput, TContext] | None = None,
        max_queue_size: int = 0,
    ):
        """
        Initialize the batch processor.

        Args:
            max_workers: Maximum number of concurrent workers
            post_processor: Optional async function called after each successful item
            max_queue_size: Maximum queue size (0 = unlimited)
        """
        self.max_workers = max_workers
        self.post_processor = post_processor
        self.max_queue_size = max_queue_size
        self._queue: asyncio.Queue[LLMWorkItem[TInput, TOutput, TContext] | None] = (
            asyncio.Queue(maxsize=max_queue_size)
        )
        self._results: list[WorkItemResult[TOutput, TContext]] = []
        self._stats = {
            "total": 0,
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "start_time": None,
            "error_counts": {},  # Track error types: {error_type: count}
            "rate_limit_count": 0,  # Track 429 errors from main LLM calls
        }
        self._workers: list[asyncio.Task] = []
        self._is_processing = False

    async def __aenter__(self):
        """Context manager entry - returns self for use in async with."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup of resources."""
        await self.cleanup()
        return False  # Don't suppress exceptions

    async def cleanup(self):
        """
        Clean up resources: cancel pending workers and clear queue.

        This method should be called when you're done with the processor,
        or use the processor as an async context manager.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Cancel any running workers
        if self._workers:
            logger.debug(f"Cleaning up {len(self._workers)} workers")
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()

            # Wait briefly for cancellations
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=2.0
                )
            except TimeoutError:
                logger.warning("Some workers did not cancel within timeout")

        # Clear any remaining items in queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def add_work(self, work_item: LLMWorkItem[TInput, TOutput, TContext]):
        """
        Add a work item to the processing queue.

        Args:
            work_item: Work item to process
        """
        await self._queue.put(work_item)
        self._stats["total"] += 1

    async def process_all(self) -> BatchResult[TOutput, TContext]:
        """
        Process all work items in the queue.

        Returns:
            BatchResult containing all results and statistics
        """
        # Record start time for rate calculation
        self._stats["start_time"] = time.time()
        self._is_processing = True

        # Add sentinel values to stop workers
        for _ in range(self.max_workers):
            await self._queue.put(None)

        # Start workers and store them for cleanup
        self._workers = [
            asyncio.create_task(self._worker(worker_id))
            for worker_id in range(self.max_workers)
        ]

        # Wait for all work to complete
        await self._queue.join()

        import logging
        logging.info("✓ Queue processing complete, waiting for workers to finish...")

        # Wait for workers to finish with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers),
                timeout=30.0  # 30 second timeout for workers to clean up
            )
            logging.info(f"✓ All {len(self._workers)} workers finished successfully")
        except TimeoutError:
            import logging
            logging.error(
                "⚠️  Workers did not finish within 30 seconds after queue.join(). "
                "Cancelling workers and proceeding..."
            )
            # Cancel any workers that are still running
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()
            # Wait briefly for cancellations to complete
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True), timeout=5.0
                )
            except TimeoutError:
                logging.error("⚠️  Some workers could not be cancelled")

        self._is_processing = False
        return BatchResult(results=self._results)

    @abstractmethod
    async def _worker(self, worker_id: int):
        """
        Worker coroutine that processes items from the queue.

        Each implementation defines its own worker strategy.

        Args:
            worker_id: Unique identifier for this worker
        """
        pass

    @abstractmethod
    async def _process_item(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        """
        Process a single work item.

        Each implementation defines how to execute the agent and handle results.

        Args:
            work_item: Work item to process

        Returns:
            Result of processing the work item
        """
        pass

    async def _run_post_processor(
        self, result: WorkItemResult[TOutput, TContext]
    ) -> None:
        """
        Run the post-processor callback if provided.

        Args:
            result: Work item result to post-process
        """
        if self.post_processor is None:
            return

        try:
            await_result = self.post_processor(result)
            # Handle both async and sync post-processors
            if asyncio.iscoroutine(await_result):
                # Inner timeout for the post-processor execution itself
                # (Outer timeout at worker level handles semaphore waits)
                await asyncio.wait_for(await_result, timeout=75.0)
        except TimeoutError:
            import logging
            logging.error(
                f"✗ Post-processor execution timed out after 75s for {result.item_id}"
            )
        except Exception as e:
            # Log error with full details - this is critical for debugging
            import logging
            import traceback

            logging.error(
                f"✗ Post-processor failed for {result.item_id}:\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error message: {str(e)}\n"
                f"  Full traceback:\n{traceback.format_exc()}"
            )
