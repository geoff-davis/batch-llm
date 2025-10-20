"""Base classes and interfaces for batch LLM processing."""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

# Conditional import for PydanticAI (optional dependency)
if TYPE_CHECKING:
    from pydantic_ai import Agent
else:
    try:
        from pydantic_ai import Agent
    except ImportError:
        Agent = Any  # type: ignore[misc,assignment]

# Type variables for generic typing
TInput = TypeVar("TInput")  # Input data type
TOutput = TypeVar("TOutput")  # Agent output type
TContext = TypeVar("TContext")  # Optional context passed through


@dataclass
class LLMWorkItem(Generic[TInput, TOutput, TContext]):
    """
    Represents a single work item to be processed by an LLM agent or direct call.

    Attributes:
        item_id: Unique identifier for this work item
        agent: PydanticAI agent to use for processing (if agent_factory/direct_call not provided)
        prompt: The prompt/input to pass to the agent (not used for direct_call)
        context: Optional context data passed through to results/post-processor
        agent_factory: Optional callable that creates an agent based on attempt number.
                      Useful for progressive temperature increases on retries.
                      Signature: (attempt_number: int) -> Agent[None, TOutput]
        direct_call: Optional callable that directly calls LLM API with temperature control.
                    Signature: (attempt_number: int, timeout: float) -> Awaitable[tuple[TOutput, dict[str, int]]]
                    Returns: (result, token_usage_dict)
        input_data: Input data for direct_call (cluster, works, etc.)
    """

    item_id: str
    agent: "Agent[None, TOutput] | None" = None
    prompt: str = ""
    context: TContext | None = None
    agent_factory: "Callable[[int], Agent[None, TOutput]] | None" = None
    direct_call: Callable[[int, float], Awaitable[tuple[TOutput, dict[str, int]]]] | None = None
    input_data: TInput | None = None

    def __post_init__(self):
        """Validate that exactly one processing method is provided."""
        methods = [self.agent, self.agent_factory, self.direct_call]
        provided = sum(1 for m in methods if m is not None)
        if provided == 0:
            raise ValueError("Must provide one of: agent, agent_factory, or direct_call")
        if provided > 1:
            raise ValueError("Only one of agent, agent_factory, or direct_call should be provided")


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

        # Add sentinel values to stop workers
        for _ in range(self.max_workers):
            await self._queue.put(None)

        # Start workers
        workers = [
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
                asyncio.gather(*workers),
                timeout=30.0  # 30 second timeout for workers to clean up
            )
            logging.info(f"✓ All {len(workers)} workers finished successfully")
        except TimeoutError:
            import logging
            logging.error(
                "⚠️  Workers did not finish within 30 seconds after queue.join(). "
                "Cancelling workers and proceeding..."
            )
            # Cancel any workers that are still running
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            # Wait briefly for cancellations to complete
            try:
                await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=5.0)
            except TimeoutError:
                logging.error("⚠️  Some workers could not be cancelled")

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
