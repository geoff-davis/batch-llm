"""Example usage of the batch_llm module.

This demonstrates how to use the batch processing framework with PydanticAI agents.
Includes examples of the new configuration system, middleware, observers, and testing utilities.
"""

import asyncio
import logging
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from batch_llm.base import LLMWorkItem, WorkItemResult
from batch_llm.core import ProcessorConfig, RateLimitConfig, RetryConfig
from batch_llm.classifiers import GeminiErrorClassifier
from batch_llm.observers import MetricsObserver
from batch_llm.middleware import BaseMiddleware
from batch_llm.parallel import ParallelBatchProcessor
from batch_llm.testing import MockAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Example 1: Simple output model
class BookSummary(BaseModel):
    """A simple book summary."""

    title: Annotated[str, Field(description="Book title")]
    summary: Annotated[str, Field(description="One-sentence summary")]
    genre: Annotated[str, Field(description="Primary genre")]


# Example 2: With context data
class EnrichmentContext(BaseModel):
    """Context data for enrichment tasks."""

    work_key: str
    original_title: str
    source_db: str = "openlibrary"


# Example custom middleware
class LoggingMiddleware(BaseMiddleware):
    """Example middleware that logs processing events."""

    async def before_process(
        self, work_item: LLMWorkItem[str, BookSummary, EnrichmentContext | None]
    ) -> LLMWorkItem[str, BookSummary, EnrichmentContext | None] | None:
        """Log before processing."""
        logging.debug(f"[MIDDLEWARE] Starting: {work_item.item_id}")
        return work_item

    async def after_process(
        self, result: WorkItemResult[BookSummary, EnrichmentContext | None]
    ) -> WorkItemResult[BookSummary, EnrichmentContext | None]:
        """Log after processing."""
        if result.success:
            logging.debug(f"[MIDDLEWARE] Completed: {result.item_id}")
        return result

    async def on_error(
        self,
        work_item: LLMWorkItem[str, BookSummary, EnrichmentContext | None],
        error: Exception,
    ) -> WorkItemResult[BookSummary, EnrichmentContext | None] | None:
        """Log errors."""
        logging.debug(f"[MIDDLEWARE] Error in {work_item.item_id}: {type(error).__name__}")
        return None  # Let default error handling proceed


async def example_post_processor(result: WorkItemResult[BookSummary, EnrichmentContext]):
    """
    Example post-processor that runs after each successful item.

    This could save to database, update metrics, etc.
    """
    if result.success and result.output:
        logging.info(
            f"Post-processing {result.item_id}: "
            f"{result.output.title} ({result.output.genre})"
        )
        # Could save to database here
        # await save_to_db(result.output, result.context)


async def example_simple():
    """
    Example 1: Simple batch processing with new configuration system.
    """
    logging.info("\n" + "=" * 80)
    logging.info("EXAMPLE 1: Simple Batch Processing (New API)")
    logging.info("=" * 80)

    # Create agent
    agent = Agent(
        "gemini-2.0-flash",
        output_type=BookSummary,
        system_prompt="You are a book summarization expert.",
    )

    # Create processor with new configuration system
    config = ProcessorConfig(
        max_workers=3,
        timeout_per_item=30.0,
    )

    processor = ParallelBatchProcessor[str, BookSummary, None](
        config=config,
        error_classifier=GeminiErrorClassifier(),
    )

    # Add work items
    books = [
        ("Pride and Prejudice", "Summarize Pride and Prejudice by Jane Austen"),
        ("1984", "Summarize 1984 by George Orwell"),
        ("The Hobbit", "Summarize The Hobbit by J.R.R. Tolkien"),
        ("Dune", "Summarize Dune by Frank Herbert"),
        ("Foundation", "Summarize Foundation by Isaac Asimov"),
    ]

    for book_id, prompt in books:
        work_item = LLMWorkItem(
            item_id=book_id,
            agent=agent,
            prompt=prompt,
            context=None,
        )
        await processor.add_work(work_item)

    # Process all
    result = await processor.process_all()

    # Show results
    logging.info(f"\nProcessed {result.total_items} items:")
    logging.info(f"  Succeeded: {result.succeeded}")
    logging.info(f"  Failed: {result.failed}")
    logging.info(f"  Total tokens: {result.total_input_tokens + result.total_output_tokens:,}")

    for item_result in result.results:
        if item_result.success:
            logging.info(
                f"  ✓ {item_result.item_id}: "
                f"{item_result.output.title} - {item_result.output.genre}"
            )
        else:
            logging.error(f"  ✗ {item_result.item_id}: {item_result.error}")


async def example_with_context_and_postprocessor():
    """
    Example 2: Batch processing with context, middleware, and observers.
    """
    logging.info("\n" + "=" * 80)
    logging.info("EXAMPLE 2: With Context, Middleware, and Observers")
    logging.info("=" * 80)

    # Create agent
    agent = Agent(
        "gemini-2.0-flash",
        output_type=BookSummary,
        system_prompt="You are a book summarization expert.",
    )

    # Create configuration with custom retry settings
    config = ProcessorConfig(
        max_workers=2,
        timeout_per_item=30.0,
        retry=RetryConfig(
            max_attempts=3,
            initial_wait=1.0,
            exponential_base=2.0,
        ),
    )

    # Create middleware and observers
    logging_middleware = LoggingMiddleware()
    metrics = MetricsObserver()

    # Create processor with middleware and observers
    processor = ParallelBatchProcessor[str, BookSummary, EnrichmentContext](
        config=config,
        post_processor=example_post_processor,  # Called after each success
        error_classifier=GeminiErrorClassifier(),
        middlewares=[logging_middleware],
        observers=[metrics],
    )

    # Add work items with context
    books = [
        {
            "work_key": "/works/OL123W",
            "original_title": "Pride and Prejudice",
            "prompt": "Summarize Pride and Prejudice by Jane Austen",
        },
        {
            "work_key": "/works/OL456W",
            "original_title": "1984",
            "prompt": "Summarize 1984 by George Orwell",
        },
        {
            "work_key": "/works/OL789W",
            "original_title": "The Hobbit",
            "prompt": "Summarize The Hobbit by J.R.R. Tolkien",
        },
    ]

    for book in books:
        context = EnrichmentContext(
            work_key=book["work_key"],
            original_title=book["original_title"],
        )
        work_item = LLMWorkItem(
            item_id=book["work_key"],
            agent=agent,
            prompt=book["prompt"],
            context=context,
        )
        await processor.add_work(work_item)

    # Process all
    result = await processor.process_all()

    # Show results with context
    logging.info(f"\nProcessed {result.total_items} items:")
    for item_result in result.results:
        if item_result.success and item_result.context:
            logging.info(
                f"  ✓ {item_result.context.work_key}: "
                f"{item_result.context.original_title} → {item_result.output.title}"
            )

    # Show metrics from observer
    collected_metrics = await metrics.get_metrics()
    logging.info("\nMetrics from Observer:")
    logging.info(f"  Success rate: {collected_metrics['success_rate']*100:.1f}%")
    logging.info(f"  Avg processing time: {collected_metrics['avg_processing_time']:.2f}s")
    logging.info(f"  Rate limits hit: {collected_metrics['rate_limits_hit']}")


async def example_error_handling():
    """
    Example 3: Handling errors and partial failures with custom configuration.
    """
    logging.info("\n" + "=" * 80)
    logging.info("EXAMPLE 3: Error Handling with Custom Configuration")
    logging.info("=" * 80)

    # Create agent
    agent = Agent(
        "gemini-2.0-flash",
        output_type=BookSummary,
        system_prompt="You are a book summarization expert.",
    )

    # Create processor with custom rate limit configuration
    config = ProcessorConfig(
        max_workers=2,
        timeout_per_item=0.1,  # Very short timeout will cause failures
        rate_limit=RateLimitConfig(
            cooldown_seconds=60.0,
            slow_start_items=10,
        ),
    )

    metrics = MetricsObserver()

    processor = ParallelBatchProcessor[str, BookSummary, None](
        config=config,
        error_classifier=GeminiErrorClassifier(),
        observers=[metrics],
    )

    # Add work items
    for i in range(5):
        work_item = LLMWorkItem(
            item_id=f"book_{i}",
            agent=agent,
            prompt=f"Summarize a book (this will likely timeout)",
            context=None,
        )
        await processor.add_work(work_item)

    # Process all
    result = await processor.process_all()

    # Show results
    logging.info(f"\nProcessed {result.total_items} items:")
    logging.info(f"  Succeeded: {result.succeeded}")
    logging.info(f"  Failed: {result.failed}")

    # Group failures by error type
    error_types: dict[str, int] = {}
    for item_result in result.results:
        if not item_result.success and item_result.error:
            error_type = item_result.error.split(":")[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1

    logging.info("\nError breakdown:")
    for error_type, count in error_types.items():
        logging.info(f"  {error_type}: {count}")

    # Show metrics
    collected_metrics = await metrics.get_metrics()
    logging.info(f"\nMetrics: {collected_metrics['items_failed']} failures out of {collected_metrics['items_processed']} items")


async def example_testing_with_mocks():
    """
    Example 4: Testing with MockAgent (no real LLM calls).
    """
    logging.info("\n" + "=" * 80)
    logging.info("EXAMPLE 4: Testing with MockAgent")
    logging.info("=" * 80)

    # Create a mock agent that simulates LLM behavior
    def mock_response_factory(prompt: str) -> BookSummary:
        """Generate mock book summaries."""
        return BookSummary(
            title="Mock Book Title",
            summary=f"Mock summary for: {prompt[:30]}...",
            genre="Fiction",
        )

    # Create mock agent with simulated latency (no rate limit for simplicity)
    mock_agent = MockAgent(
        response_factory=mock_response_factory,
        latency=0.05,  # 50ms simulated latency
        # Note: rate_limit_on_call is disabled for this simple example
        # You can enable it by setting rate_limit_on_call=3
    )

    # Create processor with fast configuration for testing
    config = ProcessorConfig(
        max_workers=2,
        timeout_per_item=10.0,
        rate_limit=RateLimitConfig(
            cooldown_seconds=1.0,  # Short cooldown for testing
            slow_start_items=2,
            slow_start_initial_delay=0.5,  # Fast slow-start for testing
            slow_start_final_delay=0.1,
        ),
    )

    metrics = MetricsObserver()

    processor = ParallelBatchProcessor[str, BookSummary, None](
        config=config,
        error_classifier=GeminiErrorClassifier(),
        observers=[metrics],
    )

    # Add work items
    books = ["Book 1", "Book 2", "Book 3", "Book 4", "Book 5"]
    for book_id in books:
        work_item = LLMWorkItem(
            item_id=book_id,
            agent=mock_agent,
            prompt=f"Summarize {book_id}",
            context=None,
        )
        await processor.add_work(work_item)

    # Process all
    result = await processor.process_all()

    # Show results
    logging.info(f"\nProcessed {result.total_items} items with MockAgent:")
    logging.info(f"  Succeeded: {result.succeeded}")
    logging.info(f"  Failed: {result.failed}")

    # Show metrics
    collected_metrics = await metrics.get_metrics()
    logging.info("\nMetrics from MockAgent test:")
    logging.info(f"  Mock agent calls: {mock_agent.call_count}")
    logging.info(f"  Rate limits hit: {collected_metrics['rate_limits_hit']}")
    logging.info(f"  Success rate: {collected_metrics['success_rate']*100:.1f}%")
    logging.info("\nNote: This test ran without making any real LLM API calls!")


async def main():
    """Run all examples."""
    # Note: Examples 1-3 require a valid Gemini API key
    # Example 4 uses MockAgent and does not require an API key

    # Uncomment to run:
    # await example_simple()
    # await example_with_context_and_postprocessor()
    # await example_error_handling()

    # This one can be run without an API key:
    await example_testing_with_mocks()

    logging.info(
        "\nExamples 1-3 are commented out. Uncomment in main() to run with a valid API key."
        "\nExample 4 (MockAgent) is enabled by default and doesn't need an API key."
    )


if __name__ == "__main__":
    asyncio.run(main())
