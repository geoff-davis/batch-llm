"""Example demonstrating context manager usage for automatic cleanup."""

import asyncio

from pydantic import BaseModel

from batch_llm import LLMWorkItem, ParallelBatchProcessor, ProcessorConfig
from batch_llm.testing import MockAgent


class SummaryOutput(BaseModel):
    """Example output model."""

    summary: str


async def main():
    """Example showing automatic resource cleanup with context manager."""

    # Create a mock agent for demonstration
    mock_agent = MockAgent(
        response_factory=lambda p: SummaryOutput(summary=f"Summary: {p}"),
        latency=0.01,
    )

    config = ProcessorConfig(max_workers=3, timeout_per_item=10.0)

    # Using async context manager ensures cleanup even if errors occur
    async with ParallelBatchProcessor[str, SummaryOutput, None](
        config=config
    ) as processor:
        # Add work items
        for i in range(5):
            await processor.add_work(
                LLMWorkItem(
                    item_id=f"item_{i}",
                    agent=mock_agent,
                    prompt=f"Summarize document {i}",
                    context=None,
                )
            )

        # Process all items
        result = await processor.process_all()

        print(f"\n{'='*60}")
        print(f"Processed: {result.total_items} items")
        print(f"Succeeded: {result.succeeded}")
        print(f"Failed: {result.failed}")
        print(f"{'='*60}\n")

    # Processor automatically cleaned up when exiting the context

    print("✓ Processor cleaned up automatically!")


if __name__ == "__main__":
    asyncio.run(main())
