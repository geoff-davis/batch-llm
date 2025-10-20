"""Middleware system for batch processing pipeline."""

from typing import Generic, Protocol

from ..base import LLMWorkItem, TContext, TInput, TOutput, WorkItemResult


class Middleware(Protocol, Generic[TInput, TOutput, TContext]):
    """Middleware for intercepting processing pipeline."""

    async def before_process(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMWorkItem[TInput, TOutput, TContext] | None:
        """
        Called before processing each item.

        Args:
            work_item: The work item about to be processed

        Returns:
            Modified work item, or None to skip processing this item
        """
        ...

    async def after_process(
        self, result: WorkItemResult[TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        """
        Called after processing each item.

        Args:
            result: The result of processing

        Returns:
            Modified result (can add metadata, modify output, etc.)
        """
        ...

    async def on_error(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        error: Exception,
    ) -> WorkItemResult[TOutput, TContext] | None:
        """
        Called when an error occurs during processing.

        Args:
            work_item: The work item that failed
            error: The exception that was raised

        Returns:
            Custom result to use instead of error, or None to use default error handling
        """
        ...


class BaseMiddleware(Generic[TInput, TOutput, TContext]):
    """Base middleware with no-op implementations."""

    async def before_process(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMWorkItem[TInput, TOutput, TContext] | None:
        """Default: pass through unchanged."""
        return work_item

    async def after_process(
        self, result: WorkItemResult[TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        """Default: pass through unchanged."""
        return result

    async def on_error(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        error: Exception,
    ) -> WorkItemResult[TOutput, TContext] | None:
        """Default: use default error handling."""
        return None
