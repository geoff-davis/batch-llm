"""Shared monotonic deadline waits and first-cause batch abort control."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from ..base import BatchTermination, LLMWorkItem, WorkItemResult
from ..core import AbortMode
from ..strategies.errors import BatchAbortedError, BatchDeadlineExceeded, ItemDeadlineExceeded

_T = TypeVar("_T")


class BatchAdmissionStopped(RuntimeError):
    """Internal signal that a controlled abort stopped source admission."""


@dataclass(frozen=True)
class AbortCause:
    kind: Literal["batch_timeout", "fail_fast"]
    reason: str
    error_category: str | None = None
    triggering_item_id: str | None = None


class AbortController:
    """Concurrency-safe, idempotent first-cause controller for one batch run."""

    def __init__(self, mode: AbortMode) -> None:
        self.mode = AbortMode(mode)
        self.event = asyncio.Event()
        self._lock = asyncio.Lock()
        self.cause: AbortCause | None = None
        self._active_provider_calls = 0

    @property
    def aborted(self) -> bool:
        return self.event.is_set()

    async def trip(self, cause: AbortCause) -> bool:
        """Record the first cause and wake waiters; return whether this call won."""
        async with self._lock:
            if self.cause is not None:
                return False
            self.cause = cause
            self.event.set()
            return True

    async def begin_provider_call(self, item_id: str | None) -> None:
        """Atomically register a provider attempt before a possible abort."""
        async with self._lock:
            if self.cause is not None:
                raise self.exception_for(item_id)
            self._active_provider_calls += 1

    async def end_provider_call(self) -> None:
        async with self._lock:
            self._active_provider_calls = max(0, self._active_provider_calls - 1)

    def exception_for(self, item_id: str | None) -> Exception:
        cause = self.cause
        if cause is None:
            raise RuntimeError("AbortController has no cause")
        if cause.kind == "batch_timeout":
            return BatchDeadlineExceeded(cause.reason, item_id=item_id)
        return BatchAbortedError(cause.reason, item_id=item_id)

    def raise_if_aborted(self, item_id: str | None) -> None:
        if self.aborted:
            raise self.exception_for(item_id)

    def result_for(self, work_item: LLMWorkItem[Any, Any, Any]) -> WorkItemResult[Any, Any]:
        exception = self.exception_for(work_item.item_id)
        category = (
            "batch_deadline_exceeded"
            if isinstance(exception, BatchDeadlineExceeded)
            else "batch_aborted"
        )
        return WorkItemResult(
            item_id=work_item.item_id,
            success=False,
            error=f"{type(exception).__name__}: {exception}",
            context=work_item.context,
            exception=exception,
            submission_index=work_item.submission_index,
            error_category=category,
        )

    def termination(self) -> BatchTermination:
        cause = self.cause
        if cause is None:
            return BatchTermination()
        return BatchTermination(
            kind=cause.kind,
            reason=cause.reason,
            error_category=cause.error_category,
            triggering_item_id=cause.triggering_item_id,
        )


def remaining_seconds(deadline: float | None, *, item_id: str | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise ItemDeadlineExceeded(
            "End-to-end item deadline exceeded",
            item_id=item_id,
        )
    return remaining


async def await_with_guardrails(
    awaitable: Awaitable[_T],
    *,
    item_deadline: float | None,
    item_id: str | None,
    abort_controller: AbortController | None = None,
    operation_timeout: float | None = None,
    active_provider: bool = False,
    on_start: Callable[[], None] | None = None,
) -> _T:
    """Await work bounded by item deadline, operation timeout, and batch abort.

    ``active_provider`` honors ``drain_active`` by ignoring a batch abort after
    the call starts. The controller is still checked immediately before start,
    so no new provider call begins after an abort.
    """

    def close_unstarted() -> None:
        if inspect.iscoroutine(awaitable):
            awaitable.close()

    provider_registered = False
    try:
        if abort_controller is not None and active_provider:
            await abort_controller.begin_provider_call(item_id)
            provider_registered = True
        elif abort_controller is not None:
            abort_controller.raise_if_aborted(item_id)
    except BaseException:
        close_unstarted()
        raise
    try:
        remaining = remaining_seconds(item_deadline, item_id=item_id)
    except BaseException:
        close_unstarted()
        if provider_registered and abort_controller is not None:
            await abort_controller.end_provider_call()
        raise

    timeout = operation_timeout
    deadline_is_limit = False
    if remaining is not None and (timeout is None or remaining <= timeout):
        timeout = remaining
        deadline_is_limit = True

    try:
        if on_start is not None:
            on_start()
    except BaseException:
        close_unstarted()
        if provider_registered and abort_controller is not None:
            await abort_controller.end_provider_call()
        raise

    task = asyncio.ensure_future(awaitable)
    timer = asyncio.create_task(asyncio.sleep(timeout)) if timeout is not None else None
    watch_abort = (
        asyncio.create_task(abort_controller.event.wait())
        if abort_controller is not None
        and not (active_provider and abort_controller.mode is AbortMode.DRAIN_ACTIVE)
        else None
    )
    controls = {control for control in (timer, watch_abort) if control is not None}
    try:
        if not controls:
            return await task
        done, _ = await asyncio.wait({task, *controls}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        if watch_abort is not None and watch_abort in done:
            assert abort_controller is not None
            raise abort_controller.exception_for(item_id)
        if deadline_is_limit:
            raise ItemDeadlineExceeded("End-to-end item deadline exceeded", item_id=item_id)
        raise TimeoutError
    except asyncio.CancelledError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        raise
    finally:
        for control in controls:
            if not control.done():
                control.cancel()
        if controls:
            await asyncio.gather(*controls, return_exceptions=True)
        if provider_registered and abort_controller is not None:
            await abort_controller.end_provider_call()


__all__ = [
    "AbortCause",
    "AbortController",
    "BatchAdmissionStopped",
    "await_with_guardrails",
    "remaining_seconds",
]
