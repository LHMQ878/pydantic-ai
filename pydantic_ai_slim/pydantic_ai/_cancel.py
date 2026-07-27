"""Run-scoped cancellation controller for first-party run cancellation.

First-party cancellation (`AgentRun.cancel()`, `RunContext.cancel_run()`) is implemented by
cancelling the asyncio task that drives the run: that wakes whatever the run is blocked on (a
model stream, tool tasks, a suspended-job poll) and reuses the exact same teardown machinery as
external cancellation — streams are closed, in-flight tool tasks are cancelled and drained,
suspended server-side jobs are best-effort cancelled, and completed work is recorded to message
history. At the outer edge of [`Agent.iter()`][pydantic_ai.agent.Agent.iter], after teardown, the
resulting `CancelledError` is translated back into
[`RunCancelled`][pydantic_ai.exceptions.RunCancelled] — but only if the cancellation was ours:

- The controller counts every `Task.cancel()` it issues. On catching `CancelledError`, the
  outer edge consumes exactly that many cancellations via `Task.uncancel()` (mirroring what
  `asyncio.timeout()` does for its own cancellation).
- If `Task.cancelling()` is still positive afterwards, an *external* cancellation raced in; it
  wins, and the `CancelledError` keeps propagating as itself.

On Python 3.10, `Task.cancelling()`/`Task.uncancel()` don't exist, so the race cannot be
disambiguated: a requested first-party cancellation is translated to `RunCancelled` even if an
external cancellation arrived at the same time (documented degraded behavior).

The controller is runtime-only state: it holds a live task reference and is never serialized.
"""

from __future__ import annotations as _annotations

import asyncio
import dataclasses
import sys
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .run import AgentRun

__all__ = ('RunBinding', 'RunCancellation', 'provide_run_binding', 'take_run_binding')


class RunCancellation:
    """Tracks first-party cancellation of a single agent run.

    One instance per run, shared (by reference) between the run's public handles and its
    internals. The task driving the run binds itself with [`bind`][pydantic_ai._cancel.RunCancellation.bind]
    at each step boundary, so `cancel()` always cancels the task currently doing the work.
    """

    def __init__(self) -> None:
        self._owner: asyncio.Task[object] | None = None
        self._issued: dict[asyncio.Task[object], int] = {}
        self._requested = False
        self._finished = False

    @property
    def cancel_requested(self) -> bool:
        """Whether a first-party cancellation has been requested (and not yet resolved)."""
        return self._requested

    def bind(self, task: asyncio.Task[object] | None = None) -> None:
        """Bind the task that is currently driving the run.

        Called at run start and at each step boundary, so manual `AgentRun.next()` driving from
        a different task than the one that started the run still gets cancelled correctly.
        If a cancellation was requested before any task was bound (e.g. `cancel()` on a
        lazily-started run) or was issued to a previous driving task, it is (re-)delivered to
        this one.
        """
        task = task or asyncio.current_task()
        if task is None:  # pragma: no cover — agent runs always execute inside a task
            return
        self._owner = task
        if self._requested and not self._finished and task not in self._issued:
            # Deliver a request that arrived before this task was bound, or was previously
            # delivered to a different driving task.
            self._issue(task)

    def cancel(self) -> None:
        """Request cancellation of the run. Idempotent; a no-op once the run has finished."""
        if self._finished or self._requested:
            return
        self._requested = True
        if self._owner is not None and not self._owner.done():
            self._issue(self._owner)

    def _issue(self, task: asyncio.Task[object]) -> None:
        self._issued[task] = self._issued.get(task, 0) + 1
        task.cancel()

    def finish(self) -> None:
        """Mark the run as finished: later `cancel()` calls become no-ops."""
        self._finished = True

    def resolve(self) -> bool:
        """Resolve a caught `CancelledError` at the run's outer edge: is it ours to translate?

        Consumes only the cancellations this controller issued to the calling task via
        `Task.uncancel()`. Returns `True` if the cancellation was first-party and no external
        cancellation is still pending (translate to `RunCancelled`); `False` if it must keep
        propagating as `CancelledError`.

        Must be called on the task the cancellation was delivered to.
        """
        if not self._requested:
            return False
        if sys.version_info < (3, 11):  # pragma: lax no cover
            # No `Task.uncancel()`/`Task.cancelling()`: we can't tell whether an external
            # cancellation raced with ours, so a requested cancellation wins (documented).
            return True
        task = asyncio.current_task()
        if task is None:  # pragma: no cover — agent runs always execute inside a task
            return True
        count = self._issued.pop(task, 0)
        while count > 0 and task.cancelling() > 0:
            task.uncancel()
            count -= 1
        # Anything left on the counter was issued externally and takes precedence.
        return task.cancelling() == 0

    def release_issued(self) -> None:
        """Release controller-issued cancellations that were never resolved.

        This includes cancellations swallowed by user code or issued to a superseded driving
        task. Releasing them prevents contamination of the tasks' outer cancellation bookkeeping,
        such as `asyncio.timeout()` and AnyIO cancellation scopes.
        """
        if sys.version_info >= (3, 11):  # pragma: lax no cover
            for task, count in self._issued.items():
                if not task.done():
                    for _ in range(count):
                        if task.cancelling() > 0:
                            task.uncancel()
        self._issued.clear()


@dataclasses.dataclass
class RunBinding:
    """Bridge an `AgentRunEvents` handle to the run it starts.

    The handle exists before its lazy background run, so it owns the cancellation controller.
    `Agent.iter()` later attaches the live run state while retaining that same controller.
    """

    cancellation: RunCancellation = dataclasses.field(default_factory=RunCancellation)
    agent_run: AgentRun[Any, Any] | None = None


_current_run_binding: ContextVar[RunBinding | None] = ContextVar('pydantic_ai.run_binding', default=None)


@contextmanager
def provide_run_binding(binding: RunBinding) -> Generator[None]:
    """Set the binding for runs started in this context, resetting it on exit."""
    token = _current_run_binding.set(binding)
    try:
        yield
    finally:
        _current_run_binding.reset(token)


def take_run_binding() -> RunBinding | None:
    """Consume and return the pending binding at most once.

    Consuming prevents nested agent runs from inheriting the outer handle's binding.
    """
    binding = _current_run_binding.get()
    if binding is not None:
        _current_run_binding.set(None)
    return binding
