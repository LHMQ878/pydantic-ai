"""Tests for the run-cancellation contract.

External cancellation: a step the run awaits — a Temporal activity under
`WAIT_CANCELLATION_COMPLETED`, an `event_stream_handler`, a capability hook — can absorb the
`CancelledError` injected by `task.cancel()` and return normally, which used to let a
cancelled run complete as if it was never cancelled. A level-triggered backstop
(`Task.cancelling()` re-checked at step boundaries) re-asserts the pending cancellation after
the completed step's messages have been recorded.

First-party cancellation: `AgentRun.cancel()` / `RunContext.cancel_run()` cancel the task
driving the run (reusing the external-cancellation teardown) and surface as `RunCancelled`,
never touching external semantics: an external `CancelledError` is never translated, and wins
when both race.

These are unit-style tests rather than VCR tests because the behavior under test is pure
control flow around injected `asyncio` cancellation, which no recorded provider response can
trigger.
"""

from __future__ import annotations as _annotations

import asyncio
import pickle
import sys
import threading
from collections.abc import AsyncIterable
from typing import Any

import anyio
import pytest
from anyio import to_thread
from anyio.from_thread import start_blocking_portal

from pydantic_ai import Agent, AgentRunEvents, AgentRunResultEvent, RunCancelled, UserError, capture_run_messages
from pydantic_ai._cancel import RunCancellation
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio

READINESS_WAIT_TIMEOUT = 5

requires_task_cancelling = pytest.mark.skipif(
    sys.version_info < (3, 11), reason='the backstop needs `Task.cancelling()` (Python 3.11+)'
)


def _task_cancelling(task: asyncio.Task[Any]) -> int:
    if sys.version_info < (3, 11):  # pragma: lax no cover
        return 0
    return task.cancelling()


def _task_uncancel(task: asyncio.Task[Any]) -> None:
    if sys.version_info < (3, 11):  # pragma: lax no cover
        return
    task.uncancel()


@requires_task_cancelling
async def test_swallowing_event_stream_handler_run_still_cancels():
    """An `event_stream_handler` that catches `CancelledError` must not absorb the run's cancellation.

    The handler is awaited on the run task itself, so its swallow used to let the whole run
    complete normally. The backstop re-asserts the pending cancellation at the next step
    boundary — after the partial response has been recorded.
    """
    in_flight = asyncio.Event()

    async def handler(ctx: RunContext, events: AsyncIterable[AgentStreamEvent]) -> None:
        try:
            async for _event in events:  # pragma: no branch
                in_flight.set()
                await asyncio.Event().wait()  # a slow consumer; cancel lands here
        except asyncio.CancelledError:
            pass  # "clean up the UI" — must not swallow the run's cancellation

    agent = Agent(TestModel())

    with capture_run_messages() as messages:
        task = asyncio.create_task(agent.run('hello', event_stream_handler=handler))
        await asyncio.wait_for(in_flight.wait(), timeout=READINESS_WAIT_TIMEOUT)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=READINESS_WAIT_TIMEOUT)

    # Cancellation never discards completed work: the interrupted turn is recorded.
    assert [type(m).__name__ for m in messages] == ['ModelRequest', 'ModelResponse']


@requires_task_cancelling
async def test_after_run_hook_cannot_convert_external_cancel_to_success():
    """An `after_run` hook that absorbs the cancellation must not let the run finalize as a
    success: the backstop fires before the result is stored."""
    in_flight = asyncio.Event()

    class SwallowInAfterRun(AbstractCapability):
        async def after_run(self, ctx: RunContext, *, result: AgentRunResult) -> AgentRunResult:
            in_flight.set()
            try:
                await asyncio.Event().wait()  # the in-flight "durable step"
            except asyncio.CancelledError:
                pass  # step completed successfully; cancellation consumed
            return result

    agent = Agent(TestModel(), capabilities=[SwallowInAfterRun()])

    task = asyncio.create_task(agent.run('hello'))
    await asyncio.wait_for(in_flight.wait(), timeout=READINESS_WAIT_TIMEOUT)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=READINESS_WAIT_TIMEOUT)


async def test_consumed_cancellation_is_not_a_false_positive():
    """A timeout scope inside a tool consumes its own cancellation via `Task.uncancel()` (as
    `asyncio.timeout()` and AnyIO scopes do), so the backstop must not misread it as a pending
    external cancellation."""
    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_lookup() -> str:
        with anyio.move_on_after(0.01):
            await asyncio.sleep(10)
        return 'timed out, moved on'

    result = await agent.run('hello')
    assert result.output == '{"slow_lookup":"timed out, moved on"}'


# --- First-party cancellation: `AgentRun.cancel()` / `RunContext.cancel_run()` ---


def test_run_cancelled_result_surface():
    """`RunCancelled` exposes the result accessors while keeping a detached history snapshot."""
    messages = [
        ModelRequest(parts=[UserPromptPart('previous')]),
        ModelResponse(parts=[TextPart('cancelled')]),
    ]
    usage = RunUsage(requests=1)
    error = RunCancelled(
        'cancelled',
        messages=messages,
        new_message_index=1,
        usage=usage,
        metadata={'customer': '123'},
        run_id='run-123',
        conversation_id='conversation-123',
    )

    assert error.all_messages() == messages
    assert error.all_messages() is not messages
    assert error.all_messages_json() == ModelMessagesTypeAdapter.dump_json(messages)
    assert error.new_messages() == messages[1:]
    assert error.new_messages_json() == ModelMessagesTypeAdapter.dump_json(messages[1:])
    assert error.response is messages[-1]
    assert error.timestamp == messages[-1].timestamp
    assert error.usage is usage
    assert error.metadata == {'customer': '123'}
    assert error.run_id == 'run-123'
    assert error.conversation_id == 'conversation-123'


def test_run_cancelled_defaults_before_model_response():
    """A cancellation before the first response has zero usage and no response-derived timestamp."""
    error = RunCancelled('cancelled')

    assert error.all_messages() == []
    assert error.new_messages() == []
    assert error.usage == RunUsage()
    assert error.metadata is None
    assert error.run_id is None
    assert error.conversation_id is None
    with pytest.raises(ValueError, match='No response found in the message history'):
        _ = error.response
    with pytest.raises(ValueError, match='No response found in the message history'):
        _ = error.timestamp


def test_run_cancelled_pickle_round_trip():
    """Pickling preserves the detached history, new-message boundary, usage, metadata, and IDs."""
    messages = [
        ModelRequest(parts=[UserPromptPart('previous')]),
        ModelResponse(parts=[TextPart('cancelled')]),
    ]
    error = RunCancelled(
        'cancelled',
        messages=messages,
        new_message_index=1,
        usage=RunUsage(requests=1),
        metadata={'customer': '123'},
        run_id='run-123',
        conversation_id='conversation-123',
    )

    restored = pickle.loads(pickle.dumps(error))

    assert restored.all_messages() == messages
    assert restored.new_messages() == messages[1:]
    assert restored.usage == RunUsage(requests=1)
    assert restored.metadata == {'customer': '123'}
    assert restored.run_id == 'run-123'
    assert restored.conversation_id == 'conversation-123'


def _parallel_tools_agent() -> tuple[Agent, list[list[ModelMessage]]]:
    """An agent whose first response calls a fast tool and a slow self-cancelling tool.

    Returns the agent and a list capturing the raw messages each model request receives, so
    tests can assert exactly what a resumed run sends to the model.
    """
    seen_by_model: list[list[ModelMessage]] = []

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_by_model.append(list(messages))
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='fast_tool', args={}, tool_call_id='call_fast'),
                    ToolCallPart(tool_name='cancelling_tool', args={}, tool_call_id='call_slow'),
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_func))

    @agent.tool_plain
    async def fast_tool() -> str:
        return 'fast result'

    @agent.tool
    async def cancelling_tool(ctx: RunContext) -> str:
        await asyncio.sleep(0.05)  # let the sibling finish first
        ctx.cancel_run()
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'never reached'  # pragma: no cover

    return agent, seen_by_model


async def test_tool_cancels_run_and_history_is_resumable():
    """`ctx.cancel_run()` from a tool raises `RunCancelled` from `agent.run()`.

    The completed sibling tool's real result is preserved in an interrupted request on
    `RunCancelled.all_messages()`, and resuming with that history plus a new prompt sends the model a
    provider-valid transcript: the real return, exactly one synthesized `'interrupted'` return
    for the cancelled call, and the new prompt.
    """
    agent, seen_by_model = _parallel_tools_agent()

    with capture_run_messages() as live_messages, pytest.raises(RunCancelled) as exc_info:
        await agent.run('go', metadata={'customer': '123'})

    error = exc_info.value
    messages = error.all_messages()
    assert messages is not live_messages
    assert messages == live_messages
    assert error.new_messages() == messages
    assert error.response is messages[1]
    assert error.usage.requests == 1
    assert error.metadata == {'customer': '123'}
    assert error.run_id is not None
    assert error.conversation_id is not None
    assert [(type(m).__name__, getattr(m, 'state', None)) for m in messages] == [
        ('ModelRequest', 'complete'),
        ('ModelResponse', 'complete'),
        ('ModelRequest', 'interrupted'),
    ]
    (fast_return,) = messages[-1].parts
    assert isinstance(fast_return, ToolReturnPart)
    assert fast_return.tool_name == 'fast_tool'
    assert fast_return.content == 'fast result'

    result = await agent.run('never mind, wrap up', message_history=messages)
    assert result.output == 'done'
    resumed_request = seen_by_model[-1][-1]
    returns = [p for p in resumed_request.parts if isinstance(p, ToolReturnPart)]
    assert [(r.tool_name, r.outcome) for r in returns] == [('fast_tool', 'success'), ('cancelling_tool', 'interrupted')]
    synthesized = returns[-1]
    assert synthesized.metadata == {'pydantic_ai_synthesized_tool_return': True}


async def test_agent_run_cancel_from_another_task():
    """`AgentRun.cancel()` is safe to call from a sibling task (a TUI's Esc handler) and
    surfaces as `RunCancelled` from whatever is driving the run."""
    started = asyncio.Event()

    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'slow'  # pragma: no cover

    runs: list[Any] = []

    async def drive():
        async with agent.iter('go') as agent_run:
            runs.append(agent_run)
            async for _node in agent_run:
                pass

    task = asyncio.create_task(drive())
    await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    runs[0].cancel()

    with pytest.raises(RunCancelled):
        await asyncio.wait_for(task, timeout=READINESS_WAIT_TIMEOUT)


async def test_iter_cancellation_is_typed_only_after_context_exit():
    """The run sees `CancelledError`; the same cancellation becomes `RunCancelled` after teardown."""
    agent = Agent(TestModel())
    seen_inside = False

    with pytest.raises(RunCancelled):
        async with agent.iter('go') as agent_run:
            agent_run.cancel()
            try:
                await anext(agent_run)
            except asyncio.CancelledError:
                seen_inside = True
                raise

    assert seen_inside


@requires_task_cancelling
async def test_iter_swallowed_cancellation_is_quiet_abandonment():
    """Leaving `agent.iter()` normally after swallowing its cancellation cleans task state."""
    agent = Agent(TestModel())

    async with agent.iter('go') as agent_run:
        agent_run.cancel()
        try:
            await anext(agent_run)
        except asyncio.CancelledError:
            pass

    assert agent_run.result is None
    task = asyncio.current_task()
    assert task is not None
    assert _task_cancelling(task) == 0


@requires_task_cancelling
async def test_iter_reasserts_swallowed_cancellation_before_next_node():
    """A swallowed first-party cancellation stops iteration before another model call."""
    model_calls: list[None] = []

    def model_function(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        model_calls.append(None)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_function))

    with pytest.raises(RunCancelled):
        async with agent.iter('go') as agent_run:
            async for _node in agent_run:
                if len(model_calls) == 1:
                    agent_run.cancel()
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError:
                        pass

    assert len(model_calls) == 1


@requires_task_cancelling
async def test_run_cancellation_tracks_issuances_per_task():
    """A controller unit test pins the task-rebind window that the public API cannot trigger
    deterministically."""
    cancellation = RunCancellation()
    a_bound = asyncio.Event()
    a_cancelled = asyncio.Event()
    resolve_a = asyncio.Event()
    a_resolved = asyncio.Event()
    external_cancel_a = asyncio.Event()
    a_state: list[tuple[bool, int]] = []

    async def drive_a() -> None:
        task = asyncio.current_task()
        assert task is not None
        cancellation.bind(task)
        a_bound.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            a_cancelled.set()
        await resolve_a.wait()
        a_state.append((cancellation.resolve(), _task_cancelling(task)))
        a_resolved.set()
        try:
            await external_cancel_a.wait()
        except asyncio.CancelledError:
            a_state.append((cancellation.resolve(), _task_cancelling(task)))
            _task_uncancel(task)

    task_a = asyncio.create_task(drive_a())
    await a_bound.wait()
    cancellation.cancel()
    cancellation.cancel()  # idempotent while a request is already pending
    await a_cancelled.wait()

    b_cancelled = asyncio.Event()
    finish_b = asyncio.Event()
    b_state: list[int] = []

    async def drive_b() -> None:
        task = asyncio.current_task()
        assert task is not None
        try:
            cancellation.bind(task)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            b_cancelled.set()
        await finish_b.wait()
        b_state.append(_task_cancelling(task))

    task_b = asyncio.create_task(drive_b())
    await b_cancelled.wait()
    resolve_a.set()
    await a_resolved.wait()
    assert a_state == [(True, 0)]

    cancellation.release_issued()
    finish_b.set()
    await task_b
    assert b_state == [0]

    task_a.cancel()
    await task_a
    assert a_state == [(True, 0), (False, 1)]

    done_cancellation = RunCancellation()
    done_bound = asyncio.Event()

    async def finish_with_issued_cancellation() -> None:
        task = asyncio.current_task()
        assert task is not None
        done_cancellation.bind(task)
        done_bound.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass

    done_task = asyncio.create_task(finish_with_issued_cancellation())
    await done_bound.wait()
    done_cancellation.cancel()
    await done_task
    done_cancellation.release_issued()
    done_cancellation.release_issued()  # clearing an already-empty controller is a no-op

    unbound_cancellation = RunCancellation()
    unbound_cancellation.cancel()
    rebound_cancelled = asyncio.Event()

    async def bind_after_request() -> None:
        task = asyncio.current_task()
        assert task is not None
        try:
            unbound_cancellation.bind(task)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            rebound_cancelled.set()
            assert unbound_cancellation.resolve()

    rebound_task = asyncio.create_task(bind_after_request())
    await rebound_cancelled.wait()
    await rebound_task

    uncancelled_cancellation = RunCancellation()
    uncancelled_bound = asyncio.Event()
    keep_uncancelled_task_live = asyncio.Event()
    uncancelled_redelivered = asyncio.Event()

    async def swallow_and_uncancel() -> None:
        task = asyncio.current_task()
        assert task is not None
        uncancelled_cancellation.bind(task)
        uncancelled_bound.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            _task_uncancel(task)
        try:
            await keep_uncancelled_task_live.wait()
        except asyncio.CancelledError:
            uncancelled_redelivered.set()
            assert uncancelled_cancellation.resolve()

    uncancelled_task = asyncio.create_task(swallow_and_uncancel())
    await uncancelled_bound.wait()
    uncancelled_cancellation.cancel()
    await asyncio.sleep(0)
    uncancelled_cancellation.bind(uncancelled_task)
    await uncancelled_redelivered.wait()
    keep_uncancelled_task_live.set()
    await uncancelled_task


async def test_event_stream_handler_cancels_run():
    """`ctx.cancel_run()` from an `event_stream_handler` (the TUI Esc gesture) cancels the run;
    the partial response streamed so far is preserved by `RunCancelled.all_messages()`."""

    async def handler(ctx: RunContext, events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _event in events:  # pragma: no branch
            ctx.cancel_run()

    agent = Agent(TestModel(custom_output_text='a few words of output'))

    with pytest.raises(RunCancelled) as exc_info:
        await agent.run('go', event_stream_handler=handler)

    response = exc_info.value.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'interrupted'


async def test_external_cancellation_is_never_translated():
    """Externally cancelling the task running the agent keeps raising `CancelledError`, not
    `RunCancelled` — it's an infrastructure signal, not an application outcome."""
    started = asyncio.Event()

    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'slow'  # pragma: no cover

    task = asyncio.create_task(agent.run('go'))
    await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=READINESS_WAIT_TIMEOUT)


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason='`CancelledError` instance preservation across `await task` needs Python 3.11+'
)
async def test_task_cancel_of_run_carries_run_cancelled():
    started = asyncio.Event()
    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'slow'  # pragma: no cover

    task = asyncio.create_task(agent.run('go'))
    await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    cancelled = RunCancelled.from_cancellation(exc_info.value)
    assert cancelled is not None
    assert [type(message) for message in cancelled.all_messages()] == [ModelRequest, ModelResponse]
    response = cancelled.all_messages()[1]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], ToolCallPart)
    assert cancelled.usage.requests == 1
    assert cancelled.run_id is not None
    assert task.cancelled()


async def test_direct_await_cancellation_carries_run_cancelled_on_all_versions():
    started = asyncio.Event()
    agent = Agent(TestModel())
    recorded: list[RunCancelled | None] = []

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'slow'  # pragma: no cover

    async def run_and_record() -> None:
        try:
            await agent.run('go')
        except asyncio.CancelledError as exc:
            recorded.append(RunCancelled.from_cancellation(exc))
            raise

    task = asyncio.create_task(run_and_record())
    await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (cancelled,) = recorded
    assert cancelled is not None
    assert [type(message) for message in cancelled.all_messages()] == [ModelRequest, ModelResponse]
    assert cancelled.usage.requests == 1
    assert cancelled.run_id is not None


@pytest.mark.skipif(sys.version_info < (3, 11), reason='`asyncio.timeout()` needs Python 3.11+')
async def test_from_cancellation_through_asyncio_timeout():
    started = asyncio.Event()
    agent = Agent(TestModel())
    # `Any` because `asyncio.Timeout` doesn't exist on Pyright's 3.10 target.
    timeout_scope: list[Any] = []

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        # Expire the enclosing timeout only now, so the model response is deterministically
        # recorded before the cancellation lands (a fixed small timeout raced run progress in CI).
        timeout_scope[0].reschedule(asyncio.get_running_loop().time())
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'slow'  # pragma: no cover

    with pytest.raises(TimeoutError) as exc_info:
        # This test is version-gated, but Pyright targets the package's Python 3.10 minimum.
        async with asyncio.timeout(READINESS_WAIT_TIMEOUT) as scope:  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
            timeout_scope.append(scope)
            await agent.run('go')

    assert isinstance(exc_info.value, TimeoutError)
    cancelled = RunCancelled.from_cancellation(exc_info.value)
    assert cancelled is not None
    assert [type(message) for message in cancelled.all_messages()] == [ModelRequest, ModelResponse]
    assert started.is_set()


def test_from_cancellation_identity_and_none():
    cancelled = RunCancelled('x')
    caused = ValueError('wrapper')
    caused.__cause__ = cancelled

    assert RunCancelled.from_cancellation(cancelled) is cancelled
    assert RunCancelled.from_cancellation(caused) is cancelled
    assert RunCancelled.from_cancellation(asyncio.CancelledError()) is None
    assert RunCancelled.from_cancellation(ValueError()) is None


def test_from_cancellation_cycle_safe():
    first = ValueError('first')
    second = RuntimeError('second')
    first.__context__ = second
    second.__context__ = first

    assert RunCancelled.from_cancellation(first) is None


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason='`CancelledError` instance preservation across `await task` needs Python 3.11+'
)
async def test_iter_external_cancel_carries_run_cancelled():
    started = asyncio.Event()
    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'slow'  # pragma: no cover

    async def drive() -> None:
        async with agent.iter('go') as agent_run:
            async for _node in agent_run:
                pass

    task = asyncio.create_task(drive())
    await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    cancelled = RunCancelled.from_cancellation(exc_info.value)
    assert cancelled is not None
    assert [type(message) for message in cancelled.all_messages()] == [ModelRequest, ModelResponse]
    assert cancelled.usage.requests == 1


@requires_task_cancelling
async def test_external_cancellation_wins_race_with_first_party_cancel():
    """When `cancel()` and an external `task.cancel()` race, the external cancellation wins and
    propagates as `CancelledError`."""
    started = asyncio.Event()

    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'slow'  # pragma: no cover

    runs: list[Any] = []

    async def drive():
        async with agent.iter('go') as agent_run:
            runs.append(agent_run)
            async for _node in agent_run:
                pass

    task = asyncio.create_task(drive())
    await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    runs[0].cancel()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=READINESS_WAIT_TIMEOUT)


@requires_task_cancelling
async def test_external_cancellation_wins_when_it_arrives_first():
    """An external cancellation delivered before `cancel()` still wins the race."""
    started = asyncio.Event()

    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'slow'  # pragma: no cover

    runs: list[Any] = []

    async def drive():
        async with agent.iter('go') as agent_run:
            runs.append(agent_run)
            async for _node in agent_run:
                pass

    task = asyncio.create_task(drive())
    await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    task.cancel()
    runs[0].cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=READINESS_WAIT_TIMEOUT)
    assert _task_cancelling(task) == 1


async def test_cancel_run_under_run_stream_events():
    """With `run_stream_events()` the run is driven by a background task; `ctx.cancel_run()`
    from a tool must cancel *that* task and surface `RunCancelled` to the event consumer."""
    agent = Agent(TestModel())

    @agent.tool
    async def cancelling_tool(ctx: RunContext) -> str:
        ctx.cancel_run()
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'never reached'  # pragma: no cover

    events: list[str] = []
    with pytest.raises(RunCancelled):
        async with agent.run_stream_events('go') as stream:
            async for event in stream:
                events.append(type(event).__name__)

    assert events  # events streamed before the cancellation are delivered


async def test_run_stream_events_cancel_mid_iteration():
    """The public event handle cancels its run while preserving events and live history."""
    agent = Agent(TestModel(custom_output_text='several words of output'))
    received: list[AgentStreamEvent | AgentRunResultEvent[str]] = []

    async with agent.run_stream_events('go') as events:
        with pytest.raises(RunCancelled):
            async for event in events:
                received.append(event)
                events.cancel()

        assert received
        assert events.all_messages()
        assert events.result is None


async def test_run_stream_events_cancel_from_sibling_task():
    """A sibling task can cancel while the consumer is blocked waiting for its next event."""
    started = asyncio.Event()
    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.Event().wait()
        return 'never reached'  # pragma: no cover

    async with agent.run_stream_events('go') as events:
        consumer = asyncio.create_task(_consume_events(events))
        await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
        events.cancel()
        with pytest.raises(RunCancelled):
            await asyncio.wait_for(consumer, timeout=READINESS_WAIT_TIMEOUT)


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason='`CancelledError` instance preservation across `await task` needs Python 3.11+'
)
async def test_run_stream_events_external_cancel_of_consumer():
    """Externally cancelling the consumer task keeps standard `CancelledError` semantics, but the
    run state rides along for `from_cancellation()` and the handle stays accessible post-cancel."""
    started = asyncio.Event()
    agent = Agent(TestModel())
    holder: list[AgentRunEvents[str]] = []

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.Event().wait()
        return 'never reached'  # pragma: no cover

    async def consume() -> None:
        async with agent.run_stream_events('go') as events:
            holder.append(events)
            await _consume_events(events)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    assert task.cancelled()
    cancelled = RunCancelled.from_cancellation(exc_info.value)
    assert cancelled is not None
    assert [type(message) for message in cancelled.all_messages()] == [ModelRequest, ModelResponse]
    assert cancelled.usage.requests == 1
    # The handle itself also remains usable after teardown.
    (events,) = holder
    assert [type(message) for message in events.all_messages()] == [ModelRequest, ModelResponse]
    assert events.result is None


async def test_run_stream_events_external_cancel_caught_in_task():
    """The consumer's own `except CancelledError` sees the attached state on all Python versions:
    the state is attached by the context manager's exit in the consumer's task, so no `await task`
    boundary is crossed before the catch."""
    started = asyncio.Event()
    agent = Agent(TestModel())
    recorded: list[RunCancelled | None] = []

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.Event().wait()
        return 'never reached'  # pragma: no cover

    async def consume() -> None:
        try:
            async with agent.run_stream_events('go') as events:
                await _consume_events(events)
        except asyncio.CancelledError as exc:
            recorded.append(RunCancelled.from_cancellation(exc))
            raise

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (cancelled,) = recorded
    assert cancelled is not None
    assert [type(message) for message in cancelled.all_messages()] == [ModelRequest, ModelResponse]


async def test_run_stream_events_external_cancel_before_iteration_attaches_nothing():
    """Cancelling a consumer that entered the context manager but never iterated has no run state
    to attach: the exception propagates bare."""
    agent = Agent(TestModel())
    recorded: list[RunCancelled | None] = []

    async def consume() -> None:
        try:
            async with agent.run_stream_events('go'):
                await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            recorded.append(RunCancelled.from_cancellation(exc))
            raise

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert recorded == [None]


async def test_concurrent_run_stream_events_handles_do_not_cross_consume():
    """Concurrent lazy runs created in the same context retain their own bindings."""
    started = {'first': asyncio.Event(), 'second': asyncio.Event()}
    finish_second = asyncio.Event()
    agents = {'first': Agent(TestModel()), 'second': Agent(TestModel())}

    @agents['first'].tool_plain
    async def block_first() -> str:
        started['first'].set()
        await asyncio.Event().wait()
        return 'never reached'  # pragma: no cover

    @agents['second'].tool_plain
    async def block_second() -> str:
        started['second'].set()
        await finish_second.wait()
        return 'second complete'

    handles: dict[str, AgentRunEvents[str]] = {}

    async def consume(name: str) -> AgentRunResultEvent[str]:
        async with agents[name].run_stream_events(name) as events:
            handles[name] = events
            result_event: AgentRunResultEvent[str] | None = None
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    result_event = event
            assert result_event is not None
            return result_event

    first_consumer = asyncio.create_task(consume('first'))
    second_consumer = asyncio.create_task(consume('second'))
    await asyncio.wait_for(
        asyncio.gather(started['first'].wait(), started['second'].wait()), timeout=READINESS_WAIT_TIMEOUT
    )
    handles['first'].cancel()
    finish_second.set()

    with pytest.raises(RunCancelled):
        await asyncio.wait_for(first_consumer, timeout=READINESS_WAIT_TIMEOUT)
    second_result_event = await asyncio.wait_for(second_consumer, timeout=READINESS_WAIT_TIMEOUT)

    second_events = handles['second']
    assert second_result_event.result.output == '{"block_second":"second complete"}'
    assert second_events.result is second_result_event.result
    assert any(
        isinstance(part, ToolReturnPart) and part.content == 'second complete'
        for message in second_events.all_messages()
        for part in message.parts
    )


async def test_run_stream_events_cancel_from_worker_thread():
    """The documented loop marshalling pattern cancels a run from a worker thread."""
    started = asyncio.Event()
    agent = Agent(TestModel())

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.Event().wait()
        return 'never reached'  # pragma: no cover

    loop = asyncio.get_running_loop()
    async with agent.run_stream_events('go') as events:
        consumer = asyncio.create_task(_consume_events(events))
        await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
        worker = threading.Thread(target=loop.call_soon_threadsafe, args=(events.cancel,))
        worker.start()
        await to_thread.run_sync(worker.join)

        with pytest.raises(RunCancelled) as exc_info:
            await asyncio.wait_for(consumer, timeout=READINESS_WAIT_TIMEOUT)

    assert exc_info.value.all_messages()


async def _consume_events(events: AsyncIterable[AgentStreamEvent | AgentRunResultEvent[Any]]) -> None:
    async for _event in events:
        pass


async def test_run_stream_events_cancel_before_iteration():
    """A pre-start cancellation prevents the lazy run from starting."""
    model_calls = 0

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1  # pragma: no cover - the pre-start cancel must prevent any model call
        return ModelResponse(parts=[TextPart('done')])  # pragma: no cover

    agent = Agent(FunctionModel(model_func))
    received: list[AgentStreamEvent | AgentRunResultEvent[str]] = []

    async with agent.run_stream_events('go') as events:
        with pytest.raises(RunCancelled) as exc_info:
            events.cancel()
            async for event in events:
                received.append(event)  # pragma: no cover - no events may be delivered

        assert received == []
        assert model_calls == 0
        assert exc_info.value.all_messages() == []
        assert events.result is None
        with pytest.raises(UserError, match='run has not started; iterate the events first'):
            events.all_messages()


async def test_run_stream_events_cancel_without_iteration():
    """Cancelling and closing an unstarted handle remains quiet and does not create run state."""
    agent = Agent(TestModel())

    async with agent.run_stream_events('go') as events:
        events.cancel()

    assert events.result is None
    with pytest.raises(UserError, match='run has not started; iterate the events first'):
        events.all_messages()


async def test_run_stream_events_state_before_and_after_completion():
    """State access rejects an unstarted run and exposes the successful final result."""
    agent = Agent(TestModel())
    result_event: AgentRunResultEvent[str] | None = None

    async with agent.run_stream_events('go') as events:
        with pytest.raises(UserError, match='run has not started; iterate the events first'):
            events.all_messages()
        with pytest.raises(UserError, match='run has not started; iterate the events first'):
            _ = events.usage
        async for event in events:
            if isinstance(event, AgentRunResultEvent):
                result_event = event

    assert result_event is not None
    assert events.all_messages()
    assert events.new_messages()
    assert events.usage.requests == 1
    assert events.result is not None
    assert events.result.output == result_event.result.output


async def test_run_stream_events_cancel_after_completion_is_noop():
    """Cancelling a completed event handle cannot disturb later work in the consumer task."""
    agent = Agent(TestModel())

    async with agent.run_stream_events('go') as events:
        async for _event in events:
            pass

    result = events.result
    events.cancel()
    await asyncio.sleep(0)
    assert events.result is result


async def test_run_stream_events_early_break_has_no_result():
    """Leaving after an early break quietly tears down the background run without a result."""
    agent = Agent(TestModel())

    async with agent.run_stream_events('go') as events:
        await anext(events)

    assert events.result is None


async def test_run_stream_events_binding_does_not_leak_to_nested_run():
    """A plain nested run completes independently before cancellation targets the outer run."""
    inner_completed = asyncio.Event()
    outer_blocked = asyncio.Event()
    inner_agent = Agent(TestModel(custom_output_text='inner complete'))
    outer_agent = Agent(TestModel())

    @outer_agent.tool_plain
    async def nested_run() -> str:
        result = await inner_agent.run('inner')
        inner_completed.set()
        outer_blocked.set()
        await asyncio.Event().wait()
        return result.output  # pragma: no cover

    async with outer_agent.run_stream_events('outer') as events:
        consumer = asyncio.create_task(_consume_events(events))
        await asyncio.wait_for(outer_blocked.wait(), timeout=READINESS_WAIT_TIMEOUT)
        events.cancel()
        with pytest.raises(RunCancelled):
            await asyncio.wait_for(consumer, timeout=READINESS_WAIT_TIMEOUT)

    assert inner_completed.is_set()


async def test_run_stream_events_binding_across_blocking_portal():
    """Bindings reach a portal loop, including cancellation marshalled from the outer thread."""

    def run_portal_cases() -> None:
        with start_blocking_portal() as portal:
            completed_agent = Agent(TestModel(custom_output_text='portal complete'))

            async def consume_completed() -> AgentRunResultEvent[str]:
                async with completed_agent.run_stream_events('go') as events:
                    result_event: AgentRunResultEvent[str] | None = None
                    async for event in events:
                        if isinstance(event, AgentRunResultEvent):
                            result_event = event
                    assert events.result is not None
                    assert events.all_messages()
                    assert result_event is not None
                    return result_event

            result_event = portal.call(consume_completed)
            assert result_event.result.output == 'portal complete'

            started = threading.Event()
            handles: list[AgentRunEvents[str]] = []
            cancelled_agent = Agent(TestModel())

            @cancelled_agent.tool_plain
            async def slow_tool() -> str:
                started.set()
                await asyncio.Event().wait()
                return 'never reached'  # pragma: no cover

            async def consume_cancelled() -> None:
                async with cancelled_agent.run_stream_events('go') as events:
                    handles.append(events)
                    await _consume_events(events)

            future = portal.start_task_soon(consume_cancelled)
            assert started.wait(timeout=READINESS_WAIT_TIMEOUT)
            portal.call(handles[0].cancel)

            with pytest.raises(RunCancelled) as exc_info:
                future.result(timeout=READINESS_WAIT_TIMEOUT)
            assert exc_info.value.all_messages()

    await to_thread.run_sync(run_portal_cases)


async def test_nested_run_stream_events_binding_isolated_under_outer_cancellation():
    """Cancelling an outer handle remains external cancellation to a nested handle."""
    inner_started = asyncio.Event()
    inner_outcome: list[str] = []
    inner_agent = Agent(TestModel())
    outer_agent = Agent(TestModel())

    @inner_agent.tool_plain
    async def inner_slow_tool() -> str:
        inner_started.set()
        await asyncio.Event().wait()
        return 'never reached'  # pragma: no cover

    @outer_agent.tool_plain
    async def nested_run() -> str:
        try:
            async with inner_agent.run_stream_events('inner') as inner_events:
                await _consume_events(inner_events)
        except asyncio.CancelledError:
            inner_outcome.append('tool cancelled')
            raise
        except RunCancelled:  # pragma: no cover - the inner handle must not claim the outer cancel
            inner_outcome.append('inner run cancelled')
            raise
        # Never executed (the inner run blocks until cancelled), but coverage's exception-table
        # attribution counts this exit arc as covered, so it must not carry a `no cover` pragma.
        return 'never reached'

    async with outer_agent.run_stream_events('outer') as outer_events:
        consumer = asyncio.create_task(_consume_events(outer_events))
        await asyncio.wait_for(inner_started.wait(), timeout=READINESS_WAIT_TIMEOUT)
        outer_events.cancel()
        with pytest.raises(RunCancelled) as exc_info:
            await asyncio.wait_for(consumer, timeout=READINESS_WAIT_TIMEOUT)

    assert inner_outcome == ['tool cancelled']
    assert exc_info.value.all_messages()


async def test_nested_run_in_for_run_hook_does_not_steal_binding():
    """A nested agent run started from a capability `for_run()` hook must not consume the
    outer `AgentRunEvents` handle's binding: the binding is taken before any lifecycle hook runs."""
    inner_agent = Agent(TestModel(custom_output_text='inner done'))
    outer_started = asyncio.Event()
    inner_outputs: list[str] = []

    class RunsAgentInForRun(AbstractCapability):
        async def for_run(self, ctx: RunContext) -> RunsAgentInForRun:
            result = await inner_agent.run('inner')
            inner_outputs.append(result.output)
            return self

    outer_agent = Agent(TestModel(), capabilities=[RunsAgentInForRun()])

    @outer_agent.tool_plain
    async def outer_slow_tool() -> str:
        outer_started.set()
        await asyncio.Event().wait()
        return 'never reached'  # pragma: no cover

    async with outer_agent.run_stream_events('outer') as events:
        consumer = asyncio.create_task(_consume_events(events))
        await asyncio.wait_for(outer_started.wait(), timeout=READINESS_WAIT_TIMEOUT)
        events.cancel()
        with pytest.raises(RunCancelled) as exc_info:
            await asyncio.wait_for(consumer, timeout=READINESS_WAIT_TIMEOUT)

    assert inner_outputs == ['inner done']
    assert exc_info.value.all_messages()


async def test_cancel_during_blocked_before_run_is_delivered():
    """`events.cancel()` while a `before_run` hook is blocked must interrupt it promptly —
    the controller is bound before `wrap_run`/`before_run` start, and the translation funnel
    covers all of setup."""
    hook_blocked = asyncio.Event()

    class BlockingBeforeRun(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            hook_blocked.set()
            await asyncio.Event().wait()  # blocks until cancelled

    agent = Agent(TestModel(), capabilities=[BlockingBeforeRun()])

    async with agent.run_stream_events('go') as events:
        consumer = asyncio.create_task(_consume_events(events))
        await asyncio.wait_for(hook_blocked.wait(), timeout=READINESS_WAIT_TIMEOUT)
        events.cancel()
        with pytest.raises(RunCancelled) as exc_info:
            await asyncio.wait_for(consumer, timeout=READINESS_WAIT_TIMEOUT)

    assert exc_info.value.all_messages() == []  # cancelled before any model request


@requires_task_cancelling
async def test_first_party_cancel_swallowed_by_after_run_is_typed():
    """A first-party cancellation absorbed by `after_run` is typed at the outer funnel."""

    class CancelInAfterRun(AbstractCapability):
        async def after_run(self, ctx: RunContext, *, result: AgentRunResult) -> AgentRunResult:
            ctx.cancel_run()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass
            return result

    agent = Agent(TestModel(), capabilities=[CancelInAfterRun()])

    with pytest.raises(RunCancelled):
        await agent.run('go')


# Blocking *external*-cancel recovery relies on the backstop, which is a no-op on Python 3.10.
@pytest.mark.parametrize('first_party', [True, pytest.param(False, marks=requires_task_cancelling)])
async def test_run_capabilities_cannot_recover_cancellation(first_party: bool):
    """`wrap_run` and `on_run_error` may observe cancellation but cannot recover it."""
    started = asyncio.Event()
    observed: list[str] = []

    class RecoverCancellation(AbstractCapability):
        async def wrap_run(self, ctx: RunContext, *, handler: Any) -> AgentRunResult:
            try:
                return await handler()
            except asyncio.CancelledError:
                observed.append('wrap_run')
                return AgentRunResult(output='recovered')

        async def on_run_error(self, ctx: RunContext, *, error: BaseException) -> AgentRunResult:
            assert isinstance(error, asyncio.CancelledError)
            observed.append('on_run_error')
            return AgentRunResult(output='recovered')

    agent = Agent(TestModel(), capabilities=[RecoverCancellation()])

    @agent.tool_plain
    async def slow_tool() -> str:
        started.set()
        await asyncio.sleep(READINESS_WAIT_TIMEOUT)
        return 'slow'  # pragma: no cover

    runs: list[Any] = []

    async def drive() -> AgentRunResult:
        async with agent.iter('go') as agent_run:
            runs.append(agent_run)
            async for _node in agent_run:
                pass
        assert agent_run.result is not None  # pragma: no cover
        return agent_run.result  # pragma: no cover

    task = asyncio.create_task(drive())
    await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    if first_party:
        runs[0].cancel()
        expected_exception = RunCancelled
    else:
        task.cancel()
        expected_exception = asyncio.CancelledError

    with pytest.raises(expected_exception):
        await asyncio.wait_for(asyncio.shield(task), timeout=READINESS_WAIT_TIMEOUT)

    assert observed == ['wrap_run', 'on_run_error']


async def test_cancel_after_completion_is_a_noop():
    """`cancel()` after the run finished must never cancel unrelated work still running on the
    task that drove the run."""
    agent = Agent(TestModel())

    async with agent.iter('go') as agent_run:
        async for _node in agent_run:
            pass

    agent_run.cancel()
    agent_run.cancel()  # repeated calls are no-ops too
    await asyncio.sleep(0)  # a cancellation would be delivered here
    assert agent_run.result is not None
    assert agent_run.result.output == 'success (no tool calls)'


async def test_cancel_run_outside_a_run_raises_user_error():
    """A synthetic `RunContext` not backed by a running agent has no run to cancel."""
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    with pytest.raises(UserError, match='`cancel_run` is only available during an agent run'):
        ctx.cancel_run()


@pytest.mark.skipif(sys.version_info >= (3, 11), reason='pins the documented degraded behavior on Python 3.10')
async def test_absorbed_cancellation_completes_on_py310():  # pragma: lax no cover
    """On Python 3.10 there is no `Task.cancelling()`, so an absorbed external cancellation
    cannot be detected: the run completes normally. This pins the documented best-effort
    behavior; it flips to `CancelledError` on 3.11+."""
    in_flight = asyncio.Event()

    async def handler(ctx: RunContext, events: AsyncIterable[AgentStreamEvent]) -> None:
        try:
            async for _event in events:
                in_flight.set()
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass

    agent = Agent(TestModel())

    task = asyncio.create_task(agent.run('hello', event_stream_handler=handler))
    await asyncio.wait_for(in_flight.wait(), timeout=READINESS_WAIT_TIMEOUT)

    task.cancel()
    result = await asyncio.wait_for(asyncio.shield(task), timeout=READINESS_WAIT_TIMEOUT)
    assert result.output == 'success (no tool calls)'
