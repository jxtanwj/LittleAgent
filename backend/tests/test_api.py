"""Tests for the SSE endpoint and its chunk translation.

These run entirely offline: chunks are written out by hand in the shapes a real
provider sends, so no model, network, or API key is involved.

Why hand-written chunks rather than a fake model: a fake model delivers each
scripted response as one complete message, so a tool call always arrives in the
finished "tool_call" shape. Real streaming providers send "tool_call_chunk"
fragments instead. A dropped-tool-call bug lived in exactly that gap, invisible
because every agent test used the complete shape.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from littleagent.api import routes
from littleagent.api.schemas import ChatRequest
from littleagent.core import agent as core_agent
from tests.conftest import (
    FakeToolCallingAgent,
    message_chunk,
    update_chunk,
)


async def _aiter(chunks: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for chunk in chunks:
        yield chunk


def parse_frames(frames: list[str]) -> list[dict[str, Any]]:
    """Decode SSE frames into events, asserting the wire format as it goes."""
    events = []
    for frame in frames:
        assert frame.startswith("data: "), f"frame missing the data prefix: {frame!r}"
        assert frame.endswith("\n\n"), f"frame missing its terminating blank line: {frame!r}"
        events.append(json.loads(frame[len("data: ") :]))
    return events


async def collect(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect the frames a chunk stream produces.

    Only the chunk translation. Terminal framing - the error and done frames -
    belongs to translate_events and is covered separately, because it must be
    emitted even when starting the run fails.
    """
    frames = [frame async for frame in routes.frames_from_chunks(_aiter(chunks))]
    return parse_frames(frames)


async def collect_full(
    monkeypatch: pytest.MonkeyPatch, chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Collect frames from a complete run, including the terminal framing."""
    monkeypatch.setattr(core_agent, "build_agent", lambda: FakeToolCallingAgent(chunks))
    req = ChatRequest(message="hi", thread_id="t1")
    frames = [frame async for frame in routes.translate_events(req)]
    return parse_frames(frames)


# --------------------------------------------------------------------------
# The bug this file exists for
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_chunk_produces_a_tool_call_event() -> None:
    """A streamed tool call must reach the client.

    Real providers - DeepSeek among them - send "tool_call_chunk", not
    "tool_call". Filtering only for the latter drops every tool call on the
    streaming path while leaving the rest of the run looking healthy, which is
    how this went unnoticed: the client simply never learns a tool is running.
    """
    chunks = [
        message_chunk(tool_call_id="call_1", tool_name="get_weather", tool_args=""),
        message_chunk(tool_args='{"city"'),
        message_chunk(tool_args=': "SF"}'),
    ]

    events = await collect(chunks)

    assert [e["type"] for e in events] == ["tool_call"]
    assert events[0]["tool_id"] == "call_1"
    assert events[0]["tool_name"] == "get_weather"


@pytest.mark.asyncio
async def test_followup_fragments_do_not_repeat_the_tool_call() -> None:
    """Only the first fragment names the tool, so only one event may be emitted."""
    chunks = [
        message_chunk(tool_call_id="call_1", tool_name="get_weather", tool_args=""),
        message_chunk(tool_args='{"city"'),
        message_chunk(tool_args=': "SF"}'),
        message_chunk(tool_args="}"),
    ]

    events = await collect(chunks)

    assert len([e for e in events if e["type"] == "tool_call"]) == 1


@pytest.mark.asyncio
async def test_completed_tool_call_shape_still_works() -> None:
    """The non-streaming "tool_call" spelling must keep working."""
    token = type("T", (), {"content_blocks": [
        {"type": "tool_call", "id": "call_9", "name": "get_weather", "args": {"city": "SF"}}
    ]})()
    chunks = [{"type": "messages", "data": (token, {"langgraph_node": "model"})}]

    events = await collect(chunks)

    assert [e["type"] for e in events] == ["tool_call"]
    assert json.loads(events[0]["tool_args"]) == {"city": "SF"}


# --------------------------------------------------------------------------
# Duplication guards
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_node_text_is_not_emitted_as_a_token() -> None:
    """The tools node re-emits its result as a message; it must be dropped.

    stream_mode="messages" reports the tool's return value from the tools node as
    well. Without the node filter it arrives as a token, so the UI shows the raw
    tool output as if the model had typed it - and then shows it again as the
    tool result.
    """
    chunks = [
        message_chunk("checking"),
        message_chunk("SF is always sunny!", node="tools"),
    ]

    events = await collect(chunks)

    assert [e["token"] for e in events] == ["checking"]


@pytest.mark.asyncio
async def test_tool_result_is_emitted_once() -> None:
    """One tool execution means exactly one tool_result event."""
    tool_message = ToolMessage(content="SF is always sunny!", name="get_weather", tool_call_id="call_1")
    chunks = [
        message_chunk(tool_call_id="call_1", tool_name="get_weather", tool_args=""),
        message_chunk("SF is always sunny!", node="tools"),
        update_chunk("tools", [tool_message]),
    ]

    events = await collect(chunks)

    results = [e for e in events if e["type"] == "tool_result"]
    assert len(results) == 1
    assert results[0]["tool_result"] == "SF is always sunny!"


@pytest.mark.asyncio
async def test_ai_message_updates_are_not_emitted_as_tool_results() -> None:
    """Only ToolMessage becomes a tool_result; AIMessage updates are skipped."""
    chunks = [update_chunk("model", [AIMessage(content="hello")])]

    assert await collect(chunks) == []


@pytest.mark.asyncio
async def test_empty_text_blocks_are_skipped() -> None:
    """Providers emit empty text blocks; they must not become empty frames."""
    chunks = [message_chunk(""), message_chunk("hi")]

    events = await collect(chunks)

    assert [e["token"] for e in events] == ["hi"]


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_tool_calls_stay_correlated() -> None:
    """With several tools in flight, each result must match its own call id.

    Two calls returning identical text are indistinguishable without the id, so a
    client would render results against the wrong call.
    """
    chunks = [
        message_chunk(tool_call_id="c1", tool_name="get_weather", tool_args=""),
        message_chunk(tool_call_id="c2", tool_name="get_weather", tool_args=""),
        update_chunk("tools", [ToolMessage(content="sunny", name="get_weather", tool_call_id="c1")]),
        update_chunk("tools", [ToolMessage(content="rainy", name="get_weather", tool_call_id="c2")]),
    ]

    events = await collect(chunks)

    assert [e["tool_id"] for e in events if e["type"] == "tool_call"] == ["c1", "c2"]
    assert [e["tool_id"] for e in events if e["type"] == "tool_result"] == ["c1", "c2"]


# --------------------------------------------------------------------------
# Terminal framing: a client must never be left waiting
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_ends_with_done(monkeypatch: pytest.MonkeyPatch) -> None:
    events = await collect_full(monkeypatch, [message_chunk("hi")])

    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_ends_with_done_even_when_nothing_was_produced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty stream still has to be terminated explicitly."""
    events = await collect_full(monkeypatch, [])

    assert [e["type"] for e in events] == ["done"]


# --------------------------------------------------------------------------
# End to end through HTTP
# --------------------------------------------------------------------------


def _client(monkeypatch: pytest.MonkeyPatch, chunks: list[dict[str, Any]]) -> TestClient:
    """Wire the real router onto a throwaway app, with the agent replaced.

    The patch targets core.agent, not routes: the HTTP layer reaches the factory
    through the module so the replacement is observed. Patching routes.build_agent
    would look right and do nothing.
    """
    monkeypatch.setattr(core_agent, "build_agent", lambda: FakeToolCallingAgent(chunks))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_endpoint_streams_sse_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real HTTP request yields a well-formed event stream."""
    tool_message = ToolMessage(content="sunny", name="get_weather", tool_call_id="call_1")
    chunks = [
        message_chunk("let me check"),
        message_chunk(tool_call_id="call_1", tool_name="get_weather", tool_args=""),
        update_chunk("tools", [tool_message]),
        message_chunk("it is sunny"),
    ]

    with _client(monkeypatch, chunks) as client:
        response = client.post(
            "/chat/stream",
            json={"message": "weather?", "thread_id": "t1"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # Streaming collapses into a single late burst if a buffering layer is
    # allowed to accumulate the response.
    assert response.headers["x-accel-buffering"] == "no"

    events = parse_frames([f + "\n\n" for f in response.text.split("\n\n") if f.startswith("data: ")])
    assert [e["type"] for e in events] == ["token", "tool_call", "tool_result", "token", "done"]


def test_endpoint_rejects_a_malformed_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation is FastAPI's job, and a bad body must not reach the agent."""
    with _client(monkeypatch, []) as client:
        response = client.post("/chat/stream", json={"message": "hi"})  # thread_id missing

    assert response.status_code == 422


def test_endpoint_reports_agent_failures_inside_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing agent must produce error + done rather than a dropped connection.

    The status code is already 200 by the time the agent runs, so an escaping
    exception would leave the client unable to tell a server bug from a network
    failure - and waiting for a frame that never arrives.
    """

    class ExplodingAgent:
        async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(core_agent, "build_agent", lambda: ExplodingAgent())
    app = FastAPI()
    app.include_router(routes.router)

    with TestClient(app) as client:
        response = client.post("/chat/stream", json={"message": "hi", "thread_id": "t1"})

    events = parse_frames([f + "\n\n" for f in response.text.split("\n\n") if f.startswith("data: ")])
    assert [e["type"] for e in events] == ["error", "done"]
    assert "boom" in events[0]["error_message"]


def test_stream_frames_are_json_objects_of_the_declared_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every emitted frame must parse back as a declared AgentEvent."""
    from pydantic import TypeAdapter

    from littleagent.api.schemas import AgentEvent

    adapter = TypeAdapter(AgentEvent)
    chunks = [message_chunk("hi")]

    with _client(monkeypatch, chunks) as client:
        response = client.post("/chat/stream", json={"message": "hi", "thread_id": "t1"})

    for raw in [f for f in response.text.split("\n\n") if f.startswith("data: ")]:
        adapter.validate_python(json.loads(raw[len("data: ") :]))


def test_request_schema_requires_a_thread_id() -> None:
    """A defaulted thread_id would let two clients silently share one history."""
    with pytest.raises(Exception):
        ChatRequest(message="hi")
