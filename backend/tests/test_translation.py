"""Tests for the chunk translation helpers.

These exercise the helpers directly, as typed objects, rather than going through
SSE frames and parsing the strings back. The refactor that introduced them was
worth doing mainly for this: the tool arguments conversion used to be an inline
conditional nested three levels deep with no test of one of its two branches.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from littleagent.api import routes
from littleagent.api.schemas import TokenEvent, ToolCallEvent, ToolResultEvent


def block(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(content_blocks=[fields])


# --------------------------------------------------------------------------
# normalise_tool_args
# --------------------------------------------------------------------------


def test_normalise_tool_args_passes_a_string_through() -> None:
    """Fragments arrive as text and must survive untouched."""
    raw = '{"city": "SF"'
    assert routes.normalise_tool_args(raw) == raw


def test_normalise_tool_args_serialises_a_dict() -> None:
    """A completed message can carry parsed arguments; the contract says str."""
    assert routes.normalise_tool_args({"city": "SF"}) == '{"city": "SF"}'


def test_normalise_tool_args_keeps_non_ascii_readable() -> None:
    """ensure_ascii would turn Chinese arguments into escape sequences."""
    result = routes.normalise_tool_args({"city": "旧金山"})
    assert "旧金山" in result
    assert "\\u" not in result


def test_normalise_tool_args_handles_an_empty_dict() -> None:
    assert routes.normalise_tool_args({}) == "{}"


# --------------------------------------------------------------------------
# events_from_message_chunk
# --------------------------------------------------------------------------


def test_tool_call_chunk_fragment_becomes_a_tool_call_event() -> None:
    """The streaming spelling, which the endpoint previously ignored entirely."""
    token = block(type="tool_call_chunk", id="call_1", name="get_weather", args="", index=0)

    events = routes.events_from_message_chunk(token, {"langgraph_node": "model"})

    assert len(events) == 1
    assert isinstance(events[0], ToolCallEvent)
    assert events[0].tool_id == "call_1"
    assert events[0].tool_name == "get_weather"
    assert events[0].tool_args == ""


def test_argument_only_fragment_emits_nothing() -> None:
    """Follow-up fragments carry neither id nor name, so they add no event."""
    token = block(type="tool_call_chunk", id=None, name=None, args='{"city"', index=0)

    assert routes.events_from_message_chunk(token, {"langgraph_node": "model"}) == []


def test_completed_tool_call_block_still_becomes_an_event() -> None:
    token = block(type="tool_call", id="call_9", name="get_weather", args={"city": "SF"})

    events = routes.events_from_message_chunk(token, {"langgraph_node": "model"})

    assert isinstance(events[0], ToolCallEvent)
    assert events[0].tool_args == '{"city": "SF"}'


def test_tools_node_output_is_ignored() -> None:
    """The tools node re-emits its result as a message; treating it as model
    output would show raw tool data as if the model had typed it."""
    token = block(type="text", text="SF is always sunny!")

    assert routes.events_from_message_chunk(token, {"langgraph_node": "tools"}) == []


def test_missing_node_metadata_is_ignored() -> None:
    """Absent metadata must not be read as model output."""
    token = block(type="text", text="hi")

    assert routes.events_from_message_chunk(token, {}) == []


def test_empty_text_block_emits_nothing() -> None:
    token = block(type="text", text="")

    assert routes.events_from_message_chunk(token, {"langgraph_node": "model"}) == []


def test_text_block_becomes_a_token_event() -> None:
    token = block(type="text", text="hello")

    events = routes.events_from_message_chunk(token, {"langgraph_node": "model"})

    assert isinstance(events[0], TokenEvent)
    assert events[0].token == "hello"


def test_several_tool_calls_in_one_chunk_each_get_an_event() -> None:
    """One chunk can request several tools at once, e.g. two cities at once."""
    token = SimpleNamespace(
        content_blocks=[
            {"type": "tool_call", "id": "c1", "name": "get_weather", "args": "{}"},
            {"type": "tool_call", "id": "c2", "name": "get_weather", "args": "{}"},
        ]
    )

    events = routes.events_from_message_chunk(token, {"langgraph_node": "model"})

    assert [e.tool_id for e in events] == ["c1", "c2"]


# --------------------------------------------------------------------------
# events_from_update_chunk
# --------------------------------------------------------------------------


def test_tool_message_becomes_a_tool_result_event() -> None:
    message = ToolMessage(content="sunny", name="get_weather", tool_call_id="call_1")

    events = routes.events_from_update_chunk({"tools": {"messages": [message]}})

    assert isinstance(events[0], ToolResultEvent)
    assert events[0].tool_id == "call_1"
    assert events[0].tool_name == "get_weather"
    assert events[0].tool_result == "sunny"


def test_ai_message_is_not_a_tool_result() -> None:
    """An AIMessage update carries content already sent as tokens."""
    events = routes.events_from_update_chunk({"model": {"messages": [AIMessage(content="hi")]}})

    assert events == []


def test_update_without_messages_is_ignored() -> None:
    assert routes.events_from_update_chunk({"model": {}}) == []


def test_tool_message_without_a_name_is_tolerated() -> None:
    """name is optional on ToolMessage but required by the contract."""
    message = ToolMessage(content="x", tool_call_id="call_1")

    events = routes.events_from_update_chunk({"tools": {"messages": [message]}})

    assert events[0].tool_name == ""


def test_non_string_tool_content_is_coerced() -> None:
    """Content can be a list of blocks; the contract declares a string."""
    message = ToolMessage(
        content=[{"type": "text", "text": "hi"}], name="t", tool_call_id="call_1"
    )

    events = routes.events_from_update_chunk({"tools": {"messages": [message]}})

    assert isinstance(events[0].tool_result, str)


# --------------------------------------------------------------------------
# frames_from_chunks: unsupported modes
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_stream_mode_is_ignored() -> None:
    """An unrecognised mode must not break the stream."""

    async def gen():
        yield {"type": "custom", "data": {"anything": 1}}

    assert [f async for f in routes.frames_from_chunks(gen())] == []
