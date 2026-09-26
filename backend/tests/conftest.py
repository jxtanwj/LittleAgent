"""Shared pytest fixtures.

The goal is for agent tests to make no network requests at all, so they need no
API key, cost nothing, and produce repeatable results.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage


class FakeToolCallingModel(FakeMessagesListChatModel):
    """A scripted fake model that also supports tool binding.

    Needed because none of the fakes shipped with langchain_core implement
    bind_tools (FakeListChatModel, FakeMessagesListChatModel, GenericFakeChatModel
    and ParrotFakeChatModel all raise NotImplementedError), while create_agent
    calls model.bind_tools() whenever tools are supplied.

    Returning self means "binding tools leaves me unchanged", which is exactly the
    behaviour a test wants when no real model is involved.

    Known limitation: every entry in `responses` is delivered as one complete
    message, whereas a real provider streams many fragmentary chunks that are
    merged together. A tool call scripted here therefore always arrives in the
    finished "tool_call" shape, never as the "tool_call_chunk" fragments DeepSeek
    actually sends - which is precisely the gap that let a dropped-tool-call bug
    survive a green test run. The chunk-level contract is covered directly in
    test_api.py, where chunks are written out by hand.
    """

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingModel":
        return self


class FakeToolCallingAgent:
    """A stand-in for a compiled agent that replays chunks verbatim.

    Since the fake model cannot produce fragmentary chunks, tests that care about
    chunk shapes inject them here instead and still exercise the real transport
    layer end to end.
    """

    def __init__(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        self._chunks = list(chunks)

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Mapping[str, Any]]:
        for chunk in self._chunks:
            yield chunk


def message_chunk(
    text: str = "",
    *,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    tool_args: str | None = None,
    node: str = "model",
) -> dict[str, Any]:
    """Build one "messages" chunk as a real provider sends it.

    `tool_args` is deliberately a raw string: real providers deliver arguments as
    fragments across many chunks, not as a ready-made dict.
    """
    if tool_name is None:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    else:
        blocks = [
            {
                "type": "tool_call_chunk",
                "id": tool_call_id,
                "name": tool_name,
                "args": tool_args or "",
                "index": 0,
            }
        ]
    return {
        "type": "messages",
        "data": (SimpleNamespace(content_blocks=blocks), {"langgraph_node": node}),
    }


def update_chunk(node: str, messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """Build one "updates" chunk, whose shape is {node: {"messages": [...]}}."""
    return {"type": "updates", "data": {node: {"messages": list(messages)}}}


@pytest.fixture
def fake_model() -> Callable[..., FakeToolCallingModel]:
    """Factory for a fake model driven by a scripted list of responses.

    Pass strings for plain replies, or AIMessage objects when a test needs to
    control tool calls precisely.
    """

    def _make(responses: Sequence[str | BaseMessage]) -> FakeToolCallingModel:
        messages = [
            AIMessage(content=r) if isinstance(r, str) else r for r in responses
        ]
        return FakeToolCallingModel(responses=messages)

    return _make
