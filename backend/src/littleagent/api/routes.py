"""HTTP routes exposing the agent over server-sent events.

Framing: every SSE frame is "data: <json>\\n\\n". The trailing blank line is not
cosmetic - it is what marks a frame as complete, so clients rely on it.

Layer split, from the bottom up:
    core.agent            assembles the agent (the kernel, no HTTP knowledge)
    frames_from_chunks()  LangChain chunks -> SSE frames (no HTTP knowledge)
    translate_events()    one agent run -> SSE frames, including terminal framing
    chat_stream()         mounts the whole thing on HTTP

Keeping the translation free of HTTP concepts is what lets it be tested without
starting a server.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Mapping

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import ToolMessage

from littleagent.api.schemas import (
    AgentEvent,
    ChatRequest,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from littleagent.core import agent as core_agent

router = APIRouter()


def sse(event: AgentEvent) -> str:
    """Serialise one event as an SSE frame.

    Building the pydantic model before serialising means a mistyped field name
    fails loudly here, instead of producing a frame the client silently cannot
    parse. It also makes the server, not just the client, subject to the schema.
    """
    return f"data: {event.model_dump_json()}\n\n"


def normalise_tool_args(args: Any) -> str:
    """Render tool call arguments as a string, whatever shape they arrive in.

    Declared as str in the contract, but the value is not always one: streaming
    providers deliver argument text in fragments, while a completed message can
    carry a parsed dict. Handling both here keeps that detail out of the branch
    that decides what to emit.
    """
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False)


def events_from_message_chunk(token: Any, metadata: Mapping[str, Any]) -> list[AgentEvent]:
    """Events carried by one "messages" chunk, or none if it is not model output.

    Returning events rather than frames keeps this testable without string
    parsing, and keeps the wire format in one place.
    """
    # Load-bearing filter. With stream_mode="messages" the tools node also emits
    # a message carrying the tool's return value, so without this check the
    # result would be sent twice: once as a token rendered as if the model said
    # it, and once as a tool_result. Only model-node tokens are real model output.
    if metadata.get("langgraph_node") != "model":
        return []

    events: list[AgentEvent] = []
    for block in token.content_blocks:
        if block["type"] == "text" and block["text"]:
            # Empty text blocks do occur; skipping them avoids flooding the
            # client with frames it cannot render.
            events.append(TokenEvent(token=block["text"]))

        elif block["type"] in ("tool_call", "tool_call_chunk"):
            # Both spellings occur in practice. A completed message carries
            # "tool_call"; a real streaming provider such as DeepSeek carries
            # "tool_call_chunk", split across many blocks where only the first
            # has an id and a name. Handling just "tool_call" silently drops
            # every tool call on the streaming path.
            #
            # Emitting only when the name appears also means emitting exactly
            # once per call, at the earliest moment a UI can say "calling
            # get_weather...". The args are still arriving at that point and are
            # not valid JSON yet, so tool_args is usually empty. A client that
            # wants to render them live needs a separate delta event type;
            # putting half-parsed JSON in ToolCallEvent would misrepresent it.
            if not block.get("name"):
                continue
            events.append(
                ToolCallEvent(
                    tool_id=block.get("id") or "",
                    tool_name=block["name"],
                    tool_args=normalise_tool_args(block.get("args", "")),
                )
            )
    return events


def events_from_update_chunk(data: Mapping[str, Any]) -> list[AgentEvent]:
    """Events carried by one "updates" chunk.

    The chunk shape is {node_name: {"messages": [...]}}. The node name is not
    needed, so only the payloads are walked.
    """
    events: list[AgentEvent] = []
    for update in data.values():
        for message in update.get("messages", []):
            # isinstance rather than comparing type(...).__name__: a rename would
            # silently stop matching, subclasses would be missed, and mypy cannot
            # narrow through a string test.
            if not isinstance(message, ToolMessage):
                continue
            events.append(
                ToolResultEvent(
                    # tool_call_id pairs this result with the ToolCallEvent that
                    # requested it, which matters when one turn issues several
                    # tool calls.
                    tool_id=message.tool_call_id,
                    tool_name=message.name or "",
                    tool_result=str(message.content),
                )
            )
    return events


async def frames_from_chunks(
    chunks: AsyncIterator[Mapping[str, Any]],
) -> AsyncIterator[str]:
    """Turn a stream of LangChain chunks into SSE frames.

    Deliberately free of any agent: it takes an async iterator of already-shaped
    chunk dicts, so tests can feed hand-written chunks instead of standing up a
    model. That matters here because the shapes are subtle - see
    events_from_message_chunk - and a fake model that produces simpler shapes
    than the real one hides exactly the bugs this function exists to prevent.

    Each chunk is {"type": ..., "data": ...}, where data is a (token, metadata)
    tuple for "messages" and a {node: {"messages": [...]}} mapping for "updates".
    They come from agent.astream(..., stream_mode=["messages", "updates"],
    version="v2").
    """
    async for chunk in chunks:
        kind = chunk["type"]

        if kind == "messages":
            token, metadata = chunk["data"]
            events = events_from_message_chunk(token, metadata)
        elif kind == "updates":
            events = events_from_update_chunk(chunk["data"])
        else:
            # A mode this function does not know about. Ignoring it keeps a new
            # mode from breaking the stream.
            continue

        for event in events:
            yield sse(event)


async def translate_events(req: ChatRequest) -> AsyncIterator[str]:
    """Translate one agent run into SSE frames.

    This is an async generator: calling it performs no work at all until it is
    iterated, which is why a StreamingResponse can be handed to the transport
    layer before the agent has been touched.

    It owns the terminal framing - error and done - rather than
    frames_from_chunks, because both must be emitted even when starting the run
    itself failed.
    """
    try:
        # Reached through the module, not a bound import, so that a test
        # replacing core.agent.build_agent is actually observed.
        agent = core_agent.build_agent()
        # thread_id selects which conversation the checkpointer appends to.
        config = core_agent.conversation_config(req.thread_id)

        # Two stream modes are needed because they carry different things:
        # "messages" yields incremental tokens (the typing effect and the tool
        # call request), while "updates" yields completed messages, which is the
        # only place a ToolMessage exists in full.
        stream = agent.astream(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            stream_mode=["messages", "updates"],
            version="v2",
        )
        async for frame in frames_from_chunks(stream):
            yield frame

    except Exception as exc:  # noqa: BLE001
        # Errors must travel inside the stream. Once response headers are sent
        # the status code is fixed, so an exception escaping the generator would
        # leave the server able to do nothing but drop the connection, and the
        # client could not tell that apart from a network failure.
        yield sse(ErrorEvent(error_message=f"{type(exc).__name__}: {exc}"))
    finally:
        # Sent on every path, success or failure. Without it a client that saw an
        # error would keep waiting for a frame that never comes.
        yield sse(DoneEvent())


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream one turn as server-sent events.

    An ASGI async generator needs no thread or event-loop bridging, which is the
    concrete reason this project is built on FastAPI rather than Flask.
    """
    return StreamingResponse(
        translate_events(req),
        media_type="text/event-stream",
        headers={
            # Any buffering layer between here and the browser collapses
            # streaming into a single late burst.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tells nginx not to buffer
        },
    )
