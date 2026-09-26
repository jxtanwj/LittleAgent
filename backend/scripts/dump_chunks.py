"""Dump the raw stream chunks of a real agent run, block by block.

This exists because a fake model lies about shapes. Every agent test here uses a
fake that delivers each scripted response as one complete message, while a real
provider streams fragments. A bug that dropped every tool call survived a green
test run for exactly that reason: the fake emitted "tool_call", the real provider
emits "tool_call_chunk", and the code only handled the former.

Run this whenever the stream is being changed, and compare what it prints against
what the tests assume.

Usage:
    python -m scripts.dump_chunks
    python -m scripts.dump_chunks --message "weather in Tokyo?" --max-blocks 40

Needs DEEPSEEK_API_KEY in the environment. Makes one real API call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from littleagent.core.agent import build_agent


def describe(block: dict[str, Any]) -> str:
    """Render one content block compactly, dropping noise keys."""
    payload = {k: v for k, v in block.items() if k not in ("type", "index")}
    return json.dumps(payload, ensure_ascii=False)


async def dump(message: str, thread_id: str, max_blocks: int) -> int:
    agent = build_agent()
    config = {"configurable": {"thread_id": thread_id}}

    block_types: dict[str, int] = {}
    printed = 0

    print(f"message   : {message!r}")
    print(f"thread_id : {thread_id!r}")
    print()

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        kind = chunk["type"]

        if kind == "messages":
            token, metadata = chunk["data"]
            node = metadata.get("langgraph_node")
            for block in token.content_blocks:
                block_type = block.get("type", "?")
                block_types[block_type] = block_types.get(block_type, 0) + 1
                if printed < max_blocks:
                    printed += 1
                    print(f"  messages node={node!r:<8} {block_type:<18} {describe(block)}")
                elif printed == max_blocks:
                    printed += 1
                    print("  ... (further blocks suppressed, use --max-blocks)")

        elif kind == "updates":
            for node, update in chunk["data"].items():
                for msg in update.get("messages", []):
                    block_types[f"update:{type(msg).__name__}"] = (
                        block_types.get(f"update:{type(msg).__name__}", 0) + 1
                    )
                    if printed < max_blocks:
                        printed += 1
                        print(
                            f"  updates  node={node!r:<8} {type(msg).__name__:<18} "
                            f"name={getattr(msg, 'name', None)!r} "
                            f"tool_call_id={getattr(msg, 'tool_call_id', None)!r}"
                        )

    print()
    print("  block / message counts:")
    for name, count in sorted(block_types.items()):
        print(f"    {name:<28} {count}")

    print()
    if "tool_call_chunk" in block_types:
        print("  >>> this provider streams tool calls as 'tool_call_chunk' fragments")
    if "tool_call" in block_types:
        print("  >>> this provider (or a fake model) sends finished 'tool_call' blocks")
    if "update:ToolMessage" in block_types:
        print("  >>> ToolMessage arrives via 'updates'; that is the only complete form")
    return 0


def main() -> int:
    # Model output can contain any Unicode. The Windows console defaults to a
    # legacy code page, where one unencodable character would abort the dump
    # partway through the run.
    for handle in (sys.stdout, sys.stderr):
        if hasattr(handle, "reconfigure"):
            handle.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--message",
        default="What is the weather in San Francisco?",
        help="prompt that should make the agent call a tool",
    )
    parser.add_argument("--thread-id", default="chunkdump", help="conversation id")
    parser.add_argument("--max-blocks", type=int, default=25, help="print at most this many")
    args = parser.parse_args()

    return asyncio.run(dump(args.message, args.thread_id, args.max_blocks))


if __name__ == "__main__":
    raise SystemExit(main())
