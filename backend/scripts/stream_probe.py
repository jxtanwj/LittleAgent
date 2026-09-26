"""Stream a turn from a running server and show the timing of every event.

This is the debugging tool to reach for when the interface looks wrong. It talks
to the real endpoint over real HTTP, which is the only way to see two things the
test suite cannot show:

  * whether tokens actually arrive one at a time, or in one late burst because
    something in the middle is buffering the response
  * what the server emits for a real model, whose chunk shapes differ from any
    fake used in tests

Usage:
    python -m scripts.stream_probe
    python -m scripts.stream_probe --message "weather in Tokyo?" --thread-id t2

Assumes the server is already running:
    uvicorn littleagent.api.app:app --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import httpx

DEFAULT_URL = "http://127.0.0.1:8000/chat/stream"


async def stream(url: str, message: str, thread_id: str) -> int:
    """Print each event with its arrival time. Returns a process exit code."""
    payload = {"message": message, "thread_id": thread_id}
    started = time.perf_counter()
    first_event_ms: float | None = None
    first_token_ms: float | None = None
    counts: dict[str, int] = {}

    print(f"POST {url}")
    print(f"  message   : {message!r}")
    print(f"  thread_id : {thread_id!r}")
    print()

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as response:
                print(f"  status       : {response.status_code}")
                print(f"  content-type : {response.headers.get('content-type')}")
                print(f"  x-accel-buffering : {response.headers.get('x-accel-buffering')}")
                print()

                if response.status_code != 200:
                    print("  server refused the request:", await response.aread())
                    return 1

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    elapsed_ms = (time.perf_counter() - started) * 1000
                    first_event_ms = first_event_ms or elapsed_ms
                    event = json.loads(line[len("data: ") :])
                    kind = event.get("type", "?")
                    counts[kind] = counts.get(kind, 0) + 1

                    if kind == "token" and first_token_ms is None:
                        first_token_ms = elapsed_ms

                    print(f"  [{elapsed_ms:8.1f}ms] {kind:<11} {summarise(event)}")

    except httpx.ConnectError:
        print(f"  cannot reach {url}")
        print("  is the server running? uvicorn littleagent.api.app:app --port 8000")
        return 2

    total_ms = (time.perf_counter() - started) * 1000
    print()
    print(f"  events      : {counts}")
    print(f"  first event : {first_event_ms or -1:.0f}ms")
    print(f"  first token : {first_token_ms or -1:.0f}ms")
    print(f"  total       : {total_ms:.0f}ms")

    verdict(counts, first_token_ms, total_ms)
    return 0


def summarise(event: dict) -> str:
    """One readable line per event, whatever its type."""
    if event.get("type") == "token":
        return repr(event.get("token", ""))
    if event.get("type") == "tool_call":
        return f"{event.get('tool_name')} id={event.get('tool_id')} args={event.get('tool_args')!r}"
    if event.get("type") == "tool_result":
        return f"{event.get('tool_name')} -> {str(event.get('tool_result'))[:60]!r}"
    if event.get("type") == "error":
        return str(event.get("error_message"))[:100]
    return json.dumps(event, ensure_ascii=False)[:80]


def verdict(counts: dict[str, int], first_token_ms: float | None, total_ms: float) -> None:
    """Say plainly whether the run looks healthy."""
    print()
    tokens = counts.get("token", 0)
    if tokens == 0:
        print("  >>> no tokens at all - check the error event above")
    elif tokens == 1:
        print("  >>> only ONE token: streaming is not reaching this client.")
        print("      Either the provider is not streaming, or something between")
        print("      here and the server is buffering the response.")
    else:
        spread = total_ms - (first_token_ms or 0)
        print(f"  >>> {tokens} tokens over {spread:.0f}ms - streaming looks healthy")

    if counts.get("tool_call", 0) and not counts.get("tool_result", 0):
        print("  >>> a tool was called but never returned a result")
    if counts.get("done", 0) == 0:
        print("  >>> no 'done' frame: a client would hang waiting forever")


def main() -> int:
    # A model reply may contain any Unicode, emoji included. The Windows console
    # defaults to a legacy code page such as GBK, where printing one raises
    # UnicodeEncodeError and kills the probe halfway through a stream. Replacing
    # unencodable characters keeps the run going; the alternative is losing the
    # diagnostics precisely when something is already going wrong.
    #
    # The loop variable is not named `stream`, which would shadow the stream()
    # coroutine below and make it uncallable.
    for handle in (sys.stdout, sys.stderr):
        if hasattr(handle, "reconfigure"):
            handle.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="stream endpoint")
    parser.add_argument(
        "--message",
        default="What is the weather in San Francisco?",
        help="user message to send",
    )
    parser.add_argument("--thread-id", default="probe", help="conversation id")
    args = parser.parse_args()

    return asyncio.run(stream(args.url, args.message, args.thread_id))


if __name__ == "__main__":
    raise SystemExit(main())
