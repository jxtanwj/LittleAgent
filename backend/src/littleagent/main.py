"""Command-line entry point.

Thin on purpose: the agent itself is assembled by littleagent.core.agent, so the
CLI, the HTTP layer and anything else share one definition of what the agent is.
This module used to hold its own copy of the model and tool wiring, which meant a
change to either had to be made in two places or they silently diverged.
"""

from __future__ import annotations

from littleagent.core import agent as core_agent

PROMPT = "What is the weather in San Francisco?"
THREAD_ID = "cli"


def main() -> None:
    # The config is mandatory here: the agent carries a checkpointer, and
    # LangGraph refuses to run without a thread_id saying which conversation
    # this is.
    result = core_agent.build_agent().invoke(
        {"messages": [{"role": "user", "content": PROMPT}]},
        config=core_agent.conversation_config(THREAD_ID),
    )

    for message in result["messages"]:
        print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()
