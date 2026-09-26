"""Assembly of the agent: the model, the tools and the checkpointer.

This lives here rather than in the HTTP layer because more than one thing needs
it. It used to sit in api/routes.py, which meant the CLI duplicated the same
assembly and the API layer was the only place that knew how to build an agent.

Callers should reach this through the module rather than importing the function
directly (`from littleagent.core import agent` then `agent.build_agent()`), so
that a test replacing build_agent is actually seen. Binding the function at
import time freezes the reference and the patch silently does nothing.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from littleagent.core.tools import get_weather

# The assembled agent, built once on first use and then reused. Caching matters
# for correctness, not just speed: the checkpointer handed to create_agent is
# what holds conversation history, so building a fresh agent per request would
# discard it and every turn would start from scratch - a thread_id would point
# at memory that no longer exists.
#
# It is built lazily rather than at import time on purpose: constructing it at
# module level would make importing this module require working credentials,
# which would break tests and any tooling that merely imports it.
_agent = None

SYSTEM_PROMPT = "You are a helpful assistant."


def build_agent():
    """Return the shared agent, assembling it on first call.

    Tests replace this with a fake-model agent and call reset_agent() afterwards
    so the cached instance does not leak between tests.
    """
    global _agent
    if _agent is None:
        model = init_chat_model(
            "deepseek-v4-flash",
            # Thinking mode is off because the UI streams plain text tokens;
            # reasoning tokens would arrive as their own block type.
            extra_body={"thinking": {"type": "disabled"}},
        )
        _agent = create_agent(
            model=model,
            tools=[get_weather],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
    return _agent


def reset_agent() -> None:
    """Drop the cached agent, so the next call rebuilds it.

    For tests, and for picking up a configuration change. InMemorySaver keeps
    history in process memory, so dropping the agent also drops every
    conversation. A persistent saver would survive this.
    """
    global _agent
    _agent = None


def conversation_config(thread_id: str) -> dict:
    """The run config that tells the checkpointer which conversation this is.

    Required, not optional: the agent is built with a checkpointer, and
    LangGraph refuses to run without a thread_id in the config. Every caller
    needs this, so it lives here rather than being rebuilt by hand in each one.
    """
    return {"configurable": {"thread_id": thread_id}}
