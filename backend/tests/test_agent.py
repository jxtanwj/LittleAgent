"""Agent behaviour tests.

Every test here uses a fake model by default, so nothing touches the network and
no DEEPSEEK_API_KEY is required. The one test that hits the real API is marked
with @pytest.mark.live and excluded by default (see pyproject.toml); run it
explicitly with `pytest -m live`.
"""

from __future__ import annotations

from typing import Callable

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from littleagent.core.tools import get_weather
from littleagent.main import main

CITY = "San Francisco"


def test_agent_replies_without_calling_tools(
    fake_model: Callable[..., object],
) -> None:
    """When the model answers with plain text, that text is returned as-is."""
    agent = create_agent(model=fake_model(["It is sunny today."]), tools=[get_weather])

    result = agent.invoke({"messages": [HumanMessage(content="How is the weather?")]})

    assert result["messages"][-1].content == "It is sunny today."


def test_agent_loop_executes_tool_and_uses_result(
    fake_model: Callable[..., object],
) -> None:
    """Full agent loop: model requests a tool, the tool runs, model answers from it.

    This is the test that proves the tool is actually wired into the agent, rather
    than merely importable on its own.
    """
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "get_weather", "args": {"city": CITY}, "id": "call_1"}],
    )
    agent = create_agent(
        model=fake_model([tool_call, "It is sunny in San Francisco."]),
        tools=[get_weather],
    )

    result = agent.invoke({"messages": [HumanMessage(content="How is the weather?")]})

    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    assert len(tool_messages) == 1, "the tool should run exactly once"
    assert CITY in str(tool_messages[0].content), "the tool should receive the args the model sent"
    assert messages[-1].content == "It is sunny in San Francisco."


def test_agent_executes_tool_exactly_once(
    fake_model: Callable[..., object],
) -> None:
    """Guard against duplicate tool execution, which would duplicate side effects or billing."""
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "get_weather", "args": {"city": "Beijing"}, "id": "call_1"}],
    )
    agent = create_agent(
        model=fake_model([tool_call, "Beijing looks fine."]),
        tools=[get_weather],
    )

    result = agent.invoke({"messages": [HumanMessage(content="Weather in Beijing?")]})

    assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 1


@pytest.mark.live
def test_main_runs_against_real_api(capsys: pytest.CaptureFixture[str]) -> None:
    """Call the real DeepSeek API, which spends quota. Skipped unless run explicitly.

    Requires DEEPSEEK_API_KEY to be present in the environment.
    """
    main()

    assert capsys.readouterr().out.strip(), "main() should print the conversation"
