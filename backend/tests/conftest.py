"""Shared pytest fixtures.

The goal is for agent tests to make no network requests at all, so they need no
API key, cost nothing, and produce repeatable results.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

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
    """

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingModel":
        return self


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
