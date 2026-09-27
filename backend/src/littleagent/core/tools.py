"""Tools the agent can call.

A plain function is enough: create_agent reads the name, the docstring and the
type annotations to build the schema it shows the model, so the docstring is part
of the tool's contract with the model rather than a comment for humans.

This is the natural place for a plugin system to hook in later - a registry that
tool modules register with - but there is no registry yet because one plugin
would not justify the indirection.
"""

from __future__ import annotations

from littleagent.core import memory


def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"{city} is always sunny!"


def search_memory(query: str) -> str:
    """Search long-term memories related to the user's current message."""
    found = memory.get_relevant_memories(query)
    if not found:
        return "No relevant memories found."
    return "\n".join(f"- {item.get('content', '')}" for item in found)


def save_memory(content: str, memory_type: str = "user", importance: float = 0.5) -> str:
    """Save a new memory to long-term storage."""
    # 这几个字段名是和检索层的约定：content 是搜索匹配的对象，
    # importance 参与排序加权。用模块方式访问 memory，是因为它的
    # save_memory 会和这个同名工具撞上。
    memory.save_memory(
        {
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
        }
    )
    return "Memory saved successfully."
