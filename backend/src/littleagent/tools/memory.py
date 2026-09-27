from langchain.tools import tool

from ..memory.retriever import get_relevant_memories
from ..memory.storage import save_memory as _save_memory


@tool
def search_memory(query: str) -> str:
    """Search long-term memories related to the user's current message."""
    memories = get_relevant_memories(query)
    if not memories:
        return "No relevant memories found."
    return "\n".join(f"- {memory.get('content', '')}" for memory in memories)


@tool
def save_memory(
    content: str,
    memory_type: str = "user",
    importance: float = 0.5,
) -> str:
    """Save a new memory to long-term storage."""
    # storage.save_memory 收的是一个 dict，字段名必须和 retriever 读的一致。
    _save_memory(
        {
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
        }
    )
    return "Memory saved successfully."
