from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from littleagent.core.memory import DEFAULT_SCOPE

# Schemas for the Little Agent API
class ChatRequest(BaseModel):
    message: str
    thread_id: str
    # 记忆的作用域：不同 user_id 的记忆互相不可见，同一个人跨会话仍然共享。
    # 默认全局，老客户端不传这个字段时行为完全和以前一样。
    # 字符集限制是卫生问题不是安全问题：这个东西会进日志和文件字段，
    # 不该带上空格和换行。
    user_id: str = Field(
        default=DEFAULT_SCOPE,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.@-]+$",
    )

# Schemas for the events emitted by the agent during processing
class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    token: str

class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_id: str
    tool_name: str
    tool_args: str = ""

class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_id: str
    tool_name: str
    tool_result: str

class DoneEvent(BaseModel):
    type: Literal["done"] = "done"

class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error_message: str

# Annotated union of all possible agent events with a discriminator
AgentEvent = Annotated[
    Union[TokenEvent, ToolCallEvent, ToolResultEvent, DoneEvent, ErrorEvent],
    Field(discriminator="type"),
]
