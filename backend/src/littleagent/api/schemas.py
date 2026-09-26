from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field

# Schemas for the Little Agent API
class ChatRequest(BaseModel):
    message: str
    thread_id: str

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
