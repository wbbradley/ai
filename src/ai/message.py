from typing import Literal, TypedDict

Role = Literal["user", "assistant"]


class Message(TypedDict):
    content: str
    role: Role
