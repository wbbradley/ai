from typing import Literal, TypedDict


class Message(TypedDict):
    content: str
    role: Literal["user", "assistant"]
