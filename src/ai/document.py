from typing import Generator, Iterator, List, Literal, TypedDict


class SessionMetadata(TypedDict):
    timestamp: float
    user: str
    hostname: str


class Message(TypedDict):
    content: str
    role: Literal["user", "assistant"]


class Document(TypedDict):
    sessions: List[SessionMetadata]
    provider: str
    model: str
    messages: List[Message]


DocumentStream = Generator[Iterator[str], Message, None]
