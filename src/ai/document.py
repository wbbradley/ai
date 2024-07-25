from typing import Generator, Iterator, List, TypedDict

from ai.message import Message


class DocumentError(Exception):
    pass


class NoSystemPromptFound(DocumentError):
    pass


class SessionMetadata(TypedDict):
    timestamp: float
    user: str
    hostname: str


class DocumentMessage(Message):
    timestamp: float


class Document(TypedDict):
    sessions: List[SessionMetadata]
    system_prompt: str
    provider: str
    model: str
    messages: List[DocumentMessage]


DocumentStream = Generator[Iterator[str], Message, None]
