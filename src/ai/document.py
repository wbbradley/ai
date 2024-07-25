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
    provider: str
    model: str
    messages: List[DocumentMessage]


def get_document_system_prompt(document: Document) -> str:
    for message in document["messages"]:
        if message["role"] == "system":
            return message["content"]
    raise NoSystemPromptFound()


DocumentStream = Generator[Iterator[str], Message, None]
