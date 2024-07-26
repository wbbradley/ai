from dataclasses import dataclass
from typing import IO, List, Optional, cast

from ai.config import Config
from ai.message import Message, Role


@dataclass
class EmbeddedDocument:
    messages: List[Message]


def parse_embedded_buffer(config: Config, file: IO) -> EmbeddedDocument:
    # Line input.
    line_iter = iter(file)

    # Parser state machine variables.
    line: Optional[str] = None
    messages: List[Message] = []
    role: Role = "user"
    content: str = ""

    while True:
        line = next(line_iter, None)
        if line is None:
            messages.append(Message(role=role, content=content.rstrip()))
            break
        if line.startswith(">>> "):
            messages.append(Message(role=role, content=content.rstrip()))
            role = cast(Role, line[4:].strip())
            content = ""
            continue
        content += line

    return EmbeddedDocument(messages=messages)
