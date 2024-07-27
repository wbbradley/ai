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

    import re

    pattern: str = r"^> (\w+)\s*(.*)$"
    while True:
        line = next(line_iter, None)
        if line is None:
            messages.append(Message(role=role, content=content.rstrip()))
            break
        match = re.search(pattern, line)
        if match and match[1] in ("user", "assistant"):
            if content:
                messages.append(Message(role=role, content=content.rstrip()))
            role = cast(Role, match[1])
            if match[2]:
                content = match[2].strip() + "\n"
            else:
                content = ""
            continue
        content += line

    return EmbeddedDocument(messages=messages)
