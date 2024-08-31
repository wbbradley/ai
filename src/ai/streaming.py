import getpass
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TypedDict, cast

from ai.chat_stream import ChatStream, generate_document_coroutine
from ai.colors import color_code, colorize
from ai.config import Config
from ai.document import Document, DocumentMessage, DocumentStream, SessionMetadata
from ai.embedded import EmbeddedDocument
from ai.message import Message
from ai.output import note
from ai.providers import chat_stream_class_factory


class Color(TypedDict):
    r: int
    g: int
    b: int


USER_COLORS = Color(r=250, g=150, b=150)
ASSISTANT_COLORS = Color(r=200, g=150, b=100)


def get_color_scheme(role: str) -> Color:
    return {
        "user": USER_COLORS,
        "assistant": ASSISTANT_COLORS,
    }[role]


class StreamingError(Exception):
    """Streaming error."""


class UnknownProviderError(StreamingError):
    """Unknown Provider."""


def compressed_time() -> str:
    return (datetime.now().isoformat().replace(":", "").replace("-", "").replace("T", "").partition("."))[0]


def get_user_input_from_editor() -> Optional[str]:
    try:
        # Get the user's preferred editor
        editor = os.environ.get("EDITOR", "nano")  # Default to nano if $EDITOR is not set

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=True) as temp_file:
            temp_filename = temp_file.name

            # Open the editor for user input
            try:
                subprocess.run([editor, temp_filename], check=True)
            except subprocess.CalledProcessError:
                logging.error("Failed to open the editor.")
                return None
            except FileNotFoundError:
                logging.error(f"Editor '{editor}' not found.")
                return None

            # Read the contents of the temporary file
            try:
                with open(temp_filename, "r") as file:
                    content = file.read()
            except IOError as e:
                logging.error(f"Error reading the temporary file: {e}")
                return None

            return content.strip()

    except Exception:
        logging.exception("[get_user_input_from_editor] An unexpected error occurred")
        return None


def stream_document_response(config: Config, document: EmbeddedDocument, one_shot: bool) -> None:
    chat_stream_cls = chat_stream_class_factory(config.provider, config)
    chat_stream: ChatStream = chat_stream_cls(config)
    if not one_shot:
        print("\n> assistant\n")
    for chunk in chat_stream.chat_stream(
        model=config.get_provider_model(),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        system_prompt=config.system_prompt,
        messages=document.messages,
    ):
        print(chunk, end="")
    if one_shot:
        print()
    else:
        print("\n\n> user\n\n")


def markdown_to_ansi(
    text: str,
    r: int,
    g: int,
    b: int,
) -> str:
    output = color_code(r, g, b)
    state = "newline"
    star2_text = ""
    header_text = ""
    col = 0
    for ch in text:
        if state == "newline":
            if ch == "#":
                header_text += ch
                continue
            elif len(header_text) != 0:
                if ch == "\n":
                    output += color_code(255 - ((255 - int(r)) // 2), g, b, bold=True)
                    output += header_text
                    output += color_code(r, g, b, bold=False)
                    output += "\n"
                    header_text = ""
                else:
                    header_text += ch
                continue
            else:
                state = "normal"

        assert header_text == ""
        assert state != "newline"
        match state:
            case "normal":
                if ch == "*":
                    state = "1star"
                else:
                    output += ch
                    col += 1
            case "1star":
                if ch == "*":
                    state = "2star"
                    star2_text = ""
                else:
                    output += "*"
                    state = "normal"
            case "2star":
                if ch == "*":
                    state = "normal"
                    output += "**"
                else:
                    state = "eating-2star"
                    star2_text = ch
            case "eating-2star":
                if ch == "*":
                    state = "eating-2star-1"
                elif ch == "\n":
                    # Not in a bold run.
                    output += "**" + star2_text + ch
                    state = "normal"
                else:
                    star2_text += ch
            case "eating-2star-1":
                if ch == "*":
                    state = "normal"
                    output += color_code(r, g, b, bold=True)
                    output += star2_text
                    output += color_code(r, g, b, bold=False)
                elif ch == "\n":
                    output += "**" + star2_text + "*" + ch
                    state = "normal"
                else:
                    star2_text += ch
    match state:
        case "1star":
            output += "*"
        case "2star":
            output += "**"
        case "eating-2star":
            output += "**" + star2_text
        case "eating-2star-1":
            output += "**" + star2_text + "*"

    output += "\001\033[0m\002"
    return output


def bold_ansi(text: str) -> str:
    return re.sub(r"\*\*(.*?)\*\*", lambda match: f"\001\033[1m\002{match.group(1)}\001\033[0m\002", text)


def run_interactive_stream(
    config: Config,
    chat_filename: Optional[str],
    provider: str,
) -> None:
    this_session = SessionMetadata(
        timestamp=time.time(), user=getpass.getuser(), hostname=socket.gethostname()
    )

    if chat_filename and Path(chat_filename).exists():
        with open(chat_filename, "r") as f:
            document = cast(Document, json.load(f))
    else:
        document = Document(
            sessions=[],
            provider=provider,
            model=config.get_provider_model(provider),
            system_prompt=config.system_prompt,
            messages=[],
        )
    document["sessions"].append(this_session)
    chat_stream_cls = chat_stream_class_factory(document["provider"], config)
    chat_stream: ChatStream = chat_stream_cls(config)
    logging.debug(document)
    doc_stream = generate_document_coroutine(
        chat_stream,
        model=document["model"],
        system_prompt=document["system_prompt"],
        messages=[Message(role=x["role"], content=x["content"]) for x in document["messages"]],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    model = document["model"]
    next(doc_stream)
    new_messages: List[DocumentMessage] = []
    input_delim = ""
    note(f"[ai] you are chatting with {colorize(provider)}'s {colorize(model)} model.")
    while True:
        query = None
        try:
            query = input(f"{input_delim}> ")
            input_delim = "\n"
        except EOFError:
            print()
            break
        if query == "fc":
            query = get_user_input_from_editor()
            if not query:
                continue
            print(markdown_to_ansi(query, **get_color_scheme("user")))
        elif query == "":
            replay_delim = ""
            for message in document["messages"]:
                print(replay_delim, end="")
                print(markdown_to_ansi(message["content"], **get_color_scheme(message["role"])))
                replay_delim = "\n"
            input_delim = ""
            continue

        # Record our new user message.
        new_messages.append(DocumentMessage(timestamp=time.time(), role="user", content=query))

        # Send message to the doc_stream.
        full_reply = ""
        spans = doc_stream.send(Message(content=query, role="user"))
        full_reply = "".join(spans)
        print(markdown_to_ansi(full_reply, **get_color_scheme("assistant")))

        # Record our new reply message.
        new_messages.append(DocumentMessage(timestamp=time.time(), role="assistant", content=full_reply))

    new_document: Document = {
        "system_prompt": document["system_prompt"],
        "sessions": [
            {
                "timestamp": time.time(),
                "user": getpass.getuser(),
                "hostname": socket.gethostname(),
            },
        ],
        "provider": config.provider,
        "model": config.get_provider_model(config.provider),
        "messages": document["messages"] + new_messages,
    }
    if len(new_messages) >= 1:
        save_report(doc_stream, chat_filename, new_document, config)


def save_report(
    doc_stream: DocumentStream,
    chat_filename: Optional[str],
    document: Document,
    config: Config,
) -> None:
    if chat_filename is None:
        slug = "".join(
            doc_stream.send(
                Message(
                    role="user",
                    content=(
                        'Please summarize our discussion by responding here with a "slug", for example '
                        '"topic-of-discussion". Keep your response limited to alphanumerics and dashes.'
                    ),
                )
            )
        )
        chat_filename = str(Path(config.report_dir or ".") / f"{slug}.json")
    else:
        logging.info(f"Backing up {chat_filename} as {chat_filename}.bak...")
        shutil.copy(chat_filename, chat_filename + ".bak")

    assert chat_filename is not None
    with open(chat_filename, "w") as f:
        json.dump(document, f, indent=2)
        print(f"\rWrote document to '{colorize(str(chat_filename))}'.")
