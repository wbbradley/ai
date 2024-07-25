import getpass
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, cast

from ai.chat_stream import ChatStream, generate_document_coroutine
from ai.colors import colored_output, colorize
from ai.config import Config
from ai.document import Document, DocumentMessage, DocumentStream, SessionMetadata
from ai.message import Message
from ai.providers import chat_stream_class_factory


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
                print("Error: Failed to open the editor.")
                return None
            except FileNotFoundError:
                print(f"Error: Editor '{editor}' not found.")
                return None

            # Read the contents of the temporary file
            try:
                with open(temp_filename, "r") as file:
                    content = file.read()
            except IOError as e:
                print(f"Error reading the temporary file: {e}")
                return None

            return content.strip()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def run_interactive_stream(
    config: Config,
    chat_filename: Optional[str],
    provider: str,
) -> None:
    this_session = SessionMetadata(
        timestamp=time.time(), user=getpass.getuser(), hostname=socket.gethostname()
    )
    if chat_filename and Path(chat_filename).exists():
        document = cast(Document, json.load(open(chat_filename, "r")))
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
    print(f"[ai] you are chatting with {colorize(provider)}'s {colorize(model)} model.")
    while True:
        query = None
        try:
            query = input(f"{input_delim}> ")
            input_delim = "\n"
        except EOFError:
            break
        if query == "fc":
            query = get_user_input_from_editor()
            if not query:
                continue
            print(query)

        # Record our new user message.
        new_messages.append(DocumentMessage(timestamp=time.time(), role="user", content=query))

        # Send message to the doc_stream.
        full_reply = ""
        spans = doc_stream.send(Message(content=query, role="user"))
        with colored_output(r=200, g=150, b=100):
            for span in spans:
                full_reply += span
                print(span, end="")

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
        # with open(transcript_filename, "w") as f:
        #     delim = ""
        #     for qa in transcript:
        #         f.write(f"{delim}## user >>\n\n{qa['query']}\n\n")
        #         f.write(f"## {model} >>\n\n{qa['reply']}")
        #         delim = "\n\n"
        #     f.write("\n")
        # print(f"\rWrote transcript to '{colorize(str(transcript_filename))}'. Goodbye.")


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
        chat_filename = str(Path(config.report_dir or ".") / f"{slug}.md")
    else:
        print(f"Backing up {chat_filename} as {chat_filename}.bak...")
        shutil.copy(chat_filename, chat_filename + ".bak")

    assert chat_filename is not None
    with open(chat_filename, "w") as f:
        json.dump(document, f, indent=2)
        print(f"\rWrote document to '{colorize(str(chat_filename))}'.")
