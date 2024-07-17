import getpass
import json
import logging
import os
import socket
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, cast

from ai.colors import colored_output, colorize
from ai.config import Config
from ai.document import Document, SessionMetadata


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
    if chat_filename and Path(chat_filename).exists():
        document = cast(Document, json.load(open(chat_filename, "r")))
    else:
        document = Document(
            sessions=[
                SessionMetadata(timestamp=time.time(), user=getpass.getuser(), hostname=socket.gethostname())
            ],
            provider=provider,
            model=config.get_provider_model(provider),
            transcript=[],
        )
    chat_stream = chat_stream_factory(provider)
    coroutine = generate_document_coroutine(, document)
    model = document["model"]
    next(coroutine)
    transcript = []
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
        qa = {"query": query, "query_timestamp": time.time()}

        # Send messages to the coroutine
        full_reply = ""
        spans = coroutine.send(query)
        with colored_output(r=200, g=150, b=100):
            for span in spans:
                full_reply += span
                print(span, end="")
        qa["reply"] = full_reply
        qa["reply_timestamp"] = time.time()
        transcript.append(qa)

    document = {
        "sessions": [
            {
                "timestamp": time.time(),
                "user": getpass.getuser(),
                "hostname": socket.gethostname(),
            },
        ],
        "provider": config.provider,
        "model": config.get_provider_model(config.provider),
        "transcript": transcript,
    }
    report_filename_path = Path(config.report_dir) / report_filename
    with open(report_filename_path, "w") as f:
        json.dump(document, f, indent=2)
        print(f"\rWrote document to '{colorize(str(report_filename_path))}'.")
    if len(transcript) >= 1:
        if not transcript_filename:
            slug = "".join(
                coroutine.send(
                    """Please summarize our discussion by responding here with a "slug", for example
                    "topic-of-discussion". Keep your response limited to alphanumerics and dashes."""
                )
            )
            transcript_filename = Path(config.transcript_dir) / f"{slug}.md"
        with open(transcript_filename, "w") as f:
            delim = ""
            for qa in transcript:
                f.write(f"{delim}## user >>\n\n{qa['query']}\n\n")
                f.write(f"## {model} >>\n\n{qa['reply']}")
                delim = "\n\n"
            f.write("\n")
        print(f"\rWrote transcript to '{colorize(str(transcript_filename))}'. Goodbye.")


def stream_spans_from_one_query(config: Config, query: str, provider: str) -> Iterator[str]:
    coroutine = create_chat_stream(config, provider)
    next(coroutine)

    # Send messages to the coroutine
    spans = coroutine.send(query)
    for span in spans:
        yield span


def run_single_query_stream(
    config: Config, query: str, report_filename: str, transcript_filename: str, model: str
) -> None:
    try:
        content = ""
        for span in stream_spans_from_one_query(config, query, model):
            content += span
            print(span, end="")
        print()
    except BaseException as e:
        results = {
            "status": "error",
            "exception": str(e),
            "query": query,
            "content": content,
            "timestamp": time.time(),
        }
        with open(report_filename, "w") as f:
            json.dump(results, f, indent=2)
        logging.info({"message": "exiting with error", "report_filename": report_filename})
        raise
    results = {
        "status": "ok",
        "query": query,
        "content": content,
        "timestamp": time.time(),
    }
    with open(report_filename, "w") as f:
        json.dump(results, f, indent=2)
    if transcript_filename:
        with open(transcript_filename, "w") as f:
            f.write(content)
            f.write("\n")
