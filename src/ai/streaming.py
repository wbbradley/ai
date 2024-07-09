import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterator, Optional, Union

from ai.colors import colored_output, colorize
from ai.config import Config
from ai.providers.anthropic import create_anthropic_chat_stream
from ai.providers.openai import create_openai_chat_stream


class StreamingError(Exception):
    """Streaming error."""


class UnknownProviderError(StreamingError):
    """Unknown Provider."""


def compressed_time() -> str:
    return (datetime.now().isoformat().replace(":", "").replace("-", "").replace("T", "").partition("."))[0]


def create_chat_stream(config: Config, provider: str) -> Generator[Iterator[str], str, None]:
    if provider == "openai":
        return create_openai_chat_stream(config)
    if provider == "anthropic":
        return create_anthropic_chat_stream(config)
    raise UnknownProviderError(provider)


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
    report_filename: str,
    transcript_filename: Optional[Union[Path, str]],
    provider: str,
) -> None:
    coroutine = create_chat_stream(config, provider)
    model = config.get_provider_model(provider)
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

    report = {
        "status": "ok",
        "system_prompt": config.system_prompt,
        "provider": config.provider,
        "model": config.get_provider_model(config.provider),
        "query": query,
        "interactive": True,
        "transcript": transcript,
        "timestamp": time.time(),
    }
    report_filename_path = Path(config.report_dir) / report_filename
    with open(report_filename_path, "w") as f:
        json.dump(report, f, indent=2)
        print(f"\rWrote report to '{colorize(str(report_filename_path))}'. Goodbye.")
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
