import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterator, Optional, Union

from openai import OpenAI

from ai.config import Config


class StreamingError(Exception):
    """Streaming error."""


class UnknownProviderError(StreamingError):
    """Unknown Provider."""


def compressed_time() -> str:
    return (datetime.now().isoformat().replace(":", "").replace("-", "").replace("T", "").partition("."))[0]


def create_chat_stream(config: Config, provider: str) -> Generator:
    if provider == "openai":
        return create_openai_chat_stream(config)
    raise UnknownProviderError(provider)


def create_openai_chat_stream(config: Config) -> Generator:
    assert config.openai
    model = config.openai.model
    client = OpenAI(api_key=config.openai.api_key)
    messages = [
        {
            "role": "system",
            "content": (
                """
                    You respond directly and to the point. Never inform me that you are an AI.
                """.strip().replace("\n", " ")
            ),
        }
    ]
    i = 0
    span_generator = None
    response = None
    while True:
        query = yield span_generator
        if not query:
            if i > 10:
                raise RuntimeError()
            i += 1
            continue

        if response:
            messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": query})
        completions = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            stream=True,
        )

        response = ""

        def response_span_generator() -> Iterator[str]:
            nonlocal response
            for chunk in completions:
                delta = chunk.choices[0].delta.content  # type: ignore
                if delta is None:
                    break
                yield delta
                if chunk.choices[0].finish_reason == "stop":  # type: ignore
                    assert chunk.choices[0].delta.content is None  # type: ignore
                    break
                response += delta

        span_generator = response_span_generator()


def set_tty_color(r: int, g: int, b: int) -> None:
    print("\001\033[38;2;{};{};{}m\002".format(r, g, b))


def reset_tty() -> None:
    print("\001\033[0m\002")


def colorize(text: str, r: int, g: int, b: int) -> str:
    return f"\001\033[38;2;{r};{g};{b}m\002{text}\001\033[0m\002"


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
    while True:
        query = None
        try:
            query = input(f"{input_delim}> ")
            input_delim = "\n"
        except EOFError:
            break
        qa = {"query": query, "query_timestamp": time.time()}

        # Send messages to the coroutine
        full_reply = ""
        spans = coroutine.send(query)
        try:
            set_tty_color(r=200, g=150, b=100)
            for span in spans:
                full_reply += span
                print(span, end="")
        finally:
            reset_tty()
        qa["reply"] = full_reply
        qa["reply_timestamp"] = time.time()
        transcript.append(qa)

    report = {
        "status": "ok",
        "query": query,
        "interactive": True,
        "transcript": transcript,
        "timestamp": time.time(),
    }
    with open(report_filename, "w") as f:
        json.dump(report, f, indent=2)
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
        print(f"\rWrote transcript to '{colorize(str(transcript_filename), r=100, g=185, b=125)}'. Goodbye.")


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
