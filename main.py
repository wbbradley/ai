import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Generator, Iterator, Optional

from openai import OpenAI

PROJECT_ID = os.environ["OPENAI_API_PROJECT_ID"]
ORG = os.environ["OPENAI_API_ORG"]


def compressed_time() -> str:
    return (datetime.now().isoformat().replace(":", "").replace("-", "").replace("T", "").partition("."))[0]


def create_chat_stream(model: str) -> Generator:
    client = OpenAI(
        organization=ORG,
        project=PROJECT_ID,
    )
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
    while True:
        query = yield span_generator
        if not query:
            if i > 10:
                raise RuntimeError()
            i += 1
            continue

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

        messages.append({"role": "assistant", "content": response})


def run_interactive_stream(report_filename: str, transcript_filename: Optional[str], model: str) -> None:
    coroutine = create_chat_stream(model)
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
            r = 200
            g = 150
            b = 100
            print(f"\001\033[38;2;{r};{g};{b}m\002")
            for span in spans:
                full_reply += span
                print(span, end="")
        finally:
            print("\001\033[0m\002")
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
            transcript_filename = f"{slug}.md"
        with open(transcript_filename, "w") as f:
            delim = ""
            for qa in transcript:
                f.write(f"{delim}## user >>\n\n{qa['query']}\n\n")
                f.write(f"## {model} >>\n\n{qa['reply']}")
                delim = "\n\n"
            f.write("\n")
        print(f"\rWrote transcript to '{transcript_filename}'. Goodbye.")


def stream_spans_from_one_query(query: str, model: str) -> Iterator[str]:
    coroutine = create_chat_stream(model)
    next(coroutine)

    # Send messages to the coroutine
    spans = coroutine.send(query)
    for span in spans:
        yield span


def run_single_query_stream(query: str, report_filename: str, transcript_filename: str, model: str) -> None:
    try:
        content = ""
        for span in stream_spans_from_one_query(query, model):
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


def main() -> None:
    parser = argparse.ArgumentParser("ai")
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument(
        "-m",
        "--model",
        required=False,
        default="gpt-4o",
        help="Which model to use",
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=False,
        default=None,
        help="Where to write the raw generated content",
    )
    parser.add_argument(
        "-r",
        "--report-filename",
        required=False,
        default=None,
        help="Where to write the report of the entire run.",
    )
    args = parser.parse_args()
    slug = str(compressed_time())
    report_filename = args.report_filename or f"ai-{slug}.json"

    if args.query is None:
        run_interactive_stream(report_filename, args.filename, args.model)
    else:
        run_single_query_stream(args.query, report_filename, args.filename, args.model)


if __name__ == "__main__":
    main()
