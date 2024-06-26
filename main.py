import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Iterator

from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

PROJECT_ID = os.environ["OPENAI_API_PROJECT_ID"]
ORG = os.environ["OPENAI_API_ORG"]


def compressed_time() -> str:
    return (
        datetime.now()
        .isoformat()
        .replace(":", "")
        .replace("-", "")
        .replace("T", "")
        .partition(".")
    )[0]


def stream(query: str) -> Iterator[ChatCompletionChunk]:
    client = OpenAI(
        organization=ORG,
        project=PROJECT_ID,
    )
    stream = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": query}],
        stream=True,
    )
    yield from stream


def main():
    parser = argparse.ArgumentParser("ai")
    parser.add_argument("query", nargs="?", default=None)
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
    if args.query is None:
        query = input("> ")
    else:
        query = args.query

    report_filename = args.report_filename or f"ai-{compressed_time()}.json"
    try:
        content = ""
        for chunk in stream(query):
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                content += chunk_content
                print(chunk_content, end="")
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
        logging.info(
            {"message": "exiting with error", "report_filename": report_filename}
        )
        raise
    results = {
        "status": "ok",
        "query": query,
        "content": content,
        "timestamp": time.time(),
    }
    with open(report_filename, "w") as f:
        json.dump(results, f, indent=2)
    if args.filename:
        with open(args.filename, "w") as f:
            f.write(content)
            f.write("\n")


if __name__ == "__main__":
    main()
