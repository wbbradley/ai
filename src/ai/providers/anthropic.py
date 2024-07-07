import re
from typing import Generator, Iterator, List, cast, reveal_type

import anthropic
from anthropic.types import MessageParam

from ai.config import Config


def create_anthropic_chat_stream(config: Config) -> Generator[str, str, None]:
    assert config.anthropic
    model = config.anthropic.model
    client = anthropic.Anthropic(api_key=config.anthropic.api_key)

    messages: List[MessageParam] = []
    i = 0
    span_generator = None
    response = None
    while True:
        query = yield cast(str, span_generator)
        if not query:
            if i > 10:
                raise RuntimeError()
            i += 1
            continue

        if response:
            messages.append(MessageParam(role="assistant", content=response))
        messages.append(MessageParam(role="user", content=query))
        response = ""

        def response_span_generator() -> Iterator[str]:
            nonlocal response
            reveal_type(client.messages.stream)

            with client.messages.stream(
                model=model,
                max_tokens=1024,
                system=re.sub(
                    r"\s+",
                    " ",
                    """
    You respond directly and to the point. Never inform me that you are an AI, directly or
    indirectly.
                    """,
                ),
                messages=messages,
            ) as stream:
                for chunk in stream.text_stream:
                    yield chunk
                    response += chunk

        span_generator = response_span_generator()
