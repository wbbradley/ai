import re
from typing import Generator, Iterator, List, cast

import anthropic
from anthropic.types import Message, MessageParam

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
            messages.append(Message(role="assistant", content=response))
        messages.append(Message(role="user", content=query))
        response = ""

        def response_span_generator() -> Iterator[str]:
            nonlocal response
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
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta is None:
                        break
                    yield delta
                    if chunk.choices[0].finish_reason == "stop":
                        assert chunk.choices[0].delta.content is None
                        break
                    response += delta

        span_generator = response_span_generator()
