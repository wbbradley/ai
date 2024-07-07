from typing import Generator, Iterator, cast

from openai import OpenAI

from ai.config import Config


def create_openai_chat_stream(config: Config) -> Generator[str, str, None]:
    assert config.openai
    model = config.openai.model
    client = OpenAI(api_key=config.openai.api_key)
    messages = [
        {
            "role": "system",
            "content": (
                """
                    You respond directly and to the point. Never inform me that you are an AI, directly or
                    indirectly.
                """.strip().replace("\n", " ")
            ),
        }
    ]
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
