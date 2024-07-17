from contextlib import contextmanager
from typing import ContextManager, Iterator, List, cast

from openai import OpenAI

from ai.chat_stream import ChatStream
from ai.config import Config
from ai.document import Message
from ai.providers import MissingProviderConfig


class OpenAIChatStream(ChatStream):
    client: OpenAI

    def __init__(self, config: Config) -> None:
        if not config.openai:
            raise MissingProviderConfig("openai configuration is missing")
        self.client = OpenAI(api_key=config.openai.api_key)

    def chat_stream(
        self,
        model: str,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> ContextManager[Iterator[str]]:
        messages = [cast(Message, {"role": "system", "content": system_prompt.strip()})] + messages

        @contextmanager
        def chat_stream_impl() -> Iterator[str]:
            completions = self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                stream=True,
            )
            for chunk in completions:
                delta = chunk.choices[0].delta.content  # type: ignore
                if delta is None:
                    break
                yield delta
                if chunk.choices[0].finish_reason == "stop":  # type: ignore
                    assert chunk.choices[0].delta.content is None  # type: ignore
                    break

        return chat_stream_impl()
