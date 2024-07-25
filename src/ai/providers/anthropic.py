from typing import Iterable, Iterator, List, cast

from anthropic import Anthropic
from anthropic.types import MessageParam

from ai.chat_stream import ChatStream
from ai.config import Config
from ai.document import Message


class AnthropicChatStream(ChatStream):
    client: Anthropic

    def __init__(self, config: Config) -> None:
        assert config.anthropic
        self.client = Anthropic(api_key=config.anthropic.api_key)

    def chat_stream(
        self,
        model: str,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> Iterator[str]:
        with self.client.messages.stream(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system_prompt.strip(),
            messages=cast(Iterable[MessageParam], messages),
        ) as stream:
            for chunk in stream.text_stream:
                yield chunk
