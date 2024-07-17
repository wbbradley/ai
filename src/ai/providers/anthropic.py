from typing import Iterator, List, cast

from anthropic import Anthropic
from anthropic.types import MessageParam

from ai.config import Config
from ai.document import Document, DocumentStream


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
    ) -> ContextManager[Iterator[str]]: ...
        with client.messages.stream(
            model=model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            system=config.system_prompt.strip(),
            messages=messages,
        ) as stream:
