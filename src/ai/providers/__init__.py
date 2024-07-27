from typing import Callable

from ai.chat_stream import ChatStream
from ai.config import Config


class ProviderError(Exception):
    """Provider error."""


class UnknownProviderError(ProviderError):
    """Unknown Provider."""


class MissingProviderConfig(ProviderError):
    """Provider is missing configuration details."""


def chat_stream_class_factory(provider: str, config: Config) -> Callable[[Config], ChatStream]:
    if provider == "openai":
        from ai.providers.openai import OpenAIChatStream

        return OpenAIChatStream
    if provider == "anthropic":
        from ai.providers.anthropic import AnthropicChatStream

        return AnthropicChatStream

    raise UnknownProviderError(provider)
