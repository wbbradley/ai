from typing import Iterator, List, cast

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
    ) -> Iterator[str]:
        messages = [cast(Message, {"role": "system", "content": system_prompt.strip()})] + messages

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


def complete_text(config: Config, client: OpenAI, prompt: str | List[Message]) -> str | None:
    system_prompt_message: Message = Message(role="system", content=config.system_prompt)  # type: ignore
    messages: List[Message] = (
        [
            system_prompt_message,
            {"role": "user", "content": prompt},
        ]
        if isinstance(prompt, str)
        else [system_prompt_message] + prompt
    )
    completion = client.chat.completions.create(
        model=config.get_provider_model(provider_override="openai"),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        messages=messages,  # type: ignore
    )
    return completion.choices[0].message.content
