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


def complete_text(client: OpenAI, prompt: str) -> str | None:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Your users are children between the ages of five and six.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content
