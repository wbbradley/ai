import logging
from typing import Iterator, List, Protocol, cast

from ai.config import Config
from ai.document import DocumentStream, Message


class ChatStream(Protocol):
    def __init__(self, config: Config) -> None: ...
    def chat_stream(
        self,
        model: str,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> Iterator[str]: ...

def generate_document_coroutine(
    chat_stream: ChatStream,
    model: str,
    system_prompt: str,
    messages: List[Message],
    max_tokens: int,
    temperature: float,
) -> DocumentStream:
    """Take the state of a document, and yield iterators that stream text chunks back to consumer.

    This coroutine accepts Messages.
    """
    i = 0
    span_generator = None
    response = ""
    system_prompt = system_prompt.strip()
    while True:
        user_message = yield cast(Iterator[str], span_generator)
        if not user_message:
            if i > 10:
                raise RuntimeError()
            i += 1
            continue

        if response:
            messages.append(Message(role="assistant", content=response))
        messages.append(user_message)
        response = ""

        def response_span_generator() -> Iterator[str]:
            nonlocal response
            logging.debug(
                dict(
                    status="sending streaming request",
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    messages=messages,
                )
            )

            for chunk in chat_stream.chat_stream(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                messages=messages,
            ):
                yield chunk
                response += chunk

        span_generator = response_span_generator()
