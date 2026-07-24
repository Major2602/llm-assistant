import logging

import chainlit as cl

from ui.handlers import (
    handle_message,
)


logger = logging.getLogger(__name__)


logger.info(
    "UI layer loaded."
)


@cl.on_chat_start
async def start() -> None:
    """
    Initialize chat session.
    """

    logger.info(
        "New chat session started."
    )


    await cl.Message(
        content="""
Hello!

I am an agentic AI assistant.

My Capabilities:

- RAG
- Web search
- Updated factual knowledge
- Streaming responses
"""
    ).send()



@cl.on_message
async def message_handler(
    message: cl.Message,
) -> None:

    await handle_message(
        message
    )
