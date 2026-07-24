"""
Chainlit application entrypoint.

This module contains only Chainlit lifecycle handlers.

All business logic is delegated to:
    ui.handlers
    ui.streaming
"""

from __future__ import annotations

import logging

import chainlit as cl

from ui.handlers import handle_message
from ui.logger import configure_ui_logging


# ==========================================================
# Logging
# ==========================================================


configure_ui_logging()

logger = logging.getLogger(
    __name__
)


logger.info(
    "UI application module loaded."
)


# ==========================================================
# Chat lifecycle
# ==========================================================


@cl.on_chat_start
async def start() -> None:
    """
    Initialize new chat session.
    """

    logger.info(
        "New Chainlit session started."
    )


    await cl.Message(
        content=(
            """
Hello! 👋

I am an **agentic AI assistant** powered by Qwen.

Capabilities:

- multilingual conversations;
- retrieval augmented generation;
- web search with fresh information;
- semantic memory;
- tool-based reasoning;
- streamed responses.

Ask me anything.
"""
        )
    ).send()



# ==========================================================
# Message handling
# ==========================================================


@cl.on_message
async def main(
    message: cl.Message,
) -> None:
    """
    Handle incoming user message.

    All processing is delegated to the UI handler layer.
    """

    logger.info(
        "Processing user message."
    )


    await handle_message(
        message
    )
