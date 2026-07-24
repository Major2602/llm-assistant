"""
Chainlit UI handlers.

This module renders typed UI events into Chainlit components.

The handler layer is intentionally unaware of LangChain internals.
It consumes only typed UI events produced by ui.streaming.
"""

from __future__ import annotations

import logging

import chainlit as cl

from ui.events import (
    DoneEvent,
    ErrorEvent,
    SourceEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from ui.sources import format_sources
from ui.streaming import stream_ui_events

logger = logging.getLogger(__name__)


# ==========================================================
# Chat handler
# ==========================================================


async def handle_message(
    message: cl.Message,
) -> None:
    """
    Handle one user message.

    This function owns the complete lifecycle of a Chainlit
    assistant message.
    """

    logger.info(
        "User message received."
    )

    reply = cl.Message(
        content=""
    )

    await reply.send()

    tool_steps: dict[str, cl.Step] = {}

    sources_markdown = ""

    streamed_tokens = 0

    try:

        async for event in stream_ui_events(
            message.content
        ):

            # -------------------------------------------------
            # LLM token
            # -------------------------------------------------

            if isinstance(event, TokenEvent):

                await reply.stream_token(
                    event.text
                )

                streamed_tokens += 1

                continue

            # -------------------------------------------------
            # Tool started
            # -------------------------------------------------

            if isinstance(event, ToolStartEvent):

                logger.info(
                    "Tool started: %s",
                    event.name,
                )

                step = cl.Step(
                    name=event.name,
                    type="tool",
                )

                step.input = "Running..."

                await step.send()

                tool_steps[event.name] = step

                continue

            # -------------------------------------------------
            # Tool finished
            # -------------------------------------------------

            if isinstance(event, ToolEndEvent):

                logger.info(
                    "Tool finished: %s",
                    event.name,
                )

                step = tool_steps.get(
                    event.name
                )

                if step is not None:

                    step.output = "Completed"

                    await step.update()

                continue

            # -------------------------------------------------
            # Sources
            # -------------------------------------------------

            if isinstance(event, SourceEvent):

                logger.info(
                    "Received %d sources.",
                    len(event.sources),
                )

                sources_markdown = format_sources(
                    event.sources
                )

                continue

            # -------------------------------------------------
            # Error
            # -------------------------------------------------

            if isinstance(event, ErrorEvent):

                logger.error(
                    "Streaming error: %s",
                    event.message,
                )

                reply.content = (
                    "⚠️ I couldn't complete your request.\n\n"
                    "Please try again."
                )

                await reply.update()

                return

            # -------------------------------------------------
            # Completed
            # -------------------------------------------------

            if isinstance(event, DoneEvent):

                logger.info(
                    "Generation completed."
                )

                break

        # -------------------------------------------------
        # Append citations
        # -------------------------------------------------

        if sources_markdown:

            reply.content += sources_markdown

        await reply.update()

        logger.info(
            "Assistant response completed. "
            "Tokens=%d",
            streamed_tokens,
        )

    except Exception:

        logger.exception(
            "Message handler failed."
        )

        reply.content = (
            "⚠️ Unexpected error."
        )

        await reply.update()
