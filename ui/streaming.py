"""
LangChain → UI event adapter.

This module is the only UI component aware of LangChain event
structures. It converts LangChain streaming events into strongly
typed UI events consumed by the Chainlit handlers.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessageChunk, ToolMessage

from agents.web_search_agent import get_agent
from ui.events import (
    DoneEvent,
    ErrorEvent,
    SourceEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from web_search.models import Source

logger = logging.getLogger(__name__)


# ==========================================================
# Helpers
# ==========================================================


def _extract_chunk_text(
    chunk: Any,
) -> str:
    """
    Extract streamed text from AIMessageChunk.

    Supports current LangChain streaming format.
    """

    if isinstance(chunk, AIMessageChunk):

        content = chunk.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):

            parts: list[str] = []

            for item in content:

                if isinstance(item, str):
                    parts.append(item)

                elif isinstance(item, dict):
                    text = item.get("text")

                    if text:
                        parts.append(text)

            return "".join(parts)

    return ""


def _tool_name(
    event: dict[str, Any],
) -> str:
    """
    Return tool name from LangChain event.
    """

    return (
        event.get("name")
        or event.get("metadata", {}).get("tool_name")
        or "tool"
    )


def _extract_sources(
    output: Any,
) -> list[Source]:
    """
    Extract Source objects from ToolMessage artifact.

    Expected artifact format:

        {
            "sources": list[Source]
        }
    """

    if not isinstance(output, ToolMessage):
        return []

    artifact = getattr(
        output,
        "artifact",
        None,
    )

    if not isinstance(artifact, dict):
        return []

    sources = artifact.get("sources")

    if not isinstance(sources, list):
        return []

    result: list[Source] = []

    for source in sources:

        if isinstance(source, Source):
            result.append(source)

    return result


# ==========================================================
# Streaming
# ==========================================================


async def stream_ui_events(
    text: str,
) -> AsyncIterator[
    TokenEvent
    | ToolStartEvent
    | ToolEndEvent
    | SourceEvent
    | ErrorEvent
    | DoneEvent
]:
    """
    Convert LangChain streaming events
    into typed UI events.
    """

    logger.info(
        "Starting UI event stream."
    )

    agent = get_agent()

    try:

        async for event in agent.astream_events(

            {
                "messages": [
                    {
                        "role": "user",
                        "content": text,
                    }
                ]
            },

            version="v2",

        ):

            event_name = event.get("event")

            logger.debug(
                "LangChain event: %s",
                event_name,
            )

            # -------------------------------------------------
            # Token streaming
            # -------------------------------------------------

            if event_name == "on_chat_model_stream":

                chunk = (
                    event
                    .get("data", {})
                    .get("chunk")
                )

                token = _extract_chunk_text(
                    chunk
                )

                if token:

                    yield TokenEvent(
                        text=token,
                    )

                continue

            # -------------------------------------------------
            # Tool started
            # -------------------------------------------------

            if event_name == "on_tool_start":

                yield ToolStartEvent(
                    name=_tool_name(event),
                )

                continue

            # -------------------------------------------------
            # Tool finished
            # -------------------------------------------------

            if event_name == "on_tool_end":

                tool = _tool_name(event)

                yield ToolEndEvent(
                    name=tool,
                )

                output = (
                    event
                    .get("data", {})
                    .get("output")
                )

                sources = _extract_sources(
                    output
                )

                if sources:

                    logger.info(
                        "Received %d sources.",
                        len(sources),
                    )

                    yield SourceEvent(
                        sources=sources,
                    )

                continue

        logger.info(
            "UI event stream completed."
        )

        yield DoneEvent()

    except Exception as exc:

        logger.exception(
            "UI event stream failed."
        )

        yield ErrorEvent(
            message=str(exc),
        )
