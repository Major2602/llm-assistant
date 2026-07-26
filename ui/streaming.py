"""
LangChain → UI streaming adapter.

Responsibilities:
- convert LangChain streaming events into UI events;
- extract streamed LLM tokens;
- propagate tool lifecycle events;
- expose web search sources to the UI.

This module is the only layer aware of LangChain event formats.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import (
    AIMessageChunk,
    ToolMessage,
)

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


def _extract_text(chunk: Any) -> str:
    """
    Extract streamed token text from AIMessageChunk.
    """

    if not isinstance(chunk, AIMessageChunk):
        return ""

    content = chunk.content

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    parts: list[str] = []

    for item in content:

        if isinstance(item, str):
            parts.append(item)
            continue

        if isinstance(item, dict):

            text = item.get("text")

            if text:
                parts.append(text)

    return "".join(parts)


def _tool_name(
    event: dict[str, Any],
) -> str:
    """
    Extract tool name from LangChain event.
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
    Extract sources from ToolMessage artifact.
    """

    if not isinstance(output, ToolMessage):
        return []

    artifact = getattr(
        output,
        "artifact",
        None,
    )

    if not isinstance(
        artifact,
        dict,
    ):
        return []

    raw_sources = artifact.get(
        "sources",
        [],
    )

    if not isinstance(
        raw_sources,
        list,
    ):
        return []

    sources: list[Source] = []

    for item in raw_sources:

        if isinstance(
            item,
            Source,
        ):

            sources.append(item)
            continue

        if isinstance(
            item,
            dict,
        ):

            try:

                sources.append(
                    Source.model_validate(
                        item
                    )
                )

            except Exception:

                logger.warning(
                    "Skipping invalid source artifact."
                )

    return sources


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
    Stream UI events from LangChain agent.
    """

    logger.info(
        "Starting streaming session."
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

            event_name = event.get(
                "event",
                "",
            )

            # ----------------------------------------------
            # LLM token
            # ----------------------------------------------

            if event_name == "on_chat_model_stream":

                chunk = (
                    event
                    .get("data", {})
                    .get("chunk")
                )

                token = _extract_text(
                    chunk
                )

                if token:

                    yield TokenEvent.create(
                        text=token,
                    )

                continue

            # ----------------------------------------------
            # Tool start
            # ----------------------------------------------

            if event_name == "on_tool_start":

                tool = _tool_name(event)

                logger.info(
                    "Tool started: %s",
                    tool,
                )

                yield ToolStartEvent.create(
                    name=tool,
                )

                continue

            # ----------------------------------------------
            # Tool end
            # ----------------------------------------------

            if event_name == "on_tool_end":

                tool = _tool_name(event)

                logger.info(
                    "Tool finished: %s",
                    tool,
                )

                yield ToolEndEvent.create(
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

                    yield SourceEvent.create(
                        sources=sources,
                    )

                continue

        logger.info(
            "Streaming completed."
        )

        yield DoneEvent.create()

    except Exception as exc:

        logger.exception(
            "Streaming failed."
        )

        yield ErrorEvent.create(
            message=str(exc),
        )
