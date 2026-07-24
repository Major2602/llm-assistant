import logging
from typing import Any, AsyncIterator

from agents.web_search_agent import get_agent

from ui.events import (
    UIEvent,
    UIEventType,
)


logger = logging.getLogger(__name__)


def _extract_text_from_chunk(
    chunk: Any,
) -> str | None:
    """
    Normalize LangChain AIMessageChunk content.

    LangChain 1.x may return:
    - str
    - list content blocks
    - None
    """

    if chunk is None:
        return None


    content = getattr(
        chunk,
        "content",
        None,
    )


    if not content:
        return None


    # Standard case:
    # "hello"
    if isinstance(
        content,
        str,
    ):
        return content


    # Multimodal/content blocks:
    # [
    #   {
    #       "type": "text",
    #       "text": "hello"
    #   }
    # ]

    if isinstance(
        content,
        list,
    ):

        parts: list[str] = []


        for block in content:

            if not isinstance(
                block,
                dict,
            ):
                continue


            if (
                block.get("type")
                == "text"
            ):

                text = block.get(
                    "text"
                )

                if text:
                    parts.append(
                        text
                    )


        if parts:
            return "".join(parts)


    logger.debug(
        "Unsupported chunk format: %s",
        type(content),
    )


    return None



def _extract_tool_name(
    event: dict,
) -> str:
    """
    Extract tool name from LangChain event.
    """

    return (
        event.get("name")
        or
        event.get(
            "metadata",
            {}
        ).get(
            "name"
        )
        or
        "unknown_tool"
    )



async def stream_agent_events(
    text: str,
) -> AsyncIterator[UIEvent]:
    """
    Convert LangChain v2 streaming events
    into internal UI events.

    Chainlit receives only UIEvent.
    """

    logger.info(
        "Starting agent event stream."
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
                "event"
            )


            # ==================================================
            # LLM TOKEN STREAM
            # ==================================================

            if event_name == (
                "on_chat_model_stream"
            ):

                chunk = (
                    event
                    .get("data", {})
                    .get("chunk")
                )


                token = _extract_text_from_chunk(
                    chunk
                )


                if token:

                    yield UIEvent(
                        type=UIEventType.TOKEN,
                        content=token,
                    )


            # ==================================================
            # TOOL START
            # ==================================================

            elif event_name == (
                "on_tool_start"
            ):

                tool_name = (
                    _extract_tool_name(
                        event
                    )
                )


                logger.info(
                    "Tool started: %s",
                    tool_name,
                )


                yield UIEvent(
                    type=UIEventType.TOOL_START,
                    content=tool_name,
                    metadata={
                        "tool": tool_name,
                    },
                )


            # ==================================================
            # TOOL END
            # ==================================================

            elif event_name == (
                "on_tool_end"
            ):

                tool_name = (
                    _extract_tool_name(
                        event
                    )
                )


                logger.info(
                    "Tool completed: %s",
                    tool_name,
                )


                yield UIEvent(
                    type=UIEventType.TOOL_END,
                    content=tool_name,
                    metadata={
                        "tool": tool_name,
                    },
                )


        logger.info(
            "Agent stream completed."
        )


        yield UIEvent(
            type=UIEventType.DONE
        )


    except Exception:

        logger.exception(
            "Agent streaming failed."
        )


        yield UIEvent(
            type=UIEventType.ERROR,
            content=(
                "Unable to complete request."
            ),
        )
