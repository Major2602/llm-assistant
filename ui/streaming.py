import logging
from typing import AsyncIterator

from agents.web_search_agent import get_agent

from ui.events import (
    UIEvent,
    UIEventType,
)


logger = logging.getLogger(__name__)


async def stream_agent_events(
    text: str,
) -> AsyncIterator[UIEvent]:
    """
    Convert LangChain agent events
    into UI events.
    """

    logger.info(
        "Starting agent stream."
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

            event_type = event.get(
                "event"
            )


            if event_type == (
                "on_chat_model_stream"
            ):

                chunk = (
                    event
                    .get("data", {})
                    .get("chunk")
                )

                if not chunk:
                    continue


                content = (
                    getattr(
                        chunk,
                        "content",
                        None,
                    )
                )


                if content:

                    yield UIEvent(
                        type=UIEventType.TOKEN,
                        content=content,
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
