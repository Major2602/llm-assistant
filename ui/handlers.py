import logging

import chainlit as cl

from ui.streaming import (
    stream_agent_events,
)

from ui.events import (
    UIEventType,
)


logger = logging.getLogger(__name__)


async def handle_message(
    message: cl.Message,
) -> None:
    """
    Process user message.

    Chainlit receives UI events only.
    """

    logger.info(
        "User message received."
    )


    msg = cl.Message(
        content=""
    )


    await msg.send()


    try:

        async for event in stream_agent_events(
            message.content
        ):

            if event.type == (
                UIEventType.TOKEN
            ):

                await msg.stream_token(
                    event.content or ""
                )


            elif event.type == (
                UIEventType.STATUS
            ):
                logger.info(
                    "Status: %s",
                    event.content,
                )


            elif event.type == (
                UIEventType.ERROR
            ):

                msg.content = (
                    "⚠️ Could not complete the request."
                )

                await msg.update()


                return


            elif event.type == (
                UIEventType.DONE
            ):

                await msg.update()


        logger.info(
            "Response completed."
        )


    except Exception:

        logger.exception(
            "UI handler failed."
        )

        msg.content = (
            "⚠️ Unexpected error."
        )

        await msg.update()
