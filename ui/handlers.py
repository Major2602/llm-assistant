import logging

import chainlit as cl

from ui.streaming import stream_agent_events
from ui.events import UIEventType
from ui.tooling import (
    get_tool_start_message,
    get_tool_end_message,
)


logger = logging.getLogger(__name__)



async def handle_message(
    message: cl.Message,
) -> None:


    logger.info(
        "User message received."
    )


    msg = cl.Message(
        content=""
    )


    await msg.send()


    active_steps = {}


    try:

        async for event in stream_agent_events(
            message.content
        ):


            # ==========================
            # TOKENS
            # ==========================

            if event.type == (
                UIEventType.TOKEN
            ):

                await msg.stream_token(
                    event.content
                )



            # ==========================
            # TOOL START
            # ==========================

            elif event.type == (
                UIEventType.TOOL_START
            ):


                tool_name = (
                    event.metadata
                    .get(
                        "tool",
                        "tool",
                    )
                )


                step = cl.Step(
                    name=tool_name,
                    type="tool",
                )


                step.output = (
                    get_tool_start_message(
                        tool_name
                    )
                )


                await step.send()


                active_steps[
                    tool_name
                ] = step



            # ==========================
            # TOOL END
            # ==========================

            elif event.type == (
                UIEventType.TOOL_END
            ):


                tool_name = (
                    event.metadata
                    .get(
                        "tool",
                        "tool",
                    )
                )


                step = active_steps.get(
                    tool_name
                )


                if step:

                    step.output = (
                        get_tool_end_message(
                            tool_name
                        )
                    )


                    await step.update()



            # ==========================
            # ERROR
            # ==========================

            elif event.type == (
                UIEventType.ERROR
            ):


                msg.content = (
                    "⚠️ I couldn't complete the request."
                )


                await msg.update()

                return



            # ==========================
            # DONE
            # ==========================

            elif event.type == (
                UIEventType.DONE
            ):

                await msg.update()



        logger.info(
            "Message processing completed."
        )


    except Exception:

        logger.exception(
            "UI handler failed."
        )


        msg.content = (
            "⚠️ Unexpected error."
        )

        await msg.update()
