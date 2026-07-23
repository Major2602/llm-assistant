import logging

import chainlit as cl

from agents.rag_agent import ask_agent


logger = logging.getLogger(__name__)


# ==========================================================
# Chat start
# ==========================================================


@cl.on_chat_start
async def start() -> None:
    """
    Initialize Chainlit session.
    """

    logger.info(
        "New chat session started."
    )

    await cl.Message(
        content=
        """
Привет!

Я Qwen3.5 ассистент.

Умею:
- вести обычный диалог
- искать информацию через RAG
- использовать Wikipedia

"""
    ).send()



# ==========================================================
# Message handler
# ==========================================================


@cl.on_message
async def main(
    message: cl.Message,
) -> None:
    """
    Handle user messages with streaming response.
    """

    logger.info(
        "User message received."
    )

    msg = cl.Message(
        content=""
    )

    await msg.send()


    try:

        async for token in ask_agent_stream(
            message.content
        ):

            await msg.stream_token(
                token
            )


        logger.info(
            "Streaming response completed."
        )


    except Exception:

        logger.exception(
            "Failed processing user message."
        )

        await msg.update(
            content=
            "Произошла ошибка при обработке запроса."
        )
