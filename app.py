import logging

import chainlit as cl

from agents.rag_agent import ask_agent

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    ),
)

# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

logger.info("App module loaded.")


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
Hello!

I am **agentic AI assistant** based on Qwen 3.6 model.

My capabilites:

- Knowledge of 201 language and dialect.
- RAG (Retrieval-Augmented Generation).
- Instantly updated factual knowledge database.
- 16384 token long context window.
- Input: text and image.
- Output: text.

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
