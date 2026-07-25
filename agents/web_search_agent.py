"""
Agent layer.

Contains LangChain agent configuration and tools.

The UI layer must not depend on this module directly.
Streaming and event conversion are handled by ui.streaming.
"""

from __future__ import annotations

import logging
import threading
from typing import Any


from langchain.agents import create_agent
from langchain.tools import tool


from llm import get_llm

from web_search.context import (
    get_context,
)


logger = logging.getLogger(__name__)



# ==========================================================
# WEB SEARCH TOOL
# ==========================================================


@tool(
    response_format="content_and_artifact"
)
async def web_search(
    query: str,
) -> tuple[str, dict[str, Any]]:
    """
    Search external information sources.

    Use this tool for:

    - factual questions;
    - recent information;
    - information not available in conversation.

    Returns:

    - relevant context;
    - source metadata.
    """


    logger.info(
        "Web search tool called. Query='%s'",
        query,
    )


    try:

        context = await get_context(
            query
        )


        artifact = {

            "sources": [

                source.model_dump()

                for source in context.sources

            ]

        }


        logger.info(
            "Web search completed. Sources=%d",
            len(
                context.sources
            ),
        )


        return (

            context.text,

            artifact,

        )


    except Exception:

        logger.exception(
            "web_search tool failed for '%s'.",
            query,
        )

        raise



# ==========================================================
# AGENT
# ==========================================================


_agent: Any | None = None

_agent_lock = threading.Lock()



def get_agent() -> Any:
    """
    Create and return singleton LangChain agent.
    """


    global _agent


    if _agent is not None:
        return _agent



    with _agent_lock:


        if _agent is not None:
            return _agent



        try:

            logger.info(
                "Initializing web_search agent."
            )


            _agent = create_agent(

                model=get_llm(),


                tools=[

                    web_search,

                ],


                system_prompt="""

You are a helpful AI assistant.

Use web_search whenever:

- factual information is required;
- recent information is required;
- external knowledge is needed.

The tool provides relevant context and sources.

Answer using the provided information.

Rules:

- Do not mention internal implementation details.
- Do not mention:
    - Qdrant
    - embeddings
    - semantic cache
    - Exa
    - internal tools

unless the user explicitly asks.

Always answer in the user's language.

When sources are provided:
- use them naturally;
- avoid inventing unsupported facts;
- clearly distinguish known information from uncertainty.

""",

            )


            logger.info(
                "Web search agent initialized successfully."
            )


            return _agent



        except Exception:

            logger.exception(
                "Failed initializing web_search agent."
            )

            raise



# ==========================================================
# PUBLIC API
# ==========================================================


async def ask_agent(
    text: str,
) -> str:
    """
    Execute agent without streaming.

    Used for testing or non-UI integrations.
    """


    logger.info(
        "Agent request received."
    )


    try:

        agent = get_agent()



        result = await agent.ainvoke(

            {
                "messages": [

                    {
                        "role": "user",
                        "content": text,
                    }

                ]
            }

        )



        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Unexpected agent response format."
            )



        messages = result.get(
            "messages",
            [],
        )



        if not messages:

            raise RuntimeError(
                "Agent returned empty response."
            )



        content = messages[-1].content



        if isinstance(
            content,
            str,
        ):

            response = content


        else:

            response = str(
                content
            )



        logger.info(
            "Agent response generated successfully."
        )


        return response



    except Exception:

        logger.exception(
            "Agent execution failed."
        )

        raise
