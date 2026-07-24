"""
Agent layer.

Contains LangChain agent configuration and tools.

The UI layer must not depend on this module directly.
Streaming and event conversion are handled by ui.streaming.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

from llm import get_llm
from web_search.context import get_context


logger = logging.getLogger(__name__)


# ==========================================================
# WEB SEARCH TOOL
# ==========================================================


@tool(
    response_format="content_and_artifact"
)
async def web_search(
    query: str,
) -> tuple[str, dict]:
    """
    Search the web and semantic memory.

    Workflow:
    - searches semantic memory;
    - falls back to Exa web search if required;
    - stores new information;
    - returns context and source metadata.

    Use this tool for:
    - factual questions;
    - current information;
    - external knowledge.
    """

    logger.info(
        "Web search tool called. Query='%s'",
        query,
    )

    try:

        context = await get_context(
            query
        )


        logger.info(
            "Web search completed. Sources=%d",
            len(
                context.sources
            ),
        )


        artifact = {
            "sources": context.sources,
        }


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



def get_agent() -> Any:
    """
    Create and return singleton LangChain agent.
    """

    global _agent


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

When sources are provided, use them naturally
and avoid inventing unsupported facts.

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


        messages = (
            result.get(
                "messages",
                []
            )
            if isinstance(
                result,
                dict,
            )
            else []
        )


        if not messages:

            raise RuntimeError(
                "Agent returned empty response."
            )


        response = messages[-1].content


        logger.info(
            "Agent response generated successfully."
        )


        return response


    except Exception:

        logger.exception(
            "Agent execution failed."
        )

        raise
