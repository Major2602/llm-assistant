import logging
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

from llm import get_llm
from rag.index import get_context


logger = logging.getLogger(__name__)


# ==========================================================
# RAG TOOL
# ==========================================================


@tool
async def web_search(
    query: str
) -> str:
    """
    Search Wikipedia using RAG.

    Uses Qdrant retrieval first.
    If no information exists:
    - loads Wikipedia;
    - creates embeddings;
    - stores chunks;
    - retrieves relevant context.
    """

    logger.info(
        "Web search tool called. Query='%s'",
        query,
    )

    try:

        context = await get_context(query)

        logger.info(
            "Web search tool returned context for '%s'.",
            query,
        )

        return context

    except Exception:
        logger.exception(
            "Web search tool failed for '%s'.",
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
            "Initializing RAG agent."
        )

        _agent = create_agent(

            model=get_llm(),

            tools=[
                web_search,
            ],

            system_prompt="""

You are a helpful assistant.

Search the web using web_search and return relevant context.
Use this tool whenever up-to-date or factual information is needed.

Answer in the user's language.

Do not mention internal RAG,
Qdrant, embeddings or tools
unless the user asks about them.

""",

        )


        logger.info(
            "RAG agent initialized successfully."
        )

        return _agent


    except Exception:
        logger.exception(
            "Failed initializing RAG agent."
        )
        raise



# ==========================================================
# PUBLIC API
# ==========================================================


async def ask_agent(
    text: str,
) -> str:
    """
    Send user message to the agent.
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
            result.get("messages", [])
            if isinstance(result, dict)
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


async def ask_agent_stream(
    text: str,
):
    """
    Stream agent response tokens.
    """

    agent = get_agent()


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

        if event["event"] == "on_chat_model_stream":

            chunk = event["data"]["chunk"]

            content = chunk.content

            if content:
                yield content
