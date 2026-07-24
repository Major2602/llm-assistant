import logging
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

from llm import get_llm
from rag.context import get_context


logger = logging.getLogger(__name__)


# ==========================================================
# RAG TOOL
# ==========================================================


@tool
async def web_search(
    query: str
) -> str:
    """
    Search the web and semantic memory.

    Workflow:
    - searches existing knowledge in Qdrant semantic cache;
    - if no relevant information exists, performs Exa web search;
    - stores new information in semantic memory;
    - returns relevant context for answering.

    Use this tool for factual questions,
    current information and external knowledge.
    """

    logger.info(
        "Web search tool called. Query='%s'",
        query,
    )

    try:

        context = await get_context(query)

        logger.info(
            "web_search tool returned context for '%s'.",
            query,
        )

        return context

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
            "Initializing RAG agent."
        )

        _agent = create_agent(

            model=get_llm(),

            tools=[
                web_search,
            ],

            system_prompt="""

            You are a helpful AI assistant.

            Use web_search whenever factual information,
            recent information or external knowledge is required.

            The tool provides information from semantic memory
            or web search.

            Answer using the provided context.

            Do not mention:
            - Qdrant
            - embeddings
            - semantic cache
            - Exa
            - internal tools

            unless the user asks about it.

            Always answer in the user's language.
            
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
