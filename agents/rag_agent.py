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
async def wikipedia_rag(
    entity: str,
    question: str,
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
        "Wikipedia RAG tool called. Entity='%s'",
        entity,
    )

    try:

        context = await get_context(
            entity=entity,
            question=question,
        )

        logger.info(
            "Wikipedia RAG returned context for '%s'.",
            entity,
        )

        return context

    except Exception:
        logger.exception(
            "Wikipedia RAG tool failed for '%s'.",
            entity,
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
                wikipedia_rag,
            ],

            system_prompt="""

You are a helpful assistant.

When factual information is required,
always use wikipedia_rag.

The tool retrieves information from
a Wikipedia RAG knowledge base.

For the "entity" argument provide
ONLY the entity name.

Correct examples:

entity="Alan Turing"
question="Who was he?"

entity="Python programming language"
question="Who created it?"

entity="Tokyo"
question="Population"


Incorrect:

entity="Who created Python?"
entity="Tell me everything about Tokyo"


Never pass the entire user message
as entity.


Use information returned by
wikipedia_rag as the primary source
for factual answers.


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
