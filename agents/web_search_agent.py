"""
Web search agent.

Module Responsibilities:

- configure LangChain agent;
- expose retrieval tool;
- bridge AgentContext to LLM;
- convert final execution result into FinalAnswer contract.
"""

from __future__ import annotations


import logging
import threading
from typing import Any


from langchain.agents import create_agent
from langchain.tools import tool


from llm import get_llm


from web_search.orchestrator import get_context


from web_search.models import (
    AgentContext,
    FinalAnswer,
)


logger = logging.getLogger(__name__)



# ==========================================================
# Retrieval Tool
# ==========================================================


@tool(
    response_format="content_and_artifact"
)
async def web_search(
    query: str,
) -> tuple[str, dict[str, Any]]:
    """
    Retrieve external context.

    Returns:

    content:
        Text context for LLM.

    artifact:
        Structured context for UI layer.
    """

    logger.info(
        "Retrieval requested query=%s",
        query,
    )


    context = await get_context(
        query
    )


    artifact = {
        "context": context.model_dump()
    }


    logger.info(
        (
            "Retrieval completed "
            "sources=%d"
        ),
        len(
            context.sources
        ),
    )


    return (
        context.text,
        artifact,
    )



# ==========================================================
# Agent configuration
# ==========================================================


SYSTEM_PROMPT = """
You are an advanced AI assistant.

You have access to an external knowledge retrieval tool.

Rules:

- Use retrieved context whenever it is available.
- Prefer retrieved information over memory.
- Never invent unsupported facts.
- If information is insufficient, clearly state uncertainty.
- Never fabricate citations.

Answer in the user's language.

Do not mention internal retrieval mechanisms.

Be concise and structured.
"""



_agent: Any | None = None

_agent_lock = threading.Lock()



def get_agent() -> Any:
    """
    Return singleton LangChain agent.
    """

    global _agent


    if _agent is not None:

        return _agent



    with _agent_lock:


        if _agent is not None:

            return _agent



        logger.info(
            "Initializing agent."
        )


        _agent = create_agent(

            model=get_llm(),

            tools=[
                web_search,
            ],

            system_prompt=SYSTEM_PROMPT,

        )


        logger.info(
            "Agent initialized."
        )


        return _agent



# ==========================================================
# Final answer builder
# ==========================================================


def _extract_context(
    messages: list[Any],
) -> AgentContext | None:
    """
    Extract AgentContext from ToolMessage artifact.
    """

    for message in reversed(messages):

        artifact = getattr(
            message,
            "artifact",
            None,
        )


        if not artifact:

            continue


        raw_context = artifact.get(
            "context"
        )


        if not raw_context:

            continue


        return AgentContext(
            **raw_context
        )


    return None



def _extract_answer(
    messages: list[Any],
) -> str:
    """
    Extract final AI response.
    """

    for message in reversed(messages):

        content = getattr(
            message,
            "content",
            None,
        )


        if not content:

            continue


        message_type = getattr(
            message,
            "type",
            None,
        )


        if message_type == "ai":

            if isinstance(
                content,
                str,
            ):
                return content


            return str(content)


    raise RuntimeError(
        "Agent returned no final answer."
    )



def _build_final_answer(
    result: dict[str, Any],
) -> FinalAnswer:
    """
    Convert LangChain result into UI contract.
    """

    messages = result.get(
        "messages",
        [],
    )


    if not messages:

        raise RuntimeError(
            "Agent returned no messages."
        )


    answer = _extract_answer(
        messages
    )


    context = _extract_context(
        messages
    )


    if context is None:

        return FinalAnswer(
            answer=answer
        )


    citation_map = {}


    if context.optimized_context:

        citation_map = (
            context.optimized_context.citation_map
        )


    return FinalAnswer(

        answer=answer,

        sources=context.sources,

        citation_map=citation_map,

        metadata=context.metadata,

    )



# ==========================================================
# Public API
# ==========================================================


async def ask_agent(
    text: str,
) -> FinalAnswer:
    """
    Execute agent and return UI contract.
    """

    logger.info(
        "Agent request received."
    )


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
            "Unexpected agent response."
        )


    final_answer = _build_final_answer(
        result
    )


    logger.info(
        (
            "Agent response completed "
            "sources=%d"
        ),
        len(
            final_answer.sources
        ),
    )


    return final_answer
