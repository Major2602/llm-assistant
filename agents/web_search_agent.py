"""
Web search agent.

Responsibilities
----------------
- configure LangChain agent;
- expose web_search tool;
- bridge retrieval pipeline to LLM;
- expose structured response for UI.

The agent intentionally knows nothing about
retrieval internals.
"""

from __future__ import annotations

import logging
import threading

from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

from llm import get_llm
from web_search.orchestrator import get_context
from web_search.models import PipelineMetadata, Source

logger = logging.getLogger(__name__)


# ==========================================================
# UI Contract
# ==========================================================


class AgentResponse(dict):
    """
    Final response returned to UI.

    {
        "answer": "...",
        "sources": [...],
        "metadata": {...}
    }
    """

    pass


# ==========================================================
# Tool
# ==========================================================


@tool(response_format="content_and_artifact")
async def web_search(
    query: str,
) -> tuple[str, dict[str, Any]]:
    """
    Retrieve optimized context.

    content:
        text injected into LLM context

    artifact:
        metadata preserved for UI
    """

    logger.info(
        "Web search requested. query='%s'",
        query,
    )

    context = await get_context(query)

    tokens = (
        context.optimized_context.total_tokens
        if context.optimized_context
        else 0
    )

    logger.info(
        "Context ready. sources=%d tokens=%d",
        len(context.sources),
        tokens,
    )

    artifact = {
        "sources": [
            source.model_dump()
            for source in context.sources
        ],
        "metadata": (
            context.metadata.model_dump()
            if context.metadata
            else None
        ),
    }

    return (
        context.text,
        artifact,
    )


# ==========================================================
# Agent
# ==========================================================


SYSTEM_PROMPT = """
You are an advanced AI assistant.

You have access to an external retrieval tool.

Instructions:

- Prefer retrieved information whenever available.
- Never invent unsupported facts.
- If retrieved information is insufficient, say so.
- Never fabricate citations.
- Never mention internal implementation details.
- Always answer in the user's language.
- Produce concise, accurate, well-structured responses.
"""


_agent: Any | None = None

_lock = threading.Lock()


def get_agent() -> Any:

    global _agent

    if _agent is not None:
        return _agent

    with _lock:

        if _agent is not None:
            return _agent

        logger.info(
            "Initializing web search agent."
        )

        _agent = create_agent(
            model=get_llm(),
            tools=[web_search],
            system_prompt=SYSTEM_PROMPT,
        )

        logger.info(
            "Web search agent initialized."
        )

        return _agent


# ==========================================================
# Public API
# ==========================================================


async def ask_agent(
    text: str,
) -> AgentResponse:
    """
    Execute agent.

    Returns

    {
        answer,
        sources,
        metadata,
    }
    """

    logger.info(
        "Agent request received."
    )

    result = await get_agent().ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": text,
                }
            ]
        }
    )

    if not isinstance(result, dict):
        raise RuntimeError(
            "Unexpected agent response."
        )

    messages = result.get(
        "messages",
        [],
    )

    if not messages:
        raise RuntimeError(
            "Agent returned no messages."
        )

    answer = ""

    sources: list[dict[str, Any]] = []

    metadata: dict[str, Any] | None = None

    for message in messages:

        content = getattr(
            message,
            "content",
            None,
        )

        if isinstance(content, str):
            answer = content

        artifact = getattr(
            message,
            "artifact",
            None,
        )

        if not artifact:
            continue

        sources = artifact.get(
            "sources",
            sources,
        )

        metadata = artifact.get(
            "metadata",
            metadata,
        )

    logger.info(
        "Agent response generated."
    )

    return AgentResponse(
        answer=answer,
        sources=sources,
        metadata=metadata,
    )
