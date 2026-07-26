"""
Web search agent.

Responsibilities:
- configure LangChain agent;
- expose web_search tool;
- bridge AgentContext to the LLM.

The agent intentionally knows nothing about:

- Exa
- Qdrant
- Hybrid Retrieval
- Embeddings
- Compression
- Context Optimization

All retrieval logic lives inside web_search.context.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

from llm import get_llm
from web_search.context import get_context

logger = logging.getLogger(__name__)


# ==========================================================
# Web Search Tool
# ==========================================================


@tool(response_format="content_and_artifact")
async def web_search(
    query: str,
) -> tuple[str, dict[str, Any]]:
    """
    Retrieve optimized external context.

    Returns:

        content:
            Optimized context prepared for LLM.

        artifact:
            Source metadata used by UI.
    """

    logger.info(
        "Web search requested. query='%s'",
        query,
    )

    context = await get_context(query)

    artifact = {
        "sources": [
            source.model_dump()
            for source in context.sources
        ],
    }

    logger.info(
        "Context ready. sources=%d tokens=%d",
        len(context.sources),
        context.token_count,
    )

    return (
        context.text,
        artifact,
    )


# ==========================================================
# Agent Singleton
# ==========================================================


_agent: Any | None = None

_agent_lock = threading.Lock()


SYSTEM_PROMPT = """
You are an advanced AI assistant.

You have access to a retrieval tool that provides
optimized external knowledge.

When answering:

• Use retrieved context whenever available.
• Prefer retrieved facts over prior knowledge.
• Never invent unsupported information.
• If the retrieved context is insufficient,
  explicitly state the uncertainty.

When sources are available:

• Base the answer only on supported facts.
• Do not fabricate citations.
• Preserve factual consistency.

Never mention internal implementation details.

Never mention:

- Exa
- Qdrant
- BM25
- embeddings
- reranking
- semantic cache
- compression
- retrieval pipeline

unless the user explicitly asks.

Always answer in the user's language.

Produce concise, accurate and well-structured answers.
"""


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
            "Initializing web search agent."
        )

        _agent = create_agent(
            model=get_llm(),
            tools=[
                web_search,
            ],
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
) -> str:
    """
    Execute agent without streaming.
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

    messages = result.get(
        "messages",
        [],
    )

    if not messages:
        raise RuntimeError(
            "Agent returned no messages."
        )

    content = messages[-1].content

    response = (
        content
        if isinstance(content, str)
        else str(content)
    )

    logger.info(
        "Agent response generated successfully."
    )

    return response
