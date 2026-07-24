"""
Web search orchestration layer.

Responsible for:
- semantic cache lookup;
- Exa fallback search;
- context preparation for LLM;
- source extraction for UI citations.

This module does not know about:
- Chainlit;
- LangChain agent internals;
- UI rendering.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from web_search.exa import search_exa
from web_search.models import (
    AgentContext,
    Source,
)
from web_search.qdrant_store import (
    add_chunks,
    cleanup_old_chunks,
    search,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


TOP_K = 5

SIMILARITY_THRESHOLD = 0.70


# ==========================================================
# Initialization
# ==========================================================


_initialized = False

_init_lock = asyncio.Lock()



async def init_web_search() -> None:
    """
    Initialize web_search subsystem once.
    """

    global _initialized


    if _initialized:
        return


    async with _init_lock:

        if _initialized:
            return


        try:

            logger.info(
                "Initializing web_search subsystem."
            )


            await cleanup_old_chunks(
                days=30
            )


            _initialized = True


            logger.info(
                "web_search initialized successfully."
            )


        except Exception:

            logger.exception(
                "Failed to initialize web_search."
            )

            raise



# ==========================================================
# Formatting
# ==========================================================


def _format_context(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Prepare retrieved chunks for LLM context.

    URLs and metadata are intentionally excluded.
    They are returned separately as sources.
    """

    result: list[str] = []


    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        result.append(
            f"""
SOURCE [{index}]

Title:
{chunk.get("title", "")}

Text:
{chunk.get("text", "")}
"""
        )


    return "\n\n".join(
        result
    )



# ==========================================================
# Sources
# ==========================================================


def _extract_sources(
    chunks: list[dict[str, Any]],
) -> list[Source]:
    """
    Convert chunks metadata into UI sources.
    """

    sources: list[Source] = []

    seen_urls: set[str] = set()


    for chunk in chunks:

        url = (
            chunk.get("url")
            or ""
        )


        if not url:
            continue


        if url in seen_urls:
            continue


        seen_urls.add(
            url
        )


        sources.append(
            Source(

                title=(
                    chunk.get("title")
                    or "Untitled source"
                ),

                url=url,

                provider=chunk.get(
                    "provider"
                ),

                score=chunk.get(
                    "score"
                ),

                published_date=chunk.get(
                    "published_date"
                ),

                author=chunk.get(
                    "author"
                ),
            )
        )


    logger.debug(
        "Extracted %d unique sources.",
        len(sources),
    )


    return sources



# ==========================================================
# Main context builder
# ==========================================================


async def get_context(
    query: str,
) -> AgentContext:
    """
    Retrieve context for agent.

    Flow:

        Query
          |
          v
    Qdrant semantic cache

          |
          +-- hit
          |
          v

    AgentContext


          |
          +-- miss
          |
          v

        Exa search

          |
          v

      Store in Qdrant

          |
          v

      AgentContext
    """

    await init_web_search()


    logger.info(
        "Building context for query='%s'",
        query,
    )


    try:

        # -------------------------------------------------
        # 1. Semantic cache
        # -------------------------------------------------

        chunks = await search(
            query=query,
            limit=TOP_K,
            score_threshold=SIMILARITY_THRESHOLD,
        )


        if chunks:

            logger.info(
                "Semantic cache hit. Chunks=%d",
                len(chunks),
            )


            return AgentContext(

                text=_format_context(
                    chunks
                ),

                sources=_extract_sources(
                    chunks
                ),
            )



        logger.info(
            "Semantic cache miss. Using Exa."
        )



        # -------------------------------------------------
        # 2. External search
        # -------------------------------------------------

        web_chunks = await search_exa(
            query
        )


        logger.info(
            "Exa returned %d chunks.",
            len(web_chunks),
        )



        # -------------------------------------------------
        # 3. Store memory
        # -------------------------------------------------

        await add_chunks(
            web_chunks
        )


        logger.info(
            "Stored Exa chunks in Qdrant."
        )



        # -------------------------------------------------
        # 4. Prepare response
        # -------------------------------------------------

        selected_chunks = (
            web_chunks[:TOP_K]
        )


        return AgentContext(

            text=_format_context(
                selected_chunks
            ),

            sources=_extract_sources(
                selected_chunks
            ),
        )



    except Exception:

        logger.exception(
            "Failed building context for query='%s'.",
            query,
        )

        raise
