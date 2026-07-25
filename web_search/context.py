"""
Web search orchestration layer.

Pipeline:

Query
 |
 v
Semantic cache (Qdrant)
 |
 +-- hit
 |
 +-- miss
        |
        v
      Exa
        |
        v
    Cheap filtering
        |
        v
    Embeddings
        |
        v
    Reranking
        |
        v
    Qdrant memory

This module does not know about:
- Chainlit
- UI
- LangGraph internals
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any


from web_search.exa import search_exa

from web_search.filter import (
    filter_documents,
)

from web_search.reranker import (
    get_reranker,
)

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


CACHE_TOP_K = 5

FINAL_CONTEXT_K = 8

SIMILARITY_THRESHOLD = 0.70


# количество кандидатов после дешевого фильтра
RERANK_INPUT_K = 30



# ==========================================================
# Initialization
# ==========================================================


_initialized = False

_init_lock = asyncio.Lock()



async def init_web_search() -> None:
    """
    Initialize web search subsystem once.
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
    Format selected chunks for LLM.
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


    return "\n\n".join(result)



# ==========================================================
# Sources
# ==========================================================


def _extract_sources(
    chunks: list[dict[str, Any]],
) -> list[Source]:
    """
    Extract unique sources.
    """

    sources: list[Source] = []

    seen_urls: set[str] = set()


    for chunk in chunks:

        url = chunk.get(
            "url",
            "",
        )


        if not url:
            continue


        if url in seen_urls:
            continue


        seen_urls.add(url)


        sources.append(
            Source(

                title=(
                    chunk.get(
                        "title"
                    )
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


    return sources



# ==========================================================
# Main pipeline
# ==========================================================


async def get_context(
    query: str,
) -> AgentContext:
    """
    Build context for agent.

    Pipeline:

    1. Semantic cache lookup
    2. Exa retrieval
    3. Cheap filtering
    4. Embedding generation
    5. Reranking
    6. Qdrant persistence
    7. Context preparation
    """


    await init_web_search()


    logger.info(
        "Building context for query='%s'",
        query,
    )


    try:


        # ==================================================
        # 1. Semantic cache
        # ==================================================

        cached_chunks = await search(
            query=query,
            limit=CACHE_TOP_K,
            score_threshold=SIMILARITY_THRESHOLD,
        )


        if cached_chunks:

            logger.info(
                "Semantic cache hit. Chunks=%d",
                len(cached_chunks),
            )


            return AgentContext(

                text=_format_context(
                    cached_chunks
                ),

                sources=_extract_sources(
                    cached_chunks
                ),
            )



        logger.info(
            "Semantic cache miss."
        )



        # ==================================================
        # 2. Exa search
        # ==================================================

        documents = await search_exa(
            query
        )


        logger.info(
            "Exa documents received=%d",
            len(documents),
        )



        # ==================================================
        # 3. Cheap filtering
        # ==================================================

        candidates = filter_documents(
            documents,
            query=query,
            limit=RERANK_INPUT_K,
        )


        logger.info(
            "After filtering=%d candidates",
            len(candidates),
        )



        if not candidates:

            raise RuntimeError(
                "No usable documents after filtering."
            )



        # ==================================================
        # 4-5 Embedding + reranking
        # ==================================================

        reranker = get_reranker()

        ranked_chunks = await reranker.rerank(
            query=query,
            chunks=candidates,
        )


        selected_chunks = ranked_chunks[
            :FINAL_CONTEXT_K
        ]


        logger.info(
            "Reranked chunks selected=%d",
            len(selected_chunks),
        )



        # ==================================================
        # 6. Store only final memory
        # ==================================================

        await add_chunks(
            selected_chunks
        )


        logger.info(
            "Stored %d chunks into Qdrant.",
            len(selected_chunks),
        )



        # ==================================================
        # 7. Return context
        # ==================================================

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
