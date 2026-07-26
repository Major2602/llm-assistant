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
      Chunking
        |
        v
      Chunk filtering
        |
        v
      Reranking
        |
        v
      Qdrant memory
        |
        v
      AgentContext


This module does not know about:
- Chainlit;
- UI;
- LangGraph internals.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any


from web_search.exa import search_exa

from web_search.chunker import (
    chunk_documents,
)

from web_search.filter import (
    filter_chunks,
)

from web_search.reranker import (
    get_reranker,
)

from web_search.qdrant_store import (
    add_chunks,
    cleanup_old_chunks,
    search,
)

from web_search.models import (
    AgentContext,
    Source,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


CACHE_TOP_K = 5

RERANK_OUTPUT_K = 8

SIMILARITY_THRESHOLD = 0.80

MAX_RERANK_CHUNKS = 50



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
                "Failed initializing web_search."
            )

            raise



# ==========================================================
# Formatting
# ==========================================================


def _format_context(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Prepare context text for LLM.
    """


    sections = []


    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        sections.append(
            f"""
SOURCE [{index}]

Title:
{chunk.get("title", "")}

Content:
{chunk.get("text", "")}
"""
        )


    return "\n\n".join(
        sections
    )



# ==========================================================
# Sources
# ==========================================================


def _extract_sources(
    chunks: list[dict[str, Any]],
) -> list[Source]:
    """
    Extract unique citation sources.
    """


    result: list[Source] = []

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


        seen_urls.add(
            url
        )


        result.append(

            Source(

                title=(
                    chunk.get("title")
                    or "Untitled source"
                ),

                url=url,

                provider=chunk.get(
                    "provider"
                ),

                score=(
                    chunk.get(
                        "rerank_score"
                    )
                    or chunk.get(
                        "filter_score"
                    )
                    or chunk.get(
                        "score"
                    )
                ),

                published_date=chunk.get(
                    "published_date"
                ),

                author=chunk.get(
                    "author"
                ),

            )

        )


    return result



# ==========================================================
# Main pipeline
# ==========================================================


async def get_context(
    query: str,
) -> AgentContext:
    """
    Build context for agent.

    Pipeline:

        Qdrant cache
            |
            v
        Exa documents
            |
            v
        Chunking
            |
            v
        Chunk filtering
            |
            v
        Reranking
            |
            v
        Qdrant memory
    """


    query = query.strip()


    if not query:

        raise ValueError(
            "Query cannot be empty."
        )


    await init_web_search()



    logger.info(
        "Building context for query='%s'",
        query,
    )



    try:


        # ==================================================
        # 1. Semantic cache
        # ==================================================

        cached = await search(

            query=query,

            limit=CACHE_TOP_K,

            score_threshold=SIMILARITY_THRESHOLD,

        )


        if cached:

            logger.info(
                "Semantic cache hit. chunks=%d",
                len(cached),
            )


            return AgentContext(

                text=_format_context(
                    cached
                ),

                sources=_extract_sources(
                    cached
                ),

            )



        logger.info(
            "Semantic cache miss."
        )



        # ==================================================
        # 2. Exa retrieval
        # ==================================================

        documents = await search_exa(
            query
        )


        logger.info(
            "Received %d Exa documents.",
            len(documents),
        )



        # ==================================================
        # 3. Chunking
        # ==================================================

        chunks = chunk_documents(
            documents
        )


        if not chunks:

            raise RuntimeError(
                "No chunks generated."
            )


        logger.info(
            "Generated chunks=%d",
            len(chunks),
        )



        # ==================================================
        # 4. Chunk filtering
        # ==================================================

        filtered_chunks = filter_chunks(

            chunks,

            query,

        )


        if not filtered_chunks:

            raise RuntimeError(
                "No chunks after filtering."
            )


        logger.info(
            "Filtered chunks=%d",
            len(filtered_chunks),
        )



        filtered_chunks = filtered_chunks[
            :MAX_RERANK_CHUNKS
        ]


        logger.info(
            "Chunks sent to reranker=%d",
            len(filtered_chunks),
        )



        # ==================================================
        # 5. Reranking
        # ==================================================

        reranker = get_reranker()


        ranked_chunks = await reranker.rerank(

            query=query,

            chunks=filtered_chunks,

            top_k=RERANK_OUTPUT_K,

        )


        if not ranked_chunks:

            raise RuntimeError(
                "Reranker returned no chunks."
            )


        logger.info(
            "Selected ranked chunks=%d",
            len(ranked_chunks),
        )



        # ==================================================
        # 6. Persistent semantic memory
        # ==================================================

        await add_chunks(
            ranked_chunks
        )


        logger.info(
            "Stored %d chunks into Qdrant.",
            len(ranked_chunks),
        )



        # ==================================================
        # 7. Agent context
        # ==================================================

        return AgentContext(

            text=_format_context(
                ranked_chunks
            ),

            sources=_extract_sources(
                ranked_chunks
            ),

        )



    except Exception:

        logger.exception(
            "Failed building context for query='%s'.",
            query,
        )

        raise
