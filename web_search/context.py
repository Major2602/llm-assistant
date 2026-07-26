"""
Web search orchestration layer.

Final Architecture:

USER QUERY
    |
    v
Query preprocessing
    |
    v
Qdrant Hybrid Retrieval
    |
    +----------------+
    |                |
    v                v
CACHE HIT        CACHE MISS
    |                |
    |                v
    |            Exa Search
    |                |
    |            Normalize
    |                |
    |            Chunker
    |                |
    |            Filter
    |                |
    |            Embedding Retrieval
    |                |
    +-------+--------+
            |
            v
    Cloudflare Reranker
            |
            v
    Extractive Compression
            |
            v
    Context Optimization
            |
       +----+----+
       |         |
       v         v
  Qdrant Memory Agent Context
"""

from __future__ import annotations


import asyncio
import logging


from web_search.chunker import (
    chunk_documents,
)

from web_search.compression import (
    compress_chunks,
)

from web_search.context_optimizer import (
    optimize_context,
)

from web_search.embedding_retrieval import (
    retrieve_by_embedding_similarity,
)

from web_search.exa import (
    search_exa,
)

from web_search.filter import (
    filter_chunks,
)

from web_search.models import (
    AgentContext,
    RankedChunk,
)

from web_search.qdrant_store import (
    add_chunks,
    cleanup_old_chunks,
    hybrid_search,
)

from web_search.query_preprocessor import (
    preprocess_query,
)

from web_search.reranker import (
    get_reranker,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


CACHE_TOP_K = 10

EMBEDDING_TOP_K = 8

RERANK_TOP_K = 5

CLEANUP_DAYS = 30



# ==========================================================
# Initialization
# ==========================================================


_initialized = False

_lock = asyncio.Lock()



async def init_web_search() -> None:
    """
    Initialize web search subsystem.

    Performs:
    - cleanup expired memory;
    - one-time initialization.
    """


    global _initialized


    if _initialized:

        return



    async with _lock:


        if _initialized:

            return



        logger.info(
            "Initializing web search subsystem."
        )


        await cleanup_old_chunks(
            days=CLEANUP_DAYS,
        )


        _initialized = True


        logger.info(
            "Web search initialized."
        )



# ==========================================================
# Cache miss pipeline
# ==========================================================


async def _build_from_exa(
    query: str,
) -> list[RankedChunk]:
    """
    Full retrieval pipeline after cache miss.

    Pipeline:

        Exa

          |

        Chunker

          |

        Filter

          |

        Embedding similarity

          |

        Reranker

    """


    logger.info(
        "Running Exa retrieval pipeline."
    )



    documents = await search_exa(
        query,
    )



    chunks = chunk_documents(
        documents,
    )



    filtered = filter_chunks(
        chunks=chunks,
        query=query,
    )



    semantic = await retrieve_by_embedding_similarity(
        query=query,
        chunks=filtered,
        top_k=EMBEDDING_TOP_K,
    )



    reranker = get_reranker()



    ranked = await reranker.rerank(
        query=query,
        chunks=semantic,
        top_k=RERANK_TOP_K,
    )



    return ranked



# ==========================================================
# Cache hit pipeline
# ==========================================================


async def _build_from_cache(
    query: str,
) -> list[RankedChunk] | None:
    """
    Retrieve from Qdrant hybrid memory.

    Pipeline:

        Dense search

        +

        BM25 search

        +

        RRF fusion

        |

        Reranker

    """


    cached = await hybrid_search(
        query=query,
        limit=CACHE_TOP_K,
    )



    if not cached:

        return None



    logger.info(
        "Hybrid retrieval cache hit. chunks=%d",
        len(cached),
    )



    reranker = get_reranker()



    ranked = await reranker.rerank(
        query=query,
        chunks=cached,
        top_k=RERANK_TOP_K,
    )



    return ranked



# ==========================================================
# Public API
# ==========================================================


async def get_context(
    query: str,
) -> AgentContext:
    """
    Build final optimized agent context.

    Pipeline:

        Query preprocessing

            |

        Hybrid retrieval

            |

        Reranking

            |

        Compression

            |

        Context optimization

    """


    await init_web_search()



    search_query = preprocess_query(
        query,
    )


    normalized_query = search_query.normalized



    if not normalized_query:

        raise ValueError(
            "Query cannot be empty."
        )



    logger.info(
        "Building context. intent=%s language=%s",
        search_query.intent,
        search_query.language,
    )



    ranked = await _build_from_cache(
        normalized_query,
    )



    if ranked is None:


        logger.info(
            "Cache miss. Running Exa pipeline."
        )


        ranked = await _build_from_exa(
            normalized_query,
        )


        await add_chunks(
            [
                chunk.model_dump()
                if hasattr(
                    chunk,
                    "model_dump",
                )
                else chunk

                for chunk in ranked
            ],
        )



    compressed = await compress_chunks(
        query=normalized_query,
        chunks=ranked,
    )



    context = optimize_context(
        compressed,
    )



    logger.info(
        "Context generated. sources=%d",
        len(context.sources),
    )



    return context
