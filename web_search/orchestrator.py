"""
Web search orchestration layer.
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


from web_search.query_normalizer import (
    normalize_query,
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
    Initialize search subsystem.

    Tasks:

    - cleanup old memory;
    - initialize once.
    """


    global _initialized


    if _initialized:

        return



    async with _lock:


        if _initialized:

            return



        logger.info(
            "Initializing web search."
        )


        await cleanup_old_chunks(
            days=CLEANUP_DAYS,
        )


        _initialized = True



        logger.info(
            "Web search initialized."
        )




# ==========================================================
# Exa pipeline
# ==========================================================


async def _retrieve_from_web(
    query: str,
) -> list[RankedChunk]:
    """
    Cache miss retrieval pipeline.

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


    documents = await search_exa(
        query
    )


    if not documents:

        return []



    chunks = chunk_documents(
        documents
    )


    if not chunks:

        return []



    filtered = filter_chunks(

        chunks=chunks,

        query=query,

    )


    if not filtered:

        return []



    semantic = await retrieve_by_embedding_similarity(

        query=query,

        chunks=filtered,

        top_k=EMBEDDING_TOP_K,

    )


    if not semantic:

        return []



    reranker = get_reranker()



    ranked = await reranker.rerank(

        query=query,

        chunks=semantic,

        top_k=RERANK_TOP_K,

    )


    return ranked




# ==========================================================
# Qdrant cache pipeline
# ==========================================================


async def _retrieve_from_memory(
    query: str,
) -> list[RankedChunk] | None:
    """
    Qdrant hybrid retrieval.

    Dense vector
        +
    BM25 sparse

        |
        v

       RRF

        |
        v

    Reranker
    """


    cached = await hybrid_search(

        query=query,

        limit=CACHE_TOP_K,

    )


    if not cached:

        return None



    logger.info(

        "Qdrant cache hit chunks=%d",

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
# Storage
# ==========================================================


async def _store_memory(
    chunks: list[RankedChunk],
) -> None:
    """
    Store retrieval memory in Qdrant.

    Stores:

    - chunks
    - metadata
    - scores
    - timestamps
    """


    if not chunks:

        return



    payload = [

        chunk.model_dump()

        for chunk in chunks

    ]


    await add_chunks(
        payload
    )




# ==========================================================
# Public API
# ==========================================================


async def get_context(
    query: str,
) -> AgentContext:
    """
    Main web search pipeline entrypoint.
    """


    await init_web_search()



    normalized_query = normalize_query(
        query
    )



    if not normalized_query:

        raise ValueError(
            "Empty query."
        )



    logger.info(
        "Building context."
    )



    ranked = await _retrieve_from_memory(

        normalized_query

    )



    if ranked is None:


        logger.info(
            "Cache miss. Running Exa."
        )


        ranked = await _retrieve_from_web(

            normalized_query

        )


        await _store_memory(
            ranked
        )



    if not ranked:

        return AgentContext()



    compressed = await compress_chunks(

        query=normalized_query,

        chunks=ranked,

    )



    if not compressed:

        return AgentContext()



    context = optimize_context(

        query=normalized_query,

        chunks=compressed,

    )



    logger.info(

        "Context ready sources=%d",

        len(context.sources),

    )


    return context
