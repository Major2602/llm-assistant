"""
Web search orchestration layer.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from datetime import datetime, UTC

from web_search.chunker import chunk_documents
from web_search.compression import compress_chunks
from web_search.context_optimizer import optimize_context
from web_search.embedding_retrieval import retrieve_by_embedding_similarity
from web_search.exa import search_exa
from web_search.filter import filter_chunks
from web_search.models import (
    AgentContext,
    DocumentChunk,
    EmbeddedChunk,
    HybridRetrievalResult,
    NormalizedQuery,
    PipelineMetadata,
    RankedChunk,
    RetrievalDecision,
)
from web_search.qdrant_store import (
    cleanup_old_chunks,
    hybrid_search,
    store_chunks,
)
from web_search.query_normalizer import preprocess_query
from web_search.reranker import get_reranker

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
# Qdrant result adaptation
# ==========================================================


def adapt_hybrid_results(
    results: list[HybridRetrievalResult],
) -> list[EmbeddedChunk]:
    """
    Adapt Qdrant memory results for reranker pipeline.

    Qdrant does not contain:
    - keyword_score
    - quality_score
    - length_score
    - filter_score
    - similarity_score

    These fields belong only to fresh retrieval pipeline.
    """

    return [

        EmbeddedChunk(

            **item.chunk.model_dump(),

            rrf_score=item.rrf_score,

        )

        for item in results

    ]



# ==========================================================
# Memory
# ==========================================================


async def _retrieve_from_memory(
    query: NormalizedQuery,
) -> tuple[RetrievalDecision, list[RankedChunk]]:
    """
    Retrieve cached chunks from Qdrant.
    """

    cached = await hybrid_search(
        query=query,
        limit=CACHE_TOP_K,
    )

    decision = RetrievalDecision(
        cache_hit=bool(cached),
        results=cached
    )

    if not cached:
        
        return decision, []

    logger.info(
        "Cache hit chunks=%d",
        len(cached),
    )

    semantic_chunks = adapt_hybrid_results(
        cached
    )
    
    reranker = get_reranker()

    ranked = await reranker.rerank(
        query=query.normalized,
        chunks=semantic_chunks,
        top_k=RERANK_TOP_K,
    )

    return decision, ranked


# ==========================================================
# Web Retrieval
# ==========================================================


async def _retrieve_from_web(
    query: NormalizedQuery,
    metadata: PipelineMetadata,
) -> tuple[list[RankedChunk], list[DocumentChunk]]:
    """
    Execute complete web retrieval pipeline.
    """

    documents = await search_exa(
        query.normalized,
    )

    if not documents:
        return [], []

    all_chunks = chunk_documents(
        documents,
    )

    metadata.chunks_created = len(all_chunks)
    
    if not all_chunks:
        return [], []

    filtered = filter_chunks(
        chunks=all_chunks,
        query=query.normalized,
    )

    metadata.chunks_filtered = len(filtered)

    if not filtered:
        return [], all_chunks

    embedded = await retrieve_by_embedding_similarity(
        query=query.normalized,
        chunks=filtered,
        top_k=EMBEDDING_TOP_K,
    )

    metadata.chunks_embedded = len(embedded)

    if not embedded:
        return [], all_chunks

    reranker = get_reranker()

    ranked = await reranker.rerank(
        query=query.normalized,
        chunks=embedded,
        top_k=RERANK_TOP_K,
    )

    return ranked, all_chunks


# ==========================================================
# Storage
# ==========================================================


async def _store_memory(
    chunks: list[DocumentChunk],
) -> None:
    """
    Store chunks in Qdrant.
    """

    if not chunks:
        return

    try:
        
        await store_chunks(
            chunks
        )

        logger.info("Memory stored chunks=%d", 
                    len(chunks)
                   )
    except Exception: 

        logger.exception(
            "Failed to store chunks in memory"
        )


# ==========================================================
# Public API
# ==========================================================


async def get_context(
    query: str,
) -> AgentContext:
    """
    Build final AgentContext.
    """

    await init_web_search()

    normalized_query = preprocess_query(
        query,
    )

    metadata = PipelineMetadata(
        request_id=str(uuid.uuid4()),
        query=normalized_query.normalized,
        created_at=int(datetime.now(UTC).timestamp())
    )

    logger.info(
        "Building context."
    )

    decision, ranked = await _retrieve_from_memory(
        normalized_query,
    )

    metadata.cache_hit = decision.cache_hit
    metadata.chunks_ranked = len(ranked)

    if not decision.cache_hit:

        logger.info(
            "Cache miss. Running web retrieval."
        )

        ranked, chunks = await _retrieve_from_web(
            normalized_query,
            metadata
        )

        await _store_memory(
            chunks,
        )

    if not ranked:

        logger.info(
            "No relevant documents found."
        )

        return AgentContext(
            metadata=metadata
        )

    compressed = await compress_chunks(
        query=normalized_query.normalized,
        chunks=ranked,
    )

    metadata.chunks_compressed = len(compressed)

    if not compressed:
        return AgentContext(
            metadata=metadata
        )

    context = optimize_context(
        query=normalized_query.normalized,
        chunks=compressed,
    )

    metadata.sources_count = len(context.sources)

    logger.info(
        "Context ready sources=%d",
        len(context.sources),
    )

    return AgentContext(
        **context.model_dump(),
        metadata=metadata
    )
