"""
Web search orchestration layer.

Baseline Architecture v1

Pipeline

User Query
    │
    ▼
Query Preprocessing
    │
    ▼
Qdrant Hybrid Retrieval
    │
    ├─────────────── Cache Hit
    │                     │
    │                     ▼
    │               Cloudflare Reranker
    │
    └─────────────── Cache Miss
                          │
                          ▼
                    Exa Search
                          │
                    Document Normalize
                          │
                        Chunker
                          │
                     Filter Chunks
                          │
                 Embedding Retrieval
                          │
                          ▼
                  Cloudflare Reranker
                          │
                          ▼
               Extractive Compression
                          │
                          ▼
                 Context Optimization
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          Qdrant Memory        Agent Context
"""

from __future__ import annotations

import asyncio
import logging

from web_search.chunker import chunk_documents
from web_search.compression import compress_chunks
from web_search.context_optimizer import optimize_context
from web_search.embedding_retrieval import retrieve_by_embeddings
from web_search.exa import search_exa
from web_search.filter import filter_chunks
from web_search.models import (
    AgentContext,
    RankedChunk,
)
from web_search.qdrant_store import (
    add_chunks,
    cleanup_old_chunks,
    hybrid_search,
)
from web_search.reranker import get_reranker

logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================

FILTER_TOP_K = 10

EMBEDDING_TOP_K = 8

RERANK_TOP_K = 5


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

        logger.info("Initializing web search.")

        await cleanup_old_chunks(days=30)

        _initialized = True

        logger.info("Web search initialized.")


# ==========================================================
# Query preprocessing
# ==========================================================


def preprocess_query(
    query: str,
) -> str:
    """
    Lightweight query preprocessing.

    Future:

    - language detection
    - intent extraction
    - query expansion
    """

    return " ".join(
        query.strip().split()
    )


# ==========================================================
# Cache Miss Pipeline
# ==========================================================


async def _build_from_exa(
    query: str,
) -> list[RankedChunk]:
    """
    Execute retrieval pipeline after cache miss.
    """

    logger.info(
        "Starting Exa retrieval pipeline."
    )

    documents = await search_exa(query)

    chunks = chunk_documents(documents)

    filtered = filter_chunks(
        chunks=chunks,
        query=query,
        top_k=FILTER_TOP_K,
    )

    semantic = await retrieve_by_embeddings(
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
# Cache Hit Pipeline
# ==========================================================


async def _build_from_cache(
    query: str,
):
    """
    Hybrid retrieval pipeline.
    """

    cached = await hybrid_search(
        query=query,
        limit=EMBEDDING_TOP_K,
    )

    if not cached:
        return None

    logger.info(
        "Hybrid cache hit (%d chunks).",
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
    Build optimized context for the agent.
    """

    await init_web_search()

    query = preprocess_query(query)

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    logger.info(
        "Building context for '%s'.",
        query,
    )

    #
    # Hybrid retrieval
    #

    ranked = await _build_from_cache(
        query,
    )

    #
    # Cache miss
    #

    if ranked is None:

        logger.info(
            "Cache miss."
        )

        ranked = await _build_from_exa(
            query,
        )

        #
        # Store only final ranked chunks.
        #

        await add_chunks(
            ranked,
        )

    #
    # Compression
    #

    compressed = await compress_chunks(
        query=query,
        chunks=ranked,
    )

    #
    # Final context optimization
    #

    context = optimize_context(
        query=query,
        chunks=compressed,
    )

    logger.info(
        "Context ready. Sources=%d",
        len(context.sources),
    )

    return context
