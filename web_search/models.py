"""
Web search domain models.

Architecture contract:

User Query
    |
    v
Query Preprocessing
    |
    v
Qdrant Hybrid Retrieval
    |
    +---------------- Cache Hit
    |
    +---------------- Cache Miss
                         |
                         v
                    Exa Search
                         |
                         v
                Document Normalize
                         |
                         v
                      Chunking
                         |
                         v
                      Filtering
                         |
                         v
              Embedding Similarity
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
                         v
                    AgentContext


This module contains only:

- pipeline data contracts;
- validation models;
- shared field definitions.

This module does NOT:

- call external APIs;
- contain ranking logic;
- contain storage logic;
- contain retrieval logic;
- contain business rules.
"""

from __future__ import annotations


from typing import Any

from pydantic import BaseModel, Field


# ==========================================================
# Query
# ==========================================================


class SearchQuery(BaseModel):
    """
    Normalized user query.

    Produced by:

        query_preprocessor.py

    Used by:

        retrieval pipeline
    """

    original: str

    normalized: str

    language: str = "en"

    intent: str = "general"

    expanded_queries: list[str] = Field(
        default_factory=list
    )


# ==========================================================
# Source metadata
# ==========================================================


class Source(BaseModel):
    """
    External citation source.

    Used by:

    - UI
    - Agent
    - Citations
    """

    title: str = "Untitled source"

    url: str = ""

    provider: str | None = None

    author: str | None = None

    published_date: str | None = None



# ==========================================================
# Documents
# ==========================================================


class WebDocument(BaseModel):
    """
    Normalized external document.

    Produced by:

        Exa

    Consumed by:

        chunker.py
    """

    query: str

    title: str = "Untitled"

    url: str = ""

    text: str = ""

    provider: str = "exa"

    author: str | None = None

    published_date: str | None = None



# ==========================================================
# Base chunk
# ==========================================================


class DocumentChunk(BaseModel):
    """
    Atomic retrieval unit.

    Produced by:

        chunker.py


    Contains all persistent metadata.
    """

    id: str

    query: str

    title: str = "Untitled"

    url: str = ""

    text: str

    provider: str = "exa"

    chunk_index: int = 0

    author: str | None = None

    published_date: str | None = None


    # Unix timestamps.

    created_at: int | None = None

    last_access: int | None = None



# ==========================================================
# Filtering
# ==========================================================


class FilteredChunk(DocumentChunk):
    """
    Chunk after heuristic filtering.

    Produced by:

        filter.py

    Added:

        filter_score
    """

    filter_score: float = 0.0



# ==========================================================
# Embedding retrieval
# ==========================================================


class EmbeddingChunk(FilteredChunk):
    """
    Chunk after dense similarity retrieval.

    Produced by:

        embedding_retrieval.py

    Added:

        embedding_score
    """

    embedding_score: float = 0.0



# ==========================================================
# Hybrid retrieval
# ==========================================================


class HybridChunk(DocumentChunk):
    """
    Chunk returned from Qdrant hybrid retrieval.

    Contains:

    - dense score
    - sparse score
    - fusion score
    """

    dense_score: float = 0.0

    sparse_score: float = 0.0

    fusion_score: float = 0.0



# ==========================================================
# Reranking
# ==========================================================


class RankedChunk(EmbeddingChunk):
    """
    Chunk after Cloudflare reranker.

    Produced by:

        reranker.py

    Added:

        rerank_score
    """

    rerank_score: float = 0.0



# ==========================================================
# Compression
# ==========================================================


class CompressedChunk(RankedChunk):
    """
    Chunk after extractive compression.

    Produced by:

        compression.py


    Keeps:

    - source metadata
    - ranking scores
    - compressed text
    """

    compressed_text: str = ""

    compression_ratio: float = 1.0



# ==========================================================
# Optimized context
# ==========================================================


class ContextDocument(BaseModel):
    """
    Final context document for LLM.

    Prepared by:

        context_optimizer.py
    """

    text: str

    source: Source

    relevance_score: float = 0.0



class OptimizedContext(BaseModel):
    """
    Final structured context package.

    Used before LLM generation.
    """

    query: str

    documents: list[ContextDocument] = Field(
        default_factory=list
    )

    total_tokens: int = 0

    citation_map: dict[str, Source] = Field(
        default_factory=dict
    )



# ==========================================================
# Agent context
# ==========================================================


class AgentContext(BaseModel):
    """
    Final context consumed by agent layer.

    Contains:

    - LLM-ready text
    - citation sources
    - optional structured context
    """

    text: str = ""

    sources: list[Source] = Field(
        default_factory=list
    )

    optimized_context: OptimizedContext | None = None



# ==========================================================
# Generic pipeline helpers
# ==========================================================


class EmbeddingResult(BaseModel):
    """
    Generic embedding response contract.

    Used only when raw vectors need transport.
    """

    vector: list[float]



class RerankResult(BaseModel):
    """
    Generic reranker result contract.
    """

    score: float

    index: int



class ChunkScore(BaseModel):
    """
    Shared ranking score container.
    """

    filter_score: float | None = None

    embedding_score: float | None = None

    rerank_score: float | None = None



# ==========================================================
# Compatibility helper
# ==========================================================


def model_to_dict(
    model: BaseModel,
) -> dict[str, Any]:
    """
    Unified model serialization.

    Prevents direct dictionary construction
    across pipeline modules.
    """

    return model.model_dump()
