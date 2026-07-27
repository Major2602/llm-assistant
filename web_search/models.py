"""
Web search pipeline contracts.

Architecture:

USER QUERY
    |
    v
QueryNormalizer
    |
    v
Qdrant Hybrid Retrieval
    |
    +-------------+
    |             |
 CACHE HIT    CACHE MISS
                  |
                  v
              Exa Search
                  |
                  v
              Document
                  |
                  v
              Chunk
                  |
                  v
              FilteredChunk
                  |
                  v
              EmbeddedChunk
                  |
                  v
              RankedChunk
                  |
                  v
              CompressedChunk
                  |
                  v
              ContextDocument
                  |
                  v
              AgentContext


This module contains ONLY:

- pydantic contracts;
- pipeline data structures;
- shared metadata definitions.

This module does NOT:

- call APIs;
- perform ranking;
- access databases;
- contain business logic.
"""


from __future__ import annotations


from typing import Any


from pydantic import BaseModel, Field



# ==========================================================
# Query
# ==========================================================


class NormalizedQuery(BaseModel):
    """
    User query after normalization.

    Produced by:

        query_normalizer.py

    Used by:

        retrieval pipeline
    """

    original: str

    normalized: str

    language: str = "en"



# ==========================================================
# Source metadata
# ==========================================================


class Source(BaseModel):
    """
    Citation source metadata.

    Used by:

    - compression
    - context
    - final answer
    """

    title: str = "Untitled"

    url: str = ""

    provider: str = "unknown"

    author: str | None = None

    published_date: str | None = None



# ==========================================================
# External documents
# ==========================================================


class WebDocument(BaseModel):
    """
    Normalized external document.

    Produced by:

        exa.py
    """

    title: str = "Untitled"

    url: str = ""

    text: str

    source: Source = Field(
        default_factory=Source
    )

    created_at: int | None = None



# ==========================================================
# Chunk pipeline
# ==========================================================


class DocumentChunk(BaseModel):
    """
    Base retrieval chunk.

    Produced by:

        chunker.py
    """

    id: str

    text: str

    source: Source = Field(
        default_factory=Source
    )

    chunk_index: int = 0

    created_at: int | None = None

    last_access: int | None = None



class FilteredChunk(DocumentChunk):
    """
    Chunk after heuristic filtering.

    Added:

    - keyword score
    - quality score
    """

    keyword_score: float = 0.0

    quality_score: float = 0.0

    filter_score: float = 0.0



class EmbeddedChunk(FilteredChunk):
    """
    Chunk after dense embedding retrieval.
    """

    similarity_score: float = 0.0



class RankedChunk(EmbeddedChunk):
    """
    Chunk after Cloudflare reranking.
    """

    rerank_score: float = 0.0



class CompressedChunk(RankedChunk):
    """
    Chunk after extractive compression.
    """

    compressed_text: str = ""

    compression_ratio: float = 1.0



# ==========================================================
# Vector contracts
# ==========================================================


class DenseVector(BaseModel):
    """
    Dense embedding vector.
    """

    values: list[float]



class SparseVector(BaseModel):
    """
    BM25 sparse vector.

    Stored by Qdrant.
    """

    indices: list[int]

    values: list[float]



# ==========================================================
# Retrieval
# ==========================================================


class HybridRetrievalResult(BaseModel):
    """
    Qdrant hybrid search result.

    Contains:

    - dense score
    - sparse score
    - fusion score
    """

    chunk: DocumentChunk

    dense_score: float = 0.0

    sparse_score: float = 0.0

    fusion_score: float = 0.0



# ==========================================================
# Context
# ==========================================================


class ContextDocument(BaseModel):
    """
    Final document supplied to LLM.
    """

    text: str

    source: Source

    relevance_score: float = 0.0



class OptimizedContext(BaseModel):
    """
    Final context package.
    """

    query: str

    documents: list[ContextDocument] = Field(
        default_factory=list
    )

    total_tokens: int = 0

    citation_map: dict[str, Source] = Field(
        default_factory=dict
    )



class AgentContext(BaseModel):
    """
    Final LLM input package.
    """

    text: str = ""

    sources: list[Source] = Field(
        default_factory=list
    )

    optimized_context: OptimizedContext | None = None



# ==========================================================
# Pipeline helpers
# ==========================================================


class PipelineScore(BaseModel):
    """
    Unified score container.
    """

    keyword: float = 0.0

    quality: float = 0.0

    similarity: float = 0.0

    rerank: float = 0.0

    fusion: float = 0.0



def model_to_dict(
    model: BaseModel,
) -> dict[str, Any]:
    """
    Unified serialization helper.
    """

    return model.model_dump()
