"""
Web search domain models.

Contracts between:

Exa retrieval
      |
      v
Document normalization
      |
      v
Chunking
      |
      v
Filtering
      |
      v
Embedding retrieval
      |
      v
Reranking
      |
      v
Compression
      |
      v
Context optimization
      |
      v
Agent

This module contains only:
- data models;
- pipeline contracts.

No:
- API calls;
- storage logic;
- ranking logic;
- business rules.
"""

from __future__ import annotations


from pydantic import BaseModel, Field


# ==========================================================
# Source
# ==========================================================


class Source(BaseModel):
    """
    External citation source.

    Used by:
    - UI;
    - agent;
    - citations.
    """

    title: str = Field(
        default="Untitled source"
    )

    url: str = Field(
        default=""
    )

    provider: str | None = Field(
        default=None
    )

    author: str | None = Field(
        default=None
    )

    published_date: str | None = Field(
        default=None
    )



# ==========================================================
# Exa document
# ==========================================================


class WebDocument(BaseModel):
    """
    Normalized document from Exa.

    Stage:

        Exa
          |
          v
        WebDocument
    """

    query: str


    title: str = Field(
        default="Untitled"
    )


    url: str = Field(
        default=""
    )


    text: str = Field(
        default=""
    )


    provider: str = Field(
        default="exa"
    )


    author: str | None = None


    published_date: str | None = None



# ==========================================================
# Chunk
# ==========================================================


class DocumentChunk(BaseModel):
    """
    Semantic text chunk.

    Created by chunker.py.

    Contains original metadata.
    """


    id: str


    query: str


    title: str = (
        "Untitled"
    )


    url: str = ""


    text: str


    provider: str = (
        "exa"
    )


    chunk_index: int = 0


    author: str | None = None


    published_date: str | None = None


    created_at: int | None = None


    last_access: int | None = None



# ==========================================================
# Filtered chunk
# ==========================================================


class FilteredChunk(
    DocumentChunk
):
    """
    Chunk after heuristic filtering.

    filter.py output.

    Ranking signals:

    - keyword relevance
    - quality
    - duplication
    """


    filter_score: float = 0.0



# ==========================================================
# Embedding retrieval
# ==========================================================


class EmbeddingResult(
    FilteredChunk
):
    """
    Chunk after dense similarity retrieval.

    embedding_retrieval.py output.

    Used after:

        filter_chunks()

        TOP 10

            |

        embedding similarity

            |

        TOP 5-8
    """


    similarity_score: float = 0.0



# ==========================================================
# Hybrid retrieval
# ==========================================================


class HybridSearchResult(
    DocumentChunk
):
    """
    Result returned by Qdrant hybrid retrieval.

    Contains:

    Dense vector score
    +
    BM25 sparse score
    +
    fusion score
    """


    dense_score: float = 0.0


    sparse_score: float = 0.0


    fusion_score: float = 0.0



# ==========================================================
# Ranked chunk
# ==========================================================


class RankedChunk(
    EmbeddingResult
):
    """
    Final ranking output.

    Produced by:

        Cloudflare reranker

    """

    rerank_score: float = 0.0



# ==========================================================
# Compression
# ==========================================================


class CompressedChunk(
    RankedChunk
):
    """
    Chunk after extractive compression.

    compression.py output.


    Original:

        1200 tokens


    Compressed:

        200-300 tokens


    Keeps:
    - relevance;
    - citation metadata.
    """


    compressed_text: str = ""


    compression_ratio: float = 1.0



# ==========================================================
# Context optimization
# ==========================================================


class ContextDocument(BaseModel):
    """
    Final optimized context unit.

    Prepared before LLM generation.
    """


    text: str


    source: Source


    relevance_score: float = 0.0



class OptimizedContext(BaseModel):
    """
    Final context package.

    Sent to Groq LLM.
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
    Context consumed by agent layer.

    Contains:

    text:
        LLM-ready context.

    sources:
        citations for UI.
    """


    text: str = ""


    sources: list[Source] = Field(
        default_factory=list
    )


    optimized_context: OptimizedContext | None = None
