"""
Web search domain models.

Contains internal data contracts between:
- Exa retrieval layer;
- chunking layer;
- chunk filtering layer;
- embedding layer;
- reranking layer;
- Qdrant storage layer;
- agent/UI layer.

No business logic is allowed here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field



# ==========================================================
# Source model
# ==========================================================


class Source(BaseModel):
    """
    External information source.

    Represents a document/page returned by:
    - Exa;
    - Qdrant semantic memory.
    """


    title: str = Field(
        default="Untitled source",
        description="Source title.",
    )


    url: str = Field(
        default="",
        description="Source URL.",
    )


    provider: str | None = Field(
        default=None,
        description="Source provider.",
    )


    score: float | None = Field(
        default=None,
        description="Final relevance score.",
    )


    published_date: str | None = Field(
        default=None,
        description="Publication date if available.",
    )


    author: str | None = Field(
        default=None,
        description="Author if available.",
    )



# ==========================================================
# Web document model
# ==========================================================


class WebDocument(BaseModel):
    """
    Raw document returned by Exa.

    Contains full document text.

    No:
    - filtering;
    - chunking;
    - embedding;
    - ranking

    happens here.
    """


    query: str = Field(
        description="Original user search query.",
    )


    title: str = Field(
        default="Untitled",
        description="Document title.",
    )


    url: str = Field(
        default="",
        description="Document URL.",
    )


    text: str = Field(
        default="",
        description="Full cleaned document content.",
    )


    provider: str = Field(
        default="exa",
        description="Document provider.",
    )


    published_date: str | None = Field(
        default=None,
        description="Publication date.",
    )


    author: str | None = Field(
        default=None,
        description="Document author.",
    )



# ==========================================================
# Chunk model
# ==========================================================


class DocumentChunk(BaseModel):
    """
    Text chunk created from WebDocument.

    Chunking happens immediately after Exa retrieval.

    These objects are candidates for:
    - heuristic filtering;
    - reranking;
    - vector storage.
    """


    id: str = Field(
        description="Unique chunk identifier.",
    )


    query: str = Field(
        description="Original user query.",
    )


    title: str = Field(
        default="Untitled",
        description="Original document title.",
    )


    url: str = Field(
        default="",
        description="Original document URL.",
    )


    text: str = Field(
        description="Chunk content.",
    )


    provider: str = Field(
        default="exa",
        description="Chunk provider.",
    )


    chunk_index: int = Field(
        default=0,
        description="Chunk position inside source document.",
    )


    published_date: str | None = Field(
        default=None,
        description="Publication date.",
    )


    author: str | None = Field(
        default=None,
        description="Document author.",
    )


    created_at: int | None = Field(
        default=None,
        description="Creation timestamp.",
    )


    last_access: int | None = Field(
        default=None,
        description="Last access timestamp.",
    )



# ==========================================================
# Filtered chunk model
# ==========================================================


class FilteredChunk(DocumentChunk):
    """
    Chunk after cheap heuristic filtering.

    Filter layer removes:
    - irrelevant chunks;
    - low information chunks;
    - duplicates.

    Does not use:
    - embeddings;
    - reranker;
    - LLM.
    """


    filter_score: float = Field(
        default=0.0,
        description="Cheap heuristic relevance score.",
    )



# ==========================================================
# Ranked chunk model
# ==========================================================


class RankedChunk(FilteredChunk):
    """
    Chunk after semantic reranking.

    Final object before Qdrant persistence.
    """


    rerank_score: float = Field(
        default=0.0,
        description="Cloudflare reranker relevance score.",
    )



# ==========================================================
# Agent context
# ==========================================================


class AgentContext(BaseModel):
    """
    Context returned to the agent tool.

    Contains:

    text:
        formatted information for LLM reasoning.

    sources:
        metadata preserved for UI citations.
    """


    text: str = Field(
        default="",
        description="Context text provided to LLM.",
    )


    sources: list[Source] = Field(
        default_factory=list,
        description="Sources used to build context.",
    )
