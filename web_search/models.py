"""
Web search domain models.

Contains internal data contracts between:
- Exa retrieval layer;
- filtering layer;
- chunking layer;
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

    This object contains full available context.
    No chunking or embedding happens here.
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
        description="Full cleaned document content from Exa.",
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
# Filter result model
# ==========================================================


class FilteredDocument(BaseModel):
    """
    Document after cheap preprocessing.

    The filter layer removes:
    - irrelevant documents;
    - empty content;
    - obvious duplicates.

    It does not use embeddings.
    """

    query: str

    title: str = "Untitled"

    url: str = ""

    text: str = ""

    provider: str = "exa"

    published_date: str | None = None

    author: str | None = None

    relevance_score: float | None = Field(
        default=None,
        description="Cheap heuristic relevance score.",
    )


# ==========================================================
# Chunk model
# ==========================================================


class DocumentChunk(BaseModel):
    """
    Text chunk prepared for embedding.

    Generated only after cheap filtering.
    """

    id: str

    query: str

    title: str = "Untitled"

    url: str = ""

    text: str

    provider: str = "exa"

    chunk_index: int = 0

    published_date: str | None = None

    author: str | None = None


# ==========================================================
# Ranked chunk model
# ==========================================================


class RankedChunk(DocumentChunk):
    """
    Chunk after reranking.

    Used before final Qdrant persistence.
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
