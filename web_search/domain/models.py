"""
Web search domain models.

Pure business contracts.
No infrastructure dependencies.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ==========================================================
# Metadata
# ==========================================================


class PipelineMetadata(BaseModel):
    """Runtime pipeline metadata."""

    request_id: str | None = None
    created_at: int | None = None
    query: str | None = None

    cache_hit: bool = False

    documents_found: int = 0
    chunks_created: int = 0
    chunks_filtered: int = 0
    chunks_embedded: int = 0
    chunks_ranked: int = 0
    chunks_compressed: int = 0

    sources_count: int = 0

    pipeline_errors: list[str] = Field(
        default_factory=list
    )


# ==========================================================
# Query
# ==========================================================


class NormalizedQuery(BaseModel):
    """Normalized user query."""

    original: str

    normalized: str

    created_at: int | None = None


# ==========================================================
# Sources
# ==========================================================


class Source(BaseModel):
    """External source metadata."""

    title: str = "Untitled"

    url: str = ""

    provider: str = "unknown"

    author: str | None = None

    published_date: str | None = None

    domain: str | None = None


# ==========================================================
# Documents
# ==========================================================


class WebDocument(BaseModel):
    """Normalized external document."""

    id: str

    text: str

    source: Source = Field(
        default_factory=Source
    )

    created_at: int | None = None

    last_access: int | None = None


# ==========================================================
# Chunk pipeline
# ==========================================================


class DocumentChunk(BaseModel):
    """Base retrieval chunk."""

    id: str

    document_id: str

    text: str

    source: Source = Field(
        default_factory=Source
    )

    chunk_index: int = 0

    created_at: int | None = None

    last_access: int | None = None


class FilteredChunk(DocumentChunk):
    """Chunk after quality filtering."""

    keyword_score: float = 0.0

    quality_score: float = 0.0

    length_score: float = 0.0

    filter_score: float = 0.0


class EmbeddedChunk(FilteredChunk):
    """Chunk after embedding retrieval."""

    similarity_score: float = 0.0

    rrf_score: float = 0.0


class RankedChunk(EmbeddedChunk):
    """Chunk after reranking."""

    rerank_score: float = 0.0


class CompressedChunk(RankedChunk):
    """Chunk after compression."""

    compressed_text: str = ""

    compression_ratio: float = 1.0


# ==========================================================
# Retrieval
# ==========================================================


class HybridRetrievalResult(BaseModel):
    """Memory retrieval result."""

    chunk: DocumentChunk

    rrf_score: float = 0.0

    retrieved_from: str = "qdrant"


class RetrievalDecision(BaseModel):
    """Memory lookup decision."""

    cache_hit: bool = False

    results: list[HybridRetrievalResult] = Field(
        default_factory=list
    )


# ==========================================================
# Context
# ==========================================================


class ContextDocument(BaseModel):
    """Document passed to LLM."""

    chunk_id: str

    text: str

    source: Source

    relevance_score: float = 0.0


class OptimizedContext(BaseModel):
    """Prepared LLM context."""

    query: str

    documents: list[ContextDocument] = Field(
        default_factory=list
    )

    total_tokens: int = 0

    citation_map: dict[str, Source] = Field(
        default_factory=dict
    )


class AgentContext(BaseModel):
    """Final agent input."""

    text: str = ""

    sources: list[Source] = Field(
        default_factory=list
    )

    optimized_context: OptimizedContext | None = None

    metadata: PipelineMetadata | None = None


# ==========================================================
# Final response
# ==========================================================


class FinalAnswer(BaseModel):
    """Agent final answer."""

    answer: str

    sources: list[Source] = Field(
        default_factory=list
    )

    citation_map: dict[str, Source] = Field(
        default_factory=dict
    )
