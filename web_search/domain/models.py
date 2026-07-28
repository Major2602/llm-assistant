# web_search/domain/models.py

from __future__ import annotations


from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any


# ==========================================================
# Helpers
# ==========================================================


def utc_timestamp() -> int:
    return int(
        datetime.now(
            UTC
        ).timestamp()
    )


# ==========================================================
# Source
# ==========================================================


@dataclass(slots=True)
class Source:
    """
    Citation source metadata.
    """

    title: str = "Untitled"

    url: str = ""

    provider: str | None = None

    domain: str | None = None

    author: str | None = None

    published_date: str | None = None



# ==========================================================
# Query
# ==========================================================


@dataclass(slots=True)
class NormalizedQuery:
    """
    Normalized user query.
    """

    original: str

    normalized: str

    created_at: int = field(
        default_factory=utc_timestamp
    )



# ==========================================================
# Documents
# ==========================================================


@dataclass(slots=True)
class WebDocument:
    """
    Raw external document.
    """

    id: str

    text: str

    source: Source

    created_at: int = field(
        default_factory=utc_timestamp
    )

    last_access: int = field(
        default_factory=utc_timestamp
    )



@dataclass(slots=True)
class DocumentChunk:
    """
    Chunk produced from document.
    """

    id: str

    text: str

    source: Source

    document_id: str | None = None

    chunk_index: int = 0

    created_at: int = field(
        default_factory=utc_timestamp
    )

    last_access: int = field(
        default_factory=utc_timestamp
    )



# ==========================================================
# Retrieval models
# ==========================================================


@dataclass(slots=True)
class FilteredChunk(DocumentChunk):
    """
    Chunk after quality filtering.
    """

    keyword_score: float = 0.0

    quality_score: float = 0.0

    length_score: float = 0.0

    filter_score: float = 0.0



@dataclass(slots=True)
class DenseVector:
    """
    Dense embedding vector.
    """

    values: list[float]



@dataclass(slots=True)
class EmbeddedChunk(FilteredChunk):
    """
    Chunk with embedding similarity score.
    """

    similarity_score: float = 0.0

    rrf_score: float | None = None



@dataclass(slots=True)
class RankedChunk(EmbeddedChunk):
    """
    Chunk after reranking.
    """

    rerank_score: float = 0.0



@dataclass(slots=True)
class CompressedChunk(RankedChunk):
    """
    Chunk after compression.
    """

    compressed_text: str = ""

    compression_ratio: float = 1.0



# ==========================================================
# Context models
# ==========================================================


@dataclass(slots=True)
class ContextDocument:
    """
    Final document representation for LLM.
    """

    chunk_id: str

    text: str

    source: Source

    relevance_score: float = 0.0



@dataclass(slots=True)
class OptimizedContext:
    """
    Optimized LLM context metadata.
    """

    query: str

    documents: list[ContextDocument] = field(
        default_factory=list
    )

    total_tokens: int = 0

    citation_map: dict[str, Source] = field(
        default_factory=dict
    )



@dataclass(slots=True)
class AgentContext:
    """
    Final pipeline output.
    """

    text: str = ""

    sources: list[Source] = field(
        default_factory=list
    )

    optimized_context: OptimizedContext | None = None

    metadata: PipelineMetadata | None = None



# ==========================================================
# Pipeline metadata
# ==========================================================


@dataclass(slots=True)
class PipelineMetadata:
    """
    Pipeline execution metrics.
    """

    request_id: str | None = None

    query: str = ""

    created_at: int = field(
        default_factory=utc_timestamp
    )

    cache_hit: bool = False

    chunks_created: int = 0

    chunks_filtered: int = 0

    chunks_embedded: int = 0

    chunks_ranked: int = 0

    chunks_compressed: int = 0

    sources_count: int = 0



# ==========================================================
# Retrieval decision
# ==========================================================


@dataclass(slots=True)
class RetrievalDecision:
    """
    Cache retrieval result.
    """

    cache_hit: bool

    results: list[Any] = field(
        default_factory=list
    )



@dataclass(slots=True)
class HybridRetrievalResult:
    """
    Qdrant hybrid search result.
    """

    chunk: DocumentChunk

    rrf_score: float

    retrieved_from: str = "qdrant"



# ==========================================================
# Pipeline state
# ==========================================================


@dataclass(slots=True)
class PipelineState:
    """
    Shared mutable state between pipeline stages.
    """

    query: NormalizedQuery

    documents: list[WebDocument] = field(
        default_factory=list
    )

    chunks: list[DocumentChunk] = field(
        default_factory=list
    )

    filtered_chunks: list[FilteredChunk] = field(
        default_factory=list
    )

    embedded_chunks: list[EmbeddedChunk] = field(
        default_factory=list
    )

    ranked_chunks: list[RankedChunk] = field(
        default_factory=list
    )

    compressed_chunks: list[CompressedChunk] = field(
        default_factory=list
    )

    context: AgentContext | None = None

    metadata: PipelineMetadata = field(
        default_factory=PipelineMetadata
    )
