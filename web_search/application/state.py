# web_search/application/state.py

from __future__ import annotations


from dataclasses import dataclass, field


from web_search.domain.models import (
    NormalizedQuery,
    WebDocument,
    DocumentChunk,
    EmbeddedChunk,
    RankedChunk,
    AgentContext,
    PipelineMetadata,
)



@dataclass
class PipelineState:
    """
    Shared mutable state of web search pipeline.

    Each application stage receives this object
    and returns updated state.
    """

    query: NormalizedQuery


    documents: list[WebDocument] = field(
        default_factory=list
    )


    chunks: list[DocumentChunk] = field(
        default_factory=list
    )


    filtered_chunks: list[DocumentChunk] = field(
        default_factory=list
    )


    embedded_chunks: list[EmbeddedChunk] = field(
        default_factory=list
    )


    ranked_chunks: list[RankedChunk] = field(
        default_factory=list
    )


    context: AgentContext | None = None


    metadata: PipelineMetadata = field(
        default_factory=PipelineMetadata
    )


    cache_hit: bool = False



    def reset_results(self) -> None:
        """
        Clear calculated pipeline results.

        Used when pipeline execution must be restarted.
        """

        self.documents.clear()

        self.chunks.clear()

        self.filtered_chunks.clear()

        self.embedded_chunks.clear()

        self.ranked_chunks.clear()

        self.context = None
