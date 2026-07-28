"""
Pipeline execution state.

Contains shared mutable state passed between pipeline stages.
"""

from __future__ import annotations


from dataclasses import dataclass, field


from web_search.domain.models import (
    AgentContext,
    DocumentChunk,
    NormalizedQuery,
    PipelineMetadata,
    RankedChunk,
    CompressedChunk,
    WebDocument,
)


@dataclass
class PipelineState:
    """
    Unified state object for pipeline execution.

    Every stage receives and returns this object.
    """


    query: NormalizedQuery


    metadata: PipelineMetadata = field(
        default_factory=PipelineMetadata
    )


    documents: list[WebDocument] = field(
        default_factory=list
    )


    chunks: list[DocumentChunk] = field(
        default_factory=list
    )


    ranked_chunks: list[RankedChunk] = field(
        default_factory=list
    )


    compressed_chunks: list[CompressedChunk] = field(
        default_factory=list
    )


    context: AgentContext | None = None


    cache_hit: bool = False


    completed_stages: list[str] = field(
        default_factory=list
    )


    def mark_completed(
        self,
        stage: str,
    ) -> None:
        """
        Mark pipeline stage as completed.
        """

        self.completed_stages.append(
            stage
        )
