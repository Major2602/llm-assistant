# web_search/application/services.py

from __future__ import annotations


from dataclasses import dataclass


from web_search.domain.contracts import (
    SearchProvider,
    VectorStore,
    Embedder,
    Reranker,
)


@dataclass
class PipelineServices:
    """
    Application service dependencies.

    Contains only abstractions.
    Concrete implementations are injected
    during application startup.
    """

    search_provider: SearchProvider

    vector_store: VectorStore

    embedder: Embedder

    reranker: Reranker
