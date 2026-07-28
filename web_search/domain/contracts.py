# web_search/domain/contracts.py

from __future__ import annotations


from typing import Protocol, Sequence


from web_search.domain.models import (
    NormalizedQuery,
    WebDocument,
    DocumentChunk,
    DenseVector,
    EmbeddedChunk,
    RankedChunk,
    HybridRetrievalResult,
    Source,
)



# ==========================================================
# Search provider
# ==========================================================


class SearchProvider(Protocol):
    """
    External search provider contract.
    """

    async def search(
        self,
        query: NormalizedQuery,
    ) -> list[WebDocument]:
        ...



# ==========================================================
# Embeddings
# ==========================================================


class Embedder(Protocol):
    """
    Embedding service contract.
    """

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[DenseVector]:
        ...


    async def embed_query(
        self,
        query: str,
    ) -> DenseVector:
        ...



# ==========================================================
# Reranker
# ==========================================================


class Reranker(Protocol):
    """
    Semantic reranking contract.
    """

    async def rerank(
        self,
        query: NormalizedQuery,
        chunks: list[EmbeddedChunk],
    ) -> list[RankedChunk]:
        ...



# ==========================================================
# Vector storage
# ==========================================================


class VectorStore(Protocol):
    """
    Vector memory contract.
    """

    async def store(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        ...


    async def search(
        self,
        query: NormalizedQuery,
    ) -> list[HybridRetrievalResult]:
        ...


    async def cleanup(
        self,
        days: int,
    ) -> None:
        ...



# ==========================================================
# HTTP client
# ==========================================================


class HttpClient(Protocol):
    """
    Shared HTTP transport contract.
    """

    async def request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> dict:
        ...



# ==========================================================
# Pipeline stages
# ==========================================================


class PipelineStage(Protocol):
    """
    Common pipeline stage contract.
    """

    async def execute(
        self,
        state,
    ):
        ...



# ==========================================================
# Source factory
# ==========================================================


class SourceResolver(Protocol):
    """
    Source metadata resolver.
    """

    def resolve(
        self,
        data,
    ) -> Source:
        ...
