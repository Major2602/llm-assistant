"""
Application service adapters.

Provides business-facing interfaces over infrastructure providers.
"""

from __future__ import annotations


from typing import Protocol


from web_search.domain.models import (
    DocumentChunk,
    EmbeddedChunk,
    NormalizedQuery,
    RankedChunk,
    WebDocument,
    HybridRetrievalResult,
    DenseVector,
)



# ==========================================================
# Provider contracts
# ==========================================================


class SearchProvider(Protocol):
    """
    External search provider contract.
    """

    async def search(
        self,
        query: str,
    ) -> list[WebDocument]:
        ...


class Embedder(Protocol):
    """
    Embedding provider contract.
    """

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[DenseVector]:
        ...


    async def embed_query(
        self,
        query: str,
    ) -> DenseVector:
        ...


class Reranker(Protocol):
    """
    Reranking provider contract.
    """

    async def rerank(
        self,
        query: str,
        chunks: list[EmbeddedChunk],
    ) -> list[RankedChunk]:
        ...


class VectorStore(Protocol):
    """
    Vector memory contract.
    """

    async def search(
        self,
        query: NormalizedQuery,
    ) -> list[HybridRetrievalResult]:
        ...


    async def store(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        ...


    async def cleanup(
        self,
        days: int,
    ) -> None:
        ...



# ==========================================================
# Application services
# ==========================================================


class SearchService:
    """
    Application wrapper around search provider.
    """


    def __init__(
        self,
        provider: SearchProvider,
    ):
        self._provider = provider



    async def execute(
        self,
        query: NormalizedQuery,
    ) -> list[WebDocument]:
        """
        Retrieve documents.
        """

        return await self._provider.search(
            query.normalized
        )



class MemoryService:
    """
    Application wrapper around vector memory.
    """


    def __init__(
        self,
        store: VectorStore,
    ):
        self._store = store



    async def lookup(
        self,
        query: NormalizedQuery,
    ) -> list[HybridRetrievalResult]:
        """
        Retrieve cached context.
        """

        return await self._store.search(
            query
        )



    async def save(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Store chunks.
        """

        await self._store.store(
            chunks
        )



    async def cleanup(
        self,
        days: int,
    ) -> None:
        """
        Cleanup expired memory.
        """

        await self._store.cleanup(
            days
        )
