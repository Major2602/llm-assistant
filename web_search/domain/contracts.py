"""
Web search domain contracts.

Abstract interfaces between pipeline layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from web_search.domain.models import (
    AgentContext,
    DocumentChunk,
    WebDocument,
    DenseVector,
    NormalizedQuery,
    EmbeddedChunk,
    RankedChunk,
)


# ==========================================================
# Retrieval contracts
# ==========================================================


class DocumentRetriever(ABC):
    """External document retrieval contract."""

    @abstractmethod
    async def retrieve(
        self,
        query: NormalizedQuery,
    ) -> list[WebDocument]:
        """
        Retrieve external documents.
        """
        raise NotImplementedError



class MemoryStore(ABC):
    """Semantic memory storage contract."""

    @abstractmethod
    async def search(
        self,
        query: NormalizedQuery,
    ):
        """
        Retrieve cached chunks.
        """
        raise NotImplementedError


    @abstractmethod
    async def store(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Persist chunks.
        """
        raise NotImplementedError



# ==========================================================
# AI service contracts
# ==========================================================


class EmbeddingProvider(ABC):
    """Embedding generation contract."""

    @abstractmethod
    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[DenseVector]:
        """
        Generate document embeddings.
        """
        raise NotImplementedError


    @abstractmethod
    async def embed_query(
        self,
        query: str,
    ) -> DenseVector:
        """
        Generate query embedding.
        """
        raise NotImplementedError



class RerankerProvider(ABC):
    """Chunk reranking contract."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[EmbeddedChunk],
    ) -> list[RankedChunk]:
        """
        Rank retrieval candidates.
        """
        raise NotImplementedError



# ==========================================================
# Pipeline contracts
# ==========================================================


class ContextBuilder(ABC):
    """Final context preparation contract."""

    @abstractmethod
    def build(
        self,
        query: str,
        chunks: list[RankedChunk],
    ) -> AgentContext:
        """
        Build final agent context.
        """
        raise NotImplementedError
