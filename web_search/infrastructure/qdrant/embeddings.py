"""
Dense vector generation adapter for Qdrant.
"""

from __future__ import annotations

from web_search.domain.models import DenseVector
from web_search.infrastructure.cloudflare.embeddings import (
    get_embedding_service,
)



async def generate_dense_vectors(
    texts: list[str],
) -> list[DenseVector]:
    """
    Generate dense embeddings for documents.
    """

    if not texts:
        return []

    service = get_embedding_service()

    return await service.embed_documents(
        texts
    )



async def generate_query_vector(
    query: str,
) -> DenseVector:
    """
    Generate query embedding.
    """

    service = get_embedding_service()

    return await service.embed_query(
        query
    )
