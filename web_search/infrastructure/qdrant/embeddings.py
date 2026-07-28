"""
Qdrant vector preparation layer.

Responsibilities:

- generate dense embeddings;
- generate sparse BM25 vectors;
- convert text into Qdrant vector format.

No storage logic.
No singleton dependencies.
"""

from __future__ import annotations


from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector


from web_search.domain.contracts import (
    Embedder,
)


class QdrantEmbeddingBuilder:
    """
    Builds vectors required by Qdrant.

    Dense embeddings are provided through injected Embedder.
    Sparse embeddings use local BM25 encoder.
    """


    def __init__(
        self,
        embedder: Embedder,
        sparse_model_name: str = "Qdrant/bm25",
    ):
        self._embedder = embedder

        self._sparse_encoder = SparseTextEmbedding(
            model_name=sparse_model_name,
        )


    async def dense(
        self,
        texts: list[str],
    ):
        """
        Generate dense embeddings.
        """

        if not texts:
            return []

        return await self._embedder.embed_documents(
            texts
        )


    def sparse(
        self,
        text: str,
    ) -> SparseVector:
        """
        Generate BM25 sparse vector.
        """

        vector = next(
            self._sparse_encoder.embed(
                [text]
            )
        )


        return SparseVector(
            indices=vector.indices.tolist(),
            values=vector.values.tolist(),
        )


    async def build_batch(
        self,
        texts: list[str],
    ) -> list[tuple[object, SparseVector]]:
        """
        Build dense + sparse vectors.

        Returns:
            [
                (
                    dense_vector,
                    sparse_vector
                )
            ]
        """

        dense_vectors = await self.dense(
            texts
        )


        result = []


        for text, dense in zip(
            texts,
            dense_vectors,
        ):

            result.append(
                (
                    dense,
                    self.sparse(text),
                )
            )


        return result
