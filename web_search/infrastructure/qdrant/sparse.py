"""
Sparse vector generation for Qdrant hybrid search.
"""

from __future__ import annotations

from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding


_encoder: SparseTextEmbedding | None = None


def get_sparse_encoder() -> SparseTextEmbedding:
    """
    Return singleton BM25 encoder.
    """

    global _encoder

    if _encoder is None:
        _encoder = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

    return _encoder



def create_sparse_vector(
    text: str,
) -> SparseVector:
    """
    Generate BM25 sparse vector.
    """

    encoder = get_sparse_encoder()

    vector = next(
        encoder.embed(
            [text]
        )
    )

    return SparseVector(
        indices=vector.indices.tolist(),
        values=vector.values.tolist(),
    )
