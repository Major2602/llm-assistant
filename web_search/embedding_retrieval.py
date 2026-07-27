"""
Dense embedding similarity retrieval layer.

Module Responsibilities:

- generate query embedding;
- generate chunk embeddings;
- calculate cosine similarity;
- rank filtered chunks;
- reduce candidate pool before reranking.
"""


from __future__ import annotations


import logging
import os


import numpy as np


from web_search.cloudflare_embeddings import (
    get_embedding_model,
)


from web_search.models import (
    FilteredChunk,
    EmbeddedChunk,
)


logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


EMBEDDING_TOP_K = int(
    os.getenv(
        "EMBEDDING_TOP_K",
        "8",
    )
)



# ==========================================================
# Similarity
# ==========================================================


def _cosine_similarity(
    query_vector: list[float],
    document_vector: list[float],
) -> float:
    """
    Calculate cosine similarity.
    """


    query = np.asarray(
        query_vector,
        dtype=np.float32,
    )


    document = np.asarray(
        document_vector,
        dtype=np.float32,
    )


    denominator = (

        np.linalg.norm(query)

        *

        np.linalg.norm(document)

    )


    if denominator == 0:

        return 0.0


    return float(

        np.dot(
            query,
            document,
        )

        /

        denominator

    )



# ==========================================================
# Embedding generation
# ==========================================================


async def _embed_chunks(
    chunks: list[FilteredChunk],
) -> list[list[float]]:
    """
    Generate embeddings for chunks.
    """


    texts = [

        chunk.text

        for chunk in chunks

    ]


    if not texts:

        return []


    model = get_embedding_model()


    return await model.embed_documents(
        texts
    )



# ==========================================================
# Public API
# ==========================================================


async def retrieve_by_embedding_similarity(
    query: str,
    chunks: list[FilteredChunk],
    top_k: int = EMBEDDING_TOP_K,
) -> list[EmbeddedChunk]:
    """
    Dense semantic retrieval.
    """


    if not chunks:

        logger.info(
            "No chunks for embedding retrieval."
        )

        return []



    logger.info(

        "Embedding retrieval started. chunks=%d",

        len(chunks),

    )



    embedding_model = get_embedding_model()



    query_vector = await embedding_model.embed_query(
        query
    )



    chunk_vectors = await _embed_chunks(
        chunks
    )



    if len(chunk_vectors) != len(chunks):

        raise RuntimeError(

            (

                "Embedding count mismatch. "

                f"chunks={len(chunks)} "

                f"vectors={len(chunk_vectors)}"

            )

        )



    scored_chunks: list[EmbeddedChunk] = []



    for chunk, vector in zip(

        chunks,

        chunk_vectors,

    ):


        score = _cosine_similarity(

            query_vector,

            vector,

        )


        scored_chunks.append(

            EmbeddedChunk(

                **chunk.model_dump(),

                similarity_score=score,

            )

        )



    scored_chunks.sort(

        key=lambda item:

            item.similarity_score,

        reverse=True,

    )



    result = scored_chunks[:top_k]



    logger.info(

        "Embedding retrieval completed. selected=%d/%d",

        len(result),

        len(chunks),

    )



    return result
