"""
Dense embedding similarity retrieval layer.

Pipeline position:

Exa
 |
 v
chunker.py
 |
 v
filter.py
 |
 v
embedding_retrieval.py
 |
 v
reranker.py
 |
 v
qdrant_store.py


Responsibilities:

- generate query embedding;
- generate chunk embeddings;
- calculate semantic similarity;
- rank chunks by embedding similarity;
- reduce candidates before reranking.


This module does NOT:

- call Exa;
- normalize documents;
- split chunks;
- filter chunks;
- rerank;
- store vectors;
- manage Qdrant.
"""


from __future__ import annotations


import logging
from typing import Any


import numpy as np


from web_search.cloudflare_embeddings import (
    get_embedding_model,
)



logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


EMBEDDING_TOP_K = 8



# ==========================================================
# Similarity
# ==========================================================


def _cosine_similarity(
    query_vector: list[float],
    document_vector: list[float],
) -> float:
    """
    Calculate cosine similarity.

    Used for local dense retrieval
    before reranking.
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
    chunks: list[dict[str, Any]],
) -> list[list[float]]:
    """
    Generate embeddings for chunks.
    """


    embedder = get_embedding_model()


    texts = [

        chunk.get(
            "text",
            "",
        )

        for chunk in chunks

    ]


    return await embedder.embed_documents(
        texts
    )



# ==========================================================
# Retrieval
# ==========================================================


async def retrieve_by_embedding_similarity(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = EMBEDDING_TOP_K,
) -> list[dict[str, Any]]:
    """
    Dense semantic retrieval.

    Input:

        query
        +
        filtered chunks


    Example:

        filter.py

            300 chunks
                 |
                 v
              TOP 10


        embedding_retrieval.py

              TOP 10
                 |
                 v

              TOP 5-8



    Output:

        chunks enriched with:

            embedding_score


    """



    if not chunks:

        logger.info(
            "No chunks provided for embedding retrieval."
        )

        return []



    logger.info(
        "Running embedding retrieval. chunks=%d",
        len(chunks),
    )



    embedder = get_embedding_model()



    query_vector = await embedder.embed_query(
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



    scored_chunks: list[
        tuple[float, dict[str, Any]]
    ] = []



    for chunk, vector in zip(
        chunks,
        chunk_vectors,
    ):


        score = _cosine_similarity(

            query_vector,

            vector,

        )



        scored_chunk = {

            **chunk,

            "embedding_score": score,

        }



        scored_chunks.append(

            (
                score,

                scored_chunk,

            )

        )



    scored_chunks.sort(

        key=lambda item: item[0],

        reverse=True,

    )



    result = [

        chunk

        for _, chunk

        in scored_chunks[:top_k]

    ]



    logger.info(
        "Embedding retrieval selected %d/%d chunks.",
        len(result),
        len(chunks),
    )


    return result
