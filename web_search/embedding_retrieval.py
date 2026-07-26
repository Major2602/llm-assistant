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


Responsibilities:

- generate query embedding;
- generate chunk embeddings;
- calculate cosine similarity;
- rank filtered chunks;
- reduce candidate pool before reranking.


This module does NOT know about:

- Exa;
- chunking;
- filtering;
- reranking;
- Qdrant;
- LLM.
"""


from __future__ import annotations


import logging
from typing import Any


import numpy as np


from web_search.cloudflare_embeddings import (
    get_embedding_model,
)


from web_search.models import (
    EmbeddingResult,
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
# Embeddings
# ==========================================================


async def _embed_chunks(
    chunks: list[dict[str, Any]],
) -> list[list[float]]:
    """
    Generate embeddings for chunk texts.
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
# Public API
# ==========================================================


async def retrieve_by_embedding_similarity(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = EMBEDDING_TOP_K,
) -> list[dict[str, Any]]:
    """
    Dense semantic retrieval.

    Input:

        Filtered chunks

    Output:

        EmbeddingResult-compatible chunks

        with:

            similarity_score


    Pipeline:

        TOP filtered chunks

                |

                v

        Dense similarity

                |

                v

        TOP K semantic chunks
    """


    if not chunks:

        logger.info(
            "No chunks provided for embedding retrieval."
        )

        return []



    logger.info(

        "Embedding retrieval started. chunks=%d",

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
        tuple[
            float,
            dict[str, Any]
        ]
    ] = []



    for chunk, vector in zip(
        chunks,
        chunk_vectors,
    ):


        score = _cosine_similarity(

            query_vector,

            vector,

        )



        enriched = {

            **chunk,

            "similarity_score": score,

        }



        scored_chunks.append(

            (

                score,

                enriched,

            )

        )



    scored_chunks.sort(

        key=lambda item:
            item[0],

        reverse=True,

    )



    result = [


        EmbeddingResult(
            **chunk
        ).model_dump()


        for _, chunk

        in scored_chunks[:top_k]


    ]



    logger.info(

        "Embedding retrieval completed. "
        "selected=%d/%d",

        len(result),

        len(chunks),

    )



    return result
