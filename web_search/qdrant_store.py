"""
Qdrant hybrid memory layer.

Architecture:

USER QUERY
    |
    v
QDRANT HYBRID RETRIEVAL

    Dense Vector Search
            +
    BM25 Sparse Search
            |
            v
        RRF Fusion

    |
    v

Embedding Retrieval
    |
    v

Reranker
    |
    v

Context


Responsibilities:

- store chunks;
- store dense embeddings;
- store BM25 sparse vectors;
- hybrid retrieval;
- memory lifecycle.


Does NOT:

- call Exa;
- chunk documents;
- filter;
- rerank;
- compress;
- build context.
"""


from __future__ import annotations


import logging
import os


from datetime import datetime, timezone, timedelta

from typing import Any


from qdrant_client import AsyncQdrantClient


from qdrant_client.models import (

    Distance,

    VectorParams,

    SparseVectorParams,

    SparseVector,

    PointStruct,

    SparseIndexParams,

    Modifier,

    Filter,

    FieldCondition,

    Range,

    PayloadSchemaType,

)


from fastembed import SparseTextEmbedding


from web_search.cloudflare_embeddings import (
    get_embedding_model,
)


from web_search.models import (
    DocumentChunk,
    HybridSearchResult,
)



logger = logging.getLogger(__name__)





# ==========================================================
# Configuration
# ==========================================================


QDRANT_URL = os.getenv(
    "QDRANT_URL"
)


if not QDRANT_URL:

    raise RuntimeError(
        "QDRANT_URL missing."
    )



QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)



COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "web_memory",
)



DENSE_VECTOR_NAME = "dense"

SPARSE_VECTOR_NAME = "bm25"



HYBRID_TOP_K = int(
    os.getenv(
        "QDRANT_TOP_K",
        "10",
    )
)





# ==========================================================
# Clients
# ==========================================================


_client: AsyncQdrantClient | None = None


_sparse_encoder: SparseTextEmbedding | None = None





def get_qdrant_client() -> AsyncQdrantClient:
    """
    Singleton Qdrant client.
    """

    global _client


    if _client is None:

        logger.info(
            "Initializing Qdrant client."
        )


        _client = AsyncQdrantClient(

            url=QDRANT_URL,

            api_key=QDRANT_API_KEY,

        )


    return _client





def get_sparse_encoder() -> SparseTextEmbedding:
    """
    Singleton BM25 encoder.
    """

    global _sparse_encoder


    if _sparse_encoder is None:

        logger.info(
            "Initializing BM25 encoder."
        )


        _sparse_encoder = SparseTextEmbedding(

            model_name="Qdrant/bm25"

        )


    return _sparse_encoder





# ==========================================================
# Collection management
# ==========================================================


async def collection_exists() -> bool:

    return await get_qdrant_client().collection_exists(

        COLLECTION_NAME

    )





async def ensure_collection(
    vector_size: int,
) -> None:
    """
    Create hybrid Qdrant collection.
    """


    client = get_qdrant_client()



    if not await client.collection_exists(
        COLLECTION_NAME
    ):


        logger.info(
            "Creating Qdrant hybrid collection."
        )


        await client.create_collection(

            collection_name=COLLECTION_NAME,


            vectors_config={

                DENSE_VECTOR_NAME:

                    VectorParams(

                        size=vector_size,

                        distance=Distance.COSINE,

                    )

            },


            sparse_vectors_config={

                SPARSE_VECTOR_NAME:

                    SparseVectorParams(

                        index=SparseIndexParams(),

                        modifier=Modifier.IDF,

                    )

            },

        )


    await ensure_payload_indexes()





async def ensure_payload_indexes():

    indexes = {

        "url":
            PayloadSchemaType.KEYWORD,


        "provider":
            PayloadSchemaType.KEYWORD,


        "last_access":
            PayloadSchemaType.INTEGER,

    }



    collection = await get_qdrant_client().get_collection(

        COLLECTION_NAME

    )



    existing = (

        collection.payload_schema

        or {}

    )



    for field, schema in indexes.items():


        if field in existing:

            continue



        await get_qdrant_client().create_payload_index(

            collection_name=COLLECTION_NAME,

            field_name=field,

            field_schema=schema,

        )





# ==========================================================
# Embeddings
# ==========================================================


async def _create_dense_embeddings(
    texts: list[str],
):

    model = get_embedding_model()


    return await model.embed_documents(
        texts
    )





def _create_sparse_embedding(
    text: str,
) -> SparseVector:


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





# ==========================================================
# Storage
# ==========================================================


async def add_chunks(
    chunks: list[dict[str, Any]],
) -> None:
    """
    Store final ranked chunks.

    Input:

        reranker/compression output

    """


    if not chunks:

        return



    texts = [

        chunk.get(
            "text",
            "",
        )

        for chunk in chunks

    ]



    dense_vectors = await _create_dense_embeddings(
        texts
    )



    await ensure_collection(

        len(
            dense_vectors[0]
        )

    )



    timestamp = int(

        datetime.now(
            timezone.utc
        ).timestamp()

    )



    points: list[PointStruct] = []



    for chunk, dense in zip(

        chunks,

        dense_vectors,

    ):


        document = DocumentChunk(

            **chunk

        )



        payload = document.model_dump()



        payload.update({

            "last_access":

                timestamp,

        })



        points.append(

            PointStruct(

                id=document.id,


                vector={

                    DENSE_VECTOR_NAME:

                        dense,


                    SPARSE_VECTOR_NAME:

                        _create_sparse_embedding(

                            document.text

                        )

                },


                payload=payload,

            )

        )



    await get_qdrant_client().upsert(

        collection_name=COLLECTION_NAME,

        points=points,

    )



    logger.info(

        "Stored %d chunks in Qdrant.",

        len(points),

    )





# ==========================================================
# Hybrid Retrieval
# ==========================================================


async def hybrid_search(
    query: str,
    limit: int = HYBRID_TOP_K,
) -> list[dict[str, Any]]:
    """
    Dense + BM25 hybrid search.

    Fusion:

        Reciprocal Rank Fusion
    """


    if not await collection_exists():

        return []



    dense_query = await get_embedding_model().embed_query(

        query

    )



    sparse_query = _create_sparse_embedding(
        query
    )



    result = await get_qdrant_client().query_points(

        collection_name=COLLECTION_NAME,


        prefetch=[

            {

                "query":

                    dense_query,


                "using":

                    DENSE_VECTOR_NAME,


                "limit":

                    limit,

            },


            {

                "query":

                    sparse_query,


                "using":

                    SPARSE_VECTOR_NAME,


                "limit":

                    limit,

            },

        ],


        query={

            "fusion":

                "rrf"

        },


        limit=limit,


        with_payload=True,

    )



    if not result.points:

        return []



    ids = [

        point.id

        for point in result.points

    ]



    await update_last_access(
        ids
    )



    output = []



    for point in result.points:


        item = {

            **point.payload,

            "fusion_score":

                point.score,

        }



        output.append(

            HybridSearchResult(

                **item

            ).model_dump()

        )



    return output





# ==========================================================
# Memory lifecycle
# ==========================================================


async def update_last_access(
    ids: list[str],
):

    if not ids:

        return



    timestamp = int(

        datetime.now(
            timezone.utc
        ).timestamp()

    )



    await get_qdrant_client().set_payload(

        collection_name=COLLECTION_NAME,


        payload={

            "last_access":

                timestamp

        },


        points=ids,

    )





async def cleanup_old_chunks(
    days: int = 30,
):

    if not await collection_exists():

        return



    cutoff = int(

        (

            datetime.now(
                timezone.utc
            )

            -

            timedelta(
                days=days
            )

        ).timestamp()

    )



    await get_qdrant_client().delete(

        collection_name=COLLECTION_NAME,


        points_selector=Filter(

            must=[

                FieldCondition(

                    key="last_access",

                    range=Range(

                        lt=cutoff

                    ),

                )

            ]

        ),

    )


    logger.info(
        "Old Qdrant memory cleaned."
    )
