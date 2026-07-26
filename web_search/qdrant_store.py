"""
Qdrant hybrid memory layer.

Baseline Architecture v1:

User Query
    |
    v
Qdrant Hybrid Retrieval
(Dense + BM25 + RRF)
    |
    v
Candidates
    |
    v
Embedding similarity
    |
    v
Reranker
    |
    v
Context


Responsibilities:

- dense vector storage
- BM25 sparse storage
- hybrid retrieval
- RRF fusion
- semantic memory
- cleanup lifecycle


Does NOT:

- call Exa
- chunk documents
- filter chunks
- rerank
- compress context
"""

from __future__ import annotations


import logging
import os


from datetime import (
    datetime,
    timedelta,
    timezone,
)


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



HYBRID_LIMIT = 10



# ==========================================================
# Clients
# ==========================================================


_client: AsyncQdrantClient | None = None



_sparse_encoder: SparseTextEmbedding | None = None




def get_qdrant() -> AsyncQdrantClient:
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

            model_name=(
                "Qdrant/bm25"
            )

        )


    return _sparse_encoder




# ==========================================================
# Collection
# ==========================================================


async def collection_exists() -> bool:

    return await get_qdrant().collection_exists(

        COLLECTION_NAME

    )





async def ensure_collection(
    vector_size: int,
) -> None:
    """
    Create hybrid collection.
    """


    client = get_qdrant()



    if not await client.collection_exists(
        COLLECTION_NAME
    ):


        logger.info(
            "Creating hybrid collection."
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



    collection = await get_qdrant().get_collection(

        COLLECTION_NAME

    )


    existing = (
        collection.payload_schema
        or {}
    )



    for field, schema in indexes.items():


        if field in existing:

            continue



        await get_qdrant().create_payload_index(

            collection_name=COLLECTION_NAME,

            field_name=field,

            field_schema=schema,

        )





# ==========================================================
# Embeddings
# ==========================================================


async def _dense_embedding(
    texts: list[str],
):

    embedder = get_embedding_model()


    return await embedder.embed_documents(
        texts
    )





def _sparse_embedding(
    text: str,
) -> SparseVector:


    encoder = get_sparse_encoder()


    result = next(
        encoder.embed(
            [text]
        )
    )


    return SparseVector(

        indices=result.indices.tolist(),

        values=result.values.tolist(),

    )





# ==========================================================
# Storage
# ==========================================================


async def add_chunks(
    chunks: list[dict[str, Any]],
) -> None:
    """
    Store final reranked chunks.
    """


    if not chunks:

        return



    texts = [

        chunk.get(
            "text",
            ""
        )

        for chunk in chunks

    ]



    dense_vectors = await _dense_embedding(
        texts
    )



    await ensure_collection(
        len(
            dense_vectors[0]
        )
    )



    now = int(

        datetime.now(
            timezone.utc
        ).timestamp()

    )



    points = []



    for chunk, dense in zip(

        chunks,

        dense_vectors,

    ):


        payload = {

            **chunk,


            "created_at":

                chunk.get(
                    "created_at",
                    now
                ),


            "last_access":

                now,

        }



        points.append(

            PointStruct(

                id=chunk["id"],


                vector={

                    DENSE_VECTOR_NAME:

                        dense,


                    SPARSE_VECTOR_NAME:

                        _sparse_embedding(

                            chunk["text"]

                        )

                },


                payload=payload,

            )

        )



    await get_qdrant().upsert(

        collection_name=COLLECTION_NAME,

        points=points,

    )


    logger.info(

        "Stored %d hybrid chunks.",

        len(points),

    )





# ==========================================================
# Hybrid Retrieval
# ==========================================================


async def hybrid_search(
    query: str,
    limit: int = HYBRID_LIMIT,
) -> list[dict[str,Any]]:
    """
    Dense + BM25 hybrid retrieval.
    """


    if not await collection_exists():

        return []



    dense = await get_embedding_model().embed_query(

        query

    )



    sparse = _sparse_embedding(
        query
    )



    results = await get_qdrant().query_points(

        collection_name=COLLECTION_NAME,


        prefetch=[


            {

                "query":

                    dense,


                "using":

                    DENSE_VECTOR_NAME,


                "limit":

                    limit,

            },


            {

                "query":

                    sparse,


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



    hits = results.points



    if not hits:

        return []



    await update_last_access(

        [
            x.id
            for x in hits
        ]

    )



    return [

        {

            **hit.payload,

            "score":

                hit.score,

        }

        for hit in hits

    ]





# ==========================================================
# Access
# ==========================================================


async def update_last_access(
    ids:list[str],
):

    if not ids:

        return



    timestamp = int(

        datetime.now(
            timezone.utc
        ).timestamp()

    )



    await get_qdrant().set_payload(

        collection_name=COLLECTION_NAME,


        payload={

            "last_access":

                timestamp

        },


        points=ids,

    )





# ==========================================================
# Cleanup
# ==========================================================


async def cleanup_old_chunks(
    days:int=30,
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



    await get_qdrant().delete(

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

        )

    )


    logger.info(
        "Old memory cleaned."
    )
