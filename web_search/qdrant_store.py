"""
Qdrant hybrid memory layer.

Module Responsibilities:

- store chunks;
- store embeddings;
- hybrid retrieval;
- memory lifecycle.
"""


from __future__ import annotations


import logging
import os


from datetime import datetime, timezone, timedelta


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
    HybridRetrievalResult,
    NormalizedQuery,
    DenseVector
)



logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


QDRANT_URL = os.getenv(
    "QDRANT_URL"
)


QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)



COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "web_memory",
)



DENSE_NAME = "dense"

SPARSE_NAME = "bm25"



TOP_K = int(
    os.getenv(
        "QDRANT_TOP_K",
        "10",
    )
)



if not QDRANT_URL:

    raise RuntimeError(
        "QDRANT_URL missing"
    )



# ==========================================================
# Clients
# ==========================================================


_client: AsyncQdrantClient | None = None


_sparse_encoder: SparseTextEmbedding | None = None



def get_client() -> AsyncQdrantClient:
    """
    Singleton Qdrant client.
    """

    global _client


    if _client is None:

        _client = AsyncQdrantClient(

            url=QDRANT_URL,

            api_key=QDRANT_API_KEY,

        )


    return _client



def get_sparse_encoder() -> SparseTextEmbedding:
    """
    BM25 encoder.

    Used only for sparse vector generation.

    Storage and scoring remain Qdrant responsibility.
    """

    global _sparse_encoder


    if _sparse_encoder is None:

        _sparse_encoder = SparseTextEmbedding(

            model_name="Qdrant/bm25"

        )


    return _sparse_encoder



# ==========================================================
# Collection
# ==========================================================


async def ensure_collection(
    vector_size: int,
) -> None:
    """
    Create Qdrant hybrid collection.
    """

    client = get_client()


    exists = await client.collection_exists(
        COLLECTION_NAME
    )


    if not exists:

        await client.create_collection(

            collection_name=COLLECTION_NAME,


            vectors_config={

                DENSE_NAME:

                    VectorParams(

                        size=vector_size,

                        distance=Distance.COSINE,

                    )

            },


            sparse_vectors_config={

                SPARSE_NAME:

                    SparseVectorParams(

                        index=SparseIndexParams(),

                        modifier=Modifier.IDF,

                    )

            },

        )



    await _ensure_payload_indexes()



async def _ensure_payload_indexes():

    indexes = {

        "source.url":
            PayloadSchemaType.KEYWORD,


        "created_at":
            PayloadSchemaType.INTEGER,


        "last_access":
            PayloadSchemaType.INTEGER,

    }



    collection = await get_client().get_collection(
        COLLECTION_NAME
    )


    existing = (
        collection.payload_schema
        or {}
    )



    for field, schema in indexes.items():

        if field in existing:

            continue


        await get_client().create_payload_index(

            collection_name=COLLECTION_NAME,

            field_name=field,

            field_schema=schema,

        )



# ==========================================================
# Vector generation
# ==========================================================


async def _dense_embeddings(
    texts: list[str],
) -> list[DenseVector]:

    model = get_embedding_model()


    return await model.embed_documents(
        texts
    )



def _sparse_embedding(
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


async def store_chunks(
    chunks: list[DocumentChunk],
) -> None:
    """
    Store chunks in Qdrant memory.

    Stores:

    - original text;
    - metadata;
    - timestamps;
    - vectors.
    """


    if not chunks:

        return



    texts = [

        chunk.text

        for chunk in chunks

    ]



    dense_vectors = await _dense_embeddings(
        texts
    )

    if not dense_vectors:

        logger.warning(
            "No embeddings generated. Skip storing chunks."
        )
        
        return

    if len(dense_vectors) != len(chunks):
        
        raise RuntimeError(
            (
                "Embedding count mismatch. "
                f"chunks={len(chunks)} "
                f"vectors={len(dense_vectors)}"
            )
        )

    await ensure_collection(

        vector_size=len(
            dense_vectors[0].values
        )

    )



    timestamp = int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )



    points = []



    for chunk, vector in zip(

        chunks,

        dense_vectors,

    ):


        payload = chunk.model_dump()



        payload["last_access"] = timestamp



        points.append(

            PointStruct(

                id=chunk.id,


                vector={

                    DENSE_NAME:
                        vector.values,


                    SPARSE_NAME:
                        _sparse_embedding(
                            chunk.text
                        )

                },


                payload=payload,

            )

        )



    await get_client().upsert(

        collection_name=COLLECTION_NAME,

        points=points,

    )



# ==========================================================
# Hybrid retrieval
# ==========================================================


async def hybrid_search(
    query: NormalizedQuery,
    limit: int = TOP_K,
) -> list[HybridRetrievalResult]:
    """
    Dense + BM25 hybrid retrieval.

    Fusion:

        Reciprocal Rank Fusion
    """


    if not await get_client().collection_exists(
        COLLECTION_NAME
    ):

        return []



    dense_query = (
        
        await get_embedding_model().embed_query(
            query.normalized
        )
        
    ).values


    sparse_query = _sparse_embedding(

        query.normalized

    )



    result = await get_client().query_points(

        collection_name=COLLECTION_NAME,


        prefetch=[

            {

                "query":
                    dense_query,

                "using":
                    DENSE_NAME,

                "limit":
                    limit,

            },


            {

                "query":
                    sparse_query,

                "using":
                    SPARSE_NAME,

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



    output = []



    for point in result.points:


        chunk = DocumentChunk(

            **point.payload

        )


        output.append(

            HybridRetrievalResult(

                chunk=chunk,

                rrf_score=point.score or 0.0,

                retrieved_from = "qdrant"

            )

        )


    await update_access(

        [

            item.chunk.id

            for item in output

        ]

    )



    return output



# ==========================================================
# Qdrant result adaptation
# ==========================================================


def adapt_hybrid_results(
    results: list[HybridRetrievalResult],
) -> list[EmbeddedChunk]:
    """
    Adapt Qdrant memory results for reranker pipeline.

    Qdrant does not contain:
    - keyword_score
    - quality_score
    - length_score
    - filter_score
    - similarity_score

    These fields belong only to fresh retrieval pipeline.
    """

    return [

        EmbeddedChunk(

            **item.chunk.model_dump(),

            retrieval_score=item.rrf_score,

        )

        for item in results

    ]



# ==========================================================
# Memory lifecycle
# ==========================================================


async def update_access(
    ids: list[str],
):

    if not ids:

        return


    timestamp = int(

        datetime.now(
            timezone.utc
        ).timestamp()

    )


    await get_client().set_payload(

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



    await get_client().delete(

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
