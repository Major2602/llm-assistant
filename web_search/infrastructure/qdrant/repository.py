"""
Qdrant repository layer.

Responsibilities:
- store chunks;
- execute hybrid retrieval;
- update access timestamps.
"""

from __future__ import annotations

import logging

from datetime import datetime, timezone

from qdrant_client.models import (
    PointStruct,
)

from web_search.domain.models import (
    DocumentChunk,
    HybridRetrievalResult,
    NormalizedQuery,
)

from web_search.infrastructure.qdrant.client import (
    get_qdrant_client,
)

from web_search.infrastructure.qdrant.collection import (
    COLLECTION_NAME,
    DENSE_NAME,
    SPARSE_NAME,
    ensure_collection,
    collection_exists,
)

from web_search.infrastructure.qdrant.embeddings import (
    generate_dense_vectors,
    generate_query_vector,
)

from web_search.infrastructure.qdrant.sparse import (
    create_sparse_vector,
)


logger = logging.getLogger(__name__)



async def store_chunks(
    chunks: list[DocumentChunk],
) -> None:
    """
    Store document chunks in Qdrant.
    """

    if not chunks:
        return


    texts = [
        chunk.text
        for chunk in chunks
    ]


    dense_vectors = await generate_dense_vectors(
        texts
    )


    if len(dense_vectors) != len(chunks):
        raise RuntimeError(
            "Dense vector count mismatch."
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


    points: list[PointStruct] = []


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
                        create_sparse_vector(
                            chunk.text
                        ),

                },

                payload=payload,

            )
        )


    await get_qdrant_client().upsert(

        collection_name=COLLECTION_NAME,

        points=points,

    )


async def hybrid_search(
    query: NormalizedQuery,
    limit: int,
) -> list[HybridRetrievalResult]:
    """
    Dense + BM25 hybrid retrieval.
    """

    if not await collection_exists():

        return []


    dense = await generate_query_vector(
        query.normalized
    )


    sparse = create_sparse_vector(
        query.normalized
    )


    result = await get_qdrant_client().query_points(

        collection_name=COLLECTION_NAME,

        prefetch=[

            {
                "query": dense.values,
                "using": DENSE_NAME,
                "limit": limit,
            },

            {
                "query": sparse,
                "using": SPARSE_NAME,
                "limit": limit,
            },

        ],

        query={
            "fusion": "rrf"
        },

        limit=limit,

        with_payload=True,

    )


    output: list[HybridRetrievalResult] = []


    for point in result.points:

        chunk = DocumentChunk(
            **point.payload
        )


        output.append(

            HybridRetrievalResult(

                chunk=chunk,

                rrf_score=point.score or 0.0,

                retrieved_from="qdrant",

            )
        )


    await update_access(
        [
            item.chunk.id
            for item in output
        ]
    )


    return output



async def update_access(
    ids: list[str],
) -> None:
    """
    Update last access timestamp.
    """

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
            "last_access": timestamp
        },

        points=ids,

    )
