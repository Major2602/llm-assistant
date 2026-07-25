"""
Qdrant storage layer.

Pipeline position:

Exa
 ↓
filter.py
 ↓
chunker.py
 ↓
cloudflare_embeddings.py
 ↓
reranker.py
 ↓
qdrant_store.py


Responsibilities:
- semantic cache lookup;
- embedding generation for final chunks;
- vector storage;
- memory cleanup.

This module does NOT:
- call Exa;
- split documents;
- filter chunks;
- rerank chunks.
"""

from __future__ import annotations

import logging
import os

from datetime import datetime, timedelta, timezone
from typing import Any


from qdrant_client import AsyncQdrantClient

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)


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
        "Environment variable QDRANT_URL is not configured."
    )


QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)


COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "web_memory",
)



# ==========================================================
# Client
# ==========================================================


_client: AsyncQdrantClient | None = None



def get_qdrant() -> AsyncQdrantClient:
    """
    Return singleton Qdrant client.
    """

    global _client


    if _client is None:

        logger.info(
            "Initializing AsyncQdrantClient."
        )


        _client = AsyncQdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )


    return _client



# ==========================================================
# Collection management
# ==========================================================


async def collection_exists() -> bool:
    """
    Check collection existence.
    """

    return await get_qdrant().collection_exists(
        COLLECTION_NAME
    )



async def ensure_collection(
    vector_size: int,
) -> None:
    """
    Create collection if missing.
    """


    client = get_qdrant()


    exists = await client.collection_exists(
        COLLECTION_NAME
    )


    if not exists:

        logger.info(
            "Creating Qdrant collection '%s'.",
            COLLECTION_NAME,
        )


        await client.create_collection(
            collection_name=COLLECTION_NAME,

            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


    await ensure_payload_indexes()



# ==========================================================
# Payload indexes
# ==========================================================


async def _payload_index_exists(
    field_name: str,
) -> bool:

    collection = await get_qdrant().get_collection(
        COLLECTION_NAME
    )


    schema = (
        collection.payload_schema
        or {}
    )


    return field_name in schema



async def ensure_payload_indexes() -> None:
    """
    Ensure searchable payload fields.
    """


    indexes = {

        "last_access":
            PayloadSchemaType.INTEGER,

        "query":
            PayloadSchemaType.KEYWORD,

        "provider":
            PayloadSchemaType.KEYWORD,

        "url":
            PayloadSchemaType.KEYWORD,

    }


    for field, schema in indexes.items():

        if await _payload_index_exists(
            field
        ):
            continue


        logger.info(
            "Creating payload index '%s'.",
            field,
        )


        await get_qdrant().create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )



# ==========================================================
# Insert
# ==========================================================


async def add_chunks(
    chunks: list[dict[str, Any]],
) -> None:
    """
    Store final reranked chunks.

    Expected input:
        5-10 chunks after reranker.py
    """


    if not chunks:

        logger.info(
            "No chunks provided for storage."
        )

        return



    logger.info(
        "Embedding and storing %d final chunks.",
        len(chunks),
    )


    embedder = get_embedding_model()


    texts = [

        chunk.get(
            "text",
            "",
        )

        for chunk in chunks

    ]


    vectors = await embedder.embed_documents(
        texts
    )


    if len(vectors) != len(chunks):

        raise RuntimeError(
            "Embedding count does not match chunk count."
        )


    await ensure_collection(
        len(vectors[0])
    )



    points = []


    now = int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )


    for chunk, vector in zip(
        chunks,
        vectors,
    ):

        payload = {

            **chunk,

            "created_at":
                chunk.get(
                    "created_at",
                    now,
                ),

            "last_access":
                now,

        }


        points.append(

            PointStruct(

                id=chunk["id"],

                vector=vector,

                payload=payload,

            )

        )


    await get_qdrant().upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


    logger.info(
        "Inserted %d chunks into Qdrant.",
        len(points),
    )



# ==========================================================
# Semantic search
# ==========================================================


async def search(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.70,
) -> list[dict[str, Any]]:
    """
    Semantic cache lookup.
    """


    if not await collection_exists():

        logger.info(
            "Qdrant collection does not exist."
        )

        return []



    embedder = get_embedding_model()


    query_vector = await embedder.embed_query(
        query
    )


    result = await get_qdrant().query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        limit=limit,

        with_payload=True,

    )


    hits = result.points


    if not hits:

        return []



    if hits[0].score < score_threshold:

        logger.info(
            "Cache miss. Score %.3f",
            hits[0].score,
        )

        return []



    await update_last_access(
        [
            point.id
            for point in hits
        ]
    )



    return [

        {
            **point.payload,
            "score": point.score,
        }

        for point in hits

    ]



# ==========================================================
# Access update
# ==========================================================


async def update_last_access(
    ids: list[str],
) -> None:


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
            "last_access": timestamp
        },

        points=ids,

    )



# ==========================================================
# Cleanup
# ==========================================================


async def cleanup_old_chunks(
    days: int = 30,
) -> None:
    """
    Remove unused memory.
    """


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

        ),

    )


    logger.info(
        "Removed chunks older than %d days.",
        days,
    )
