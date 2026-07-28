"""
Qdrant collection management.
"""

from __future__ import annotations

import logging

from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Modifier,
    PayloadSchemaType,
)

from web_search.infrastructure.qdrant.client import (
    get_qdrant_client,
)


logger = logging.getLogger(__name__)


COLLECTION_NAME = "web_memory"

DENSE_NAME = "dense"

SPARSE_NAME = "bm25"



async def ensure_collection(
    vector_size: int,
) -> None:
    """
    Ensure hybrid Qdrant collection exists.
    """

    client = get_qdrant_client()

    exists = await client.collection_exists(
        COLLECTION_NAME
    )

    if not exists:

        logger.info(
            "Creating Qdrant collection=%s",
            COLLECTION_NAME,
        )

        await client.create_collection(

            collection_name=COLLECTION_NAME,

            vectors_config={
                DENSE_NAME: VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                )
            },

            sparse_vectors_config={
                SPARSE_NAME: SparseVectorParams(
                    index=SparseIndexParams(),
                    modifier=Modifier.IDF,
                )
            },
        )


    await ensure_payload_indexes()



async def ensure_payload_indexes() -> None:
    """
    Create required payload indexes.
    """

    client = get_qdrant_client()

    collection = await client.get_collection(
        COLLECTION_NAME
    )

    existing = (
        collection.payload_schema
        or {}
    )

    indexes = {
        "source.url": PayloadSchemaType.KEYWORD,
        "created_at": PayloadSchemaType.INTEGER,
        "last_access": PayloadSchemaType.INTEGER,
    }


    for field, schema in indexes.items():

        if field in existing:
            continue

        await client.create_payload_index(

            collection_name=COLLECTION_NAME,

            field_name=field,

            field_schema=schema,

        )



async def collection_exists() -> bool:
    """
    Check collection existence.
    """

    return await get_qdrant_client().collection_exists(
        COLLECTION_NAME
    )
