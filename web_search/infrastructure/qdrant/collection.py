"""
Qdrant collection management.

Responsibilities:

- create collections;
- configure vector indexes;
- configure payload indexes.

No business logic.
No global state.
"""

from __future__ import annotations


from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Modifier,
    PayloadSchemaType,
)


from web_search.infrastructure.qdrant.client import (
    QdrantConnection,
)


class QdrantCollectionManager:
    """
    Manage Qdrant collection lifecycle.
    """


    def __init__(
        self,
        connection: QdrantConnection,
        collection_name: str,
    ):
        self._connection = connection
        self._collection_name = collection_name


    async def ensure(
        self,
        vector_size: int,
    ) -> None:
        """
        Ensure collection exists.
        """

        client = self._connection.client


        exists = await client.collection_exists(
            self._collection_name
        )


        if not exists:

            await client.create_collection(
                collection_name=self._collection_name,

                vectors_config={
                    "dense": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    )
                },

                sparse_vectors_config={
                    "bm25": SparseVectorParams(
                        index=SparseIndexParams(),
                        modifier=Modifier.IDF,
                    )
                },
            )


        await self._ensure_payload_indexes()



    async def _ensure_payload_indexes(
        self,
    ) -> None:
        """
        Create required payload indexes.
        """

        client = self._connection.client


        indexes = {
            "source.url": PayloadSchemaType.KEYWORD,
            "created_at": PayloadSchemaType.INTEGER,
            "last_access": PayloadSchemaType.INTEGER,
        }


        collection = await client.get_collection(
            self._collection_name
        )


        existing = (
            collection.payload_schema
            or {}
        )


        for field, schema in indexes.items():

            if field in existing:
                continue


            await client.create_payload_index(
                collection_name=self._collection_name,
                field_name=field,
                field_schema=schema,
            )
