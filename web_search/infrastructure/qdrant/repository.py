"""
Qdrant vector store repository.

Responsibilities:

- persist document chunks;
- execute hybrid retrieval;
- update access timestamps.

Implements VectorStore contract.
No business logic.
"""

from __future__ import annotations


from datetime import datetime, timezone


from qdrant_client.models import (
    PointStruct,
)


from web_search.domain.contracts import (
    VectorStore,
)


from web_search.domain.models import (
    DocumentChunk,
    HybridRetrievalResult,
    NormalizedQuery,
)


from web_search.infrastructure.qdrant.client import (
    QdrantConnection,
)


from web_search.infrastructure.qdrant.collection import (
    QdrantCollectionManager,
)


from web_search.infrastructure.qdrant.embeddings import (
    QdrantEmbeddingBuilder,
)



class QdrantRepository(
    VectorStore,
):
    """
    Qdrant implementation of vector storage.
    """


    def __init__(
        self,
        connection: QdrantConnection,
        collection: QdrantCollectionManager,
        embeddings: QdrantEmbeddingBuilder,
        collection_name: str,
    ):
        self._connection = connection
        self._collection = collection
        self._embeddings = embeddings
        self._collection_name = collection_name



    async def store(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Store document chunks.
        """

        if not chunks:
            return


        texts = [
            chunk.text
            for chunk in chunks
        ]


        vectors = await self._embeddings.build_batch(
            texts
        )


        if not vectors:
            return


        await self._collection.ensure(
            vector_size=len(
                vectors[0][0].values
            )
        )


        timestamp = int(
            datetime.now(
                timezone.utc
            ).timestamp()
        )


        points = []


        for chunk, (dense, sparse) in zip(
            chunks,
            vectors,
        ):

            payload = chunk.model_dump()

            payload["last_access"] = timestamp


            points.append(
                PointStruct(
                    id=chunk.id,

                    vector={
                        "dense": dense.values,
                        "bm25": sparse,
                    },

                    payload=payload,
                )
            )


        await self._connection.client.upsert(
            collection_name=self._collection_name,
            points=points,
        )



    async def search(
        self,
        query: NormalizedQuery,
        limit: int,
    ) -> list[HybridRetrievalResult]:
        """
        Hybrid dense + sparse retrieval.

        Fusion:
            Reciprocal Rank Fusion
        """

        client = self._connection.client


        exists = await client.collection_exists(
            self._collection_name
        )


        if not exists:
            return []


        dense_query = (
            await self._embeddings._embedder.embed_query(
                query.normalized
            )
        ).values


        sparse_query = self._embeddings.sparse(
            query.normalized
        )


        result = await client.query_points(
            collection_name=self._collection_name,

            prefetch=[

                {
                    "query": dense_query,
                    "using": "dense",
                    "limit": limit,
                },

                {
                    "query": sparse_query,
                    "using": "bm25",
                    "limit": limit,
                },

            ],

            query={
                "fusion": "rrf"
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
                    retrieved_from="qdrant",
                )
            )


        await self.update_access(
            [
                item.chunk.id
                for item in output
            ]
        )


        return output



    async def update_access(
        self,
        ids: list[str],
    ) -> None:
        """
        Update last access timestamps.
        """

        if not ids:
            return


        timestamp = int(
            datetime.now(
                timezone.utc
            ).timestamp()
        )


        await self._connection.client.set_payload(
            collection_name=self._collection_name,

            payload={
                "last_access": timestamp,
            },

            points=ids,
        )
