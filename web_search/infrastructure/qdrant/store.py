# web_search/infrastructure/qdrant/store.py

from __future__ import annotations


import logging
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


from web_search.domain.contracts import VectorStore, Embedder


from web_search.domain.models import (
    DocumentChunk,
    NormalizedQuery,
    HybridRetrievalResult,
)


logger = logging.getLogger(__name__)



DENSE_NAME = "dense"

SPARSE_NAME = "bm25"



class QdrantVectorStore(VectorStore):
    """
    Qdrant vector storage implementation.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: Embedder,
        collection_name: str,
        sparse_encoder: SparseTextEmbedding,
    ):
        self._client = client
        self._embedder = embedder
        self._collection_name = collection_name
        self._sparse_encoder = sparse_encoder



    async def _collection_exists(
        self,
    ) -> bool:

        return await self._client.collection_exists(
            self._collection_name
        )



    def _sparse_embedding(
        self,
        text: str,
    ) -> SparseVector:

        vector = next(
            self._sparse_encoder.embed(
                [text]
            )
        )

        return SparseVector(
            indices=vector.indices.tolist(),
            values=vector.values.tolist(),
        )



    async def _ensure_collection(
        self,
        vector_size: int,
    ) -> None:

        exists = await self._collection_exists()

        if exists:
            return


        await self._client.create_collection(

            collection_name=self._collection_name,

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



    async def store(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Store document chunks.
        """

        if not chunks:
            return


        vectors = await self._embedder.embed_documents(

            [
                chunk.text
                for chunk in chunks
            ]

        )


        if not vectors:
            return


        await self._ensure_collection(
            len(
                vectors[0].values
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
            vectors,
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
                            self._sparse_embedding(
                                chunk.text
                            ),
                    },

                    payload=payload,
                )
            )


        await self._client.upsert(

            collection_name=self._collection_name,

            points=points,

        )



    async def search(
        self,
        query: NormalizedQuery,
    ) -> list[HybridRetrievalResult]:
        """
        Hybrid dense + sparse retrieval.
        """

        if not await self._collection_exists():

            return []


        dense = (
            await self._embedder.embed_query(
                query.normalized
            )
        )


        sparse = self._sparse_embedding(
            query.normalized
        )


        result = await self._client.query_points(

            collection_name=self._collection_name,

            prefetch=[

                {
                    "query": dense.values,
                    "using": DENSE_NAME,
                    "limit": 10,
                },

                {
                    "query": sparse,
                    "using": SPARSE_NAME,
                    "limit": 10,
                },

            ],

            query={
                "fusion": "rrf"
            },

            limit=10,

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

                    rrf_score=(
                        point.score
                        or 0.0
                    ),

                    retrieved_from="qdrant",
                )
            )


        await self._update_access(
            [
                item.chunk.id
                for item in output
            ]
        )


        return output



    async def _update_access(
        self,
        ids: list[str],
    ) -> None:

        if not ids:
            return


        timestamp = int(
            datetime.now(
                timezone.utc
            ).timestamp()
        )


        await self._client.set_payload(

            collection_name=self._collection_name,

            payload={
                "last_access": timestamp,
            },

            points=ids,
        )



    async def cleanup(
        self,
        days: int,
    ) -> None:
        """
        Remove stale chunks.
        """

        if not await self._collection_exists():

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


        await self._client.delete(

            collection_name=self._collection_name,

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
