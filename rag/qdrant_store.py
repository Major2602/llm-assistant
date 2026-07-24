import logging
import os
from datetime import datetime, timedelta, timezone

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    Range,
    VectorParams,
)

from rag.cloudflare_embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# ==========================================================
# Configuration
# ==========================================================

QDRANT_URL = os.getenv("QDRANT_URL")

if not QDRANT_URL:
    raise RuntimeError("Environment variable QDRANT_URL is not configured.")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "web_memory",
)

# ==========================================================
# Client
# ==========================================================

_client: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    global _client

    if _client is None:
        logger.info("Initializing AsyncQdrantClient.")

        _client = AsyncQdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    return _client
    

# ==========================================================
# Payload Index
# ==========================================================

async def ensure_payload_indexes() -> None:

    client = get_qdrant()

    try:

        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="last_access",
            field_schema="integer",
        )

        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="query",
            field_schema="keyword",
        )

        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="provider",
            field_schema="keyword",
        )

        logger.info(
            "Payload indexes are ready."
        )

    except Exception:
        logger.exception(
            "Failed creating payload indexes."
        )
        raise


# ==========================================================
# Collection
# ==========================================================

async def ensure_collection(
    vector_size: int,
) -> None:

    client = get_qdrant()

    try:
        
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

        logger.info(
            "Qdrant collection '%s' is ready.",
            COLLECTION_NAME,
        )

    except Exception:
        logger.exception(
            "Failed ensuring collection '%s'.",
            COLLECTION_NAME,
        )
        raise


# ==========================================================
# Insert
# ==========================================================

async def add_chunks(
    chunks: list[dict],
) -> None:

    if not chunks:
        return

    logger.info(
        "Adding %d web memory chunks into Qdrant.",
        len(chunks),
    )

    embedder = get_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    try:

        vectors = await embedder.embed_documents(
            texts
        )

        if not vectors:
            raise RuntimeError(
                "Embedding model returned no vectors."
            )

        if len(vectors) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        await ensure_collection(
            len(vectors[0])
        )

        points = [

            PointStruct(
                id=chunk["id"],
                vector=vector,
                payload=chunk,
            )

            for chunk, vector in zip(
                chunks,
                vectors,
            )

        ]

        await get_qdrant().upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        logger.info(
            "Inserted %d chunks.",
            len(points),
        )

    except Exception:
        logger.exception(
            "Failed inserting chunks into Qdrant."
        )
        raise


# ==========================================================
# Search
# ==========================================================

async def search(
    query: str,
    limit: int = 5,
    score_threshold: float = 0.70,
) -> list[dict]:

    logger.info(
        "Searching query='%s' limit=%d threshold=%.2f",
        query,
        limit,
        score_threshold,
    )

    embedder = get_embedding_model()

    try:

        query_vector = await embedder.embed_query(
            query
        )

        await ensure_collection(
            len(query_vector)
        )

        result = await get_qdrant().query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

        hits = result.points

        if not hits:
            logger.info(
                "No chunks found."
            )
            return []

        await update_last_access(
            [
                point.id
                for point in hits
            ]
        )

        logger.info(
            "Found %d chunks.",
            len(hits),
        )

        return [

            {
                **point.payload,
                "score": point.score,
            }

            for point in hits

        ]

    except Exception:
        logger.exception(
            "Qdrant search failed."
        )
        raise


# ==========================================================
# Update access
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

    try:

        await get_qdrant().set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "last_access": timestamp
            },
            points=ids,
        )

        logger.debug(
            "Updated last_access for %d chunks.",
            len(ids),
        )

    except Exception:
        logger.exception(
            "Failed updating last_access."
        )
        raise


# ==========================================================
# Cleanup
# ==========================================================

async def cleanup_old_chunks(
    days: int = 30,
) -> None:

    client = get_qdrant()

    try:
        
        if not await client.collection_exists(
            COLLECTION_NAME
        ):
            logger.info(
                "Qdrant collection '%s' does not exist. Skip cleanup.",
                COLLECTION_NAME,
            )
            return
            
        cutoff = int(
            (
                datetime.now(timezone.utc)
                - timedelta(days=days)
            ).timestamp()
        )

        logger.info(
            "Removing inactive web memory chunks older than %d days.",
            days,
        )


        await client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="last_access",
                        range=Range(
                            lt=cutoff,
                        ),
                    )
                ]
            ),
        )

        logger.info(
            "Cleanup finished."
        )

    except Exception:
        logger.exception(
            "Failed cleaning old chunks."
        )
        raise
