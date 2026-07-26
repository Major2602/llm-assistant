"""
Cloudflare Workers AI Embedding Service.

Baseline architecture v1 position:

Query
 |
 v
query_preprocessor.py
 |
 v
cloudflare_embeddings.py
 |
 v
embedding_retrieval.py
 |
 v
qdrant_store.py


Documents:

chunker.py
 |
 v
filter.py
 |
 v
cloudflare_embeddings.py
 |
 v
qdrant_store.py


Responsibilities:

- generate query embeddings;
- generate document embeddings;
- handle Cloudflare API communication;
- manage batching;
- normalize embedding responses.


This module DOES NOT know about:

- Exa;
- documents;
- chunks;
- filtering;
- reranking;
- compression;
- Qdrant;
- agents.
"""


from __future__ import annotations


import logging
import os
from typing import Any


import httpx


from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


logger = logging.getLogger(__name__)



# ==========================================================
# Configuration
# ==========================================================


REQUEST_TIMEOUT = 60.0


EMBEDDING_MODEL = (
    "@cf/qwen/qwen3-embedding-0.6b"
)


# Conservative Cloudflare batch.
#
# Prevents:
# - request size overflow;
# - worker timeout;
# - token limit issues.
#
# Can be tuned later.
MAX_BATCH_SIZE = 32



CF_ACCOUNT_ID = os.getenv(
    "CF_ACCOUNT_ID"
)


CF_API_TOKEN = os.getenv(
    "CF_API_TOKEN"
)



if not CF_ACCOUNT_ID:

    raise RuntimeError(
        "CF_ACCOUNT_ID is not configured."
    )



if not CF_API_TOKEN:

    raise RuntimeError(
        "CF_API_TOKEN is not configured."
    )



API_URL = (

    "https://api.cloudflare.com/client/v4/accounts/"

    f"{CF_ACCOUNT_ID}"

    "/ai/run/"

    f"{EMBEDDING_MODEL}"

)



# ==========================================================
# Exceptions
# ==========================================================


class CloudflareEmbeddingError(Exception):
    """
    Embedding service failure.
    """



# ==========================================================
# HTTP Client
# ==========================================================


_client: httpx.AsyncClient | None = None



def get_http_client() -> httpx.AsyncClient:
    """
    Return shared async HTTP client.
    """

    global _client


    if _client is None:

        logger.info(
            "Initializing Cloudflare embedding HTTP client."
        )


        _client = httpx.AsyncClient(

            timeout=httpx.Timeout(
                REQUEST_TIMEOUT
            ),

            headers={

                "Authorization":
                    f"Bearer {CF_API_TOKEN}",

                "Content-Type":
                    "application/json",

            },

            follow_redirects=True,

        )


    return _client



# ==========================================================
# Batch utilities
# ==========================================================


def _split_batches(
    items: list[str],
) -> list[list[str]]:
    """
    Split texts into safe Cloudflare batches.
    """

    return [

        items[i:i + MAX_BATCH_SIZE]

        for i in range(
            0,
            len(items),
            MAX_BATCH_SIZE,
        )

    ]



# ==========================================================
# Response parsing
# ==========================================================


def _parse_embeddings(
    payload: dict[str, Any],
) -> list[list[float]]:
    """
    Normalize Cloudflare response.

    Supports possible formats:

    [
        [...],
        [...]
    ]

    or:

    [
        {
            "embedding": [...]
        }
    ]
    """

    if not payload.get(
        "success",
        False,
    ):

        raise CloudflareEmbeddingError(

            str(
                payload.get(
                    "errors"
                )
            )

        )


    result = payload.get(
        "result"
    )


    if not isinstance(
        result,
        dict,
    ):

        raise CloudflareEmbeddingError(
            "Missing result object."
        )


    data = result.get(
        "data"
    )


    if not isinstance(
        data,
        list,
    ):

        raise CloudflareEmbeddingError(
            "Invalid embedding response."
        )



    embeddings: list[list[float]] = []



    for item in data:


        if isinstance(
            item,
            list,
        ):

            embeddings.append(
                item
            )


        elif isinstance(
            item,
            dict,
        ):

            vector = item.get(
                "embedding"
            )


            if isinstance(
                vector,
                list,
            ):

                embeddings.append(
                    vector
                )



    return embeddings



# ==========================================================
# Service
# ==========================================================


class CloudflareEmbeddings:
    """
    Cloudflare embedding service.

    Used by:

    - embedding retrieval;
    - Qdrant storage;
    - semantic compression.
    """



    @retry(

        stop=stop_after_attempt(3),

        wait=wait_exponential(

            multiplier=1,

            min=1,

            max=8,

        ),

        retry=retry_if_exception_type(

            httpx.HTTPError

        ),

        reraise=True,

    )
    async def _request(

        self,

        texts: list[str],

    ) -> list[list[float]]:

        """
        Execute embedding request.
        """

        if not texts:

            return []



        client = get_http_client()



        response = await client.post(

            API_URL,

            json={

                "text": texts,

            },

        )



        response.raise_for_status()



        payload = response.json()



        embeddings = _parse_embeddings(
            payload
        )



        if len(embeddings) != len(texts):

            raise CloudflareEmbeddingError(

                (

                    "Embedding count mismatch. "

                    f"Expected {len(texts)}, "

                    f"received {len(embeddings)}"

                )

            )


        return embeddings



    async def embed_documents(

        self,

        texts: list[str],

    ) -> list[list[float]]:

        """
        Generate embeddings for documents/chunks.
        """

        if not texts:

            return []



        logger.info(

            "Embedding documents count=%d",

            len(texts),

        )



        batches = _split_batches(
            texts
        )


        result: list[list[float]] = []



        for batch in batches:


            embeddings = await self._request(

                batch

            )


            result.extend(
                embeddings
            )



        logger.info(

            "Generated %d document embeddings.",

            len(result),

        )


        return result



    async def embed_query(

        self,

        query: str,

    ) -> list[float]:

        """
        Generate embedding for user query.
        """

        if not query.strip():

            raise CloudflareEmbeddingError(
                "Cannot embed empty query."
            )



        embeddings = await self._request(

            [
                query
            ]

        )



        if not embeddings:

            raise CloudflareEmbeddingError(

                "Empty query embedding."

            )



        return embeddings[0]



# ==========================================================
# Singleton
# ==========================================================


_embedding_service: CloudflareEmbeddings | None = None



def get_embedding_model() -> CloudflareEmbeddings:
    """
    Return singleton embedding service.

    Singleton is intentionally preserved
    for Render free tier compatibility.
    """

    global _embedding_service


    if _embedding_service is None:


        logger.info(
            "Initializing CloudflareEmbeddings singleton."
        )


        _embedding_service = CloudflareEmbeddings()



    return _embedding_service
