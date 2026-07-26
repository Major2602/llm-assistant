"""
Cloudflare Workers AI Embedding Service.

Pipeline position:

Query
 |
 v
query_preprocessor.py
 |
 v
cloudflare_embeddings.py
 |
 +----------------------+
 |                      |
 v                      v

embedding_retrieval.py   qdrant_store.py


Responsibilities:

- generate query embeddings;
- generate document embeddings;
- communicate with Cloudflare Workers AI;
- batch requests;
- normalize API responses.


This module does NOT know about:

- documents;
- chunks;
- filtering;
- ranking;
- reranking;
- compression logic;
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


REQUEST_TIMEOUT = float(
    os.getenv(
        "CF_EMBEDDING_TIMEOUT",
        "60",
    )
)


EMBEDDING_MODEL = os.getenv(
    "CF_EMBEDDING_MODEL",
    "@cf/qwen/qwen3-embedding-0.6b",
)


MAX_BATCH_SIZE = int(
    os.getenv(
        "CF_EMBEDDING_BATCH_SIZE",
        "32",
    )
)



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
    Cloudflare embedding service error.
    """




# ==========================================================
# HTTP Client
# ==========================================================


_client: httpx.AsyncClient | None = None



def get_http_client() -> httpx.AsyncClient:
    """
    Shared async HTTP client.
    """

    global _client


    if _client is None:

        logger.info(
            "Initializing Cloudflare embedding client."
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
# Batch helpers
# ==========================================================


def _split_batches(
    texts: list[str],
) -> list[list[str]]:
    """
    Split embedding requests into batches.
    """

    if not texts:

        return []


    return [

        texts[index:index + MAX_BATCH_SIZE]

        for index in range(

            0,

            len(texts),

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
    Normalize Cloudflare embedding response.

    Expected:

    {
        "success": true,
        "result": {
            "data": [
                [...]
            ]
        }
    }
    """

    if not payload.get(
        "success",
        False,
    ):

        raise CloudflareEmbeddingError(

            f"Cloudflare error: "
            f"{payload.get('errors')}"

        )



    result = payload.get(
        "result"
    )


    if not isinstance(
        result,
        dict,
    ):

        raise CloudflareEmbeddingError(
            "Invalid embedding result."
        )



    data = result.get(
        "data"
    )


    if not isinstance(
        data,
        list,
    ):

        raise CloudflareEmbeddingError(
            "Embedding data missing."
        )



    embeddings: list[list[float]] = []



    for item in data:


        if isinstance(
            item,
            list,
        ):

            embeddings.append(

                [

                    float(value)

                    for value in item

                ]

            )


            continue



        if isinstance(
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

                    [

                        float(value)

                        for value in vector

                    ]

                )



    if not embeddings:

        raise CloudflareEmbeddingError(
            "No embeddings returned."
        )



    return embeddings




# ==========================================================
# Service
# ==========================================================


class CloudflareEmbeddings:
    """
    Cloudflare Workers AI embedding wrapper.

    Used by:

    - embedding retrieval;
    - Qdrant dense vectors;
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
        Send embedding request.
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

                    f"Expected={len(texts)} "

                    f"Received={len(embeddings)}"

                )

            )



        return embeddings




    async def embed_documents(

        self,

        texts: list[str],

    ) -> list[list[float]]:

        """
        Generate embeddings for documents.

        Used for:

        - chunks;
        - Qdrant dense vectors.
        """

        if not texts:

            return []



        clean_texts = [

            text.strip()

            for text in texts

            if text and text.strip()

        ]



        if not clean_texts:

            return []



        logger.info(

            "Generating document embeddings. count=%d",

            len(clean_texts),

        )



        embeddings: list[list[float]] = []



        for batch in _split_batches(
            clean_texts
        ):


            batch_embeddings = await self._request(
                batch
            )


            embeddings.extend(
                batch_embeddings
            )



        if len(embeddings) != len(clean_texts):

            raise CloudflareEmbeddingError(
                "Final embedding count mismatch."
            )



        return embeddings




    async def embed_query(

        self,

        query: str,

    ) -> list[float]:

        """
        Generate embedding for search query.
        """

        query = query.strip()



        if not query:

            raise CloudflareEmbeddingError(
                "Query cannot be empty."
            )



        embeddings = await self._request(

            [
                query
            ]

        )



        if len(embeddings) != 1:

            raise CloudflareEmbeddingError(
                "Invalid query embedding response."
            )



        return embeddings[0]




# ==========================================================
# Singleton
# ==========================================================


_embedding_service: CloudflareEmbeddings | None = None



def get_embedding_model() -> CloudflareEmbeddings:
    """
    Return singleton embedding service.
    """

    global _embedding_service


    if _embedding_service is None:

        logger.info(
            "Creating Cloudflare embedding singleton."
        )


        _embedding_service = CloudflareEmbeddings()



    return _embedding_service
