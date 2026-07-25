"""
Cloudflare Workers AI embeddings client.

Responsible for:
- generating embeddings for documents;
- generating embeddings for queries.

Pipeline position:

chunker.py
      |
      v
cloudflare_embeddings.py
      |
      v
reranker.py
      |
      v
qdrant_store.py

This module does not know about:
- Exa;
- filtering;
- chunking;
- reranking;
- Qdrant.
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


MODEL_NAME = (
    "@cf/qwen/qwen3-embedding-0.6b"
)


CF_ACCOUNT_ID = os.getenv(
    "CF_ACCOUNT_ID"
)

CF_API_TOKEN = os.getenv(
    "CF_API_TOKEN"
)


if not CF_ACCOUNT_ID:
    raise RuntimeError(
        "Environment variable CF_ACCOUNT_ID is not configured."
    )


if not CF_API_TOKEN:
    raise RuntimeError(
        "Environment variable CF_API_TOKEN is not configured."
    )


API_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    f"{CF_ACCOUNT_ID}/ai/run/{MODEL_NAME}"
)


# ==========================================================
# Exceptions
# ==========================================================


class CloudflareEmbeddingError(Exception):
    """
    Cloudflare embedding API error.
    """



# ==========================================================
# HTTP Client
# ==========================================================


_client: httpx.AsyncClient | None = None



def get_http_client() -> httpx.AsyncClient:
    """
    Create reusable async HTTP client.
    """

    global _client


    if _client is None:

        logger.info(
            "Initializing Cloudflare AsyncClient."
        )


        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                REQUEST_TIMEOUT
            ),

            headers={
                "Authorization": (
                    f"Bearer {CF_API_TOKEN}"
                ),

                "Content-Type": (
                    "application/json"
                ),
            },

            follow_redirects=True,
        )


    return _client



# ==========================================================
# Embeddings
# ==========================================================


class CloudflareEmbeddings:
    """
    Cloudflare Workers AI embedding wrapper.
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
    async def _embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Send texts to Cloudflare embedding model.
        """


        if not texts:
            return []


        logger.info(
            "Requesting embeddings. Count=%d",
            len(texts),
        )


        client = get_http_client()


        try:

            response = await client.post(
                API_URL,

                json={
                    "text": texts,
                },
            )


            if response.status_code != 200:

                logger.error(
                    "Cloudflare HTTP error %d: %s",
                    response.status_code,
                    response.text,
                )

                response.raise_for_status()



            payload: dict[str, Any] = (
                response.json()
            )


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
                    "Invalid embeddings data format."
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

                    if vector:
                        embeddings.append(
                            vector
                        )



            if len(embeddings) != len(texts):

                raise CloudflareEmbeddingError(
                    (
                        "Embedding count mismatch. "
                        f"Expected={len(texts)} "
                        f"Received={len(embeddings)}"
                    )
                )


            logger.info(
                "Received %d embeddings.",
                len(embeddings),
            )


            return embeddings



        except httpx.HTTPError:

            logger.exception(
                "Cloudflare embedding request failed."
            )

            raise



        except Exception:

            logger.exception(
                "Unexpected embedding failure."
            )

            raise



    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Create embeddings for document chunks.
        """

        return await self._embed(
            texts
        )



    async def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Create embedding for search query.
        """

        embeddings = await self._embed(
            [
                query
            ]
        )


        if not embeddings:

            raise CloudflareEmbeddingError(
                "Cloudflare returned empty query embedding."
            )


        return embeddings[0]



# ==========================================================
# Singleton
# ==========================================================


_embeddings: CloudflareEmbeddings | None = None



def get_embedding_model() -> CloudflareEmbeddings:
    """
    Return singleton embedding client.
    """

    global _embeddings


    if _embeddings is None:

        logger.info(
            "Initializing CloudflareEmbeddings."
        )

        _embeddings = CloudflareEmbeddings()


    return _embeddings
