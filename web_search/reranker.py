"""
Cloudflare Workers AI reranker.

Responsible for:
- semantic reranking of filtered web chunks;
- reducing candidate chunks before Qdrant storage.

Pipeline position:

Exa
 ↓
filter.py
 ↓
chunker.py
 ↓
embedding
 ↓
reranker.py
 ↓
qdrant_store.py


Uses:
@cf/baai/bge-reranker-base
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

RERANK_MODEL = "@cf/baai/bge-reranker-base"

MAX_RERANK_TOKENS = 450

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
    f"{CF_ACCOUNT_ID}/ai/run/{RERANK_MODEL}"
)


# ==========================================================
# Client
# ==========================================================

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """
    Lazily initialize Cloudflare HTTP client.
    """

    global _client


    if _client is None:

        logger.info(
            "Initializing Cloudflare reranker client."
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
# Exceptions
# ==========================================================

class CloudflareRerankerError(Exception):
    """
    Cloudflare Workers AI reranker error.
    """



# ==========================================================
# Reranker
# ==========================================================


class CloudflareReranker:
    """
    Cloudflare BGE reranker.

    Input:
        query
        list of chunks

    Output:
        sorted chunks with rerank scores
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
        query: str,
        documents: list[str],
    ) -> list[float]:

        logger.debug(
            "Reranking %d documents.",
            len(documents),
        )


        try:

            client = get_http_client()


            response = await client.post(
                API_URL,
                json={
                    "query": query,
                    "contexts": documents,
                },
            )


            response.raise_for_status()


            payload = response.json()


            if not payload.get(
                "success",
                False,
            ):

                raise CloudflareRerankerError(
                    str(
                        payload.get(
                            "errors"
                        )
                    )
                )


            result = payload.get(
                "result",
                {},
            )


            scores = (
                result.get(
                    "scores"
                )
                or result.get(
                    "data"
                )
            )


            if not isinstance(
                scores,
                list,
            ):

                raise CloudflareRerankerError(
                    "Invalid reranker response format."
                )


            normalized_scores: list[float] = []


            for item in scores:

                if isinstance(
                    item,
                    (int, float),
                ):

                    normalized_scores.append(
                        float(item)
                    )


                elif isinstance(
                    item,
                    dict,
                ):

                    score = (
                        item.get("score")
                        or item.get(
                            "relevance_score"
                        )
                    )

                    if score is not None:
                        normalized_scores.append(
                            float(score)
                        )


            if len(normalized_scores) != len(documents):

                raise CloudflareRerankerError(
                    "Reranker score count mismatch."
                )


            return normalized_scores


        except httpx.HTTPError:

            logger.exception(
                "Cloudflare reranker HTTP error."
            )

            raise


        except Exception:

            logger.exception(
                "Unexpected reranker error."
            )

            raise



    async def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Rank chunks by relevance.

        Keeps only top_k chunks.
        """


        if not chunks:

            return []


        documents = [
            chunk.get(
                "text",
                "",
            )
            for chunk in chunks
        ]


        scores = await self._request(
            query,
            documents,
        )


        ranked = []


        for chunk, score in zip(
            chunks,
            scores,
        ):

            item = {
                **chunk,
                "rerank_score": score,
            }

            ranked.append(
                item
            )


        ranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )


        selected = ranked[:top_k]


        logger.info(
            "Reranked %d chunks -> %d chunks.",
            len(chunks),
            len(selected),
        )


        return selected



# ==========================================================
# Singleton
# ==========================================================

_reranker: CloudflareReranker | None = None



def get_reranker() -> CloudflareReranker:
    """
    Return singleton reranker instance.
    """

    global _reranker


    if _reranker is None:

        logger.info(
            "Initializing CloudflareReranker."
        )

        _reranker = CloudflareReranker()


    return _reranker
