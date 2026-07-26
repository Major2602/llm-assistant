"""
Cloudflare Workers AI reranker.

Responsible for:
- semantic reranking of filtered web chunks;
- dynamic batching for Cloudflare token limits;
- reducing candidate chunks before Qdrant storage.

Pipeline:

Exa
 ↓
chunker.py
 ↓
filter.py
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


RERANK_MODEL = (
    "@cf/baai/bge-reranker-base"
)


# Conservative limit.
# Includes:
# query tokens
# +
# contexts tokens
MAX_RERANK_TOKENS = 450


# Approximation:
# multilingual average.
#
# 1 token ≈ 4 characters
#
# Conservative because
# CJK languages tokenize worse.
CHARS_PER_TOKEN = 4


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

                "Authorization":
                    f"Bearer {CF_API_TOKEN}",

                "Content-Type":
                    "application/json",

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
# Token estimation
# ==========================================================


def _estimate_tokens(
    text: str,
) -> int:
    """
    Estimate token count.

    Used only for batching.
    """

    if not text:
        return 0


    return max(
        1,
        len(text)
        //
        CHARS_PER_TOKEN,
    )



# ==========================================================
# Dynamic batching
# ==========================================================


def _create_batches(
    query: str,
    chunks: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """
    Split chunks into Cloudflare-safe batches.

    Keeps total estimated tokens
    below MAX_RERANK_TOKENS.
    """


    query_tokens = _estimate_tokens(
        query
    )


    available_tokens = (
        MAX_RERANK_TOKENS
        -
        query_tokens
    )


    batches: list[
        list[dict[str, Any]]
    ] = []


    current_batch: list[
        dict[str, Any]
    ] = []


    current_tokens = 0



    for chunk in chunks:


        text = chunk.get(
            "text",
            "",
        )


        chunk_tokens = _estimate_tokens(
            text
        )


        if (

            current_batch

            and

            current_tokens
            +
            chunk_tokens
            >
            available_tokens

        ):


            batches.append(
                current_batch
            )


            current_batch = []

            current_tokens = 0



        current_batch.append(
            chunk
        )


        current_tokens += chunk_tokens



    if current_batch:

        batches.append(
            current_batch
        )


    logger.info(

        "Created %d reranker batches from %d chunks.",

        len(batches),

        len(chunks),

    )


    return batches



# ==========================================================
# HTTP request
# ==========================================================


class CloudflareReranker:


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

            or

            result.get(
                "data"
            )

        )


        if not isinstance(
            scores,
            list,
        ):

            raise CloudflareRerankerError(
                "Invalid reranker response."
            )


        normalized = []


        for item in scores:


            if isinstance(
                item,
                (int,float),
            ):

                normalized.append(
                    float(item)
                )


            elif isinstance(
                item,
                dict,
            ):

                score = (

                    item.get(
                        "score"
                    )

                    or

                    item.get(
                        "relevance_score"
                    )

                )


                if score is not None:

                    normalized.append(
                        float(score)
                    )


        if len(normalized) != len(documents):

            raise CloudflareRerankerError(
                "Score count mismatch."
            )


        return normalized



# ==========================================================
# Batch reranking
# ==========================================================


    async def _rerank_batches(

        self,

        query: str,

        batches: list[list[dict[str,Any]]],

    ) -> list[dict[str,Any]]:


        ranked = []


        for index, batch in enumerate(
            batches,
            start=1,
        ):


            logger.info(

                "Processing reranker batch %d/%d size=%d",

                index,

                len(batches),

                len(batch),

            )


            texts = [

                item.get(
                    "text",
                    "",
                )

                for item in batch

            ]


            scores = await self._request(

                query,

                texts,

            )


            for chunk, score in zip(

                batch,

                scores,

            ):

                ranked.append(

                    {

                        **chunk,

                        "rerank_score":
                            score,

                    }

                )


        return ranked



# ==========================================================
# Public API
# ==========================================================


    async def rerank(

        self,

        query: str,

        chunks: list[dict[str,Any]],

        top_k: int = 5,

    ) -> list[dict[str,Any]]:


        if not chunks:

            return []


        batches = _create_batches(

            query,

            chunks,

        )


        ranked = await self._rerank_batches(

            query,

            batches,

        )


        ranked.sort(

            key=lambda x:
                x["rerank_score"],

            reverse=True,

        )


        result = ranked[:top_k]


        logger.info(

            "Reranked %d chunks -> %d chunks.",

            len(chunks),

            len(result),

        )


        return result



# ==========================================================
# Singleton
# ==========================================================

_reranker: CloudflareReranker | None = None



def get_reranker() -> CloudflareReranker:


    global _reranker


    if _reranker is None:

        logger.info(
            "Initializing CloudflareReranker."
        )


        _reranker = CloudflareReranker()


    return _reranker
