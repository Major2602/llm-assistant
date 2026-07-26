"""
Cloudflare Workers AI reranker layer.

Pipeline position:

Embedding similarity
        |
        v
TOP 5-8 chunks
        |
        v
Cloudflare reranker
        |
        v
TOP 3-5 RankedChunk
        |
        v
Extractive compression


Responsibilities:

- semantic reranking;
- deep relevance scoring;
- reducing candidate chunks.

Does NOT:

- retrieve documents;
- generate embeddings;
- compress context;
- store data;
- build LLM context.
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


from web_search.models import (
    RankedChunk,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


REQUEST_TIMEOUT = float(
    os.getenv(
        "CF_RERANK_TIMEOUT",
        "60",
    )
)


RERANK_MODEL = os.getenv(
    "CF_RERANK_MODEL",
    "@cf/baai/bge-reranker-base",
)


DEFAULT_TOP_K = int(
    os.getenv(
        "RERANK_TOP_K",
        "5",
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
    f"{CF_ACCOUNT_ID}/ai/run/{RERANK_MODEL}"
)


# ==========================================================
# Exceptions
# ==========================================================


class CloudflareRerankerError(Exception):
    """
    Cloudflare reranker service error.
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
            "Initializing Cloudflare reranker HTTP client."
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

        )


    return _client



# ==========================================================
# Response parsing
# ==========================================================


def _parse_scores(
    payload: dict[str, Any],
) -> list[float]:
    """
    Extract reranker scores.

    Supported formats:

    result:
    {
        scores: []
    }

    or:

    result:
    {
        data: [
            {
                score: float
            }
        ]
    }
    """

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
        "result"
    )


    if not isinstance(
        result,
        dict,
    ):

        raise CloudflareRerankerError(
            "Missing reranker result."
        )


    raw_scores = (

        result.get(
            "scores"
        )

        or

        result.get(
            "data"
        )

    )


    if not isinstance(
        raw_scores,
        list,
    ):

        raise CloudflareRerankerError(
            "Invalid reranker response."
        )


    scores: list[float] = []


    for item in raw_scores:

        if isinstance(
            item,
            (int, float),
        ):

            scores.append(
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

                scores.append(
                    float(score)
                )


    return scores



# ==========================================================
# Service
# ==========================================================


class CloudflareReranker:
    """
    Cloudflare bge-reranker wrapper.

    Produces RankedChunk objects.
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
    async def _request_scores(

        self,

        query: str,

        documents: list[str],

    ) -> list[float]:

        """
        Request relevance scores.
        """

        if not documents:

            return []


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


        return _parse_scores(
            payload
        )



    async def rerank(

        self,

        query: str,

        chunks: list[dict[str, Any]],

        top_k: int | None = None,

    ) -> list[RankedChunk]:

        """
        Rerank embedding candidates.

        Input:

            TOP 5-8 chunks


        Output:

            TOP 3-5 RankedChunk
        """

        if not chunks:

            return []


        limit = (

            top_k

            if top_k is not None

            else DEFAULT_TOP_K

        )


        documents = [

            chunk.get(
                "text",
                "",
            )

            for chunk in chunks

        ]


        scores = await self._request_scores(

            query,

            documents,

        )


        if len(scores) != len(chunks):

            raise CloudflareRerankerError(

                (
                    "Rerank score count mismatch. "

                    f"Expected={len(chunks)} "

                    f"Received={len(scores)}"

                )

            )


        ranked: list[RankedChunk] = []


        for chunk, score in zip(

            chunks,

            scores,

        ):

            ranked.append(

                RankedChunk(

                    **chunk,

                    rerank_score=score,

                )

            )



        ranked.sort(

            key=lambda item:

                item.rerank_score,

            reverse=True,

        )


        result = ranked[:limit]


        logger.info(

            "Reranked chunks. input=%d output=%d",

            len(chunks),

            len(result),

        )


        return result



# ==========================================================
# Singleton
# ==========================================================


_reranker: CloudflareReranker | None = None



def get_reranker() -> CloudflareReranker:
    """
    Return singleton reranker.
    """

    global _reranker


    if _reranker is None:

        logger.info(
            "Initializing CloudflareReranker singleton."
        )


        _reranker = CloudflareReranker()


    return _reranker
