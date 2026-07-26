"""
Cloudflare Workers AI reranker layer.

Baseline architecture v1:

Embedding similarity
        |
        v
TOP 5-8 chunks
        |
        v
Cloudflare reranker
        |
        v
TOP 3-5 chunks
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
- format LLM context.
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
        "CF_RERANK_TIMEOUT",
        "60",
    )
)


RERANK_MODEL = os.getenv(
    "CF_RERANK_MODEL",
    "@cf/baai/bge-reranker-base",
)


RERANK_TOP_K = int(
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
    Cloudflare reranker API error.
    """



# ==========================================================
# HTTP Client
# ==========================================================


_client: httpx.AsyncClient | None = None



def get_http_client() -> httpx.AsyncClient:
    """
    Singleton async HTTP client.
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


def _extract_scores(
    payload: dict[str, Any],
) -> list[float]:
    """
    Extract scores from Cloudflare response.

    Supports possible API formats:

    {
        result:
        {
            scores:[]
        }
    }


    or:


    {
        result:
        {
            data:[]
        }
    }
    """



    result = payload.get(
        "result",
        {},
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
            "Invalid reranker scores format."
        )



    scores: list[float] = []



    for item in raw_scores:


        if isinstance(
            item,
            (float, int),
        ):

            scores.append(
                float(item)
            )


        elif isinstance(
            item,
            dict,
        ):


            value = (

                item.get(
                    "score"
                )

                or

                item.get(
                    "relevance_score"
                )

            )


            if value is not None:

                scores.append(
                    float(value)
                )



    return scores




# ==========================================================
# Cloudflare API
# ==========================================================


class CloudflareReranker:
    """
    Cloudflare bge-reranker wrapper.
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
    async def _score(

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



        logger.info(

            "Cloudflare rerank request. "
            "documents=%d",

            len(documents),

        )



        response = await client.post(

            API_URL,

            json={

                "query": query,

                "contexts": documents,

            },

        )



        if response.status_code >= 400:


            logger.error(

                "Cloudflare reranker error %s: %s",

                response.status_code,

                response.text,

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



        scores = _extract_scores(
            payload
        )



        if len(scores) != len(documents):

            raise CloudflareRerankerError(

                (
                    "Score count mismatch. "

                    f"Expected={len(documents)} "

                    f"Received={len(scores)}"

                )

            )



        return scores




# ==========================================================
# Public API
# ==========================================================


    async def rerank(

        self,

        query: str,

        chunks: list[dict[str, Any]],

        top_k: int | None = None,

    ) -> list[dict[str, Any]]:

        """
        Rerank already filtered semantic candidates.

        Input:

            TOP 5-8 chunks
            from embedding retrieval


        Output:

            TOP 3-5 chunks
            with rerank_score
        """



        if not chunks:

            return []



        limit = (

            top_k

            if top_k is not None

            else RERANK_TOP_K

        )



        documents = [

            chunk.get(
                "text",
                "",
            )

            for chunk in chunks

        ]



        scores = await self._score(

            query,

            documents,

        )



        ranked: list[dict[str, Any]] = []



        for chunk, score in zip(

            chunks,

            scores,

        ):


            ranked.append(

                {

                    **chunk,

                    "rerank_score":
                        score,

                }

            )



        ranked.sort(

            key=lambda item:

                item.get(
                    "rerank_score",
                    0,
                ),

            reverse=True,

        )



        result = ranked[:limit]



        logger.info(

            "Reranked chunks. "
            "input=%d output=%d",

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
    Return singleton reranker instance.
    """


    global _reranker



    if _reranker is None:

        logger.info(
            "Initializing CloudflareReranker."
        )


        _reranker = CloudflareReranker()



    return _reranker
