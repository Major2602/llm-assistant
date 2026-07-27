"""
Cloudflare Workers AI reranker layer.

Module Responsibilities:

- semantic reranking;
- deep relevance scoring;
- reducing candidate chunks.
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
    EmbeddedChunk,
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
    Cloudflare reranker exception.
    """



# ==========================================================
# HTTP client
# ==========================================================


_client: httpx.AsyncClient | None = None



def get_http_client() -> httpx.AsyncClient:
    """
    Shared async HTTP client.
    """

    global _client


    if _client is None:

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
    Extract scores from Cloudflare response.
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
# Reranker service
# ==========================================================


class CloudflareReranker:
    """
    Cloudflare bge-reranker-base wrapper.
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


        return _parse_scores(
            response.json()
        )



    async def rerank(

        self,

        query: str,

        chunks: list[EmbeddedChunk],

        top_k: int = DEFAULT_TOP_K,

    ) -> list[RankedChunk]:
        """
        Rerank embedding candidates.
        """


        if not chunks:

            return []


        documents = [

            chunk.text

            for chunk in chunks

        ]


        scores = await self._request_scores(

            query,

            documents,

        )


        if len(scores) != len(chunks):

            raise CloudflareRerankerError(

                (

                    "Score count mismatch. "

                    f"chunks={len(chunks)} "

                    f"scores={len(scores)}"

                )

            )



        ranked = [

            RankedChunk(

                **chunk.model_dump(),

                rerank_score=score,

            )

            for chunk, score

            in zip(
                chunks,
                scores,
            )

        ]



        ranked.sort(

            key=lambda item:

                item.rerank_score,

            reverse=True,

        )


        result = ranked[:top_k]


        logger.info(

            "Reranking completed. selected=%d/%d",

            len(result),

            len(chunks),

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

        _reranker = CloudflareReranker()


    return _reranker
