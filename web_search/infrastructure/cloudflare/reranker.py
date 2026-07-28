# web_search/infrastructure/cloudflare/reranker.py

from __future__ import annotations


import logging


from web_search.domain.models import (
    EmbeddedChunk,
    RankedChunk,
)


from web_search.infrastructure.http import HttpClient
from web_search.domain.contracts import Reranker


logger = logging.getLogger(__name__)



class CloudflareReranker(Reranker):
    """
    Cloudflare semantic reranker implementation.
    """

    def __init__(
        self,
        http_client: HttpClient,
        api_url: str,
    ):
        self._http_client = http_client
        self._api_url = api_url



    async def _request_scores(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """
        Request relevance scores from Cloudflare.
        """

        if not documents:
            return []


        response = await self._http_client.request(
            method="POST",
            url=self._api_url,
            json={
                "query": query,
                "contexts": documents,
            },
        )


        if not response.get(
            "success",
            False,
        ):
            raise RuntimeError(
                f"Cloudflare reranker error: {response}"
            )


        result = response.get(
            "result",
            {},
        )


        raw_scores = (
            result.get("scores")
            or result.get("data")
            or []
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
                    item.get("score")
                    or item.get("relevance_score")
                )


                if score is not None:
                    scores.append(
                        float(score)
                    )


        return scores



    async def rerank(
        self,
        query: str,
        chunks: list[EmbeddedChunk],
    ) -> list[RankedChunk]:
        """
        Rerank embedded chunks.
        """

        if not chunks:
            return []


        documents = [
            chunk.text
            for chunk in chunks
        ]


        scores = await self._request_scores(
            query=query,
            documents=documents,
        )


        if len(scores) != len(chunks):

            raise RuntimeError(
                (
                    "Reranker score count mismatch. "
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
            key=lambda item: item.rerank_score,
            reverse=True,
        )


        return ranked
