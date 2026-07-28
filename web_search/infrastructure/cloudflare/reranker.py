"""
Cloudflare reranker provider.
"""

from __future__ import annotations

import logging

from web_search.domain.contracts import (
    RerankerProvider,
)

from web_search.domain.models import (
    EmbeddedChunk,
    RankedChunk,
)

from web_search.infrastructure.cloudflare.client import (
    CloudflareClient,
)


logger = logging.getLogger(__name__)


class CloudflareRerankerProvider(
    RerankerProvider
):

    def __init__(
        self,
        client: CloudflareClient,
        model: str,
        top_k: int = 5,
    ):
        self.client = client
        self.model = model
        self.top_k = top_k


    def _parse_scores(
        self,
        payload: dict,
    ) -> list[float]:

        data = (
            payload
            .get("result", {})
            .get("scores")
            or []
        )

        return [
            float(
                item.get("score")
                if isinstance(item, dict)
                else item
            )
            for item in data
        ]


    async def rerank(
        self,
        query: str,
        chunks: list[EmbeddedChunk],
    ) -> list[RankedChunk]:

        if not chunks:
            return []


        payload = await self.client.post(
            self.model,
            {
                "query": query,
                "contexts": [
                    chunk.text
                    for chunk in chunks
                ],
            },
        )


        scores = self._parse_scores(
            payload
        )


        ranked = [

            RankedChunk(
                **chunk.model_dump(),
                rerank_score=score,
            )

            for chunk, score
            in zip(chunks, scores)

        ]


        ranked.sort(
            key=lambda x: x.rerank_score,
            reverse=True,
        )


        return ranked[:self.top_k]
