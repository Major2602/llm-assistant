"""
Cloudflare reranker provider.
"""

from __future__ import annotations


from web_search.infrastructure.http import (
    HttpClient,
)


from web_search.domain.contracts import (
    Reranker,
)


from web_search.domain.models import (
    RankedChunk,
    EmbeddedChunk,
)



class CloudflareRerankerProvider(
    Reranker
):
    """
    BGE reranker implementation.
    """

    def __init__(
        self,
        http: HttpClient,
        account_id: str,
        token: str,
        model: str,
    ):

        self.http = http

        self.url = (
            "https://api.cloudflare.com/client/v4/"
            f"accounts/{account_id}/ai/run/{model}"
        )

        self.token = token



    async def rerank(
        self,
        query: str,
        chunks: list[EmbeddedChunk],
        top_k: int,
    ) -> list[RankedChunk]:

        if not chunks:

            return []


        response = await self.http.request(
            "POST",
            self.url,
            json={
                "query": query,
                "contexts": [
                    chunk.text
                    for chunk in chunks
                ],
            },
            headers={
                "Authorization":
                    f"Bearer {self.token}",
            },
        )


        response.raise_for_status()


        payload = response.json()


        scores = (
            payload
            .get("result", {})
            .get("scores", [])
        )


        ranked = []


        for chunk, score in zip(
            chunks,
            scores,
        ):

            ranked.append(

                RankedChunk(

                    **chunk.model_dump(),

                    rerank_score=float(
                        score
                    ),

                )

            )


        ranked.sort(
            key=lambda item:
                item.rerank_score,
            reverse=True,
        )


        return ranked[:top_k]
