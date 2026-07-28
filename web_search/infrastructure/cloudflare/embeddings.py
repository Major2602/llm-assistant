"""
Cloudflare embeddings provider.

Infrastructure implementation.
"""

from __future__ import annotations


from typing import Any


from web_search.infrastructure.http import (
    HttpClient,
)


from web_search.domain.contracts import (
    Embedder,
)


from web_search.domain.models import (
    DenseVector,
)



class CloudflareEmbeddingProvider(
    Embedder
):
    """
    Cloudflare Workers AI embeddings.
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



    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[DenseVector]:
        """
        Generate document embeddings.
        """

        if not texts:
            return []


        response = await self.http.request(
            "POST",
            self.url,
            json={
                "text": texts,
            },
            headers={
                "Authorization":
                    f"Bearer {self.token}",
            },
        )


        response.raise_for_status()


        payload = response.json()


        vectors = (
            payload
            .get("result", {})
            .get("data", [])
        )


        return [

            DenseVector(
                values=item
            )

            for item in vectors

        ]



    async def embed_query(
        self,
        query: str,
    ) -> DenseVector:
        """
        Generate query embedding.
        """

        result = await self.embed_documents(
            [query]
        )


        if not result:

            raise RuntimeError(
                "Embedding response is empty."
            )


        return result[0]
