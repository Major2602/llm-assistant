"""
Cloudflare embeddings provider.
"""

from __future__ import annotations

import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import httpx

from web_search.domain.contracts import (
    EmbeddingProvider,
)

from web_search.domain.models import (
    DenseVector,
)

from web_search.infrastructure.cloudflare.client import (
    CloudflareClient,
)


logger = logging.getLogger(__name__)


class CloudflareEmbeddingProvider(
    EmbeddingProvider
):

    def __init__(
        self,
        client: CloudflareClient,
        model: str,
        dimension: int = 1024,
    ):
        self.client = client
        self.model = model
        self.dimension = dimension


    def _parse(
        self,
        payload: dict,
    ) -> list[list[float]]:

        data = (
            payload
            .get("result", {})
            .get("data", [])
        )

        vectors = []

        for item in data:

            vector = (
                item.get("embedding")
                if isinstance(item, dict)
                else item
            )

            if vector:
                vectors.append(
                    [
                        float(x)
                        for x in vector
                    ]
                )

        return vectors


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(
            min=1,
            max=8,
        ),
        retry=retry_if_exception_type(
            httpx.HTTPError
        ),
    )
    async def _request(
        self,
        texts: list[str],
    ) -> list[DenseVector]:

        payload = await self.client.post(
            self.model,
            {
                "text": texts
            },
        )

        vectors = self._parse(
            payload
        )

        return [
            DenseVector(
                values=v
            )
            for v in vectors
        ]


    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[DenseVector]:

        if not texts:
            return []

        return await self._request(
            texts
        )


    async def embed_query(
        self,
        query: str,
    ) -> DenseVector:

        result = await self._request(
            [query]
        )

        return result[0]
