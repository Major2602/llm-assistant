# web_search/infrastructure/cloudflare/embeddings.py

from __future__ import annotations


import logging


from web_search.domain.contracts import Embedder
from web_search.domain.models import DenseVector
from web_search.infrastructure.http import HttpClient


logger = logging.getLogger(__name__)



class CloudflareEmbeddingClient(Embedder):
    """
    Cloudflare Workers AI embeddings implementation.
    """

    def __init__(
        self,
        http_client: HttpClient,
        api_url: str,
    ):
        self._http_client = http_client
        self._api_url = api_url



    async def _request_embeddings(
        self,
        inputs: list[str],
    ) -> list[DenseVector]:
        """
        Request embeddings from Cloudflare.
        """

        if not inputs:
            return []


        response = await self._http_client.request(
            method="POST",
            url=self._api_url,
            json={
                "text": inputs,
            },
        )


        if not response.get(
            "success",
            False,
        ):
            raise RuntimeError(
                f"Cloudflare embeddings error: {response}"
            )


        result = response.get(
            "result",
            {},
        )


        vectors = (
            result.get("data")
            or result.get("embeddings")
            or []
        )


        output: list[DenseVector] = []


        for vector in vectors:

            if isinstance(
                vector,
                dict,
            ):

                values = (
                    vector.get("embedding")
                    or vector.get("values")
                )

            else:

                values = vector


            if not isinstance(
                values,
                list,
            ):
                continue


            output.append(
                DenseVector(
                    values=[
                        float(item)
                        for item in values
                    ]
                )
            )


        return output



    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[DenseVector]:
        """
        Generate embeddings for documents.
        """

        return await self._request_embeddings(
            texts,
        )



    async def embed_query(
        self,
        query: str,
    ) -> DenseVector:
        """
        Generate embedding for query.
        """

        result = await self._request_embeddings(
            [query],
        )


        if not result:

            raise RuntimeError(
                "Cloudflare returned empty query embedding."
            )


        return result[0]
