"""
Qdrant client infrastructure layer.

Responsibilities:

- create Qdrant connection;
- manage client lifecycle;
- isolate external dependency;
- provide async client via DI.

No singleton state.
"""

from __future__ import annotations


from qdrant_client import AsyncQdrantClient


from web_search.domain.exceptions import (
    InfrastructureConfigurationError,
)


class QdrantClientFactory:
    """
    Factory for creating Qdrant clients.

    Client lifecycle is controlled by application layer.
    """


    def create(
        self,
        url: str,
        api_key: str | None = None,
    ) -> AsyncQdrantClient:
        """
        Create async Qdrant client.
        """

        if not url:
            raise InfrastructureConfigurationError(
                "Qdrant URL is required."
            )


        return AsyncQdrantClient(
            url=url,
            api_key=api_key,
        )


class QdrantConnection:
    """
    Thin wrapper around Qdrant client.

    Used by repositories.
    """


    def __init__(
        self,
        client: AsyncQdrantClient,
    ):
        self._client = client


    @property
    def client(
        self,
    ) -> AsyncQdrantClient:
        """
        Access underlying client.
        """

        return self._client


    async def close(
        self,
    ) -> None:
        """
        Close Qdrant connection.
        """

        await self._client.close()
