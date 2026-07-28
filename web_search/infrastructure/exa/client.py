"""
Exa API client.
"""

from __future__ import annotations

import logging

from exa_py import AsyncExa


logger = logging.getLogger(__name__)


class ExaClient:
    """
    Async Exa API wrapper.
    """

    def __init__(
        self,
        api_key: str,
    ):
        self._client = AsyncExa(
            api_key=api_key
        )


    async def search(
        self,
        query: str,
        limit: int,
    ) -> list:
        """
        Execute Exa search.
        """

        logger.info(
            "Exa search query=%s",
            query,
        )

        response = await self._client.search_and_contents(
            query=query,
            num_results=limit,
            text=True,
            type="auto",
        )

        return response.results
