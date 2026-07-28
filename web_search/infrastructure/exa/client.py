"""
Exa API client.

Infrastructure layer only.
"""

from __future__ import annotations


from typing import Any


from web_search.infrastructure.http import (
    HttpClient,
)


class ExaClient:
    """
    Low-level Exa HTTP client.
    """


    API_URL = (
        "https://api.exa.ai/search"
    )


    def __init__(
        self,
        http: HttpClient,
        api_key: str,
    ):
        self.http = http
        self.api_key = api_key



    async def search(
        self,
        query: str,
        limit: int,
    ) -> list[Any]:
        """
        Execute Exa request.
        """

        response = await self.http.request(
            "POST",
            self.API_URL,
            json={
                "query": query,
                "num_results": limit,
                "contents": {
                    "text": True,
                },
            },
            headers={
                "x-api-key": self.api_key,
            },
        )


        response.raise_for_status()


        payload = response.json()


        return (
            payload.get("results")
            or []
        )
