"""
Shared Cloudflare HTTP client.
"""

from __future__ import annotations

import httpx


class CloudflareClient:

    def __init__(
        self,
        account_id: str,
        token: str,
        timeout: float = 60,
    ):
        self.base_url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/"
        )

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )


    async def post(
        self,
        model: str,
        payload: dict,
    ) -> dict:
        response = await self.client.post(
            self.base_url + model,
            json=payload,
        )

        response.raise_for_status()

        return response.json()
