"""
Shared HTTP infrastructure layer.

Responsibilities:

- unified async HTTP client;
- timeout management;
- retry policy;
- common headers;
- transport-level errors.

No business logic here.
"""

from __future__ import annotations


import asyncio
import logging

from dataclasses import dataclass, field
from typing import Any


import httpx


from web_search.domain.exceptions import (
    TransientError,
    ProviderUnavailable,
)


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


@dataclass(frozen=True)
class HttpSettings:
    """
    HTTP client configuration.
    """

    timeout: float = 30.0

    retries: int = 3

    retry_delay: float = 1.0

    headers: dict[str, str] = field(
        default_factory=dict
    )


# ==========================================================
# Client
# ==========================================================


class HttpClient:
    """
    Shared async HTTP client.

    Created by application bootstrap
    and injected into providers.
    """

    def __init__(
        self,
        settings: HttpSettings,
    ):
        self.settings = settings

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.timeout
            ),
            headers=settings.headers,
        )


    async def close(self) -> None:
        """
        Close HTTP resources.
        """

        await self._client.aclose()



    async def request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """
        Execute HTTP request with retry.
        """

        last_error: Exception | None = None


        for attempt in range(
            self.settings.retries + 1
        ):

            try:

                response = await self._client.request(
                    method,
                    url,
                    json=json,
                    headers=headers,
                )


                if response.status_code >= 500:

                    raise TransientError(
                        f"Server error {response.status_code}"
                    )


                return response


            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                TransientError,
            ) as error:

                last_error = error


                if attempt >= self.settings.retries:

                    break


                delay = (
                    self.settings.retry_delay
                    *
                    (attempt + 1)
                )


                logger.warning(
                    "HTTP retry attempt=%d delay=%s error=%s",
                    attempt + 1,
                    delay,
                    error,
                )


                await asyncio.sleep(
                    delay
                )


        raise ProviderUnavailable(
            "HTTP request failed"
        ) from last_error
