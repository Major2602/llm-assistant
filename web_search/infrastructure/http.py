"""
Shared HTTP infrastructure.

Responsibilities:
- provide reusable async HTTP clients;
- centralize timeout configuration;
- manage client lifecycle.
"""

from __future__ import annotations

import logging
import os

import httpx


logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = float(
    os.getenv(
        "HTTP_TIMEOUT",
        "60",
    )
)


_clients: dict[str, httpx.AsyncClient] = {}



def get_http_client(
    name: str = "default",
    *,
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """
    Return named singleton HTTP client.
    """

    if name in _clients:
        return _clients[name]


    logger.info(
        "Creating HTTP client=%s",
        name,
    )


    client = httpx.AsyncClient(

        timeout=httpx.Timeout(
            timeout
            or DEFAULT_TIMEOUT
        ),

        headers=headers,

        follow_redirects=True,

    )


    _clients[name] = client


    return client



async def close_http_clients() -> None:
    """
    Close all HTTP clients.
    """

    for name, client in _clients.items():

        logger.info(
            "Closing HTTP client=%s",
            name,
        )

        await client.aclose()


    _clients.clear()
