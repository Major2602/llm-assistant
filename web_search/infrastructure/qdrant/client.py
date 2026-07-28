"""
Qdrant client infrastructure layer.
"""

from __future__ import annotations

import logging
import os

from qdrant_client import AsyncQdrantClient


logger = logging.getLogger(__name__)


_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    """
    Return singleton Qdrant async client.
    """

    global _client

    if _client is None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url:
            raise RuntimeError(
                "QDRANT_URL is missing."
            )

        logger.info(
            "Initializing Qdrant client."
        )

        _client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
        )

    return _client


async def close_qdrant_client() -> None:
    """
    Close Qdrant client.
    """

    global _client

    if _client is not None:
        await _client.close()
        _client = None
