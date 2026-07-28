"""
Exa client adapter.
"""

from __future__ import annotations

import os

from exa_py import AsyncExa


_client: AsyncExa | None = None



def get_exa_client() -> AsyncExa:
    """
    Return singleton Exa client.
    """

    global _client

    if _client is None:

        api_key = os.getenv(
            "EXA_TOKEN"
        )

        if not api_key:
            raise RuntimeError(
                "EXA_TOKEN is missing."
            )

        _client = AsyncExa(
            api_key=api_key
        )

    return _client
