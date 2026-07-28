"""
Cloudflare HTTP client adapter.
"""

from __future__ import annotations

from web_search.infrastructure.http import (
    get_http_client,
)


def get_cloudflare_client():
    """
    Return shared Cloudflare HTTP client.
    """

    return get_http_client(
        name="cloudflare",
        headers={
            "Authorization": (
                f"Bearer {__import__('os').getenv('CF_API_TOKEN')}"
            ),
            "Content-Type": "application/json",
        },
    )
