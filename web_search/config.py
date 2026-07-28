"""
Web search pipeline configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WebSearchConfig:
    """Runtime configuration for web search pipeline."""

    # Exa
    exa_api_key: str=""
    exa_results: int = 100
    
    cache_top_k: int = 10
    embedding_top_k: int = 8
    rerank_top_k: int = 5
    cleanup_days: int = 30


def _get_int_env(
    name: str,
    default: int,
) -> int:
    """
    Safely parse integer environment variable.
    """

    value = os.getenv(name)

    if value is None:
        return default

    try:
        parsed = int(value)

        if parsed < 0:
            raise ValueError

        return parsed

    except ValueError:
        return default


def load_config() -> WebSearchConfig:
    """
    Load configuration from environment.
    """

    return WebSearchConfig(

        exa_api_key=os.getenv(
            "EXA_TOKEN",
            "",
        ),

        exa_results = _get_int_env(
            "EXA_RESULTS",
            100,
        )
        
        cache_top_k=_get_int_env(
            "CACHE_TOP_K",
            10,
        ),
        
        embedding_top_k=_get_int_env(
            "EMBEDDING_TOP_K",
            8,
        ),
        
        rerank_top_k=_get_int_env(
            "RERANK_TOP_K",
            5,
        ),
        
        cleanup_days=_get_int_env(
            "CLEANUP_DAYS",
            30,
        ),
    )
