"""
Web search pipeline configuration.
"""

import os


# Memory retrieval

CACHE_TOP_K = int(
    os.getenv(
        "CACHE_TOP_K",
        "10",
    )
)


# Dense retrieval

EMBEDDING_TOP_K = int(
    os.getenv(
        "EMBEDDING_TOP_K",
        "8",
    )
)


# Reranking

RERANK_TOP_K = int(
    os.getenv(
        "RERANK_TOP_K",
        "5",
    )
)


# Cleanup

CLEANUP_DAYS = int(
    os.getenv(
        "CLEANUP_DAYS",
        "30",
    )
)
