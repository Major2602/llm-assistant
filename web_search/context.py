import asyncio
import logging
from typing import Any

from web_search.qdrant_store import (
    add_chunks,
    cleanup_old_chunks,
    search
)
from web_search.exa import search_exa

logger = logging.getLogger(__name__)

# ==========================================================
# Configuration
# ==========================================================

TOP_K = 5
SIMILARITY_THRESHOLD = 0.60

# ==========================================================
# Initialization
# ==========================================================

_initialized = False
_init_lock = asyncio.Lock()


async def init_web_search() -> None:
    """
    Initialize the web_search subsystem once.
    """

    global _initialized

    if _initialized:
        return

    async with _init_lock:
        if _initialized:
            return

        try:
            logger.info("Initializing web_search subsystem.")

            await cleanup_old_chunks(days=30)

            _initialized = True

            logger.info("web_search initialized successfully.")

        except Exception:
            logger.exception("Failed to initialize web_search.")
            raise


# ==========================================================
# Formatting
# ==========================================================

def _format_context(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Convert retrieved chunks into context for the LLM.
    """

    result: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        result.append(
            f"""SOURCE {index}

Title:
{chunk.get("title", "")}

Text:
{chunk.get("text", "")}

URL:
{chunk.get("url", "")}
"""
        )

    return "\n\n".join(result)


# ==========================================================
# Main Web Search
# ==========================================================

async def get_context(
    query: str,
) -> str:
    """
    Retrieve cache for a user query.
    """

    await init_web_search()

    logger.info(
        "Searching web_memory. Query='%s'",
        query,
    )

    try:
        
        # -------------------------------------------------
        # 1. Search semantic cache
        # -------------------------------------------------

        chunks = await search(
            query=query,
            limit=TOP_K,
            score_threshold=SIMILARITY_THRESHOLD,
        )

        if chunks:
            logger.info(
                "Semantic cache hit. Found %d semantic cache chunks.",
                len(chunks),
            )
            return _format_context(chunks)

        logger.info(
            "Semantic cache miss. Searching Exa."
        )

        # -------------------------------------------------
        # 2. Exa fallback
        # -------------------------------------------------

        web_chunks = await search_exa(query)

        logger.info(
            "Loaded %d chunks from Exa.",
            len(web_chunks),
        )

        # -------------------------------------------------
        # 3. Store cache in Qdrant
        # -------------------------------------------------

        await add_chunks(web_chunks)

        logger.info(
            "Exa chunks stored in Qdrant."
        )

        # -------------------------------------------------
        # 4. Return fresh context
        # -------------------------------------------------

        return _format_context(
            web_chunks[:TOP_K]
        )

    except Exception:
        logger.exception(
            "Failed to build semantic cache for query '%s'.",
            query,
        )
        raise
