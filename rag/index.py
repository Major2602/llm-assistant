import asyncio
import logging
from typing import Any

from rag.qdrant_store import (
    add_chunks,
    cleanup_old_chunks,
    search,
)
from rag.wikipedia import load_wikipedia

logger = logging.getLogger(__name__)

# ==========================================================
# Configuration
# ==========================================================

TOP_K = 5
SIMILARITY_THRESHOLD = 0.75

# ==========================================================
# Initialization
# ==========================================================

_initialized = False
_init_lock = asyncio.Lock()


async def init_rag() -> None:
    """
    Initialize the RAG subsystem once.
    """

    global _initialized

    if _initialized:
        return

    async with _init_lock:
        if _initialized:
            return

        try:
            logger.info("Initializing RAG subsystem.")

            await cleanup_old_chunks(days=30)

            _initialized = True

            logger.info("RAG initialized successfully.")

        except Exception:
            logger.exception("Failed to initialize RAG.")
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

Source:
{chunk.get("source", "")}
"""
        )

    return "\n\n".join(result)


# ==========================================================
# Main RAG
# ==========================================================

async def get_context(
    entity: str,
    question: str,
) -> str:
    """
    Retrieve context for an entity and user question.
    """

    await init_rag()

    logger.info(
        "Searching RAG context. Entity='%s'",
        entity,
    )

    try:
        # -------------------------------------------------
        # 1. Search existing knowledge
        # -------------------------------------------------

        chunks = await search(
            question=question,
            entity=entity,
            limit=TOP_K,
            score_threshold=SIMILARITY_THRESHOLD,
        )

        if chunks:
            logger.info(
                "Found %d cached chunks.",
                len(chunks),
            )
            return _format_context(chunks)

        logger.info(
            "No cached chunks found. Loading Wikipedia."
        )

        # -------------------------------------------------
        # 2. Wikipedia fallback
        # -------------------------------------------------

        wikipedia_chunks = await load_wikipedia(entity)

        logger.info(
            "Loaded %d chunks from Wikipedia.",
            len(wikipedia_chunks),
        )

        # -------------------------------------------------
        # 3. Store in Qdrant
        # -------------------------------------------------

        await add_chunks(wikipedia_chunks)

        logger.info(
            "Wikipedia chunks stored in Qdrant."
        )

        # -------------------------------------------------
        # 4. Search again
        # -------------------------------------------------

        chunks = await search(
            question=question,
            entity=entity,
            limit=TOP_K,
            score_threshold=SIMILARITY_THRESHOLD,
        )

        if not chunks:
            logger.warning(
                "No chunks returned after indexing. "
                "Using Wikipedia chunks directly."
            )
            chunks = wikipedia_chunks[:TOP_K]

        return _format_context(chunks)

    except Exception:
        logger.exception(
            "Failed to build RAG context for entity '%s'.",
            entity,
        )
        raise
