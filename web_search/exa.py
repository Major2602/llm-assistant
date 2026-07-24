import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_exa import ExaSearchRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================

EXA_TOKEN = os.getenv("EXA_TOKEN")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

EXA_RESULTS = 5


# ==========================================================
# Text splitter
# ==========================================================

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


# ==========================================================
# Exa client
# ==========================================================

_retriever: ExaSearchRetriever | None = None


def get_exa_retriever() -> ExaSearchRetriever:
    """
    Lazily initialize Exa retriever.
    """

    global _retriever

    if _retriever is not None:
        return _retriever

    if not EXA_TOKEN:
        logger.error(
            "Environment variable EXA_TOKEN is not configured."
        )
        raise RuntimeError(
            "Environment variable EXA_TOKEN is not configured."
        )

    logger.info(
        "Initializing ExaSearchRetriever."
    )

    _retriever = ExaSearchRetriever(
        exa_api_key=EXA_TOKEN,
        k=EXA_RESULTS,
        text_contents=True,
    )

    logger.info(
        "ExaSearchRetriever initialized successfully."
    )

    return _retriever


# ==========================================================
# Search
# ==========================================================


async def _search(
    query: str,
):
    """
    Execute Exa web search.
    """

    logger.info(
        "Searching Exa for query='%s'.",
        query,
    )

    retriever = get_exa_retriever()

    documents = await retriever.ainvoke(
        query
    )

    logger.info(
        "Exa returned %d documents.",
        len(documents),
    )

    return documents


# ==========================================================
# Processing
# ==========================================================


def _normalize_document(
    document: Any,
    query: str,
) -> dict[str, Any] | None:
    """
    Convert Exa document into internal format.
    """

    metadata = (
        document.metadata
        if hasattr(document, "metadata")
        else {}
    )

    text = (
        getattr(
            document,
            "page_content",
            "",
        )
        or ""
    ).strip()


    if not text:
        logger.debug(
            "Skipping empty Exa document."
        )
        return None


    return {
        "query": query,

        "title": (
            metadata.get("title")
            or "Untitled"
        ),

        "url": (
            metadata.get("url")
            or ""
        ),

        "text": text,

        "provider": "exa",

        "published_date": (
            metadata.get("published_date")
        ),

        "author": (
            metadata.get("author")
        ),
    }


def _chunk_documents(
    documents: list[Any],
    query: str,
) -> list[dict[str, Any]]:
    """
    Split Exa documents into Qdrant chunks.
    """

    now = int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )


    chunks: list[dict[str, Any]] = []


    for document in documents:

        normalized = _normalize_document(
            document,
            query,
        )

        if normalized is None:
            continue


        text_chunks = _splitter.split_text(
            normalized["text"]
        )


        for index, chunk in enumerate(text_chunks):

            chunks.append(
                {
                    "id": str(
                        uuid.uuid4()
                    ),

                    "query": (
                        normalized["query"]
                    ),

                    "title": (
                        normalized["title"]
                    ),

                    "url": (
                        normalized["url"]
                    ),

                    "text": chunk,

                    "provider": (
                        normalized["provider"]
                    ),

                    "chunk_index": index,

                    "published_date": (
                        normalized["published_date"]
                    ),

                    "author": (
                        normalized["author"]
                    ),

                    "created_at": now,

                    "last_access": now,
                }
            )


    logger.info(
        "Prepared %d chunks from Exa results.",
        len(chunks),
    )

    return chunks


# ==========================================================
# Public API
# ==========================================================


async def search_exa(
    query: str,
) -> list[dict[str, Any]]:
    """
    Search web using Exa and prepare chunks
    for semantic storage.
    """

    logger.info(
        "Starting Exa web search for '%s'.",
        query,
    )


    documents = await _search(
        query
    )


    if not documents:

        logger.warning(
            "Exa returned no results for '%s'.",
            query,
        )

        raise ValueError(
            f"No Exa search results for '{query}'."
        )


    chunks = _chunk_documents(
        documents,
        query,
    )


    if not chunks:

        logger.warning(
            "Exa returned documents but no usable chunks."
        )

        raise ValueError(
            "Exa returned no usable text chunks."
        )


    return chunks
