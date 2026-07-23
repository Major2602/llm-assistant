import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_exa import ExaSearchRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

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

_retriever = ExaSearchRetriever(
    api_key=os.getenv("EXA_TOKEN"),
    k=5,
    text_contents=True,
)


async def _search(query: str):
    """
    Execute Exa search.
    """

    logger.info("Searching Exa: '%s'", query)

    documents = await _retriever.ainvoke(query)

    logger.info(
        "Exa returned %d documents.",
        len(documents),
    )

    return documents


def _chunk_documents(
    documents,
) -> list[dict[str, str | int]]:
    """
    Convert Exa documents into Qdrant chunks.
    """

    now = datetime.now(timezone.utc).timestamp()

    result: list[dict[str, str | int]] = []

    for document in documents:

        title = (
            document.metadata.get("title")
            or "Untitled"
        )

        source = (
            document.metadata.get("url")
            or ""
        )

        text = document.page_content.strip()

        if not text:
            continue

        chunks = _splitter.split_text(text)

        for index, chunk in enumerate(chunks):

            result.append(
                {
                    "id": str(uuid.uuid4()),
                    "entity": title,
                    "chunk_index": index,
                    "text": chunk,
                    "title": title,
                    "language": "unknown",
                    "source": source,
                    "created_at": now,
                    "last_access": now,
                }
            )

    logger.info(
        "Prepared %d chunks.",
        len(result),
    )

    return result


async def search_exa(
    query: str,
) -> list[dict[str, str | int]]:
    """
    Search Exa and prepare chunks for indexing.
    """

    logger.info(
        "Searching web for '%s'.",
        query,
    )

    documents = await _search(query)

    if not documents:

        logger.warning(
            "No Exa results for '%s'.",
            query,
        )

        raise ValueError(
            f"No search results for '{query}'."
        )

    return _chunk_documents(documents)
