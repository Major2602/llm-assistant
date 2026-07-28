# web_search/processing/chunking.py

from __future__ import annotations


from web_search.domain.models import (
    WebDocument,
    DocumentChunk,
)



DEFAULT_CHUNK_SIZE = 800

DEFAULT_OVERLAP = 120



def _split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Keeps previous chunking behavior:
    - fixed size;
    - overlap between segments.
    """

    if not text:
        return []


    words = text.split()


    if len(words) <= chunk_size:

        return [
            text.strip()
        ]


    chunks: list[str] = []


    start = 0


    while start < len(words):

        end = start + chunk_size


        chunk = " ".join(
            words[start:end]
        ).strip()


        if chunk:

            chunks.append(
                chunk
            )


        start = end - overlap


        if start < 0:

            start = 0


        if end >= len(words):

            break


    return chunks



def chunk_documents(
    documents: list[WebDocument],
) -> list[DocumentChunk]:
    """
    Convert web documents into searchable chunks.
    """

    result: list[DocumentChunk] = []


    for document in documents:

        chunks = _split_text(
            document.text
        )


        for index, text in enumerate(
            chunks
        ):

            result.append(

                DocumentChunk(

                    document_id=document.id,

                    source=document.source,

                    text=text,

                    chunk_index=index,

                )

            )


    return result
