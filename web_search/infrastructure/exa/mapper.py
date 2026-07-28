"""
Exa response mapping.
"""

from __future__ import annotations


from uuid import uuid4
from datetime import datetime, UTC


from web_search.domain.models import (
    WebDocument,
    Source,
)



def map_exa_document(
    document: dict,
) -> WebDocument | None:
    """
    Convert Exa document into domain model.
    """


    text = (
        document.get(
            "text",
            ""
        )
        or ""
    ).strip()


    if not text:
        return None



    timestamp = int(
        datetime.now(
            UTC
        ).timestamp()
    )


    return WebDocument(

        id=str(
            uuid4()
        ),

        text=text,

        source=Source(

            title=(
                document.get(
                    "title"
                )
                or "Untitled"
            ),

            url=(
                document.get(
                    "url"
                )
                or ""
            ),

            provider="exa",

            author=document.get(
                "author"
            ),

            published_date=document.get(
                "published_date"
            ),
        ),

        created_at=timestamp,

        last_access=timestamp,

    )
