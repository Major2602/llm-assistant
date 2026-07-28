"""
Exa response mapper.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from web_search.domain.models import (
    WebDocument,
    Source,
)



def _timestamp() -> int:
    return int(
        datetime.now(
            timezone.utc
        ).timestamp()
    )



def _get(
    obj,
    field: str,
    default=None,
):
    return getattr(
        obj,
        field,
        default,
    )



def map_exa_document(
    document,
) -> WebDocument | None:
    """
    Convert Exa object to domain model.
    """

    text = (
        _get(
            document,
            "text",
            "",
        )
        or ""
    ).strip()


    if not text:
        return None


    timestamp = _timestamp()


    return WebDocument(

        id=str(
            uuid4()
        ),

        text=text,

        source=Source(

            title=(
                _get(
                    document,
                    "title",
                    None,
                )
                or "Untitled"
            ),

            url=(
                _get(
                    document,
                    "url",
                    None,
                )
                or ""
            ),

            provider="exa",

            author=_get(
                document,
                "author",
            ),

            published_date=_get(
                document,
                "published_date",
            ),

        ),

        created_at=timestamp,

        last_access=timestamp,

    )
