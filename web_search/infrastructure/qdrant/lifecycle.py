"""
Qdrant memory lifecycle management.

Responsibilities:
- cleanup stale chunks;
- maintain memory retention policy.
"""

from __future__ import annotations

import logging

from datetime import datetime, timezone, timedelta

from qdrant_client.models import (
    Filter,
    FieldCondition,
    Range,
)

from web_search.infrastructure.qdrant.client import (
    get_qdrant_client,
)

from web_search.infrastructure.qdrant.collection import (
    COLLECTION_NAME,
    collection_exists,
)


logger = logging.getLogger(__name__)



async def cleanup_old_chunks(
    days: int = 30,
) -> None:
    """
    Remove chunks not accessed within retention period.
    """

    if not await collection_exists():

        logger.info(
            "Qdrant collection missing. Cleanup skipped."
        )

        return


    cutoff = int(
        (
            datetime.now(
                timezone.utc
            )
            -
            timedelta(
                days=days
            )
        ).timestamp()
    )


    logger.info(
        "Cleaning Qdrant chunks older than %d days.",
        days,
    )


    await get_qdrant_client().delete(

        collection_name=COLLECTION_NAME,

        points_selector=Filter(

            must=[

                FieldCondition(

                    key="last_access",

                    range=Range(
                        lt=cutoff
                    ),

                )

            ]

        ),

    )
