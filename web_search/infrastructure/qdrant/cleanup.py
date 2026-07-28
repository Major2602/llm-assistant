"""
Qdrant cleanup service.

Responsibilities:

- remove expired vectors;
- manage storage lifecycle.

No retrieval logic.
No pipeline dependencies.
"""

from __future__ import annotations


from datetime import (
    datetime,
    timezone,
    timedelta,
)


from qdrant_client.models import (
    Filter,
    FieldCondition,
    Range,
)



from web_search.infrastructure.qdrant.client import (
    QdrantConnection,
)



class QdrantCleanupService:
    """
    Removes stale chunks from Qdrant.
    """


    def __init__(
        self,
        connection: QdrantConnection,
        collection_name: str,
    ):
        self._connection = connection
        self._collection_name = collection_name



    async def remove_older_than(
        self,
        days: int,
    ) -> None:
        """
        Delete chunks not accessed for N days.
        """

        client = self._connection.client


        exists = await client.collection_exists(
            self._collection_name
        )


        if not exists:
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



        await client.delete(

            collection_name=self._collection_name,

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
