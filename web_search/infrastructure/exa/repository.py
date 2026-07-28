"""
Exa document repository.
"""

from __future__ import annotations

import logging

from web_search.domain.contracts import (
    DocumentRetriever,
)

from web_search.domain.models import (
    NormalizedQuery,
    WebDocument,
)

from web_search.infrastructure.exa.client import (
    ExaClient,
)

from web_search.infrastructure.exa.mapper import (
    map_exa_document,
)


logger = logging.getLogger(__name__)


class ExaRepository(
    DocumentRetriever
):
    """
    Exa implementation of document retrieval.
    """

    def __init__(
        self,
        client: ExaClient,
        results: int,
    ):
        self._client = client
        self._results = results


    async def retrieve(
        self,
        query: NormalizedQuery,
    ) -> list[WebDocument]:
        """
        Retrieve and normalize documents.
        """

        documents = await self._client.search(
            query.normalized,
            self._results,
        )


        result: list[WebDocument] = []


        for document in documents:

            mapped = map_exa_document(
                document
            )

            if mapped:
                result.append(
                    mapped
                )


        logger.info(
            "Exa documents=%d",
            len(result),
        )


        return result
