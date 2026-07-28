"""
Exa SearchProvider implementation.
"""

from __future__ import annotations


from web_search.domain.contracts import (
    SearchProvider,
)


from web_search.domain.models import (
    WebDocument,
)


from web_search.infrastructure.exa.client import (
    ExaClient,
)


from web_search.infrastructure.exa.mapper import (
    map_exa_document,
)



class ExaProvider(SearchProvider):
    """
    Domain-compatible Exa provider.
    """


    def __init__(
        self,
        client: ExaClient,
        results_limit: int = 100,
    ):
        self.client = client
        self.results_limit = results_limit



    async def search(
        self,
        query: str,
    ) -> list[WebDocument]:
        """
        Search web documents.
        """


        documents = await self.client.search(
            query,
            self.results_limit,
        )


        result: list[WebDocument] = []


        for item in documents:

            document = map_exa_document(
                item
            )


            if document:

                result.append(
                    document
                )


        return result
