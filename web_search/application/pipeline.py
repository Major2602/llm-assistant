"""
Application pipeline orchestration.

Responsibilities:

- execute web search use-case;
- coordinate pipeline stages;
- mutate PipelineState;
- keep business flow independent from infrastructure.
"""

from __future__ import annotations


import logging


from web_search.domain.models import (
    AgentContext,
    PipelineMetadata,
)


from web_search.application.state import (
    PipelineState,
)


from web_search.application.policies import (
    RetrievalPolicy,
)


from web_search.domain.contracts import (
    SearchProvider,
    VectorStore,
    Embedder,
    Reranker,
)


from web_search.processing.chunking import (
    chunk_documents,
)


from web_search.processing.filtering import (
    filter_chunks,
)


from web_search.processing.retrieval import (
    retrieve_by_embedding_similarity,
)


from web_search.processing.reranking import (
    rerank_chunks,
)


from web_search.processing.compression import (
    compress_chunks,
)


from web_search.processing.context import (
    optimize_context,
)


from web_search.query_normalizer import (
    preprocess_query,
)


logger = logging.getLogger(__name__)



class WebSearchPipeline:
    """
    Main application pipeline.

    Flow:

        normalize query
            ↓
        memory lookup
            ↓
        web retrieval
            ↓
        chunking
            ↓
        filtering
            ↓
        embedding retrieval
            ↓
        reranking
            ↓
        compression
            ↓
        context optimization
    """


    def __init__(
        self,
        search_provider: SearchProvider,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        retrieval_policy: RetrievalPolicy,
    ):
        self.search_provider = search_provider
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.policy = retrieval_policy



    async def execute(
        self,
        query: str,
    ) -> AgentContext:
        """
        Execute complete search pipeline.
        """


        normalized_query = preprocess_query(
            query
        )


        state = PipelineState(

            query=normalized_query,

            metadata=PipelineMetadata()

        )


        logger.info(
            "Pipeline started."
        )



        await self._memory_stage(
            state
        )


        if not self.policy.use_memory_result(
            state
        ):

            await self._web_stage(
                state
            )



        if not state.ranked:

            logger.info(
                "No relevant chunks."
            )

            return AgentContext(
                metadata=state.metadata
            )



        state.compressed = await compress_chunks(
            query=state.query.normalized,
            chunks=state.ranked,
        )


        state.context = optimize_context(
            query=state.query.normalized,
            chunks=state.compressed,
        )


        logger.info(
            "Pipeline completed."
        )


        return state.context



    async def _memory_stage(
        self,
        state: PipelineState,
    ) -> None:
        """
        Retrieve from vector memory.
        """


        results = await self.vector_store.search(
            state.query
        )


        if not results:

            state.metadata.cache_hit = False

            return



        state.metadata.cache_hit = True


        state.ranked = await rerank_chunks(
            query=state.query.normalized,
            chunks=results,
            reranker=self.reranker,
        )



    async def _web_stage(
        self,
        state: PipelineState,
    ) -> None:
        """
        Execute fresh web retrieval.
        """


        documents = await self.search_provider.search(
            state.query.normalized
        )


        state.documents = documents


        if not documents:

            return



        state.chunks = chunk_documents(
            documents
        )


        filtered = filter_chunks(
            chunks=state.chunks,
            query=state.query.normalized,
        )


        embedded = await retrieve_by_embedding_similarity(
            query=state.query.normalized,
            chunks=filtered,
            embedder=self.embedder,
        )


        state.ranked = await rerank_chunks(
            query=state.query.normalized,
            chunks=embedded,
            reranker=self.reranker,
        )


        await self.vector_store.store(
            state.chunks
        )
