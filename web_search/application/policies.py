# web_search/application/policies.py

from __future__ import annotations


from dataclasses import dataclass



@dataclass(frozen=True)
class RetrievalPolicy:
    """
    Retrieval stage policies.

    Contains only pipeline decisions.
    No infrastructure configuration.
    """

    max_documents: int = 10

    max_chunks: int = 50

    max_filtered_chunks: int = 10

    max_embedded_chunks: int = 20

    max_ranked_chunks: int = 10



@dataclass(frozen=True)
class CachePolicy:
    """
    Memory/cache behavior policy.
    """

    enabled: bool = True

    cleanup_days: int = 30



@dataclass(frozen=True)
class QualityPolicy:
    """
    Content quality thresholds.

    Previous filter constants are moved here.
    """

    min_text_length: int = 200

    min_words: int = 40

    min_score: float = 0.30



@dataclass(frozen=True)
class CompressionPolicy:
    """
    Context compression policy.
    """

    max_context_chunks: int = 10



@dataclass(frozen=True)
class PipelinePolicy:
    """
    Aggregated pipeline rules.

    Application layer depends on this object.
    """

    retrieval: RetrievalPolicy

    cache: CachePolicy

    quality: QualityPolicy

    compression: CompressionPolicy



def default_pipeline_policy() -> PipelinePolicy:
    """
    Default production pipeline policy.
    """

    return PipelinePolicy(

        retrieval=RetrievalPolicy(),

        cache=CachePolicy(),

        quality=QualityPolicy(),

        compression=CompressionPolicy(),

    )
