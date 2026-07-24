"""
Web search domain models.

Contains internal data contracts between:
- retrieval layer;
- agent layer;
- UI layer.

No business logic is allowed here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ==========================================================
# Source model
# ==========================================================


class Source(BaseModel):
    """
    External information source.

    Represents a document/page returned by:
    - Exa;
    - Qdrant semantic memory.
    """

    title: str = Field(
        default="Untitled source",
        description="Source title.",
    )

    url: str = Field(
        default="",
        description="Source URL.",
    )

    provider: str | None = Field(
        default=None,
        description="Source provider (exa, qdrant, etc.).",
    )

    score: float | None = Field(
        default=None,
        description="Semantic similarity score.",
    )

    published_date: str | None = Field(
        default=None,
        description="Publication date if available.",
    )

    author: str | None = Field(
        default=None,
        description="Author if available.",
    )


# ==========================================================
# Agent context
# ==========================================================


class AgentContext(BaseModel):
    """
    Context returned to the agent tool.

    Contains:

    text:
        formatted information for LLM reasoning.

    sources:
        metadata preserved for UI citations.
    """

    text: str = Field(
        default="",
        description="Context text provided to LLM.",
    )

    sources: list[Source] = Field(
        default_factory=list,
        description="Sources used to build context.",
    )
