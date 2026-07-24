from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Source:

    title: str

    url: str

    provider: str | None = None

    score: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )



@dataclass(slots=True)
class AgentContext:

    context_text: str

    sources: list[Source] = field(
        default_factory=list
    )
