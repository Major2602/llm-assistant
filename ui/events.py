"""
Typed UI events used by the Chainlit layer.

This module defines the internal event model shared between:

    LangChain Streaming
            │
            ▼
     ui.streaming
            │
            ▼
        UI Events
            │
            ▼
      ui.handlers
            │
            ▼
        Chainlit UI

The UI layer must not depend on LangChain event structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from web_search.models import Source


# ==========================================================
# Event Types
# ==========================================================


class UIEventType(StrEnum):
    """
    Supported UI event types.
    """

    TOKEN = "token"

    TOOL_START = "tool_start"

    TOOL_END = "tool_end"

    SOURCE = "source"

    ERROR = "error"

    DONE = "done"


# ==========================================================
# Base Event
# ==========================================================


@dataclass(slots=True, kw_only=True)
class BaseEvent:
    """
    Base UI event.
    """

    type: UIEventType

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ==========================================================
# Token Event
# ==========================================================


@dataclass(slots=True, kw_only=True)
class TokenEvent(BaseEvent):
    """
    Streamed LLM token.
    """

    text: str

    def __init__(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(
            type=UIEventType.TOKEN,
            metadata=metadata or {},
        )

        self.text = text


# ==========================================================
# Tool Events
# ==========================================================


@dataclass(slots=True, kw_only=True)
class ToolStartEvent(BaseEvent):
    """
    Tool execution started.
    """

    name: str

    def __init__(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(
            type=UIEventType.TOOL_START,
            metadata=metadata or {},
        )

        self.name = name


@dataclass(slots=True, kw_only=True)
class ToolEndEvent(BaseEvent):
    """
    Tool execution finished.
    """

    name: str

    def __init__(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(
            type=UIEventType.TOOL_END,
            metadata=metadata or {},
        )

        self.name = name


# ==========================================================
# Sources
# ==========================================================


@dataclass(slots=True, kw_only=True)
class SourceEvent(BaseEvent):
    """
    Retrieved information sources.
    """

    sources: list[Source]

    def __init__(
        self,
        sources: list[Source],
        metadata: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(
            type=UIEventType.SOURCE,
            metadata=metadata or {},
        )

        self.sources = sources


# ==========================================================
# Errors
# ==========================================================


@dataclass(slots=True, kw_only=True)
class ErrorEvent(BaseEvent):
    """
    Streaming error.
    """

    message: str

    def __init__(
        self,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(
            type=UIEventType.ERROR,
            metadata=metadata or {},
        )

        self.message = message


# ==========================================================
# Completion
# ==========================================================


@dataclass(slots=True, kw_only=True)
class DoneEvent(BaseEvent):
    """
    Stream finished successfully.
    """

    def __init__(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(
            type=UIEventType.DONE,
            metadata=metadata or {},
        )


# ==========================================================
# Public Union Type
# ==========================================================


UIEvent = (
    TokenEvent
    | ToolStartEvent
    | ToolEndEvent
    | SourceEvent
    | ErrorEvent
    | DoneEvent
)
