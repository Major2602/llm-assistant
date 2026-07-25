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

    @classmethod
    def create(
        cls,
        text: str,
        metadata: dict[str, Any] | None = None
    ) -> "TokenEvent":

        return cls(
            type=UIEventType.TOKEN,
            text=text,
            metadata=metadata or {}
        )


# ==========================================================
# Tool Events
# ==========================================================


@dataclass(slots=True, kw_only=True)
class ToolStartEvent(BaseEvent):
    """
    Tool execution started.
    """

    name: str

    @classmethod
    def create(
        cls,
        name: str,
        metadata: dict[str, Any] | None = None
    ) -> "ToolStartEvent":

        return cls(
            type=UIEventType.TOOL_START,
            name=name,
            metadata=metadata or {}
        )


@dataclass(slots=True, kw_only=True)
class ToolEndEvent(BaseEvent):
    """
    Tool execution finished.
    """

    name: str

    @classmethod
    def create(
        cls,
        name: str,
        metadata: dict[str, Any] | None = None
    ) -> "ToolEndEvent":

        return cls(
            type=UIEventType.TOOL_END,
            name=name,
            metadata=metadata or {}
        )


# ==========================================================
# Sources
# ==========================================================


@dataclass(slots=True, kw_only=True)
class SourceEvent(BaseEvent):
    """
    Retrieved information sources.
    """

    sources: list[Source]

    @classmethod
    def create(
        cls,
        sources: list[Source],
        metedata: dict[str, Any] | None = None
    ) -> "SourceEvent":

        return cls(
            type=UIEventType.SOURCE,
            sources=sources,
            metadata=metadata or {}
        )


# ==========================================================
# Errors
# ==========================================================


@dataclass(slots=True, kw_only=True)
class ErrorEvent(BaseEvent):
    """
    Streaming error.
    """

    message: str

    @classmethod
    def create(
        cls,
        message: str,
        metadata: dict[str, Any] | None = None
    ) -> "ErrorEvent":

        return cls(
            type=UIEventType.ERROR,
            message=message,
            metadata=metadata or {}
        )


# ==========================================================
# Completion
# ==========================================================


@dataclass(slots=True, kw_only=True)
class DoneEvent(BaseEvent):
    """
    Stream finished successfully.
    """

    @classmethod
    def create(
        cls,
        metadata: dict[str, Any] | None = None
    ) -> "DoneEvent":

        return cls(
            type=UIEventType.DONE,
            metadata=metadata or {}
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
