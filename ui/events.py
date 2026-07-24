from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class UIEventType(str, Enum):
    """
    UI event types.
    """

    TOKEN = "token"

    STATUS = "status"

    TOOL_START = "tool_start"

    TOOL_END = "tool_end"

    SOURCE = "source"

    ERROR = "error"

    DONE = "done"


@dataclass(slots=True)
class UIEvent:
    """
    Internal UI event model.

    LangChain events must be converted
    into this format before reaching Chainlit.
    """

    type: UIEventType

    content: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )
