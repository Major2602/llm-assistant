"""
Web search domain exceptions.
"""


class WebSearchError(Exception):
    """Base web search exception."""


class ExaSearchError(WebSearchError):
    """Exa provider failure."""
