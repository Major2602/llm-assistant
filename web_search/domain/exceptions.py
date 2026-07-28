"""
Web search domain exceptions.

Business-level exceptions only.
No infrastructure dependencies.
"""


class WebSearchError(Exception):
    """
    Base exception for web search domain.
    """


class InvalidQueryError(WebSearchError):
    """
    Raised when user query is invalid.
    """


class RetrievalError(WebSearchError):
    """
    Raised when retrieval pipeline fails.
    """


class ProcessingError(WebSearchError):
    """
    Raised during document/chunk processing.
    """


class ContextBuildError(WebSearchError):
    """
    Raised when final context preparation fails.
    """
