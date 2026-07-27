"""
Minimal query preprocessing layer.

Pipeline position:

USER QUERY
    |
    v
Query preprocessing
    |
    v
SearchQuery
    |
    v
Agent / Retrieval pipeline


Responsibilities:

- sanitize user input;
- normalize whitespace;
- prevent malformed queries;
- preserve semantic meaning;
- create SearchQuery contract.


This module does NOT:

- classify intent;
- detect language;
- expand queries;
- call LLM;
- call agents;
- call search providers;
- rank results.
"""


from __future__ import annotations


import logging
import re


from web_search.models import SearchQuery


logger = logging.getLogger(__name__)


# ==========================================================
# Configuration
# ==========================================================


MAX_QUERY_LENGTH = 1024


CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x1F\x7F]"
)


MULTISPACE_RE = re.compile(
    r"\s+"
)



# ==========================================================
# Normalization
# ==========================================================


def _sanitize_query(
    query: str,
) -> str:
    """
    Clean user query without changing meaning.

    Operations:

    - trim spaces;
    - remove control characters;
    - collapse whitespace;
    - limit length.
    """

    if not query:

        return ""


    query = query.strip()


    query = CONTROL_CHARS_RE.sub(
        " ",
        query,
    )


    query = MULTISPACE_RE.sub(
        " ",
        query,
    )


    return query[:MAX_QUERY_LENGTH].strip()



# ==========================================================
# Public API
# ==========================================================


def preprocess_query(
    query: str,
) -> SearchQuery:
    """
    Convert raw user query into SearchQuery.

    The agentic LLM is responsible for:

    - intent detection;
    - tool selection;
    - query rewriting;
    - language understanding.
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )


    normalized = _sanitize_query(
        query
    )


    if not normalized:

        raise ValueError(
            "Query became empty after sanitization."
        )


    result = SearchQuery(

        original=query,

        normalized=normalized,

    )


    logger.info(
        "Query sanitized. length=%d",
        len(normalized),
    )


    return result
