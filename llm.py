import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

logger = logging.getLogger(__name__)


MODEL_NAME: str = os.getenv(
    "MODEL_NAME",
    "qwen/qwen3.6-27b"
)

GROQ_BASE_URL: str = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1"
)

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """
    Returns a singleton ChatOpenAI client.

    The client is created lazily on the first call and reused
    for subsequent requests.

    Raises:
        RuntimeError:
            If GROQ_TOKEN environment variable is missing.
        Exception:
            If ChatOpenAI initialization fails.
    """

    global _llm

    if _llm is not None:
        return _llm

    logger.info("Initializing LLM client...")

    groq_token = os.getenv("GROQ_TOKEN")

    if not groq_token:
        logger.error(
            "GROQ_TOKEN environment variable is not configured."
        )
        raise RuntimeError(
            "Missing required environment variable: GROQ_TOKEN"
        )

    try:
        _llm = ChatOpenAI(
            model=MODEL_NAME,
            base_url=GROQ_BASE_URL,
            api_key=groq_token,
            temperature=0.2,
            max_tokens=2048,
        )

        logger.info(
            "LLM client initialized successfully. Model: %s",
            MODEL_NAME,
        )

        return _llm

    except Exception:
        logger.exception(
            "Failed to initialize LLM client."
        )
        raise
