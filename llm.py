import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=False)

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.6-27b")

GROQ_API_BASE = "https://api.groq.com/openai/v1"

TEMPERATURE = 0.2
MAX_TOKENS = 2048
TIMEOUT = 60          # секунд
MAX_RETRIES = 3

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """
    Возвращает singleton-экземпляр ChatOpenAI.
    Создается только при первом обращении.
    """
    global _llm

    if _llm is not None:
        return _llm

    api_key = os.getenv("GROQ_TOKEN")

    if not api_key:
        logger.error("Environment variable GROQ_TOKEN is not configured.")
        raise RuntimeError("Environment variable GROQ_TOKEN is not configured.")

    try:
        logger.info("Initializing ChatOpenAI")
        logger.info("Model: %s", MODEL_NAME)
        logger.debug("Base URL: %s", GROQ_API_BASE)
        logger.debug("Temperature: %s", TEMPERATURE)
        logger.debug("Max tokens: %s", MAX_TOKENS)
        logger.debug("Timeout: %s sec", TIMEOUT)
        logger.debug("Max retries: %s", MAX_RETRIES)

        _llm = ChatOpenAI(
            model=MODEL_NAME,
            base_url=GROQ_API_BASE,
            api_key=api_key,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
            streaming=True
        )

        logger.info("ChatOpenAI initialized successfully.")

        return _llm

    except Exception:
        logger.exception("Failed to initialize ChatOpenAI.")
        raise
