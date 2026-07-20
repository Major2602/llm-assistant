import os
from dotenv import load_dotenv

load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen3.5-2B-Instruct"
)

HF_PROVIDER = os.getenv(
    "HF_PROVIDER",
    "featherless-ai"
)

WIKI_LANGUAGE = os.getenv(
    "WIKI_LANGUAGE",
    "en"
)
