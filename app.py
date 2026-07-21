import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# LLM
# файл: llm.py
# ==========================================================

from langchain_openai import ChatOpenAI

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "qwen/qwen3.6-27b"
)

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=f"{MODEL_NAME}",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_TOKEN"],
            temperature=0.2,
            max_tokens=2048,
        )
    return _llm

# ==========================================================
# RAG WIKIPEDIA
# файл: rag/wikipedia.py
# ==========================================================

import requests

from langdetect import detect, LangDetectException
from llama_index.core import Document

SUPPORTED_LANGS = {"en", "ru", "de", "fr", "es", "it", "pt", "ja", "zh", "ar"}


def _search(api: str, entity: str):

    response = requests.get(
        api,
        params={
            "action": "query",
            "list": "search",
            "srsearch": entity,
            "utf8": 1,
            "format": "json",
        },
        headers={
            "User-Agent": "Chainlit-RAG/1.0"
        },
        timeout=20,
    )

    response.raise_for_status()

    return response.json()["query"]["search"]


def _page(api: str, title: str):

    response = requests.get(
        api,
        params={
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "explaintext": 1,
            "format": "json",
        },
        headers={
            "User-Agent": "Chainlit-RAG/1.0"
        },
        timeout=20,
    )

    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))

    return page.get("extract", "")


def _load(entity: str, lang: str):

    api = f"https://{lang}.wikipedia.org/w/api.php"
    results = _search(api, entity)

    if not results:
        return None

    title = results[0]["title"]
    text = _page(api, title)

    if not text.strip():
        return None

    return Document(
        text=text,
        metadata={
            "title": title,
            "language": lang,
            "source": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ','_')}"
        }
    )

def load_wikipedia(entity: str):

    try:
        lang = detect(entity)
    except LangDetectException:
        lang = "en"

    if lang not in SUPPORTED_LANGS:
        lang = "en"

    document = _load(entity, lang)

    if document is None and lang != "en":
        document = _load(entity, "en")

    if document is None:
        raise ValueError(f"Wikipedia article '{entity}' not found.")

    return [document]

# ==========================================================
# RAG INDEX
# файл: rag/index.py
# ==========================================================

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

embed_model = None

def get_embed_model():

    global embed_model

    if embed_model is None:
        embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    return embed_model

_indexes = {}

def get_query_engine(entity: str):

    Settings.embed_model = get_embed_model()

    if entity not in _indexes:

        docs = load_wikipedia(entity)

        index = VectorStoreIndex.from_documents(
            docs
        )

        _indexes[entity] = index


    return _indexes[entity].as_query_engine(
        similarity_top_k=4
    )

# ==========================================================
# AGENT
# файл: agents/rag_agent.py
# ==========================================================

from langchain.tools import tool
from langchain.agents import create_agent

@tool
async def wikipedia_rag(
    entity: str,
    question: str,
)->str:
    """
    Search Wikipedia.
    Use this tool when the user asks
    about factual information.
    
    entity:
        The exact entity or page to search
        (person, city, company, country, event, etc.)

    question:
        User's original question about this entity.
    """

    engine = get_query_engine(entity)

    response = await engine.aquery(question)

    return str(response)


_agent = None


def get_agent():

    global _agent

    if _agent is None:

        _agent = create_agent(
            model=get_llm(),
            tools=[
                wikipedia_rag
            ],
            system_prompt="""
            You are a helpful assistant.
            
            When factual information is required, use wikipedia_rag.
            
            For the "entity" argument provide ONLY the entity name.
            
            Good examples:
            
            entity="Alan Turing"
            question="Who was he?"
            
            entity="Python (programming language)"
            question="Who created it?"
            
            entity="Tokyo"
            question="Population"
            
            Never pass the entire user message as entity.
            
            Always answer in the user's language.
            
            Always prefer the tool result over your own knowledge.
            """
        )

    return _agent

async def ask_agent(text):

    agent = get_agent()

    result = await agent.ainvoke(
        {
            "messages":[
                {
                    "role":"user",
                    "content":text
                }
            ]
        }
    )

    return result["messages"][-1].content

# ==========================================================
# CHAINLIT APP
# файл: app.py
# ==========================================================
print("START")
import chainlit as cl
print("IMPORT SUCCESSFUL")

@cl.on_chat_start
async def start():

    await cl.Message(

        content=
        """
Привет!

Я Qwen3.5 ассистент.

Умею:
- вести обычный диалог
- искать информацию через RAG
- использовать Wikipedia

"""
    ).send()

@cl.on_message
async def main(
    message:cl.Message
):

    answer = await ask_agent(
        message.content
    )


    await cl.Message(
        content=answer
    ).send()
