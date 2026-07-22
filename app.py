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

import uuid
from datetime import datetime, timezone

import requests
from langdetect import detect, LangDetectException
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_LANGS = {"en", "ru", "de", "fr", "es", "it", "pt", "ja", "zh", "ar"}

USER_AGENT = "Chainlit-RAG/2.0"

# Wikipedia API

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
            "User-Agent": USER_AGENT
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
            "User-Agent": USER_AGENT
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

    return {
        "title": title,
        "language": lang,
        "source": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "text": text,
    }


# Language detection

def _detect_language(entity: str):

    try:
        lang = detect(entity)
    except LangDetectException:
        lang = "en"

    if lang not in SUPPORTED_LANGS:
        lang = "en"

    return lang


# Chunk splitter

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


def _chunk_document(document: dict):

    now = datetime.now(timezone.utc).timestamp()
    chunks = _splitter.split_text(document["text"])
    result = []
    for index, chunk in enumerate(chunks):

        result.append(
            {
                "id": str(uuid.uuid4()),
                "entity": document["title"],
                "chunk_index": index,
                "text": chunk,
                "title": document["title"],
                "language": document["language"],
                "source": document["source"],
                "created_at": now,
                "last_access": now,
            }
        )

    return result


# Public API

def load_wikipedia(entity: str):

    lang = _detect_language(entity)
    document = _load(entity, lang)

    if document is None and lang != "en":
        document = _load(entity, "en")

    if document is None:
        raise ValueError(f"Wikipedia article '{entity}' not found.")

    return _chunk_document(document)


# ==========================================================
# CLOUDFLARE EMBEDDINGS
# файл: rag/cloudflare_embeddings.py
# ==========================================================

import os
from typing import List

import requests


CF_ACCOUNT_ID = os.environ["CF_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CF_API_TOKEN"]

MODEL_EMBEDDINGS = "@cf/qwen/qwen3-embedding-0.6b"

API_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/"
    f"{CF_ACCOUNT_ID}/ai/run/{MODEL_EMBEDDINGS}"
)


class CloudflareEmbeddingError(Exception):
    """Ошибка получения эмбеддингов."""


class CloudflareEmbeddings:
    """
    Cloudflare Workers AI Embeddings.

    Используется вместо FastEmbedEmbedding.
    """

    def __init__(
        self,
        timeout: int = 60,
    ):
        self.timeout = timeout

        self.headers = {
            "Authorization": f"Bearer {CF_API_TOKEN}",
            "Content-Type": "application/json",
        }

    # --------------------------------------------------------
    # Internal
    # --------------------------------------------------------

    def _request(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        response = requests.post(
            API_URL,
            headers=self.headers,
            json={
                "text": texts,
                "instruction": ("Given a user question, retrieve relevant Wikipedia passages.")
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        payload = response.json()

        if not payload.get("success", False):
            raise CloudflareEmbeddingError(payload.get("errors"))

        return payload["result"]["data"]


    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Получить embeddings для списка документов.
        """

        if not texts:
            return []

        return self._request(texts)

    def embed_query(
        self,
        text: str,
    ) -> List[float]:
        """
        Получить embedding поискового запроса.
        """

        return self._request([text])[0]


# ------------------------------------------------------------
# Singleton
# ------------------------------------------------------------

_embeddings = None


def get_embedding_model() -> CloudflareEmbeddings:

    global _embeddings

    if _embeddings is None:
        _embeddings = CloudflareEmbeddings()

    return _embeddings

# ==========================================================
# QDRANT STORE
# файл: rag/quadrant_store.py
# ==========================================================

import os

from datetime import datetime, timezone, timedelta
from typing import List, Dict


from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)


#from rag.cloudflare_embeddings import get_embedding_model


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

QDRANT_URL = os.environ["QDRANT_URL"]

QDRANT_API_KEY = os.environ.get(
    "QDRANT_API_KEY"
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "wikipedia"
)


# Размерность qwen3-embedding-0.6b
# определяется автоматически при первом создании
VECTOR_SIZE = None


# ---------------------------------------------------------
# Client
# ---------------------------------------------------------

_client = None


def get_qdrant():

    global _client

    if _client is None:

        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    return _client



# ---------------------------------------------------------
# Collection
# ---------------------------------------------------------

def ensure_collection(
    vector_size: int
):

    client = get_qdrant()

    collections = [
        item.name
        for item in client.get_collections().collections
    ]


    if COLLECTION_NAME not in collections:

        client.create_collection(
            collection_name=COLLECTION_NAME,

            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )



# ---------------------------------------------------------
# Insert Wikipedia chunks
# ---------------------------------------------------------

def add_chunks(
    chunks: List[Dict]
):

    """
    Сохраняет Wikipedia chunks в Qdrant.

    chunks:
    [
        {
            id,
            text,
            entity,
            title,
            source,
            language,
            created_at,
            last_access
        }
    ]
    """

    if not chunks:
        return


    embedder = get_embedding_model()


    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    vectors = embedder.embed_documents(
        texts
    )


    ensure_collection(
        len(vectors[0])
    )


    points = []


    for chunk, vector in zip(
        chunks,
        vectors
    ):

        points.append(

            PointStruct(

                id=chunk["id"],

                vector=vector,

                payload=chunk,
            )
        )


    get_qdrant().upsert(

        collection_name=COLLECTION_NAME,

        points=points,

    )



# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

def search(
    question: str,
    entity: str,
    limit: int = 5,
    score_threshold: float = 0.75,
):

    """
    Поиск релевантных чанков.

    Возвращает:
    [
        {
            text,
            metadata...
        }
    ]
    """


    embedder = get_embedding_model()


    query_vector = embedder.embed_query(
        question
    )


    ensure_collection(
        len(query_vector)
    )


    result = get_qdrant().query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        query_filter=Filter(

            must=[

                FieldCondition(

                    key="entity",

                    match=MatchValue(
                        value=entity
                    ),
                )
            ]
        ),

        limit=limit,

        score_threshold=score_threshold,

    )


    hits = result.points


    if not hits:
        return []


    update_last_access(
        [
            point.id
            for point in hits
        ]
    )


    return [

        {
            **point.payload,
            "score": point.score,
        }

        for point in hits

    ]



# ---------------------------------------------------------
# Update access time
# ---------------------------------------------------------

def update_last_access(
    ids: List[str]
):

    now = datetime.now(
        timezone.utc
    ).timestamp()


    for point_id in ids:

        get_qdrant().set_payload(

            collection_name=COLLECTION_NAME,

            payload={
                "last_access": now
            },

            points=[
                point_id
            ],
        )



# ---------------------------------------------------------
# Delete old data
# ---------------------------------------------------------

def cleanup_old_chunks(
    days: int = 30
):

    """
    Удаляет данные,
    которые не использовались N дней.
    """


    cutoff = (
        datetime.now(timezone.utc)
        -
        timedelta(days=days)
    ).timestamp()


    get_qdrant().delete(

        collection_name=COLLECTION_NAME,

        points_selector=Filter(

            must=[

                FieldCondition(

                    key="last_access",

                    range=Range(
                        lt=cutoff
                    ),

                )

            ]

        )

    )

# ==========================================================
# RAG INDEX
# файл: rag/index.py
# ==========================================================

from typing import List, Dict

#from rag.wikipedia import load_wikipedia
#from rag.qdrant_store import (
#    search,
#    add_chunks,
#    cleanup_old_chunks,
#)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TOP_K = 5

SIMILARITY_THRESHOLD = 0.75



# ---------------------------------------------------------
# RAG initialization
# ---------------------------------------------------------

_initialized = False


def init_rag():

    """
    Инициализация RAG.

    Выполняется один раз при запуске.
    """

    global _initialized

    if _initialized:
        return


    # Удаление неиспользуемых данных
    # старше 30 дней

    cleanup_old_chunks(
        days=30
    )


    _initialized = True



# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

def _format_context(
    chunks: List[Dict]
) -> str:

    """
    Преобразует найденные чанки
    в контекст для Qwen3.6.
    """


    result = []


    for index, chunk in enumerate(chunks, 1):

        result.append(

            f"""
SOURCE {index}

Title:
{chunk.get("title")}

Text:
{chunk.get("text")}

Source:
{chunk.get("source")}
"""
        )


    return "\n\n".join(result)



# ---------------------------------------------------------
# Main RAG logic
# ---------------------------------------------------------

def get_context(
    entity: str,
    question: str,
) -> str:

    init_rag()



    # -----------------------------------------------------
    # 1. Search existing knowledge
    # -----------------------------------------------------

    chunks = search(

        question=question,

        entity=entity,

        limit=TOP_K,

        score_threshold=SIMILARITY_THRESHOLD,

    )


    if chunks:

        return _format_context(
            chunks
        )



    # -----------------------------------------------------
    # 2. Wikipedia fallback
    # -----------------------------------------------------

    wikipedia_chunks = load_wikipedia(
        entity
    )



    # -----------------------------------------------------
    # 3. Store in Qdrant
    # -----------------------------------------------------

    add_chunks(
        wikipedia_chunks
    )



    # -----------------------------------------------------
    # 4. Search again
    # -----------------------------------------------------

    chunks = search(

        question=question,

        entity=entity,

        limit=TOP_K,

        score_threshold=SIMILARITY_THRESHOLD,

    )

    if not chunks:

        chunks = wikipedia_chunks[:TOP_K]

    return _format_context(
        chunks
    )

# ==========================================================
# AGENT
# файл: agents/rag_agent.py
# ==========================================================

from langchain.tools import tool
from langchain.agents import create_agent

#from rag.index import get_context
#from llm import get_llm



# ----------------------------------------------------------
# RAG TOOL
# ----------------------------------------------------------

@tool
async def wikipedia_rag(
    entity: str,
    question: str,
) -> str:
    """
    Search Wikipedia using RAG.

    The tool first searches existing knowledge
    in Qdrant.

    If no relevant information exists,
    it loads Wikipedia, creates embeddings
    using Cloudflare Workers AI and stores
    chunks in Qdrant.

    entity:
        Exact entity name:
        person, city, company, country,
        event, technology, etc.

    question:
        User's original question
        about this entity.
    """


    context = get_context(
        entity=entity,
        question=question,
    )


    return "TOOL WORKS" #context



# ----------------------------------------------------------
# AGENT
# ----------------------------------------------------------

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

When factual information is required,
always use wikipedia_rag.

The tool performs retrieval from
a Wikipedia RAG database.

For the "entity" argument provide
ONLY the entity name.

Correct examples:

entity="Alan Turing"
question="Who was he?"

entity="Python programming language"
question="Who created it?"

entity="Tokyo"
question="Population"


Incorrect:

entity="Who created Python?"
entity="Tell me everything about Tokyo"


Never pass the entire user message
as entity.


Use the information returned by
wikipedia_rag as the primary source
for factual answers.


Answer in the user's language.

Do not mention internal RAG,
Qdrant, embeddings or tools
unless the user asks about them.

"""

        )


    return _agent



# ----------------------------------------------------------
# PUBLIC API
# ----------------------------------------------------------

async def ask_agent(
    text: str
):


    agent = get_agent()


    result = await agent.ainvoke(

        {
            "messages": [
                {
                    "role": "user",
                    "content": text,
                }
            ]
        }

    )

    for i, m in enumerate(result["messages"]):
        print(i, type(m).__name__, getattr(m, "content", None))

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
