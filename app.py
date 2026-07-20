import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "prism-ml/Ternary-Bonsai-27B-gguf"
)

HF_PROVIDER = "together"

WIKI_LANGUAGE = os.getenv("WIKI_LANGUAGE", "en")

# ==========================================================
# LLM
# файл: llm.py
# ==========================================================

from langchain_openai import ChatOpenAI

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen3.5-2B"
)

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=f"{MODEL_NAME}:{HF_PROVIDER}",
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
            temperature=0.2,
            max_tokens=2048,
        )
    return _llm

# ==========================================================
# RAG WIKIPEDIA
# файл: rag/wikipedia.py
# ==========================================================

from llama_index.readers.wikipedia import WikipediaReader

def load_wikipedia(topic:str):

    reader = WikipediaReader(
        language=WIKI_LANGUAGE
    )


    documents = reader.load_data(
        pages=[
            topic
        ]
    )

    return documents

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

def get_query_engine(topic:str):

    Settings.embed_model = get_embed_model()

    if topic not in _indexes:

        docs = load_wikipedia(topic)

        index = VectorStoreIndex.from_documents(
            docs
        )

        _indexes[topic] = index


    return _indexes[topic].as_query_engine(
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
    topic:str
)->str:
    """
    Search Wikipedia knowledge.
    Use this tool when the user asks
    about factual information.
    """

    engine = get_query_engine(
        topic
    )

    response = await engine.aquery(
        topic
    )

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
            You are Qwen assistant.
            You can answer normally.
            If the user needs factual knowledge
            or information about a topic,
            use wikipedia_rag tool.
            Always prefer tool results
            over your internal memory.
            Answer in the user's language.
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
