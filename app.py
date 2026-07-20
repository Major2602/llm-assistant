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

from huggingface_hub import AsyncInferenceClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage


class HuggingFaceQwenChat(BaseChatModel):
    """
    LangChain wrapper над HF AsyncInferenceClient.

    Использует:
        HuggingFace Inference Providers
        provider = featherless-ai

    Qwen3/Qwen3.5 умеют tool calling.
    """

    client: AsyncInferenceClient
    model_name: str
    provider: str

    async def _generate(
        self,
        messages,
        stop=None,
        **kwargs
    ):

        hf_messages = []

        for m in messages:
            hf_messages.append(
                {
                    "role": m.type,
                    "content": m.content
                }
            )

        result = await self.client.chat.completions.create(
            model=f"{self.model_name}:{self.provider}",
            messages=hf_messages,
            temperature=0.2,
            max_tokens=2048,
        )

        text = (
            result
            .choices[0]
            .message
            .content
        )

        return self._create_chat_result(
            [
                AIMessage(
                    content=text
                )
            ]
        )

    @property
    def _llm_type(self):
        return "qwen-featherless"

client = AsyncInferenceClient(api_key=os.environ["HF_TOKEN"])

llm = HuggingFaceQwenChat(
    client=client,
    model_name=MODEL_NAME,
    provider=HF_PROVIDER
)

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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name=
    "BAAI/bge-small-en-v1.5"
)

_indexes = {}

def get_query_engine(topic:str):

    if topic not in _indexes:
        docs = load_wikipedia(
            topic
        )
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
            model=llm,
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

import chainlit as cl

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
