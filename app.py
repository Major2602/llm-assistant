import os
from typing import Any, List, Optional, Dict
import chainlit as cl
from huggingface_hub import AsyncInferenceClient
from llama_index.core import VectorStoreIndex
from llama_index.readers.wikipedia import WikipediaReader
from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms.base import LLM
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from chainlit.langchain.callbacks import ChainlitCallbackHandler

# Настройка параметров
model_id = "Qwen/Qwen3.5-2B"
token = os.getenv("HF_TOKEN")

# Кастомный класс с поддержкой стандартных аргументов
class HFInferenceClientLLM(LLM):
    model_id: str
    token: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    provider: Optional[str] = None # Например, 'hf-inference'

    _client: Any = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Инициализируем клиент. Если указан provider, можно модифицировать URL или заголовки
        self._client = AsyncInferenceClient(model=self.model_id, token=self.token)

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference_client"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "max_new_tokens": self.max_new_tokens
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        import asyncio
        async def _invoke():
            resp = ""
            # Объединяем дефолтные параметры с переданными в kwargs
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stop_sequences": stop
            }
            gen_kwargs.update(kwargs)

            async for token in self._client.text_generation(prompt, stream=True, **gen_kwargs):
                resp += token
            return resp
        return asyncio.run(_invoke())

# 1. Настройка RAG (LlamaIndex + Wikipedia)
def get_wikipedia_data(query: str):
    try:
        reader = WikipediaReader()
        # Ищем по ключевому слову из запроса
        documents = reader.load_data(pages=[query])
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        response = query_engine.query(f"Provide detailed information about {query}")
        return str(response)
    except Exception as e:
        return f"Search error: {e}"

# 2. Определение инструментов LangChain
tools = [
    Tool(
        name="wikipedia_search",
        func=get_wikipedia_data,
        description="Useful for when you need to answer questions about specific facts, people, places, or history from Wikipedia."
    )
]

# 3. Инициализация с новыми аргументами
llm = HFInferenceClientLLM(
    model_id=model_id,
    token=token,
    max_new_tokens=1024,
    temperature=0.1,
    provider="featherless-ai" #
)

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

@cl.on_chat_start
async def start():
    cl.user_session.set("agent", agent_executor)
    await cl.Message(content="""Hello! I am AI Agent based on Qwen 3.5 Model via HuggingFaceEndpoints.
    My agentic workflow is built on LangChain. Currently I support RAG (Retrieval-Augmented Generation)
    from Wikipedia via LlamaIndex. How can I help you today?""").send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")

    # Создаем колбэк для визуализации шагов в Chainlit
    cb = ChainlitCallbackHandler()

    # Вызываем агента с передачей колбэка
    response = await cl.make_async(agent.invoke)(
        {"input": message.content},
        config={"callbacks": [cb]}
    )

    await cl.Message(content=response["output"]).send()
