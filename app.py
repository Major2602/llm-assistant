import os
import chainlit as cl
from huggingface_hub import InferenceClient
import asyncio

# Настройка клиента
model_id = "Qwen/Qwen3.5-2B"
token = os.getenv("HF_TOKEN")
client = InferenceClient(model=model_id, token=token)

SYSTEM_PROMPT = "You are a helpful asssistant based on Qwen 3.5 model."

@cl.on_chat_start
async def start():
    # Инициализируем историю сообщений с системным промптом
    cl.user_session.set("messages", [{"role": "system", "content": SYSTEM_PROMPT}])
    await cl.Message(content="Hello! I am AI bot based on Qwen 3.5 model. How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    messages = cl.user_session.get("messages")
    messages.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")

    # Мы используем loop.run_in_executor для синхронного итератора InferenceClient,
    # чтобы избежать конфликтов с AnyIO event loop на Render.
    def get_response():
        return client.chat_completion(
            messages=messages,
            max_tokens=4096,
            stream=True,
            temperature=0.7
        )

    loop = asyncio.get_event_loop()
    # Выполняем синхронный генератор в отдельном потоке, чтобы не блокировать loop
    response_gen = await loop.run_in_executor(None, get_response)

    for chunk in response_gen:
        token = chunk.choices[0].delta.content
        if token:
            await msg.stream_token(token)

    messages.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("messages", messages)
    await msg.send()
