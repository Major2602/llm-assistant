import os
import chainlit as cl
from huggingface_hub import AsyncInferenceClient

# Настройка клиента
model_id = "prism-ml/Ternary-Bonsai-27B-AWQ-4bit"
token = os.getenv("HF_TOKEN")
client = AsyncInferenceClient(model=model_id, token=token, provider="together")

SYSTEM_PROMPT = "You are a helpful assistant based on Qwen 3.5 model."

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
    
    # Стриминг ответа от Hugging Face
    stream = await client.chat_completion(
        messages=messages,
        max_tokens=4096,
        stream=True,
        temperature=0.7
    )

    async for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            token = chunk.choices[0].delta.content
            if token:
                await msg.stream_token(token)

    messages.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("messages", messages)
    await msg.send()
