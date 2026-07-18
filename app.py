import os
import chainlit as cl
from huggingface_hub import InferenceClient

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
    
    # Стриминг ответа от Hugging Face
    for chunk in client.chat_completion(
        messages=messages,
        max_tokens=4096,
        stream=True,
        temperature=0.7
    ):
        token = chunk.choices[0].delta.content
        if token:
            await msg.stream_token(token)

    messages.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("messages", messages)
    await msg.send()
