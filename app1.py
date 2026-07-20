import chainlit as cl

from agents.router import answer



@cl.on_chat_start
async def start():

    await cl.Message(
        content=
        "Привет! Я Qwen ассистент с RAG по Wikipedia."
    ).send()



@cl.on_message
async def main(message: cl.Message):

    response = answer(
        message.content
    )

    await cl.Message(
        content=response
    ).send()
