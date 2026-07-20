from agents.rag_agent import ask_agent
from llm import chat



def answer(message):

    keywords=[
        "who",
        "what",
        "history",
        "where",
        "when"
    ]


    if any(
        x in message.lower()
        for x in keywords
    ):

        return ask_agent(
            message
        )


    return chat(
        [
            {
                "role":"system",
                "content":
                "You are a helpful assistant."
            },
            {
                "role":"user",
                "content":message
            }
        ]
    )
