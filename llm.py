from huggingface_hub import InferenceClient

from config import (
    HF_TOKEN,
    MODEL_NAME,
    HF_PROVIDER
)


client = InferenceClient(
    api_key=HF_TOKEN
)


def chat(messages):

    response = client.chat.completions.create(
        model=f"{MODEL_NAME}:{HF_PROVIDER}",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content
