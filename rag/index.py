from llama_index.core import (
    VectorStoreIndex,
    Settings
)

from llama_index.core.embeddings import (
    resolve_embed_model
)


def build_index(documents):

    Settings.embed_model = resolve_embed_model(
        "local:BAAI/bge-small-en-v1.5"
    )

    index = VectorStoreIndex.from_documents(
        documents
    )

    return index
