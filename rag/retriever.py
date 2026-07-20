from .wikipedia import load_wikipedia
from .index import build_index


_indexes = {}



def get_rag_engine(topic):

    if topic not in _indexes:

        docs = load_wikipedia(topic)

        index = build_index(
            docs
        )

        _indexes[topic] = index


    return _indexes[topic].as_query_engine(
        similarity_top_k=3
    )
