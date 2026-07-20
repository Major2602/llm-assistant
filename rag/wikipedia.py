from llama_index.readers.wikipedia import WikipediaReader
from config import WIKI_LANGUAGE



def load_wikipedia(topic: str):

    reader = WikipediaReader(
        language=WIKI_LANGUAGE
    )

    documents = reader.load_data(
        pages=[topic]
    )

    return documents
