from langchain.agents import create_agent
from langchain.tools import tool

from rag.retriever import get_rag_engine


@tool
def wikipedia_search(query:str)->str:
    """
    Search information from Wikipedia knowledge base.
    """

    engine = get_rag_engine(query)

    result = engine.query(
        query
    )

    return str(result)



agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[
        wikipedia_search
    ],
    system_prompt="""
You are an assistant.

Use wikipedia_search tool when
the user needs factual information.

Otherwise answer normally.
"""
)



def ask_agent(question):

    result = agent.invoke(
        {
            "messages":[
                {
                    "role":"user",
                    "content":question
                }
            ]
        }
    )


    return result["messages"][-1].content
