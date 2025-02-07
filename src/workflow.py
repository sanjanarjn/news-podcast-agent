import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv, find_dotenv

from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool  # Ensure this import is present

from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langchain_core.tools import tool

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore, Chroma
from langchain.schema import Document

from src.api.fetch_news import get_news
from src.database.vector_store import store_news, search_news_by_tags_and_category, search_news

import streamlit as st

from enum import Enum  # Ensure to import Enum

load_dotenv(override=True)
openai_api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4o")

class GraphState(Enum):
    USER_QUERY_ANALYSIS_STARTED = "Analysing your query... 🕵🏼‍♀️"
    USER_QUERY_ANALYSIS_COMPLETE = "Analysed your query ✅ "
    EXISTING_DOCS_FOUND = "I have some information on this already... 🙌🏻"
    API_FETCH_IN_PROGRESS = "I do not have the information on this, let me look up! This might take a lil while... ⏰"
    STORE_IN_PROGRESS = "I got the information now, let me store it in case u need it again!"
    FETCH_CONTEXT_IN_PROGRESS = "Done, let me get the most relevant pieces for you now."
    SUMMARISATION_IN_PROGESS = "I am almost there. Getting the right info for you... 🙂"

class QueryState(TypedDict):
    user_query: str
    messages: Annotated[list[AnyMessage], add_messages]
    tags: list[str]
    category: str
    news_articles: list[dict]
    context: list[dict]
    graph_state: GraphState



USER_QUERY_ANALYZER_NODE = "user_query_analyzer"
FETCH_EXISTING_DOCS="fetch_existing_docs"
EXECUTE_NEWS_FETCH = "execute_news_fetch"
UNABLE_TO_ANSWER_NODE = "unable_to_answer"
SUBMIT_FINAL_ANSWER = "submit_final_answer"
STORE_IN_VECTOR_DB = "store_in_vector_db"
FETCH_CONTEXT_FROM_VECTOR_DB = "fetch_context_from_vector_db"


class QueryTags(BaseModel):
    tags: list[str] = Field(description="The key topics which is of interest from the input")
    category: str = Field(description="""Categorise the query into: 
                          business, entertainment, general, health, science, sports, technology""")

from langchain.utils.openai_functions import convert_pydantic_to_openai_function
tagging_functions =[convert_pydantic_to_openai_function(QueryTags)]

user_query_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyse the input carefully and see the topics and categories which the user is interested in. DO NOT GUESS"),
    ("human", "{input}")
])

llm_with_functions = llm.bind(functions=tagging_functions)

final_answer_prompt = SystemMessage(content="Based on the context passed {context} and the user's query summarise and give a list of bulleted points to the user.\
                                    DO NOT GUESS or ADD any points on your own")


def get_final_answer_prompt(context, user_query):
    return SystemMessage(content=f"Based on the context passed: {context} and the user's query: {user_query}, "
                                 "summarize and give a list of bulleted points to the user. "
                                 "DO NOT GUESS or ADD any points on your own.")

def extract_tags_and_category(tool_message):
    tags = []
    category = ""
    
    if 'function_call' in tool_message:
        function_call = tool_message['function_call']
        if 'arguments' in function_call:
            import json
            try:
                parsed_arguments = json.loads(function_call['arguments'])
                tags = parsed_arguments.get("tags", [])
                category = parsed_arguments.get("category", "")
                print("Tags:", tags)
                print("Category:", category)
            except json.JSONDecodeError:
                print("Failed to parse arguments")
    
    return tags, category

def user_query_analyzer(state: QueryState):
    print(f"🟡 In user query analyzer... ")
    chain = user_query_analysis_prompt | llm_with_functions
    topics_identified = chain.invoke(state["user_query"])
    tags = []
    category = ""
    if hasattr(topics_identified, "additional_kwargs"):
        tags, category = extract_tags_and_category(topics_identified.additional_kwargs)

    return {"messages": [topics_identified], "tags": tags, "category": category, "graph_state": GraphState.USER_QUERY_ANALYSIS_COMPLETE}

def fetch_existing_docs_from_vectordb(state: QueryState):
    print("🟡 Checking if docs are in Chroma...")
    existing_docs = []
    if(state["tags"] or state["category"]):
        existing_docs = search_news_by_tags_and_category(tags=state["tags"], category=state["category"])
        
    if existing_docs:
        return {"messages": state["messages"], "tags": state["tags"], "context": existing_docs, "graph_state": GraphState.EXISTING_DOCS_FOUND}
    else:
        return {"messages": state["messages"], "tags": state["tags"], "context": existing_docs, "graph_state": GraphState.API_FETCH_IN_PROGRESS}

    
def execute_news_fetch(state: QueryState):
    print(f"🟡 Fetching news via API")
    news_articles = get_news(keywords=state["tags"], category=state["category"])
    print(news_articles)
    return {"messages": state["messages"], "tags": state["tags"], "news_articles": news_articles, "graph_state": GraphState.STORE_IN_PROGRESS}
    
def store_in_vector_db(state: QueryState):
    print("🟡 Storing the fetched articles in vector db")
    if(state["news_articles"]):
        store_news(state["news_articles"], state["category"], state["tags"])
    state["graph_state"] = GraphState.FETCH_CONTEXT_IN_PROGRESS
    return state

def fetch_context_from_vector_db(state: QueryState):
    print("🟡 Fetching the most relevant chunks from vector db")
    context = search_news(state["user_query"], state["tags"], state["category"], top_k=10)
    state["context"] = context
    state["news_articles"] = context
    state["graph_state"] = GraphState.SUMMARISATION_IN_PROGESS
    return state
    

def submit_final_answer(state: QueryState):
    state["graph_state"] = GraphState.SUMMARISATION_IN_PROGESS
    print(f"🟡 Submitting final answer")
    updated_final_answer_prompt = get_final_answer_prompt(state["context"], state["user_query"])
    print("🔍🔍🔍🔍🔍")
    print(updated_final_answer_prompt)
    return {"messages": [llm.invoke([updated_final_answer_prompt] + state["messages"])]}
     
def unable_to_answer(state: QueryState):
    return {"messages": [AIMessage(content="I am sorry, but I am unable to answer that. Please try asking something else.")]}


def fetch_existing_docs_condition(state: QueryState):
    print("News category based router")

    if isinstance(state, list):
        print("state is a list")
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get("messages", [])):
        print("state is a dict")
        ai_message = messages[-1]
    elif messages := getattr(state, "messages", []):
        print("state is a base model")
        ai_message = messages[-1]
    else:
        print("state is none")
        return UNABLE_TO_ANSWER_NODE
    
    if not state["tags"] or not state["category"]:
        return UNABLE_TO_ANSWER_NODE
    else:
        return FETCH_EXISTING_DOCS
    

def fetch_news_from_api_condition(state: QueryState):
    
    if state["context"]:
        return SUBMIT_FINAL_ANSWER
    else:
        return EXECUTE_NEWS_FETCH
    

def compileGraph(graphState): 
    builder = StateGraph(QueryState)

    builder.add_node(USER_QUERY_ANALYZER_NODE, user_query_analyzer)
    builder.add_node(FETCH_EXISTING_DOCS, fetch_existing_docs_from_vectordb)
    builder.add_node(EXECUTE_NEWS_FETCH, execute_news_fetch)
    builder.add_node(UNABLE_TO_ANSWER_NODE, unable_to_answer)
    builder.add_node(SUBMIT_FINAL_ANSWER, submit_final_answer)
    builder.add_node(STORE_IN_VECTOR_DB, store_in_vector_db)
    builder.add_node(FETCH_CONTEXT_FROM_VECTOR_DB, fetch_context_from_vector_db)

    builder.add_edge(START, USER_QUERY_ANALYZER_NODE)
    builder.add_conditional_edges(USER_QUERY_ANALYZER_NODE, fetch_existing_docs_condition)
    builder.add_conditional_edges(FETCH_EXISTING_DOCS, fetch_news_from_api_condition)
    builder.add_edge(EXECUTE_NEWS_FETCH, STORE_IN_VECTOR_DB)
    builder.add_edge(STORE_IN_VECTOR_DB, FETCH_CONTEXT_FROM_VECTOR_DB)
    builder.add_edge(FETCH_CONTEXT_FROM_VECTOR_DB, SUBMIT_FINAL_ANSWER)

    builder.add_edge(SUBMIT_FINAL_ANSWER, END)
    builder.add_edge(UNABLE_TO_ANSWER_NODE, END)

    memory = MemorySaver()
    react_graph = builder.compile(checkpointer=memory)

    return react_graph


def displayGraph(react_graph):
    from PIL import Image

    graph_image = react_graph.get_graph(xray=True).draw_mermaid_png()
    image_path = "graph.png"
    with open(image_path, "wb") as f:
        f.write(graph_image)

    # Open and display the image
    img = Image.open(image_path)
    img.show()


def run_streamlit_ui():
    st.title("Daily News Podcast Agent!")  # Set the title of the app

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, What are you interested in hearing about today?"),
        ]

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    user_query = st.chat_input("Enter your daily news query here...")  # Input for user query

    thread = {"configurable": {"thread_id": "1"}}
    initial_input = {"user_query": user_query, "messages": st.session_state.chat_history, "tags": [], "news_articles": [], "category":"", "graph_state": GraphState.USER_QUERY_ANALYSIS_STARTED}
    react_graph = compileGraph(initial_input)

    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)
        
        ai_response = ""
        graph_state = GraphState.USER_QUERY_ANALYSIS_STARTED
        
        with st.status(graph_state.value):
            for event in react_graph.stream(initial_input, thread, stream_mode="values"):
                graph_state = event["graph_state"]
                ai_response = (event['messages'][-1].content)  
                #print(ai_response)
                print(graph_state.value)
                st.write(graph_state.value)

        st.session_state.chat_history.append(AIMessage(content=ai_response))
        with st.chat_message("AI"):
            st.write(ai_response)

# Add CSS for blinking effect
st.markdown("""
<style>
@keyframes blink {
  50% { opacity: 0; }
}
</style>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    run_streamlit_ui()
    pass

