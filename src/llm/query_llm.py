import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from database.vector_store import search_news

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)

def ask_llm(query):
    related_news = search_news(query, top_k=3)
    context = "\n".join([f"- {news['title']} ({news['source']})" for news in related_news])

    prompt = f"Based on recent news:\n{context}\n\nAnswer the question: {query}"

    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    query = "What are the latest AI advancements?"
    print(ask_llm(query))