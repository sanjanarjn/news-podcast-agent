import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# Create a persistent ChromaDB store
vector_db = Chroma(persist_directory="data/chroma_db", embedding_function=embedding_model)

def store_news(articles):
    docs = [
        Document(page_content=f"{article['title']}. {article['description'] or ''}",
                 metadata={"url": article["url"], "source": article["source"]["name"]})
        for article in articles
    ]
    vector_db.add_documents(docs)
    vector_db.persist()  # Save data persistently

def search_news(query, top_k=5):
    results = vector_db.similarity_search(query, k=top_k)
    return [{"title": doc.page_content, "url": doc.metadata["url"], "source": doc.metadata["source"]} for doc in results]

if __name__ == "__main__":
    from src.api.fetch_news import get_news
    articles = get_news()
    store_news(articles)
    print(search_news("AI breakthroughs"))