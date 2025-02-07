import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import chromadb
import openai
from datetime import datetime


load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create a persistent ChromaDB store
chroma_client = chromadb.PersistentClient(path="data/chroma_db")  # Folder where ChromaDB will save data
collection = chroma_client.get_or_create_collection(name="news_collection")


def store_news(articles, category, tags):
    print("\n\n📝📝📝📝📝📝")
    docs = []
    ids = []
    metadata = []

    today = get_today_date()
    for article in articles:
        docs.append(f"{article['title']}. {article['description'] or ''}. {article['content'] or ''}")
        ids.append(article["url"])
        metadata.append({
                    "url": article["url"], 
                    "source": article["source"]["name"], 
                    "category": category, 
                    **{tag: True for tag in tags},
                    "date": today})
    

    # Generate embeddings using the updated OpenAI API
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=docs
    )

    # Extract embeddings
    embeddings = [item.embedding for item in response.data] 
    collection.upsert(documents=docs, ids=ids, embeddings=embeddings, metadatas=metadata)
    # Get total number of documents
    num_docs = collection.count()
    print(f"Total number of documents in collection: {num_docs}")

    return

def get_today_date():
    today = datetime.now().strftime("%Y-%m-%d")
    return today

def search_news_by_tags_and_category(tags, category, top_k=3):
    where_clause = {
        "$and": [
            {
                "$or": [
                    {"category": category},
                    {tags[0]: True} if len(tags) == 1 else {"$or": [{tag: True} for tag in tags]}
                ]
            },
            {"date": get_today_date()}  # Added date condition
        ]
    }
    print(f"Search query: {where_clause}")
    results = collection.get(where=where_clause, include=["metadatas", "documents"])

    news_articles = []

    for doc, meta, id in zip(results["documents"][:top_k], results["metadatas"][:top_k], results["ids"][:top_k]):
        news_articles.append({"content": doc, "url": id, "source": meta["source"]})

    print(f"Articles found in vector store: {len(news_articles)}")
    return news_articles

def search_news(query, tags, category, top_k=3):
    print("In search vector store: 🔍🔍🔍🔍🔍🔍")

    where_clause = {
        "$and": [
            {
                "$or": [
                    {"category": category},
                    {tags[0]: True} if len(tags) == 1 else {"$or": [{tag: True} for tag in tags]}
                ]
            },
            {"date": get_today_date()}  # Added date condition
        ]
    }
    print(f"Search query: {where_clause}")
    # Generate query embedding manually
    query_response = openai.embeddings.create(
        model="text-embedding-ada-002",  # Ensure it matches stored embeddings
        input=[query]
    )

    query_embedding = query_response.data[0].embedding  # Extract embedding

    # Now query ChromaDB using `query_embeddings`
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k,
        where=where_clause,
        include=["metadatas", "documents"]
    )
    news_articles = []

    # Check if results contain documents and if the first element is a list
    if results["documents"] and isinstance(results["documents"][0], list):
        for doc, meta, id in zip(results["documents"][0], results["metadatas"][0], results["ids"][0]):
            news_articles.append({"content": doc, "url": id, "source": meta["source"]})
    else:
        print("No documents found or unexpected format.")

    print(f"Articles found in vector store: {len(news_articles)}")
    return news_articles

if __name__ == "__main__":
    from src.api.fetch_news import get_news
    articles = get_news()
    store_news(articles)
    print(search_news("AI breakthroughs"))