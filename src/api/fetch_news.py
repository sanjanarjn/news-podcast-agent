import requests
import os
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_news(keywords="AI", language="en", page_size=10):
    url = f"https://newsapi.org/v2/everything?q={keywords}&language={language}&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["articles"]
    else:
        raise Exception(f"Error fetching news: {response.text}")

if __name__ == "__main__":
    news = get_news()
    print(news[:2])  # Print first 2 articles for testing