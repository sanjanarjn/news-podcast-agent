import requests
import os
from dotenv import load_dotenv
import concurrent.futures
from datetime import datetime

load_dotenv()

NEWS_APIS = [
    {
        "name":"NewsApi",
        "headLineUrl":"https://newsapi.org/v2/top-headlines",
        "storiesUrl":"https://newsapi.org/v2/everything",
        "categoryParam":"category",
        "apiKeyParam":"apiKey",
        "fromDateParam":"from",
        "toDateParam":"to",
        "languageParam":"language",
        "countryParam":"country",
        "apiKey":os.getenv("NEWSAPI_API_KEY")
    },
    {
        "name":"GNews",
        "headLineUrl":"https://gnews.io/api/v4/top-headlines",
        "storiesUrl":"https://gnews.io/api/v4/search",
        "categoryParam":"category",
        "apiKeyParam":"apikey",
        "fromDateParam":"from",
        "toDateParam":"to",
        "languageParam":"lang",
        "countryParam":"country",
        "apiKey":os.getenv("G_NEWS_KEY")
    }    
]

def call_newsApi(news, category, date, language, country):
    params = {news["apiKeyParam"]:news["apiKey"]}
    #URL
    if date[0] == datetime.today().strftime("%Y-%m-%d"):
        if news["name"] == "NewsApi":
            url = news["headLineUrl"]
        else:
            url = news["headLineUrl"]
        '''
        elif news["name"] == "GNews" and ("headlines" in userInput or "top stories" in userInput):
            url = news["headLineUrl"]
        '''  
    else:
        url = news["storiesUrl"]
    #Categories/Key Words
    if "top-headlines" in url:
        if isinstance(category, dict):
            params[news["categoryParam"]] = list(category.items())[0][0]
            params["q"] = ' OR '.join(f'{word.strip()}' for word in list(category.items())[0][1].split(","))
        else:
            params[news["categoryParam"]] = category
    else:
        if isinstance(category, dict):
            query = ""
            if list(category.items())[0][0] != "general":
                query = f'"{list(category.items())[0][0]}" OR '
            query += ' OR '.join(f'"{word.strip()}"' for word in list(category.items())[0][1].split(","))
            params["q"] = query
        else:
            params["q"] = category
    #dates
    params[news["fromDateParam"]] = date[0]
    if len(date) > 1:
        params[news["toDateParam"]] = date[1]
    if language != "":
        params[news["languageParam"]] = language
    if country != "":
        params[news["countryParam"]] = country
    
    print(f"\nFetching news via {url} with params: {params}")
    response = requests.get(url, params=params)
    return response.json().get("articles", [])

def make_request(news, category, date, language, country):
    articlesList = call_newsApi(news, category, date, language, country)
    if news["name"] == "NewsApi":
        return [{"Author": article['author'], "Title": article['title'], "PublishedAt": article['publishedAt'], "Content": article['content'], "Category": category, "ApiSource": news['name'], "URL": article["url"]} for article in articlesList]
    elif news["name"] == "GNews":
       return [{"Author": article['source']['name'], "Title": article['title'], "PublishedAt": article['publishedAt'], "Content": article['content'], "Category": category, "ApiSource": news['name'], "URL": article["url"]} for article in articlesList]

def fetch_news(news, categories, date, language, country):
    combinedArticles = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(make_request, news, category, date, language, country) for category in categories]
        for future in concurrent.futures.as_completed(futures):
            combinedArticles += future.result()
    return combinedArticles

def get_news(categories, date, language, country):
    combinedArticles = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_news, news, categories, date, language, country) for news in NEWS_APIS]
        for future in concurrent.futures.as_completed(futures):
            combinedArticles += future.result()
    return combinedArticles

if __name__ == "__main__":
    news = get_news()
    for each_news in news:
        print(type(each_news))
    print(news[:2])  # Print first 2 articles for testing