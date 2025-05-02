import requests
from bs4 import BeautifulSoup
import yfinance as yf
import datetime
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

nltk.download('vader_lexicon')

def get_combined_sentiment(ticker, scale_factor=3):
    """
    Improved sentiment fetching and processing for stocks.
    - Filters neutral scores
    - Averages over real (non-neutral) sentiments
    - Scales output for stronger ML signals
    """
    def get_finviz_sentiment(ticker):
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_table = soup.find('table', class_='fullview-news-outer')
            if news_table:
                rows = news_table.find_all('tr')
                headlines = [row.a.get_text() for row in rows if row.a]
                return headlines
            else:
                return []
        except Exception as e:
            print(f"Finviz error: {e}")
            return []

    def get_yahoo_finance_sentiment(ticker):
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            headlines = [item['title'] for item in news if 'title' in item]
            return headlines
        except Exception as e:
            print(f"Yahoo Finance error: {e}")
            return []

    def get_reddit_sentiment(ticker):
        try:
            url = f"https://www.reddit.com/r/wallstreetbets/search.json?q={ticker}&restrict_sr=1&sort=new"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                posts = response.json()['data']['children']
                titles = [post['data']['title'] for post in posts]
                return titles
            else:
                return []
        except Exception as e:
            print(f"Reddit error: {e}")
            return []

    # Gather headlines from all sources
    finviz_headlines = get_finviz_sentiment(ticker)
    yahoo_headlines = get_yahoo_finance_sentiment(ticker)
    reddit_posts = get_reddit_sentiment(ticker)

    all_texts = finviz_headlines + yahoo_headlines + reddit_posts

    if not all_texts:
        return 0  # Neutral sentiment if no data

    # Analyse sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []

    for text in all_texts:
        score = sia.polarity_scores(text)['compound']
        # Only keep non-neutral scores
        if abs(score) > 0.05:
            sentiment_scores.append(score)

    if not sentiment_scores:
        return 0

    # Average and scale
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    scaled_sentiment = average_sentiment * scale_factor

    # Keep final value between -1 and 1
    scaled_sentiment = max(min(scaled_sentiment, 1), -1)

    return scaled_sentiment


def get_cached_sentiment(ticker, cache_path="new/sentiment_csv/sentiment_cache.csv"):
    today = datetime.date.today().isoformat()

    # Load or create cache
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["ticker", "date", "sentiment"])

    # Check for existing entry
    existing = cache[(cache["ticker"] == ticker) & (cache["date"] == today)]
    if not existing.empty:
        return float(existing["sentiment"].iloc[0])

    # If not cached, compute and store
    score = get_combined_sentiment(ticker)
    new_entry = pd.DataFrame([[ticker, today, score]], columns=["ticker", "date", "sentiment"])
    new_entry.dropna(axis=1, how='all', inplace=True)
    cache = pd.concat([cache, new_entry], ignore_index=True)
    cache.to_csv(cache_path, index=False)
    return score



if __name__ == "__main__":
    ticker = "PLTR"
    sentiment_score = get_cached_sentiment(ticker)
    print(f"Scaled combined sentiment score for {ticker}: {sentiment_score}")
