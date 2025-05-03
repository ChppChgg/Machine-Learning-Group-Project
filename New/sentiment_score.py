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


def get_cached_sentiment(ticker, date=None, cache_path="new/csv_files/sentiment_cache.csv"):
    if date is None:
        date = datetime.date.today()
    elif isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()

    ticker = ticker.upper()

    # Load or create cache
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)
        cache.rename(columns={"date": "Date", "ticker": "Ticker"}, inplace=True)
        cache["Date"] = pd.to_datetime(cache["Date"])
        cache["Ticker"] = cache["Ticker"].str.upper()
    else:
        cache = pd.DataFrame(columns=["Ticker", "Date", "sentiment"])

    # Skip if already cached
    if ((cache["Ticker"] == ticker) & (cache["Date"] == pd.to_datetime(date))).any():
        return float(cache[(cache["Ticker"] == ticker) & (cache["Date"] == pd.to_datetime(date))]["sentiment"].iloc[0])

    # Compute sentiment score for the given date (approximate — gets today's headlines)
    score = get_combined_sentiment(ticker)
    new_entry = pd.DataFrame([[ticker, date, score]], columns=["Ticker", "Date", "sentiment"])
    new_entry["Date"] = pd.to_datetime(new_entry["Date"])

    # Append and recompute rolling average
    cache = pd.concat([cache, new_entry], ignore_index=True)
    cache = cache.sort_values(["Ticker", "Date"])
    cache["sentiment_7d_avg"] = cache.groupby("Ticker")["sentiment"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    cache.to_csv(cache_path, index=False)    
    return score



if __name__ == "__main__":
    ticker = "PLTR"
    sentiment_score = get_cached_sentiment(ticker)
    print(f"Scaled combined sentiment score for {ticker}: {sentiment_score}")
