import requests
import pandas as pd
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from datetime import datetime, timedelta
import numpy as np

# CSV file to store previously fetched sentiment scores
CACHE_FILE = "sentiment-csv/sentiment_cache.csv"

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load sentiment cache if it exists, otherwise create a new DataFrame
if os.path.exists(CACHE_FILE):
    cache_df = pd.read_csv(CACHE_FILE)
else:
    cache_df = pd.DataFrame(columns=["ticker", "date", "sentiment"])

def fetch_sentiment_score(ticker: str, date_str: str = None) -> float:
    """
    Fetch 7-day aggregated sentiment score for a stock ticker using Finviz headlines.
    Uses cached values if available. Only scrapes for Mondays after Jan 1st, 2024.
    """
    global cache_df

    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # Skip if date is before 2024
    if date_obj < datetime(2024, 1, 1):
        return 0.0

    # We only compute fresh sentiment on Mondays
    if date_obj.weekday() != 0:  # 0 = Monday
        return np.nan

    cache_key = (ticker.upper(), date_str)
    cached = cache_df[
        (cache_df["ticker"] == cache_key[0]) &
        (cache_df["date"] == cache_key[1])
    ]
    if not cached.empty:
        return cached["sentiment"].values[0]

    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }

        page = requests.get(url, headers=headers, timeout=10)
        if page.status_code != 200:
            print(f"{ticker} page fetch failed with status {page.status_code}")
            return 0.0

        soup = BeautifulSoup(page.content, "html.parser")
        news_table = soup.find("table", class_="fullview-news-outer")

        if news_table is None:
            print(f"No headlines found on Finviz for {ticker} — skipping sentiment.")
            score = 0.0
        else:
            # Gather all headlines within the past 7 calendar days
            headlines = []
            rows = news_table.find_all("tr")
            for row in rows:
                time_tag = row.td.text.strip()
                headline = row.a.text.strip()

                # Parse Finviz timestamp (either date or just time if same-day)
                try:
                    if " " in time_tag:
                        headline_date = datetime.strptime(time_tag.split(" ")[0], "%b-%d-%y")
                    else:
                        headline_date = date_obj  # If time only, assume same-day
                except:
                    headline_date = date_obj

                if (date_obj - timedelta(days=7)) <= headline_date <= date_obj:
                    headlines.append(headline)

            scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
            score = sum(scores) / len(scores) if scores else 0.0

        # Cache the computed score
        cache_df = pd.concat([
            cache_df,
            pd.DataFrame([{
                "ticker": cache_key[0],
                "date": cache_key[1],
                "sentiment": score
            }])
        ])
        cache_df.to_csv(CACHE_FILE, index=False)

        return score

    except Exception as e:
        print(f"Error getting sentiment for {ticker} on {date_str}: {e}")
        return 0.0
