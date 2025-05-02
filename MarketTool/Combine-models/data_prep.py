import pandas as pd
import yfinance as yf
import time
import random
import os
import threading
from datetime import datetime

# Create a data cache directory
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Global lock to prevent concurrent YF downloads
yf_lock = threading.Lock()
# Track last request time for rate limiting
last_request_time = datetime.now()
MIN_REQUEST_INTERVAL = 5.0  # Minimum seconds between requests

def download_stock_data(ticker: str, start: str, end: str, retries=5, use_cache=True) -> pd.DataFrame:
    """
    Download stock data with improved caching and robust rate limit handling
    """
    global last_request_time
    
    # Check cache first if enabled (with more flexible path handling)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")
    
    if use_cache and os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                print(f"Using cached data for {ticker}")
                return df
        except Exception as e:
            print(f"Cache error for {ticker}: {e}")
    
    # Acquire lock to prevent concurrent downloads
    with yf_lock:
        # Enforce rate limiting with MUCH more aggressive delays
        now = datetime.now()
        elapsed = (now - last_request_time).total_seconds()
        if elapsed < MIN_REQUEST_INTERVAL:
            # Much longer delay between requests - up to 10 seconds
            sleep_time = MIN_REQUEST_INTERVAL - elapsed + random.uniform(2.0, 10.0)
            print(f"Rate limiting: waiting {sleep_time:.2f}s before fetching {ticker}")
            time.sleep(sleep_time)
        
        # Download with robust retry logic
        attempt = 0
        max_wait = 10  # Start with 10 seconds
        while attempt < retries:
            try:
                # Update last request time
                last_request_time = datetime.now()
                
                # Try to download data with progress=False to avoid console spam
                print(f"Downloading {ticker} (attempt {attempt+1}/{retries})")
                df = yf.download(ticker, start=start, end=end, progress=False)
                
                # Process successful download
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if not df.empty:
                    # Save to cache
                    if use_cache:
                        df.to_csv(cache_file)
                    return df
                else:
                    print(f"Empty data received for {ticker}")
                    attempt += 1
                    time.sleep(max_wait)
                    max_wait *= 2  # Exponential backoff
            
            except Exception as e:
                attempt += 1
                if "Rate limit" in str(e):
                    # Much longer wait on rate limits - 20-60 seconds
                    max_wait = 20 + attempt * 10
                    print(f"Rate limit hit for {ticker}, waiting {max_wait}s (attempt {attempt}/{retries})")
                else:
                    max_wait = 5 + attempt * 5
                    print(f"Error downloading {ticker}: {e} - waiting {max_wait}s")
                
                time.sleep(max_wait)
    
    print(f"Failed to download {ticker} after {retries} attempts")
    return pd.DataFrame()

def create_labels(df: pd.DataFrame, threshold=0.03, lookback=5):
    """Create more robust labels by considering multi-day trends"""
    # Calculate future return (next day)
    future_return = df["Close"].shift(-1) / df["Close"] - 1
    
    # Add short-term trend filter (previous 5 days)
    short_trend = df["Close"].pct_change(lookback)
    
    # Create labels with trend confirmation
    labels = []
    for i, (ret, trend) in enumerate(zip(future_return, short_trend)):
        if ret > threshold and trend > 0:  # Strong uptrend + positive prediction
            labels.append("BUY")
        elif ret < -threshold and trend < 0:  # Strong downtrend + negative prediction
            labels.append("SELL") 
        else:
            labels.append(None)
    
    return pd.Series(labels, index=df.index)

def get_sp500_tickers() -> list:
    """Get S&P 500 tickers with caching"""
    cache_file = f"{CACHE_DIR}/sp500_tickers.txt"
    
    # Check cache first
    if os.path.exists(cache_file):
        # Only use cache if it's less than 30 days old
        if (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 30:
            with open(cache_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
    
    try:
        # If we need to fetch new data
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = table[0]["Symbol"].tolist()
        tickers = [ticker.replace(".", "-") for ticker in tickers]
        
        # Cache the result
        with open(cache_file, 'w') as f:
            f.write('\n'.join(tickers))
            
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        # Return some default stocks if fetch fails
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"]
