import pandas as pd
import yfinance as yf

def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]]

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
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tickers = table[0]["Symbol"].tolist()
    return [ticker.replace(".", "-") for ticker in tickers]
