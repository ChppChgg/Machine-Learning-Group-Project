import pandas as pd
import yfinance as yf

def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]]

def create_labels(df: pd.DataFrame, threshold=0.01) -> pd.Series:
    future_return = df["Close"].shift(-1) / df["Close"] - 1
    return future_return.apply(lambda x: "BUY" if x > threshold else "SELL" if x < -threshold else "HOLD")

def get_sp500_tickers() -> list:
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tickers = table[0]["Symbol"].tolist()
    return [ticker.replace(".", "-") for ticker in tickers]
