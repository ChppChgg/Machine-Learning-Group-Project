import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start="2020-01-01", end=None):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)  # 'Date' becomes a column
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Ticker"] = ticker
    return df

if __name__ == "__main__":
    df = download_stock_data("AAPL")
    print(df.head())

