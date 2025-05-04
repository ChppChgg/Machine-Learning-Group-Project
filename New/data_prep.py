import yfinance as yf
import pandas as pd

from indicators import add_technical_indicators
from sentiment_score import get_cached_sentiment

def get_sp500_tickers(limit=200):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    df = table[0]
    tickers = df["Symbol"].dropna().unique().tolist()
    return tickers[:limit]


# Downloads historical stock data for a given ticker
def download_stock_data(ticker, start="2020-01-01", end=None):
    df = yf.download(ticker, start=start, end=end)

    # Reset index (flatten row index)
    df.reset_index(inplace=True)

    # Flatten columns (handles multi-level columns)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Ticker"] = ticker.upper()
    return df

# Merges the price data with cached sentiment scores
def merge_with_sentiment(price_df, sentiment_path="new/csv_files/sentiment_cache.csv"):
    sentiment_df = pd.read_csv(sentiment_path)

    # Standardise column names and formats for merging
    sentiment_df.rename(columns={"date": "Date", "ticker": "Ticker"}, inplace=True)
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])
    sentiment_df["Ticker"] = sentiment_df["Ticker"].str.upper()
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df["Ticker"] = price_df["Ticker"].str.upper()

    merged = pd.merge(price_df, sentiment_df, on=["Date", "Ticker"], how="left")
    return merged

# Generates BUY, SELL, or HOLD labels based on next-day returns
def label_data(df, lookahead=3):
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Future_Close"] = df["Close"].shift(-lookahead)
    df["Return"] = (df["Future_Close"] / df["Close"]) - 1

    valid_returns = df["Return"].dropna()
    upper = valid_returns.quantile(0.65)
    lower = valid_returns.quantile(0.35)

    def label_row(ret):
        if pd.isna(ret):
            return None
        elif ret > upper:
            return "BUY"
        elif ret < lower:
            return "SELL"
        else:
            return "HOLD"

    df["Label"] = df["Return"].apply(label_row)
    df.drop(columns=["Future_Close", "Return"], inplace=True)
    return df



if __name__ == "__main__":
    all_data = []

    tickers = get_sp500_tickers()

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")

        # Update sentiment cache for this ticker (ensures sentiment_7d_avg exists)
        try:
            get_cached_sentiment(ticker)
        except Exception as e:
            print(f"Skipping sentiment for {ticker} due to error: {e}")
            continue

        # Download price data and process it
        try:
            df = download_stock_data(ticker)
            merged = merge_with_sentiment(df)
            with_indicators = add_technical_indicators(merged)
            labelled = label_data(with_indicators)
            labelled.dropna(subset=["Label"], inplace=True)
            labelled["sentiment_7d_avg"] = labelled["sentiment_7d_avg"].fillna(0.0)
            required_cols = ["RSI", "MACD", "MACD_Signal", "SMA", "EMA"]
            cleaned = labelled.dropna(subset=required_cols)
            print(f"  {ticker} rows before drop: {len(labelled)} | after drop: {len(cleaned)}")

            if not cleaned.empty:
                all_data.append(cleaned)
        except Exception as e:
            print(f"Skipping {ticker} due to data error: {e}")
            continue

    # Combine and save final training dataset
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

        feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "sentiment_7d_avg",
        "RSI", "MACD", "MACD_Signal", "SMA", "EMA",
        "BB_High", "BB_Low", "OBV", "Momentum", "WilliamsR"
        ]
        target_col = "Label"

        output_df = final_df[["Date", "Ticker"] + feature_cols + [target_col]]
        output_df.to_csv("new/csv_files/training_data.csv", index=False)

        print("\nSaved training_data.csv with shape:", output_df.shape)
    else:
        print("No usable data was generated.")