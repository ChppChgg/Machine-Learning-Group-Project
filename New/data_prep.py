import yfinance as yf
import pandas as pd

from indicators import add_technical_indicators
from sentiment_score import get_cached_sentiment

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
def label_data(df, threshold=0.0075):
    df = df.copy()
    df["Future_Close"] = df["Close"].shift(-1)
    df["Return"] = (df["Future_Close"] / df["Close"]) - 1

    def label(row):
        if row["Return"] > threshold:
            return "BUY"
        elif row["Return"] < -threshold:
            return "SELL"
        else:
            return "HOLD"

    df["Label"] = df.apply(label, axis=1)
    df.drop(columns=["Future_Close", "Return"], inplace=True)
    return df



if __name__ == "__main__":
    # List of tickers to process (expand as needed)
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD",
        "ADBE", "CRM", "ORCL", "CSCO", "QCOM", "AVGO", "TXN", "IBM", "SHOP", "SQ",
        "PYPL", "PLTR", "UBER", "LYFT", "TWLO", "ROKU", "SPOT", "BA", "DIS", "NKE",
        "WMT", "TGT", "COST", "MCD", "KO", "PEP", "JNJ", "PFE", "MRK", "CVX",
        "XOM", "BP", "V", "MA", "AXP", "GS", "JPM", "BAC", "WFC", "BLK"
    ]

    all_data = []

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
            "sentiment_7d_avg", "RSI", "MACD", "MACD_Signal", "SMA", "EMA"
        ]
        target_col = "Label"

        output_df = final_df[["Date", "Ticker"] + feature_cols + [target_col]]
        output_df.to_csv("new/csv_files/training_data.csv", index=False)

        print("\nSaved training_data.csv with shape:", output_df.shape)
    else:
        print("No usable data was generated.")