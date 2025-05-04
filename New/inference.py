import pandas as pd
import joblib
import sys
from sentiment_score import get_cached_sentiment
from indicators import add_technical_indicators
from data_prep import download_stock_data
from datetime import datetime, timedelta

# Load trained models and preprocessors
rf = joblib.load("new/models/random_forest_model.pkl")
xgb = joblib.load("new/models/xgboost_model.pkl")
lr = joblib.load("new/models/logistic_model.pkl")
scaler = joblib.load("new/models/scaler.pkl")
label_encoder = joblib.load("new/models/label_encoder.pkl")

def predict_for_ticker(ticker):
    from collections import Counter

    # Get recent price data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=45)
    df = download_stock_data(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

    if df.empty or len(df) < 10:
        print(f"No sufficient data for {ticker}")
        return

    df = add_technical_indicators(df)
    df = df.dropna(subset=["RSI", "MACD", "MACD_Signal", "SMA", "EMA"])

    if df.empty:
        print(f"Not enough data after computing indicators for {ticker}")
        return

    # Use most recent data row
    latest = df.iloc[-1:].copy()

    # Inject today's sentiment
    sentiment_score = get_cached_sentiment(ticker)
    latest["sentiment_7d_avg"] = sentiment_score

    # Feature columns
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "sentiment_7d_avg", "RSI", "MACD", "MACD_Signal", "SMA", "EMA"
    ]
    X = latest[feature_cols]
    X_scaled = scaler.transform(X)

    # Get predictions
    rf_pred = rf.predict(X_scaled)[0]
    xgb_pred = xgb.predict(X_scaled)[0]
    lr_pred = lr.predict(X_scaled)[0]

    rf_label = label_encoder.inverse_transform([rf_pred])[0]
    xgb_label = label_encoder.inverse_transform([xgb_pred])[0]
    lr_label = label_encoder.inverse_transform([lr_pred])[0]

    # Voting from models
    votes = [rf_label, xgb_label, lr_label]
    vote_count = Counter(votes)
    top_vote = vote_count.most_common(1)[0][0]

    # Adjust based on sentiment
    adjusted_decision = top_vote  # default
    if sentiment_score > 0.6 and top_vote == "HOLD":
        adjusted_decision = "BUY"
    elif sentiment_score < -0.6 and top_vote == "HOLD":
        adjusted_decision = "SELL"

    # Output
    print(f"\nPredictions for {ticker} on {latest['Date'].values[0]}")
    print(f"  Random Forest       : {rf_label}")
    print(f"  XGBoost             : {xgb_label}")
    print(f"  Logistic Regression : {lr_label}")
    print(f"  Sentiment Score     : {sentiment_score:.3f}")
    print(f"  Model Vote Result   : {top_vote}")
    print(f"  Final Recommendation: {adjusted_decision}")

if __name__ == "__main__":
    # Example usage
    #ticker_input = input("Enter a stock ticker (e.g., AAPL): ").strip().upper()
    #predict_for_ticker(ticker_input)
    sys.stdout = open("new/prediction_output.txt", "w")
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD",
        "ADBE", "CRM", "ORCL", "CSCO", "QCOM", "AVGO", "TXN", "IBM", "SHOP", "SQ",
        "PYPL", "PLTR", "UBER", "LYFT", "TWLO", "ROKU", "SPOT", "BA", "DIS", "NKE",
        "WMT", "TGT", "COST", "MCD", "KO", "PEP", "JNJ", "PFE", "MRK", "CVX",
        "XOM", "BP", "V", "MA", "AXP", "GS", "JPM", "BAC", "WFC", "BLK"
    ]
    for ticker in tickers:
        print("\n" + "=" * 40)
        predict_for_ticker(ticker)