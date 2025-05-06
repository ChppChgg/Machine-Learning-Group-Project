import pandas as pd
import joblib
import sys
from sentiment_score import get_cached_sentiment
from indicators import add_technical_indicators
from data_prep import download_stock_data, get_sp500_tickers
from datetime import datetime, timedelta
from collections import Counter

# Load trained models and preprocessors
rf = joblib.load("new/models/random_forest_model.pkl")
xgb = joblib.load("new/models/xgboost_model.pkl")
lr = joblib.load("new/models/logistic_model.pkl")
scaler = joblib.load("new/models/scaler.pkl")
label_encoder = joblib.load("new/models/label_encoder.pkl")

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "sentiment_7d_avg", "RSI", "MACD", "MACD_Signal",
    "SMA", "EMA", "OBV", "Momentum", "WilliamsR"
]

def predict_for_ticker(ticker):
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=45)
        df = download_stock_data(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

        if df.empty or len(df) < 10:
            print(f"{ticker}: Not enough raw data")
            return

        df = add_technical_indicators(df)
        df.dropna(subset=FEATURE_COLS[6:], inplace=True)

        if df.empty:
            print(f"{ticker}: Not enough data after indicators")
            return

        latest = df.iloc[-1:].copy()

        # Fetch and assign today's sentiment
        sentiment_score = get_cached_sentiment(ticker)
        latest["sentiment_7d_avg"] = sentiment_score

        # Ensure all required features are present
        if not all(col in latest.columns for col in FEATURE_COLS):
            print(f"{ticker}: Missing features for prediction")
            return

        X = latest[FEATURE_COLS]
        X_scaled = scaler.transform(X)

        # Predict with each model
        rf_pred = rf.predict(X_scaled)[0]
        xgb_pred = xgb.predict(X_scaled)[0]
        lr_pred = lr.predict(X_scaled)[0]

        # Convert to labels
        rf_label = label_encoder.inverse_transform([rf_pred])[0]
        xgb_label = label_encoder.inverse_transform([xgb_pred])[0]
        lr_label = label_encoder.inverse_transform([lr_pred])[0]

        # Voting
        votes = [rf_label, xgb_label, lr_label]
        vote_count = Counter(votes)
        top_vote = vote_count.most_common(1)[0][0]

        # Sentiment adjustment
        adjusted_decision = top_vote
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

    except Exception as e:
        print(f"{ticker}: Error - {e}")

if __name__ == "__main__":
    sys.stdout = open("new/prediction_output.txt", "w")
    tickers = get_sp500_tickers(limit=50)  # Limit to 50 for speed
    for ticker in tickers:
        print("\n" + "=" * 40)
        predict_for_ticker(ticker)
