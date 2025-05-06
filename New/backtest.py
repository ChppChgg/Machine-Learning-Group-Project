import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from data_prep import download_stock_data
from sentiment_score import get_cached_sentiment
from indicators import add_technical_indicators

# Load trained models and tools
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

def predict_tomorrow(ticker, portfolio_value=10000):
    df = download_stock_data(ticker, start=(datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d"))
    if df.empty or len(df) < 10:
        return None

    df = add_technical_indicators(df)
    df.dropna(subset=FEATURE_COLS[6:], inplace=True)

    if df.empty:
        return None

    latest = df.iloc[-1].copy()

    try:
        sentiment = get_cached_sentiment(ticker)
    except:
        sentiment = 0.0

    latest["sentiment_7d_avg"] = sentiment
    X_df = pd.DataFrame([latest[FEATURE_COLS]])
    X_scaled = scaler.transform(X_df)

    rf_probs = rf.predict_proba(X_scaled)[0]
    xgb_probs = xgb.predict_proba(X_scaled)[0]
    lr_probs = lr.predict_proba(X_scaled)[0]

    labels = label_encoder.classes_
    avg_probs = (rf_probs + xgb_probs + lr_probs) / 3
    decision_index = avg_probs.argmax()
    decision = labels[decision_index]

    # Model votes
    rf_label = label_encoder.inverse_transform([rf.predict(X_scaled)[0]])[0]
    xgb_label = label_encoder.inverse_transform([xgb.predict(X_scaled)[0]])[0]
    lr_label = label_encoder.inverse_transform([lr.predict(X_scaled)[0]])[0]
    votes = [rf_label, xgb_label, lr_label]

    # Sentiment-based adjustment
    adjusted = decision
    if sentiment > 0.6 and decision == "HOLD":
        adjusted = "BUY"
    elif sentiment < -0.6 and decision == "HOLD":
        adjusted = "SELL"

    # Position sizing
    model_agreement = votes.count(adjusted)
    sentiment_strength = abs(sentiment)
    base_position = 0.05
    size = base_position + 0.05 * model_agreement + 0.05 * (sentiment_strength > 0.5)
    investment = portfolio_value * min(size, 0.25)

    # Price movement range estimation
    std_dev = df["Close"].pct_change().std()
    last_price = latest["Close"]
    expected_range = (
        round(last_price * (1 - std_dev), 2),
        round(last_price * (1 + std_dev), 2)
    )

    return {
        "Final Decision": adjusted,
        "Votes": f"RF={rf_label}, XGB={xgb_label}, LR={lr_label}",
        "Sentiment": round(sentiment, 3),
        "Confidence": dict(zip(labels, avg_probs.round(3))),
        "Suggested Investment (£)": round(investment, 2),
        "Expected Price Range": expected_range,
        "Reasoning": generate_reasoning(adjusted, model_agreement, sentiment)
    }

def generate_reasoning(decision, agreement_count, sentiment):
    reasons = []

    if agreement_count == 3:
        reasons.append("All models agree on this decision.")
    elif agreement_count == 2:
        reasons.append("Majority of models agree.")
    else:
        reasons.append("Mixed model signals — be cautious.")

    if decision == "BUY" and sentiment > 0.5:
        reasons.append("Strong positive sentiment reinforces buy.")
    elif decision == "SELL" and sentiment < -0.5:
        reasons.append("Negative sentiment adds weight to sell signal.")
    elif decision == "HOLD":
        reasons.append("Market uncertainty — no clear signal.")

    return " ".join(reasons)

def backtest_ticker(ticker, initial_money=10000, start="2025-01-01", end="2025-05-01", lookahead=3):
    df = download_stock_data(ticker, start=start, end=end)
    if df.empty or len(df) < lookahead + 10:
        print(f"No sufficient data for {ticker}")
        return pd.DataFrame()

    df = add_technical_indicators(df)
    df.dropna(subset=FEATURE_COLS[6:], inplace=True)
    df["sentiment_7d_avg"] = 0.0

    sentiment_df = pd.read_csv("new/csv_files/sentiment_cache.csv")
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])

    results = []
    money = initial_money

    for i in range(len(df) - lookahead):
        row = df.iloc[i].copy()
        future_close = df.iloc[i + lookahead]["Close"]
        current_close = row["Close"]
        date = pd.to_datetime(row["Date"])

        sentiment_score = sentiment_df[
            (sentiment_df["Ticker"].str.upper() == ticker.upper()) &
            (sentiment_df["Date"] == date)
        ]["sentiment_7d_avg"].values
        sentiment = float(sentiment_score[0]) if len(sentiment_score) else 0.0

        row["sentiment_7d_avg"] = sentiment
        X_df = pd.DataFrame([row[FEATURE_COLS]])
        X_scaled = scaler.transform(X_df)

        rf_pred = label_encoder.inverse_transform([rf.predict(X_scaled)[0]])[0]
        xgb_pred = label_encoder.inverse_transform([xgb.predict(X_scaled)[0]])[0]
        lr_pred = label_encoder.inverse_transform([lr.predict(X_scaled)[0]])[0]

        votes = [rf_pred, xgb_pred, lr_pred]
        top_vote = max(set(votes), key=votes.count)

        adjusted_decision = top_vote
        if sentiment > 0.6 and top_vote == "HOLD":
            adjusted_decision = "BUY"
        elif sentiment < -0.6 and top_vote == "HOLD":
            adjusted_decision = "SELL"

        pct_return = (future_close - current_close) / current_close
        profit_loss = 0
        trade_executed = False

        if adjusted_decision == "BUY":
            profit_loss = money * pct_return
            money += profit_loss
            trade_executed = True
        elif adjusted_decision == "SELL":
            profit_loss = -money * pct_return
            money += profit_loss
            trade_executed = True

        results.append({
            "Date": date,
            "Ticker": ticker.upper(),
            "Decision": adjusted_decision,
            "Entry Price": current_close,
            "Future Price": future_close,
            "Return (%)": round(pct_return * 100, 2),
            "Trade Executed": trade_executed,
            "Profit/Loss (£)": round(profit_loss, 2),
            "Portfolio Value (£)": round(money, 2),
            "Sentiment": sentiment,
            "Votes": f"RF={rf_pred}, XGB={xgb_pred}, LR={lr_pred}"
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    user_input = input("Enter a stock ticker (e.g., AAPL): ").strip().upper()
    df = backtest_ticker(user_input)

    if not df.empty:
        path = f"new/csv_files/model_backtest_results.csv"
        df.to_csv(path, index=False)
        print(f"\nBacktest saved to {path}")

        # Summary
        final_value = df["Portfolio Value (£)"].iloc[-1]
        total_profit = final_value - 10000
        trade_count = df["Trade Executed"].sum()
        profitable_trades = df[df["Profit/Loss (£)"] > 0]["Trade Executed"].sum()
        avg_return = df["Return (%)"][df["Trade Executed"]].mean()

        print("\nBacktest Summary")
        print(f"Total Trades         : {trade_count}")
        print(f"Profitable Trades    : {profitable_trades}")
        print(f"Average Return/Trade : {avg_return:.2f}%")
        print(f"Final Portfolio (£)  : {final_value:.2f}")
        print(f"Total Profit (£)     : {total_profit:.2f}")

        # Tomorrow prediction
        print("\nPredicted Trade for Tomorrow")
        prediction = predict_tomorrow(user_input)
        if prediction:
            print(f"Final Decision        : {prediction['Final Decision']}")
            print(f"Votes                 : {prediction['Votes']}")
            print(f"Sentiment             : {prediction['Sentiment']:.3f}")
            print(f"Confidence            : {prediction['Confidence']}")
            print(f"Suggested Investment  : £{prediction['Suggested Investment (£)']}")
            print(f"Expected Price Range  : £{prediction['Expected Price Range'][0]} - £{prediction['Expected Price Range'][1]}")
            print(f"Reasoning             : {prediction['Reasoning']}")
        else:
            print("Not enough data to predict tomorrow's trade.")

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Portfolio Value (£)"], label=f"{user_input} Portfolio (£)", linewidth=2)
        plt.title(f"Portfolio Growth Over Time - {user_input}", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (£)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No data available to backtest that ticker.")
