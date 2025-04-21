import os
import random
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from data_prep import download_stock_data, get_sp500_tickers
from indicators import generate_features
from sklearn.preprocessing import StandardScaler

# --- Load models ---
model_path = "MarketTool/Combine-models/models/"
model = joblib.load(os.path.join(model_path, "stacked_classifier.pkl"))
scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))

# --- Settings ---
start = "2023-01-01"
end = "2024-12-01"
num_stocks = 20  # number of random S&P 500 stocks to test

tickers = get_sp500_tickers()
random_tickers = random.sample(tickers, num_stocks)

results = []

# --- Run backtest for each ticker ---
for ticker in random_tickers:
    try:
        print(f"\n=== {ticker} ===")
        df = download_stock_data(ticker, start, end)
        if df.empty or len(df) < 100:
            print("Insufficient data. Skipping.")
            continue

        features = generate_features(df, ticker).dropna()
        X = features.drop(columns=["Label", "Ticker", "Date"], errors="ignore")
        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)
        pred_labels = encoder.inverse_transform(preds)

        features["Model_Prediction"] = pred_labels
        features["Close"] = df.loc[features.index, "Close"]
        features["Returns"] = features["Close"].pct_change() * 100

        position_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
        features["Position"] = features["Model_Prediction"].map(position_map).fillna(0)
        features["Strategy_Returns"] = features["Position"].shift(1) * features["Returns"]
        features.dropna(subset=["Strategy_Returns"], inplace=True)

        features["Cumulative_Returns"] = (1 + features["Returns"] / 100).cumprod()
        features["Cumulative_Strategy_Returns"] = (1 + features["Strategy_Returns"] / 100).cumprod()

        strategy_return = (features["Cumulative_Strategy_Returns"].iloc[-1] - 1) * 100
        buy_hold_return = (features["Cumulative_Returns"].iloc[-1] - 1) * 100
        sharpe = features["Strategy_Returns"].mean() / features["Strategy_Returns"].std() * np.sqrt(252)

        results.append({
            "Ticker": ticker,
            "Strategy Return (%)": round(strategy_return, 2),
            "Buy & Hold Return (%)": round(buy_hold_return, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Outperformance (%)": round(strategy_return - buy_hold_return, 2)
        })

    except Exception as e:
        print(f"Error with {ticker}: {e}")

# --- Save and display results ---
results_df = pd.DataFrame(results)
os.makedirs("MarketTool/Combine-models/test-results", exist_ok=True)
results_df.to_csv("MarketTool/Combine-models/test-results/mass_backtest_results.csv", index=False)

print("\n=== Summary Results ===")
print(results_df)

# --- Plot Visualisation ---
plt.figure(figsize=(14, 10))

# 1. Sharpe Ratio Histogram
plt.subplot(2, 2, 1)
plt.hist(results_df["Sharpe Ratio"], bins=10, color="skyblue", edgecolor="black")
plt.title("Sharpe Ratio Distribution")
plt.xlabel("Sharpe Ratio")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)

# 2. Strategy Return vs Buy & Hold
plt.subplot(2, 2, 2)
plt.scatter(results_df["Buy & Hold Return (%)"], results_df["Strategy Return (%)"], c="green", alpha=0.7)
plt.plot(results_df["Buy & Hold Return (%)"], results_df["Buy & Hold Return (%)"], linestyle="--", color="red", label="Equal Returns")
plt.xlabel("Buy & Hold Return (%)")
plt.ylabel("Strategy Return (%)")
plt.title("Strategy vs Buy & Hold Returns")
plt.legend()
plt.grid(alpha=0.3)

# 3. Outperformance Histogram
plt.subplot(2, 2, 3)
plt.hist(results_df["Outperformance (%)"], bins=10, color="orange", edgecolor="black")
plt.title("Strategy Outperformance (vs Buy & Hold)")
plt.xlabel("Outperformance (%)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)

# 4. Top Performing Stocks by Strategy Return
top_performers = results_df.sort_values("Strategy Return (%)", ascending=False).head(10)
plt.subplot(2, 2, 4)
plt.bar(top_performers["Ticker"], top_performers["Strategy Return (%)"], color="blue")
plt.title("Top 10 Performing Stocks (Strategy)")
plt.xlabel("Ticker")
plt.ylabel("Strategy Return (%)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()