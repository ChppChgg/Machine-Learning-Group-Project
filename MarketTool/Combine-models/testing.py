import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from data_prep import download_stock_data
from indicators import generate_features
from sklearn.preprocessing import StandardScaler

# --- Load models and scaler ---
model_path = "MarketTool/Combine-models/models/"
model = joblib.load(os.path.join(model_path, "stacked_classifier.pkl"))
scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))

# --- Choose ticker and period ---
ticker = "ADBE"
start = "2023-01-01"
end = "2023-12-31"

# --- Download and process data ---
df = download_stock_data(ticker, start, end)
features = generate_features(df, ticker)
features = features.dropna()

# --- Prepare features for model (drop non-feature columns if present) ---
X = features.drop(columns=["Label", "Ticker", "Date"], errors="ignore")
X_scaled = scaler.transform(X)

# --- Predict using the trained model ---
preds = model.predict(X_scaled)
pred_labels = encoder.inverse_transform(preds)

# --- Add predictions to DataFrame ---
features = features.copy()
features["Model_Prediction"] = pred_labels
features["Close"] = df.loc[features.index, "Close"]

# --- Backtesting Model Predictions ---

# Calculate daily returns
features['Returns'] = features['Close'].pct_change() * 100

# Map model predictions to positions: 1 for BUY, -1 for SELL, 0 for HOLD
position_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
features['Position'] = features['Model_Prediction'].map(position_map).fillna(0)

# Calculate strategy returns (shift position to avoid lookahead bias)
features['Strategy_Returns'] = features['Position'].shift(1) * features['Returns']
features.dropna(subset=['Strategy_Returns'], inplace=True)

# Calculate cumulative returns
features['Cumulative_Returns'] = (1 + features['Returns'] / 100).cumprod()
features['Cumulative_Strategy_Returns'] = (1 + features['Strategy_Returns'] / 100).cumprod()

# --- Plotting: All graphs in a single window ---
fig, axs = plt.subplots(4, 1, figsize=(14, 16))

# 1. Price and predictions
axs[0].plot(features.index, features["Close"], label="Close Price", color="black")
buy_idx = features[features["Model_Prediction"] == "BUY"].index
sell_idx = features[features["Model_Prediction"] == "SELL"].index
hold_idx = features[features["Model_Prediction"] == "HOLD"].index if "HOLD" in features["Model_Prediction"].unique() else []
axs[0].scatter(buy_idx, features.loc[buy_idx, "Close"], marker="^", color="green", label="BUY", alpha=0.7)
axs[0].scatter(sell_idx, features.loc[sell_idx, "Close"], marker="v", color="red", label="SELL", alpha=0.7)
axs[0].set_title(f"{ticker} Price & Model Predictions")
axs[0].legend()
axs[0].grid(alpha=0.3)

# 2. RSI (if present)
if "RSI" in features.columns:
    axs[1].plot(features.index, features["RSI"], label="RSI", color="purple")
    axs[1].axhline(70, color="red", linestyle="--", alpha=0.5)
    axs[1].axhline(30, color="green", linestyle="--", alpha=0.5)
    axs[1].set_title("RSI")
    axs[1].set_ylim(0, 100)
    axs[1].grid(alpha=0.3)
    axs[1].legend()
else:
    axs[1].set_visible(False)

# 3. MACD (if present)
if "MACD" in features.columns and "MACD_Signal" in features.columns:
    axs[2].plot(features.index, features["MACD"], label="MACD", color="blue")
    axs[2].plot(features.index, features["MACD_Signal"], label="Signal", color="red")
    axs[2].set_title("MACD")
    axs[2].grid(alpha=0.3)
    axs[2].legend()
else:
    axs[2].set_visible(False)

# 4. Cumulative returns (Buy & Hold vs Model)
axs[3].plot(features.index, features['Cumulative_Returns'], label='Buy and Hold', color='blue')
axs[3].plot(features.index, features['Cumulative_Strategy_Returns'], label='Model Strategy', color='green')
axs[3].set_title(f'{ticker} Model Backtest vs Buy and Hold')
axs[3].set_xlabel('Date')
axs[3].set_ylabel('Cumulative Returns')
axs[3].legend()
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print performance metrics
strategy_return = (features['Cumulative_Strategy_Returns'].iloc[-1] - 1) * 100
buy_hold_return = (features['Cumulative_Returns'].iloc[-1] - 1) * 100
sharpe_ratio = features['Strategy_Returns'].mean() / features['Strategy_Returns'].std() * np.sqrt(252)
print(f"\nBacktest Performance ({ticker}):")
print(f"Strategy return: {strategy_return:.2f}%")
print(f"Buy and hold return: {buy_hold_return:.2f}%")
print(f"Strategy outperformance: {strategy_return - buy_hold_return:.2f}%")
print(f"Sharpe ratio: {sharpe_ratio:.2f}")