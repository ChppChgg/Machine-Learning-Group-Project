import pandas as pd
import numpy as np
from finviz_sentiment import fetch_sentiment_score

# === Relative Strength Index (RSI) ===
def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# === Moving Average Convergence Divergence (MACD) ===
def calculate_macd(data: pd.Series, slow=26, fast=12, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"MACD": macd, "MACD_Signal": signal_line})

# === Simple Moving Average (SMA) ===
def calculate_sma(data: pd.Series, window: int = 20):
    return data.rolling(window).mean()

# === Exponential Moving Average (EMA) ===
def calculate_ema(data: pd.Series, span: int = 20):
    return data.ewm(span=span, adjust=False).mean()

# === Bollinger Bands ===
def calculate_bollinger_bands(data: pd.Series, window=20, num_std=2):
    sma = calculate_sma(data, window)
    std = data.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return pd.DataFrame({
        "Bollinger_Mid": sma,
        "Bollinger_Upper": upper,
        "Bollinger_Lower": lower
    })

# === Stochastic Oscillator (%K and %D) ===
def calculate_stochastic_oscillator(high, low, close, k=14, d=3):
    low_min = low.rolling(window=k).min()
    high_max = high.rolling(window=k).max()
    percent_k = 100 * (close - low_min) / (high_max - low_min).replace(0, 1e-10)
    percent_d = percent_k.rolling(window=d).mean()
    return pd.DataFrame({
        "Stochastic_%K": percent_k,
        "Stochastic_%D": percent_d
    })

# === Average True Range (ATR) ===
def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

# === On-Balance Volume (OBV) ===
def calculate_obv(close, volume):
    direction = close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    obv = (volume * direction).cumsum()
    return obv

# === Main Feature Generator ===
def generate_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    # Standard technical indicators
    features["RSI"] = calculate_rsi(df["Close"])
    features = features.join(calculate_macd(df["Close"]))
    features["SMA_20"] = calculate_sma(df["Close"])
    features["EMA_20"] = calculate_ema(df["Close"])
    features = features.join(calculate_bollinger_bands(df["Close"]))
    features = features.join(calculate_stochastic_oscillator(df["High"], df["Low"], df["Close"]))
    features["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"])
    features["OBV"] = calculate_obv(df["Close"], df["Volume"])

    # Engineered features (custom signals)
    features["RSI_diff"] = features["RSI"] - features["RSI"].shift(3)
    features["MACD_diff"] = features["MACD"] - features["MACD_Signal"]
    features["Price_vs_SMA"] = df["Close"] / features["SMA_20"]

    # Weekly sentiment fetched from Finviz (only Mondays)
    features["Sentiment"] = features.index.to_series().apply(
        lambda d: fetch_sentiment_score(ticker, d.strftime("%Y-%m-%d")) if d.weekday() == 0 else np.nan
    )
    features["Sentiment"] = features["Sentiment"].ffill()  # Carry forward last known sentiment

    return features.dropna()
