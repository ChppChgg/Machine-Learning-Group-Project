import pandas as pd
import numpy as np

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean().replace(0, 1e-10)

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, slow=26, fast=12, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"MACD": macd, "MACD_Signal": signal_line})

def calculate_sma(data: pd.Series, window: int = 20):
    return data.rolling(window).mean()

def calculate_ema(data: pd.Series, span: int = 20):
    return data.ewm(span=span, adjust=False).mean()

def calculate_bollinger_bands(data: pd.Series, window=20, num_std=2):
    sma = calculate_sma(data, window)
    std = data.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return pd.DataFrame({"Bollinger_Mid": sma, "Bollinger_Upper": upper, "Bollinger_Lower": lower})

def calculate_stochastic_oscillator(high, low, close, k=14, d=3):
    low_min = low.rolling(window=k).min()
    high_max = high.rolling(window=k).max()
    percent_k = 100 * (close - low_min) / (high_max - low_min).replace(0, 1e-10)
    percent_d = percent_k.rolling(window=d).mean()
    return pd.DataFrame({"Stochastic_%K": percent_k, "Stochastic_%D": percent_d})

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["RSI"] = calculate_rsi(df["Close"])
    features = features.join(calculate_macd(df["Close"]))
    features["SMA_20"] = calculate_sma(df["Close"])
    features["EMA_20"] = calculate_ema(df["Close"])
    features = features.join(calculate_bollinger_bands(df["Close"]))
    features = features.join(calculate_stochastic_oscillator(df["High"], df["Low"], df["Close"]))
    return features.dropna()