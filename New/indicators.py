import pandas as pd
import ta

# Relative Strength Index (RSI)
def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Moving Average Convergence Divergence (MACD)
def compute_macd(df, short=12, long=26, signal=9):
    exp1 = df['Close'].ewm(span=short, adjust=False).mean()
    exp2 = df['Close'].ewm(span=long, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

# Simple Moving Average (SMA)
def compute_sma(df, window=20):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    return df

# Exponential Moving Average (EMA)
def compute_ema(df, span=20):
    df['EMA'] = df['Close'].ewm(span=span, adjust=False).mean()
    return df

# Bollinger bands
def compute_bollinger_bands(df, window=20):
    bb = ta.volatility.BollingerBands(close=df["Close"], window=window, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    return df

# On Balance Volume
def compute_obv(df):
    obv = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
    df["OBV"] = obv.on_balance_volume()
    return df

# Momentum
def compute_momentum(df, window=10):
    mom = ta.momentum.ROCIndicator(close=df["Close"], window=window)
    df["Momentum"] = mom.roc()
    return df

# Williams % ratio
def compute_williams_r(df):
    will = ta.momentum.WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"])
    df["WilliamsR"] = will.williams_r()
    return df


# Apply all technical indicators
def add_technical_indicators(df):
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_sma(df)
    df = compute_ema(df)
    df = compute_bollinger_bands(df)
    df = compute_obv(df)
    df = compute_momentum(df)
    df = compute_williams_r(df)
    return df
