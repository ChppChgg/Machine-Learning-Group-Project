import pandas as pd

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

# Apply all technical indicators
def add_technical_indicators(df):
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_sma(df)
    df = compute_ema(df)
    return df
