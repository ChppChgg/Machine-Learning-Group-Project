import pandas as pd
import numpy as np

def detect_market_regime(data, short_period=10, long_period=50):
    """
    Detect market regime (trend direction and strength)
    
    Parameters:
    -----------
    data : DataFrame
        Price data with 'Close' column
    short_period : int
        Short period for moving average
    long_period : int
        Long period for moving average
        
    Returns:
    --------
    DataFrame with columns:
    - trend: 1 (bullish), -1 (bearish), 0 (neutral)
    - strength: float 0-1 indicating trend strength
    """
    df = data.copy()
    
    # Calculate moving averages
    df['sma_short'] = df['Close'].rolling(short_period).mean()
    df['sma_long'] = df['Close'].rolling(long_period).mean()
    
    # Calculate trend direction and strength
    df['trend'] = np.where(df['sma_short'] > df['sma_long'], 1, 
                           np.where(df['sma_short'] < df['sma_long'], -1, 0))
    
    # Calculate trend strength (0-1)
    df['pct_diff'] = abs(df['sma_short'] - df['sma_long']) / df['sma_long']
    df['strength'] = df['pct_diff'] / df['pct_diff'].rolling(50).max().fillna(df['pct_diff'])
    df['strength'] = df['strength'].clip(0, 1)  # Force between 0-1
    
    return df[['trend', 'strength']]