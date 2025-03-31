import numpy as np
import pandas as pd

def calculate_technical_indicators(df, price_col='Close'):
    """
    Calculate technical indicators for stock data.
    
    Args:
        df: DataFrame with stock price data
        price_col: Column name for price data
        
    Returns:
        DataFrame with added technical indicators
    """
    # Moving Averages
    df['MA5'] = df[price_col].rolling(window=5).mean()
    df['MA10'] = df[price_col].rolling(window=10).mean()
    
    # Returns
    df['Returns'] = df[price_col].pct_change()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=5).std()
    
    # RSI (Relative Strength Index)
    df['RSI'] = 100 - (100 / (1 + df['Returns'].rolling(window=14).mean() / 
                                df['Returns'].rolling(window=14).std()))
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Returns']) * df['Volume']).cumsum()
    
    # Bollinger Bands
    df['SMA_20'] = df[price_col].rolling(window=20).mean()
    df['STD_20'] = df[price_col].rolling(window=20).std()
    df['Bollinger_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['Bollinger_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
    
    # Clean up NaN values from rolling calculations
    df.dropna(inplace=True)
    
    return df
