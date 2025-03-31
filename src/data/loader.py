import pandas as pd
import yfinance as yf
from datetime import datetime

def load_news_data(cnbc_path, reuters_path):
    """
    Load and combine financial news data from CSV files.
    
    Args:
        cnbc_path: Path to CNBC headlines CSV
        reuters_path: Path to Reuters headlines CSV
        
    Returns:
        Processed news DataFrame
    """
    data1 = pd.read_csv(cnbc_path)
    data2 = pd.read_csv(reuters_path)
    
    # Combine news data
    news_data = pd.concat([data1, data2])
    
    # Clean and process dates
    news_data = news_data.dropna()
    news_data['Date'] = pd.to_datetime(news_data['Time'], 
                                     format='%I:%M %p ET %a, %d %B %Y', 
                                     errors='coerce')
    news_data = news_data.sort_values('Date')
    news_data['DateOnly'] = news_data['Date'].dt.date
    
    return news_data

def download_stock_data(symbol, start_date, end_date):
    """
    Download stock data from Yahoo Finance.
    
    Args:
        symbol: Stock symbol
        start_date: Start date for data download
        end_date: End date for data download
        
    Returns:
        DataFrame with stock data
    """
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    
    # Process column names
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.columns = [''.join(col).strip() if col[1] else col[0] for col in stock_data.columns]
    stock_data.columns = [col.replace(symbol, '') for col in stock_data.columns]
    
    return stock_data
