import argparse
import json
import os
import pandas as pd
import numpy as np
from src.data.loader import load_news_data, download_stock_data
from src.features.sentiment import FinBERTSentimentAnalyzer
from src.features.technical import calculate_technical_indicators
from src.models.cnn_lstm import build_cnn_lstm_han_model
from src.evaluation.backtest import SentimentTradingStrategy
from src.utils.visualization import plot_predictions, plot_portfolio_performance

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Financial Market Prediction')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'backtest'], 
                        default='train', help='Operation mode')
    parser.add_argument('--config', type=str, default='config/model_config.json',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Path to saved model')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Data loading
    print("Loading data...")
    news_data = load_news_data(config['cnbc_path'], config['reuters_path'])
    stock_data = download_stock_data(config['symbol'], config['start_date'], config['end_date'])
    
    # Sentiment analysis
    print("Analyzing sentiment...")
    sentiment_analyzer = FinBERTSentimentAnalyzer()
    news_data['FinBERT_Sentiment'] = sentiment_analyzer.batch_sentiment_analysis(
        news_data['Headlines'].tolist()
    )
    
    # Aggregate daily sentiment
    daily_sentiment = news_data.groupby('DateOnly')['FinBERT_Sentiment'].mean().reset_index()
    daily_sentiment.rename(columns={'DateOnly': 'Date'}, inplace=True)
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    
    # Merge stock data with sentiment
    combined_data = pd.merge(stock_data, daily_sentiment, on='Date', how='left')
    combined_data['FinBERT_Sentiment'].fillna(method='ffill', inplace=True)
    
    # Calculate technical indicators
    combined_data = calculate_technical_indicators(combined_data)
    
    if args.mode == 'train':
        # Train the model
        print("Training model...")
        # [Add training code here]
        
    elif args.mode == 'predict':
        # Make predictions using the model
        print("Making predictions...")
        # [Add prediction code here]
        
    elif args.mode == 'backtest':
        # Run backtesting
        print("Running backtest...")
        strategy = SentimentTradingStrategy(combined_data)
        strategy.generate_signals('FinBERT_Sentiment')
        results = strategy.backtest()
        
        print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        # Plot results
        plot_portfolio_performance(results['portfolio_value'])

if __name__ == '__main__':
    main()
