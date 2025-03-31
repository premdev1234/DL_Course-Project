import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SentimentTradingStrategy:
    """
    Backtesting framework for sentiment-based trading strategy.
    """
    
    def __init__(self, data, initial_cash=100000, position_size=0.2, transaction_cost=0.001):
        """
        Initialize the backtesting strategy.
        
        Args:
            data: DataFrame with price and sentiment data
            initial_cash: Initial investment amount
            position_size: Fraction of portfolio to invest in each trade
            transaction_cost: Transaction cost as a fraction
        """
        self.data = data
        self.initial_cash = initial_cash
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        
    def generate_signals(self, sentiment_col, buy_threshold=0.5, sell_threshold=-0.5):
        """
        Generate trading signals based on sentiment scores.
        
        Args:
            sentiment_col: Column name for sentiment scores
            buy_threshold: Threshold for buy signals
            sell_threshold: Threshold for sell signals
            
        Returns:
            DataFrame with added signals
        """
        # Normalize sentiment scores to range (-1, 1)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data[sentiment_col] = scaler.fit_transform(self.data[[sentiment_col]])
        
        # Generate signals
        self.data['Signal'] = np.where(self.data[sentiment_col] > buy_threshold, 1,
                                     np.where(self.data[sentiment_col] < sell_threshold, -1, 0))
        
        return self.data
    
    def backtest(self):
        """
        Run backtest of trading strategy.
        
        Returns:
            Dictionary with backtest results
        """
        cash = self.initial_cash
        shares_held = 0
        portfolio_value = []
        
        for i in range(len(self.data) - 1):
            signal = self.data['Signal'].iloc[i]
            next_day_price = self.data['Close'].iloc[i + 1]
            
            # Buy condition
            if signal == 1 and cash > 0:
                invest_amount = cash * self.position_size
                shares_to_buy = invest_amount // next_day_price
                cost = shares_to_buy * next_day_price * (1 + self.transaction_cost)
                
                if cost <= cash:
                    shares_held += shares_to_buy
                    cash -= cost
            
            # Sell condition
            elif signal == -1 and shares_held > 0:
                sell_value = shares_held * next_day_price * (1 - self.transaction_cost)
                cash += sell_value
                shares_held = 0
            
            # Track portfolio value
            total_value = cash + (shares_held * next_day_price)
            portfolio_value.append(total_value)
        
        # Calculate performance metrics
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized Sharpe Ratio
        max_drawdown = np.min(np.divide(portfolio_value, np.maximum.accumulate(portfolio_value))) - 1
        
        return {
            'portfolio_value': portfolio_value,
            'final_value': portfolio_value[-1],
            'returns': returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
