import pandas as pd
import numpy as np


class FeatureGenerator:
    @staticmethod
    def calculate_rsi(close_prices, periods=14):
        """
        Calculate Relative Strength Index (RSI)
        :param close_prices: Series of closing prices
        :param periods: RSI calculation period
        :return: RSI values
        """
        delta = close_prices.diff()

        # Make two series: one for lower closes and one for higher closes
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        # Calculate the EWMA
        roll_up = up.ewm(com=periods-1, adjust=False).mean()
        roll_down = down.ewm(com=periods-1, adjust=False).mean()

        # Calculate the RSI based on EWMA
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    @staticmethod
    def calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD)
        :param close_prices: Series of closing prices
        :return: MACD line
        """
        # Calculate the fast and slow exponential moving averages
        exp1 = close_prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = close_prices.ewm(span=slow_period, adjust=False).mean()

        # MACD line is the difference between the fast and slow EMAs
        macd_line = exp1 - exp2

        return macd_line

    @staticmethod
    def generate_features(df):
        """
        Generate advanced trading features
        :param df: Input price DataFrame
        :return: DataFrame with additional features
        """
        # Technical indicators
        df['rsi'] = FeatureGenerator.calculate_rsi(df['close'])
        df['macd'] = FeatureGenerator.calculate_macd(df['close'])

        # Lagged features
        df['close_lag1'] = df['close'].shift(1)
        df['close_lag2'] = df['close'].shift(2)

        # Percentage changes
        df['pct_change'] = df['close'].pct_change()

        # Additional features
        df['price_momentum'] = df['close'] - \
            df['close'].rolling(window=14).mean()
        df['volatility'] = df['close'].rolling(window=14).std()

        return df.dropna()
