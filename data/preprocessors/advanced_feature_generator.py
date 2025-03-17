# data/preprocessors/advanced_feature_generator.py
import pandas as pd
import numpy as np
import ta
from typing import List, Optional
from ta import momentum, trend, volatility, volume


class AdvancedFeatureGenerator:
    """
    Advanced feature generator for creating technical indicators and features
    for machine learning models in trading strategies.
    """
    
    def __init__(self, lookback_window: int = 20):
        """
        Initialize the feature generator.
        
        Args:
            lookback_window (int): Window size for calculating indicators.
        """
        self.lookback_window = lookback_window
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical indicators and custom features.
        
        Args:
            df (pd.DataFrame): Input dataframe with OHLCV data.
            
        Returns:
            pd.DataFrame: DataFrame with additional features.
        """
        # Create a copy to avoid modifying the original dataframe
        df_features = df.copy()
        
        # Trend Indicators
        df_features['sma'] = trend.sma_indicator(df_features['close'], window=self.lookback_window)
        df_features['ema'] = trend.ema_indicator(df_features['close'], window=self.lookback_window)
        df_features['macd'] = trend.macd_diff(df_features['close'])
        df_features['adx'] = trend.adx(df_features['high'], df_features['low'], df_features['close'])
        
        # Momentum Indicators
        df_features['rsi'] = momentum.rsi(df_features['close'])
        df_features['stoch'] = momentum.stoch(df_features['high'], df_features['low'], df_features['close'])
        df_features['cci'] = trend.cci(df_features['high'], df_features['low'], df_features['close'])
        
        # Volatility Indicators
        df_features['bb_high'] = volatility.bollinger_hband(df_features['close'])
        df_features['bb_low'] = volatility.bollinger_lband(df_features['close'])
        df_features['atr'] = volatility.average_true_range(df_features['high'], df_features['low'], df_features['close'])
        
        # Volume Indicators
        df_features['obv'] = volume.on_balance_volume(df_features['close'], df_features['volume'])
        df_features['vwap'] = self._calculate_vwap(df_features)
        
        # Custom Features
        df_features['price_momentum'] = self._calculate_price_momentum(df_features['close'])
        df_features['volatility'] = self._calculate_volatility(df_features['close'])
        
        return df_features
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    def _calculate_price_momentum(self, prices: pd.Series) -> pd.Series:
        """Calculate price momentum as percentage change."""
        return prices.pct_change(periods=self.lookback_window)
    
    def _calculate_volatility(self, prices: pd.Series) -> pd.Series:
        """Calculate rolling volatility."""
        return prices.rolling(window=self.lookback_window).std()
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using z-score normalization.
        
        Args:
            df (pd.DataFrame): Input dataframe with features.
            
        Returns:
            pd.DataFrame: Normalized dataframe.
        """
        # Create a copy to avoid modifying the original dataframe
        df_normalized = df.copy()
        
        # Exclude non-numeric columns and those we don't want to normalize
        exclude_columns = ['date', 'timestamp', 'open_time', 'close_time']
        numeric_columns = df_normalized.select_dtypes(include=[np.number]).columns
        columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]
        
        # Apply z-score normalization
        for column in columns_to_normalize:
            mean = df_normalized[column].mean()
            std = df_normalized[column].std()
            if std != 0:  # Avoid division by zero
                df_normalized[column] = (df_normalized[column] - mean) / std
        
        return df_normalized

    @staticmethod
    def generate_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a comprehensive set of technical indicators and features.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
                Required columns: ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            pd.DataFrame: DataFrame with additional technical indicators
        """
        try:
            # Create a copy of the dataframe
            df = df.copy()
            
            # Trend Indicators
            df['sma_7'] = ta.trend.sma_indicator(df['close'], window=7)
            df['sma_25'] = ta.trend.sma_indicator(df['close'], window=25)
            df['sma_99'] = ta.trend.sma_indicator(df['close'], window=99)
            
            df['ema_7'] = ta.trend.ema_indicator(df['close'], window=7)
            df['ema_25'] = ta.trend.ema_indicator(df['close'], window=25)
            df['ema_99'] = ta.trend.ema_indicator(df['close'], window=99)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # ADX
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            
            # Momentum Indicators
            df['rsi'] = ta.momentum.rsi(df['close'])
            df['stoch_rsi'] = ta.momentum.stochrsi(df['close'])
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
            
            # Volatility Indicators
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            df['bbands_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bbands_lower'] = ta.volatility.bollinger_lband(df['close'])
            df['bbands_middle'] = ta.volatility.bollinger_mavg(df['close'])
            df['bbands_width'] = (df['bbands_upper'] - df['bbands_lower']) / df['bbands_middle']
            
            # Volume Indicators
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            # Price Action Features
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['body_ratio'] = df['body_size'] / (df['high'] - df['low'])
            
            # Trend Features
            df['close_sma_7_ratio'] = df['close'] / df['sma_7']
            df['close_sma_25_ratio'] = df['close'] / df['sma_25']
            df['close_sma_99_ratio'] = df['close'] / df['sma_99']
            
            # Volatility Features
            df['high_low_range'] = df['high'] - df['low']
            df['daily_return'] = df['close'].pct_change()
            df['volatility_14'] = df['daily_return'].rolling(window=14).std()
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error generating features: {e}")
            return df
            
    @staticmethod
    def add_custom_indicators(df: pd.DataFrame, 
                            custom_windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Add custom indicators with specified lookback windows.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            custom_windows (List[int], optional): List of lookback windows
        
        Returns:
            pd.DataFrame: DataFrame with additional custom indicators
        """
        try:
            if custom_windows is None:
                custom_windows = [5, 10, 20, 50]
                
            for window in custom_windows:
                # Price momentum
                df[f'momentum_{window}'] = df['close'].diff(window)
                
                # Price acceleration
                df[f'acceleration_{window}'] = df[f'momentum_{window}'].diff()
                
                # Volume momentum
                df[f'volume_momentum_{window}'] = df['volume'].diff(window)
                
                # Custom oscillator
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df[f'custom_osc_{window}'] = (
                    typical_price - typical_price.rolling(window=window).mean()
                ) / typical_price.rolling(window=window).std()
            
            return df
            
        except Exception as e:
            print(f"Error adding custom indicators: {e}")
            return df

    @staticmethod
    def select_top_features(features, n_features=15):
        """
        Select top features based on variance and correlation

        :param features: DataFrame with features
        :param n_features: Number of top features to select
        :return: Selected top features
        """
        # Select only numeric columns
        numeric_features = features.select_dtypes(include=[np.number])

        # Calculate feature variances
        variances = numeric_features.var()

        # Calculate correlation matrix
        correlation_matrix = numeric_features.corr().abs()

        # Combined importance score
        importance_score = variances * (1 - correlation_matrix.mean())

        # Select top features
        top_features = importance_score.nlargest(n_features)

        return numeric_features[top_features.index]
