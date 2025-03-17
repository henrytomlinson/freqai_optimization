# data/preprocessors/advanced_feature_generator.py
import numpy as np
import pandas as pd
import talib


class AdvancedFeatureGenerator:
    @staticmethod
    def generate_comprehensive_features(df):
        """
        Generate an extensive set of advanced trading features

        :param df: Input price DataFrame
        :return: DataFrame with advanced features
        """
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Create copy to avoid modifying original DataFrame
        features = df.copy()

        # Convert timestamp to datetime if it's a string
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])

        # 1. Momentum Indicators
        features['rsi'] = talib.RSI(features['close'], timeperiod=14)
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(
            features['close'])

        # 2. Volatility Indicators
        features['atr'] = talib.ATR(
            features['high'], features['low'], features['close'], timeperiod=14)
        features['bbands_upper'], features['bbands_middle'], features['bbands_lower'] = talib.BBANDS(
            features['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )

        # 3. Trend Indicators
        features['sma_50'] = talib.SMA(features['close'], timeperiod=50)
        features['sma_200'] = talib.SMA(features['close'], timeperiod=200)
        features['ema_50'] = talib.EMA(features['close'], timeperiod=50)

        # 4. Volume Indicators
        features['obv'] = talib.OBV(features['close'], features['volume'])
        features['volume_ma_20'] = talib.SMA(features['volume'], timeperiod=20)

        # 5. Oscillators
        features['stoch_k'], features['stoch_d'] = talib.STOCH(
            features['high'], features['low'], features['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )

        # 6. Advanced Derivative Features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(1 + features['returns'])

        # 7. Directional Movement Indicators
        features['adx'] = talib.ADX(
            features['high'], features['low'], features['close'], timeperiod=14)
        features['plus_di'] = talib.PLUS_DI(
            features['high'], features['low'], features['close'], timeperiod=14)
        features['minus_di'] = talib.MINUS_DI(
            features['high'], features['low'], features['close'], timeperiod=14)

        # Remove NaN values
        return features.dropna()

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
