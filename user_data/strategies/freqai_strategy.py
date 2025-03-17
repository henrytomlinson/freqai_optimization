import logging
import numpy as np
import pandas as pd
import talib
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.strategy.interface import IStrategy
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# Implement AdvancedFeatureGenerator directly in the strategy file
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


# Implement PortfolioRiskManager directly in the strategy file
class PortfolioRiskManager:
    def __init__(self, initial_capital=10000, max_risk_per_trade=0.02):
        """
        Initialize portfolio risk management

        :param initial_capital: Starting portfolio value
        :param max_risk_per_trade: Maximum risk allowed per trade (2% default)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = 0.1  # 10% maximum portfolio drawdown

        # Trade tracking
        self.open_trades = []
        self.trade_history = []

    def calculate_position_size(self, model_prediction=None, market_volatility=None):
        """
        Dynamically calculate optimal position size

        :param model_prediction: Confidence of trade prediction
        :param market_volatility: Current market volatility
        :return: Position size in dollars
        """
        # Base position sizing
        base_position = self.current_capital * self.max_risk_per_trade

        # Adjust based on model confidence if provided
        if model_prediction is not None:
            confidence_multiplier = abs(model_prediction)
            base_position *= confidence_multiplier

        # Adjust based on market volatility if provided
        if market_volatility is not None:
            volatility_adjustment = 1 / (1 + market_volatility)
            base_position *= volatility_adjustment

        # Ensure position size is within safe limits
        max_position = self.current_capital * 0.2  # No more than 20% in single trade
        return min(base_position, max_position)


class FreqAIOptimizedStrategy(IStrategy):
    """
    FreqAI strategy that uses optimized machine learning models for trading decisions.
    This strategy integrates with the FreqAI framework and uses the custom feature engineering
    and risk management components from the codebase.
    """
    
    # Strategy parameters
    minimal_roi = {
        "0": 0.01  # 1% profit target
    }
    
    stoploss = -0.02  # 2% stop loss
    
    # Trailing stop parameters
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5% trailing profit
    trailing_stop_positive_offset = 0.01  # 1% offset
    trailing_only_offset_is_reached = True
    
    # Timeframe for the strategy
    timeframe = '5m'
    
    # Hyperparameters for the strategy
    buy_rsi_threshold = IntParameter(25, 40, default=30, space="buy")
    sell_rsi_threshold = IntParameter(60, 80, default=70, space="sell")
    
    # FreqAI parameters
    freqai_config = {
        "enabled": True,
        "feature_parameters": {
            "include_timeframes": ["5m", "15m", "1h"],
            "include_corr_pairlist": ["BTC/USDT", "ETH/USDT"],
            "label_period_candles": 24,  # Predict 24 candles ahead
            "include_shifted_candles": 3,  # Include 3 previous candles
            "DI_threshold": 0.0,  # Data importance threshold
            "weight_factor": 0.9,  # Weighting factor for target
            "principal_component_analysis": False,
            "use_SVM_to_remove_outliers": False,
            "stratify_training_data": 0,
            "indicator_max_period_candles": 100,  # Max lookback period
        },
        "data_split_parameters": {
            "test_size": 0.15,  # 15% test data
            "random_state": 42,
            "shuffle": False,
        },
        "model_training_parameters": {
            "n_estimators": 400,
            "learning_rate": 0.02,
            "max_depth": 6,
            "verbosity": 0,
            "random_state": 42
        },
        "identifier": "optimized_model",
        "live_retrain_hours": 24,  # Retrain every 24 hours
        "expiration_hours": 168,  # Model expires after 7 days
        "fit_live_predictions_candles": 300,  # Use 300 candles for live predictions
        "save_backtest_models": True,
        "purge_old_models": True,
        "train_period_days": 30,  # Train on 30 days of data
        "backtest_period_days": 7,  # Backtest on 7 days of data
        "model_save_type": "joblib",
    }
    
    def __init__(self, config: dict) -> None:
        """
        Initialize the strategy with configuration
        """
        super().__init__(config)
        
        # Initialize risk manager
        self.risk_manager = PortfolioRiskManager(
            initial_capital=config.get('dry_run_wallet', 10000),
            max_risk_per_trade=0.02  # 2% risk per trade
        )
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Generate indicators for the strategy using AdvancedFeatureGenerator
        """
        # Use the AdvancedFeatureGenerator to create features
        try:
            # Rename columns to match expected format
            df = dataframe.copy()
            
            # Generate comprehensive features
            df = AdvancedFeatureGenerator.generate_comprehensive_features(df)
            
            # Merge generated features back to dataframe
            for col in df.columns:
                if col not in dataframe.columns:
                    dataframe[col] = df[col]
            
            return dataframe
        except Exception as e:
            logger.error(f"Error in populate_indicators: {e}")
            return dataframe
    
    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Generate buy signals based on FreqAI predictions
        """
        if not self.dp:
            return dataframe
        
        # Initialize the enter_long column with 0
        dataframe['enter_long'] = 0
        
        # Check if FreqAI predictions exist
        if 'prediction' in dataframe.columns:
            dataframe.loc[
                (
                    (dataframe['prediction'] > 0) &  # Positive prediction
                    (dataframe['rsi'] < self.buy_rsi_threshold.value) &  # RSI below threshold
                    (dataframe['volume'] > 0)  # Ensure volume exists
                ),
                'enter_long'] = 1
        
        return dataframe
    
    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Generate sell signals based on FreqAI predictions
        """
        if not self.dp:
            return dataframe
        
        # Initialize the exit_long column with 0
        dataframe['exit_long'] = 0
        
        # Check if FreqAI predictions exist
        if 'prediction' in dataframe.columns:
            dataframe.loc[
                (
                    (dataframe['prediction'] < 0) &  # Negative prediction
                    (dataframe['rsi'] > self.sell_rsi_threshold.value) &  # RSI above threshold
                    (dataframe['volume'] > 0)  # Ensure volume exists
                ),
                'exit_long'] = 1
        
        return dataframe
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, **kwargs) -> bool:
        """
        Apply risk management to trade entries
        """
        # Get dataframe for the pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Get prediction confidence
        prediction_confidence = abs(last_candle.get('prediction', 0))
        
        # Get market volatility (using ATR as a proxy)
        market_volatility = last_candle.get('atr', 0) / last_candle['close']
        
        # Calculate position size based on risk management
        position_size = self.risk_manager.calculate_position_size(
            model_prediction=prediction_confidence,
            market_volatility=market_volatility
        )
        
        # Adjust trade amount if needed
        if position_size < amount * rate:
            self.log(f"Risk management reduced position size for {pair}")
            return False
        
        return True
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           **kwargs) -> float:
        """
        Apply custom stake amount based on risk management
        """
        # Get dataframe for the pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Get prediction confidence
        prediction_confidence = abs(last_candle.get('prediction', 0))
        
        # Get market volatility (using ATR as a proxy)
        market_volatility = last_candle.get('atr', 0) / last_candle['close']
        
        # Calculate position size based on risk management
        position_size = self.risk_manager.calculate_position_size(
            model_prediction=prediction_confidence,
            market_volatility=market_volatility
        )
        
        # Ensure position size is within limits
        position_size = max(min_stake, min(position_size, max_stake))
        
        return position_size 