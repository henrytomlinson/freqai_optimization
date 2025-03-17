import logging
import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.strategy.interface import IStrategy
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

from data.preprocessors.advanced_feature_generator import AdvancedFeatureGenerator
from risk_management.portfolio_manager import PortfolioRiskManager

logger = logging.getLogger(__name__)


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
        
        # Check if FreqAI predictions exist
        if 'prediction' in dataframe.columns:
            dataframe.loc[
                (
                    (dataframe['prediction'] > 0) &  # Positive prediction
                    (dataframe['rsi'] < self.buy_rsi_threshold.value) &  # RSI below threshold
                    (dataframe['volume'] > 0)  # Ensure volume exists
                ),
                'buy'] = 1
        
        return dataframe
    
    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Generate sell signals based on FreqAI predictions
        """
        if not self.dp:
            return dataframe
        
        # Check if FreqAI predictions exist
        if 'prediction' in dataframe.columns:
            dataframe.loc[
                (
                    (dataframe['prediction'] < 0) &  # Negative prediction
                    (dataframe['rsi'] > self.sell_rsi_threshold.value) &  # RSI above threshold
                    (dataframe['volume'] > 0)  # Ensure volume exists
                ),
                'sell'] = 1
        
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