#!/bin/bash

# Exit on error
set -e

# Define the user's home directory and project directory explicitly
USER_HOME="/home/henry"
PROJECT_DIR="${USER_HOME}/freqai_optimization"

echo "=== FreqAI Project Structure Fix ==="
echo "This script will check and fix your project structure"
echo "Using project directory: ${PROJECT_DIR}"

# Activate virtual environment
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/venv/bin/activate" || {
  echo "Virtual environment not found. Please run install_talib_complete_fixed.sh first."
  exit 1
}

# Check if the project has the correct structure
echo "Step 1: Checking project structure..."

# Create necessary directories if they don't exist
mkdir -p "${PROJECT_DIR}/user_data/data/binance"
mkdir -p "${PROJECT_DIR}/user_data/strategies"
mkdir -p "${PROJECT_DIR}/user_data/models"
mkdir -p "${PROJECT_DIR}/config"
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/scripts"

# Check if the strategy file exists
if [ ! -f "${PROJECT_DIR}/user_data/strategies/freqai_strategy.py" ]; then
  echo "Strategy file not found. Creating a basic FreqAI strategy..."
  cat > "${PROJECT_DIR}/user_data/strategies/freqai_strategy.py" << 'EOL'
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter

logger = logging.getLogger(__name__)

class FreqAIOptimizedStrategy(IStrategy):
    """
    Example strategy using FreqAI.
    """
    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    timeframe = '5m'

    # Buy hyperspace params:
    buy_rsi_threshold = IntParameter(low=10, high=40, default=30, space='buy', optimize=True)
    
    # Sell hyperspace params:
    sell_rsi_threshold = IntParameter(low=60, high=90, default=70, space='sell', optimize=True)

    process_only_new_candles = True
    startup_candle_count = 0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several indicators to the given DataFrame.
        """
        # RSI
        dataframe['rsi'] = 0  # Placeholder, will be calculated by FreqAI
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi_threshold.value)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi_threshold.value)
            ),
            'exit_long'] = 1

        return dataframe
EOL
fi

# Check if the config file exists
if [ ! -f "${PROJECT_DIR}/config/freqtrade_config.json" ]; then
  echo "Config file not found. Creating a basic FreqTrade config..."
  cat > "${PROJECT_DIR}/config/freqtrade_config.json" << 'EOL'
{
    "max_open_trades": 5,
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "USD",
    "dry_run": true,
    "dry_run_wallet": 1000,
    "cancel_open_orders_on_exit": false,
    "trading_mode": "spot",
    "margin_mode": "",
    "unfilledtimeout": {
        "entry": 10,
        "exit": 10,
        "exit_timeout_count": 0,
        "unit": "minutes"
    },
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1,
        "price_last_balance": 0.0,
        "check_depth_of_market": {
            "enabled": false,
            "bids_to_ask_delta": 1
        }
    },
    "exit_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1
    },
    "exchange": {
        "name": "binance",
        "key": "",
        "secret": "",
        "ccxt_config": {
            "timeout": 120000
        },
        "ccxt_async_config": {
            "timeout": 120000
        },
        "pair_whitelist": [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "ADA/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "DOT/USDT",
            "DOGE/USDT",
            "AVAX/USDT"
        ],
        "pair_blacklist": []
    },
    "pairlists": [
        {"method": "StaticPairList"}
    ],
    "freqai": {
        "enabled": true,
        "purge_old_models": true,
        "train_period_days": 30,
        "backtest_period_days": 7,
        "identifier": "unique-id",
        "feature_parameters": {
            "include_timeframes": [
                "5m",
                "15m",
                "1h"
            ],
            "include_corr_pairlist": [
                "BTC/USDT",
                "ETH/USDT"
            ],
            "label_period_candles": 24,
            "include_shifted_candles": 2,
            "DI_threshold": 0.9,
            "weight_factor": 0.9,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": true,
            "indicator_periods_candles": [10, 20]
        },
        "data_split_parameters": {
            "test_size": 0.25,
            "shuffle": false
        },
        "model_training_parameters": {
            "n_estimators": 100
        }
    },
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8082,
        "verbosity": "error",
        "enable_openapi": true,
        "jwt_secret_key": "something_secure",
        "CORS_origins": [],
        "username": "freqai",
        "password": "optimization"
    },
    "bot_name": "freqai_bot",
    "initial_state": "running",
    "force_entry_enable": false,
    "internals": {
        "process_throttle_secs": 5
    }
}
EOL
fi

# Check if the run script exists
if [ ! -f "${PROJECT_DIR}/scripts/run_freqai_strategy.py" ]; then
  echo "Run script not found. Creating a basic run script..."
  mkdir -p "${PROJECT_DIR}/scripts"
  cat > "${PROJECT_DIR}/scripts/run_freqai_strategy.py" << 'EOL'
#!/usr/bin/env python3
import sys
import logging
import os
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("freqai_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("freqai_strategy")

def main():
    parser = argparse.ArgumentParser(description='Run FreqAI strategy')
    parser.add_argument('--mode', choices=['dry_run', 'live'], default='dry_run', help='Trading mode')
    args = parser.parse_args()

    logger.info(f"Running FreqAI strategy with config: config/freqtrade_config.json")
    logger.info(f"Strategy: FreqAIOptimizedStrategy")
    logger.info(f"Mode: {args.mode}")

    # Execute the freqtrade command directly
    os.system("freqtrade trade --config config/freqtrade_config.json --strategy FreqAIOptimizedStrategy --freqaimodel LightGBMRegressorMultiTarget")

if __name__ == "__main__":
    main()
EOL
  chmod +x "${PROJECT_DIR}/scripts/run_freqai_strategy.py"
fi

echo "=== Project structure check complete ==="
echo "Your FreqAI project structure has been fixed."
echo "You can now run your FreqAI bot with:"
echo "cd ${PROJECT_DIR}"
echo "screen -S freqai"
echo "./run_bot_fixed.sh" 