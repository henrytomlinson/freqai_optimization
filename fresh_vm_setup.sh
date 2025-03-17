#!/bin/bash

# Exit on error
set -e

# Define colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== FreqAI Bot Complete Setup Script ===${NC}"
echo "This script will set up your FreqAI bot from scratch on a fresh VM"

# Step 1: Update system and install dependencies
echo -e "\n${YELLOW}Step 1: Updating system and installing dependencies...${NC}"
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential wget curl git python3-dev python3-pip python3-venv cmake screen jq

# Step 2: Create project directory
echo -e "\n${YELLOW}Step 2: Creating project directory...${NC}"
PROJECT_DIR="$HOME/freqai_optimization"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Step 3: Create and activate virtual environment
echo -e "\n${YELLOW}Step 3: Creating virtual environment...${NC}"
python3 -m venv venv
source "$PROJECT_DIR/venv/bin/activate"
pip install --upgrade pip wheel setuptools

# Step 4: Install TA-Lib
echo -e "\n${YELLOW}Step 4: Installing TA-Lib...${NC}"
cd "$HOME"
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd "$HOME/ta-lib/"
./configure --prefix=/usr
make
sudo make install
sudo ldconfig

# Step 5: Install FreqTrade and TA-Lib Python wrapper
echo -e "\n${YELLOW}Step 5: Installing FreqTrade and TA-Lib Python wrapper...${NC}"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
pip install numpy cython

# Try different installation methods for TA-Lib Python wrapper
echo "Installing TA-Lib Python wrapper..."
pip install ta-lib || {
  echo "Standard installation failed. Trying with specific compiler flags..."
  pip install --no-binary :all: ta-lib || {
    echo "Installation with flags failed. Trying from GitHub source..."
    pip install git+https://github.com/mrjbq7/ta-lib.git || {
      echo "GitHub installation failed. Trying manual installation..."
      cd "$HOME"
      if [ ! -d "$HOME/ta-lib-python" ]; then
        git clone https://github.com/mrjbq7/ta-lib.git ta-lib-python
      fi
      cd ta-lib-python
      python setup.py install
    }
  }
}

# Install FreqTrade
echo "Installing FreqTrade..."
pip install -U freqtrade

# Step 6: Create project structure
echo -e "\n${YELLOW}Step 6: Creating project structure...${NC}"
cd "$PROJECT_DIR"

# Create necessary directories
mkdir -p "$PROJECT_DIR/user_data/data/binance"
mkdir -p "$PROJECT_DIR/user_data/strategies"
mkdir -p "$PROJECT_DIR/user_data/models"
mkdir -p "$PROJECT_DIR/config"
mkdir -p "$PROJECT_DIR/scripts"

# Step 7: Create FreqTrade configuration
echo -e "\n${YELLOW}Step 7: Creating FreqTrade configuration...${NC}"
cat > "$PROJECT_DIR/config/freqtrade_config.json" << 'EOF'
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
            "DI_threshold": 0.0,
            "weight_factor": 0.9,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": true,
            "stratify_training_data": 0,
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
        "jwt_secret_key": "somethingrandom",
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
EOF

# Step 8: Create FreqAI strategy
echo -e "\n${YELLOW}Step 8: Creating FreqAI strategy...${NC}"
cat > "$PROJECT_DIR/user_data/strategies/freqai_strategy.py" << 'EOF'
import logging
import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.strategy.interface import IStrategy
from functools import reduce
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

logger = logging.getLogger(__name__)

class LightGBMRegressorMultiTarget(BaseRegressionModel):
    """
    LightGBM regressor model for multiple targets
    """
    def fit(self, data_dictionary, dk):
        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        
        model = FreqaiMultiOutputRegressor(
            estimator=lgb.LGBMRegressor(
                n_estimators=self.model_training_parameters.get("n_estimators", 100),
                learning_rate=self.model_training_parameters.get("learning_rate", 0.05),
                max_depth=self.model_training_parameters.get("max_depth", 10),
                num_leaves=self.model_training_parameters.get("num_leaves", 32),
                subsample=self.model_training_parameters.get("subsample", 0.8),
                colsample_bytree=self.model_training_parameters.get("colsample_bytree", 0.8),
                verbosity=self.model_training_parameters.get("verbosity", -1),
            )
        )
        
        model.fit(X, y)
        return model

class FreqAIOptimizedStrategy(IStrategy):
    """
    FreqAI strategy using LightGBM for prediction
    """
    minimal_roi = {
        "0": 0.01
    }
    
    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True
    
    # Buy hyperspace params:
    buy_rsi_threshold = IntParameter(low=10, high=40, default=30, space="buy", optimize=True)
    
    # Sell hyperspace params:
    sell_rsi_threshold = IntParameter(low=60, high=90, default=70, space="sell", optimize=True)
    
    process_only_new_candles = True
    startup_candle_count = 0
    
    def feature_engineering_expand_all(self, dataframe, period, metadata, **kwargs):
        """
        Create additional features and add them to the dataframe
        """
        dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["%-mfi"] = ta.MFI(dataframe, timeperiod=14)
        dataframe["%-adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["%-sma-9"] = ta.SMA(dataframe, timeperiod=9)
        dataframe["%-sma-21"] = ta.SMA(dataframe, timeperiod=21)
        dataframe["%-ema-9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["%-ema-21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["%-ema-diff"] = dataframe["%-ema-9"] - dataframe["%-ema-21"]
        dataframe["%-close-change"] = dataframe["close"].pct_change() * 100
        dataframe["%-volume-change"] = dataframe["volume"].pct_change() * 100
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macdsignal"] = macd["macdsignal"]
        dataframe["%-macdhist"] = macd["macdhist"]
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe["%-bb-upper"] = bollinger["upperband"]
        dataframe["%-bb-mid"] = bollinger["middleband"]
        dataframe["%-bb-lower"] = bollinger["lowerband"]
        dataframe["%-bb-width"] = (dataframe["%-bb-upper"] - dataframe["%-bb-lower"]) / dataframe["%-bb-mid"]
        
        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe["%-stoch-k"] = stoch["slowk"]
        dataframe["%-stoch-d"] = stoch["slowd"]
        
        return dataframe
    
    def populate_any_indicators(self, dataframe, metadata, **kwargs):
        """
        Create basic indicators and return them to FreqAI for training/prediction
        """
        dataframe = self.freqai.start(dataframe, metadata, **kwargs)
        return dataframe
    
    def populate_indicators(self, dataframe, metadata):
        """
        Create basic indicators
        """
        dataframe = self.populate_any_indicators(dataframe, metadata)
        return dataframe
    
    def populate_entry_trend(self, dataframe, metadata):
        """
        Define buy conditions
        """
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1) &
                (dataframe["&-prediction"] > 0.01) &
                (dataframe["%-rsi"] < self.buy_rsi_threshold.value)
            ),
            "enter_long"
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe, metadata):
        """
        Define sell conditions
        """
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1) &
                (dataframe["&-prediction"] < 0.005) &
                (dataframe["%-rsi"] > self.sell_rsi_threshold.value)
            ),
            "exit_long"
        ] = 1
        
        return dataframe
EOF

# Step 9: Create run script
echo -e "\n${YELLOW}Step 9: Creating run script...${NC}"
cat > "$PROJECT_DIR/run_bot.sh" << 'EOF'
#!/bin/bash

# Exit on error
set -e

# Define the user's home directory and project directory explicitly
USER_HOME="$HOME"
PROJECT_DIR="$USER_HOME/freqai_optimization"

echo "=== FreqAI Bot Startup Script ==="

# Check if port 8082 is in use and kill the process if needed
PORT_CHECK=$(lsof -i:8082 -t || echo "")
if [ ! -z "$PORT_CHECK" ]; then
    echo "Port 8082 is already in use by process $PORT_CHECK. Stopping it..."
    sudo kill -9 $PORT_CHECK
    sleep 2
    echo "Process stopped."
fi

# Activate virtual environment
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"

# Set environment variables to increase timeout for Binance API
export BINANCE_API_TIMEOUT=120000

# Run the bot with increased timeout
echo "Starting FreqAI bot in dry run mode..."
freqtrade trade --config "$PROJECT_DIR/config/freqtrade_config.json" --strategy FreqAIOptimizedStrategy --freqaimodel LightGBMRegressorMultiTarget

# Note: This script should be run in a screen session:
# screen -S freqai
# ./run_bot.sh
# 
# To detach from screen: Ctrl+A, then D
# To reattach to screen: screen -r freqai
EOF

# Step 10: Create a simple script to run the bot
echo -e "\n${YELLOW}Step 10: Creating a simple script to run FreqAI strategy...${NC}"
cat > "$PROJECT_DIR/scripts/run_freqai_strategy.py" << 'EOF'
#!/usr/bin/env python3
import os
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'freqai_strategy.log'))
    ]
)

logger = logging.getLogger('freqai_strategy')

def main():
    parser = argparse.ArgumentParser(description='Run FreqAI strategy')
    parser.add_argument('--mode', choices=['dry_run', 'live'], default='dry_run', help='Trading mode')
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'freqtrade_config.json')
    
    logger.info(f"Running FreqAI strategy with config: {config_path}")
    logger.info(f"Strategy: FreqAIOptimizedStrategy")
    logger.info(f"Mode: {args.mode}")
    
    cmd = f"{sys.executable} -m freqtrade trade --config {config_path} --strategy FreqAIOptimizedStrategy --freqaimodel LightGBMRegressorMultiTarget"
    
    if args.mode == 'dry_run':
        cmd += " --dry-run"
    
    logger.info(f"Running FreqTrade with command: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()
EOF

# Make scripts executable
chmod +x "$PROJECT_DIR/run_bot.sh"
chmod +x "$PROJECT_DIR/scripts/run_freqai_strategy.py"

# Step 11: Create a systemd service file
echo -e "\n${YELLOW}Step 11: Creating systemd service file...${NC}"
cat > "$HOME/freqai.service" << EOF
[Unit]
Description=FreqAI Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/freqtrade trade --config $PROJECT_DIR/config/freqtrade_config.json --strategy FreqAIOptimizedStrategy --freqaimodel LightGBMRegressorMultiTarget
Restart=on-failure
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=freqai

[Install]
WantedBy=multi-user.target
EOF

echo -e "\n${YELLOW}To install the systemd service, run:${NC}"
echo "sudo cp $HOME/freqai.service /etc/systemd/system/"
echo "sudo systemctl daemon-reload"
echo "sudo systemctl enable freqai.service"
echo "sudo systemctl start freqai.service"

# Step 12: Verify installation
echo -e "\n${YELLOW}Step 12: Verifying installation...${NC}"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"

echo -e "\n${YELLOW}Checking TA-Lib installation:${NC}"
python -c "import talib; print('TA-Lib version:', talib.__version__)" || echo -e "${RED}TA-Lib installation failed${NC}"

echo -e "\n${YELLOW}Checking FreqTrade installation:${NC}"
python -c "import freqtrade; print('FreqTrade version:', freqtrade.__version__)" || echo -e "${RED}FreqTrade installation failed${NC}"

echo -e "\n${YELLOW}Checking LightGBM installation:${NC}"
python -c "import lightgbm; print('LightGBM version:', lightgbm.__version__)" || echo -e "${RED}LightGBM installation failed${NC}"

# Final instructions
echo -e "\n${GREEN}=== Setup Complete ===${NC}"
echo -e "\n${YELLOW}To run your FreqAI bot:${NC}"
echo "1. Start a screen session:"
echo "   screen -S freqai"
echo "2. Run the bot:"
echo "   cd $PROJECT_DIR"
echo "   ./run_bot.sh"
echo "3. Detach from screen (without stopping the bot):"
echo "   Press Ctrl+A, then press D"
echo "4. To reattach to the screen session later:"
echo "   screen -r freqai"
echo -e "\n${YELLOW}Alternatively, you can use the systemd service:${NC}"
echo "sudo systemctl start freqai.service"
echo -e "\n${YELLOW}To check the bot status:${NC}"
echo "sudo systemctl status freqai.service"
echo -e "\n${YELLOW}To view logs:${NC}"
echo "journalctl -u freqai.service -f" 