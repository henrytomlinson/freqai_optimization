#!/bin/bash

# Exit on error
set -e

# Define the user's home directory and project directory explicitly
USER_HOME="/home/henry"
PROJECT_DIR="${USER_HOME}/freqai_optimization"

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
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/venv/bin/activate"

# Set environment variables to increase timeout for Binance API
export BINANCE_API_TIMEOUT=120000

# Run the bot with increased timeout
echo "Starting FreqAI bot in dry run mode..."
freqtrade trade --config "${PROJECT_DIR}/config/freqtrade_config.json" --strategy FreqAIOptimizedStrategy --freqaimodel LightGBMRegressorMultiTarget

# Note: This script should be run in a screen session:
# screen -S freqai
# ./run_bot_fixed.sh
# 
# To detach from screen: Ctrl+A, then D
# To reattach to screen: screen -r freqai 