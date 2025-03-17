#!/bin/bash

# Print header
echo "=== FreqAI Bot Startup Script ==="

# Define project directory as current directory
PROJECT_DIR="$(pwd)"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run setup script first."
    exit 1
fi

# Check if config file exists
if [ ! -f "config/freqtrade_config.json" ]; then
    echo "Configuration file not found!"
    exit 1
fi

# Check if port 8080 is in use
if lsof -i:8080 >/dev/null 2>&1; then
    echo "Port 8080 is already in use. Please free up the port before running the bot."
    echo "You can use: lsof -i:8080 to see which process is using it"
    exit 1
fi

# Start the bot
echo "Starting FreqTrade bot..."
freqtrade trade \
    --config config/freqtrade_config.json \
    --strategy-path strategy \
    --db-url sqlite:///tradesv3.dryrun.sqlite

# Note: This script should be run in a screen session:
# screen -S freqai
# ./run_bot_fixed.sh
# 
# To detach from screen: Ctrl+A, then D
# To reattach to screen: screen -r freqai 