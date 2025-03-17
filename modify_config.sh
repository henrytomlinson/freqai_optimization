#!/bin/bash

# Exit on error
set -e

echo "=== FreqAI Configuration Modifier ==="

# Activate virtual environment
cd ~/freqai_optimization
source venv/bin/activate

# Check if config file exists
CONFIG_FILE="config/freqtrade_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

# Create a backup of the original config
cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
echo "Created backup of config file at ${CONFIG_FILE}.backup"

# Modify the config file to use a different port and increase timeouts
echo "Modifying config file to use port 8082 and increase timeouts..."
cat "$CONFIG_FILE" | \
    jq '.api_server.listen_port = 8082 | 
        .exchange.ccxt_config.timeout = 120000 | 
        .exchange.ccxt_async_config.timeout = 120000' > "${CONFIG_FILE}.new"

# Replace the original config with the modified one
mv "${CONFIG_FILE}.new" "$CONFIG_FILE"

echo "Configuration updated successfully!"
echo "The API server will now use port 8082 and API timeouts have been increased to 120 seconds."
echo "To run the bot with the new configuration, use: ./run_bot.sh" 