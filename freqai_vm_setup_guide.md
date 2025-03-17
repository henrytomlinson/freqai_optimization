# FreqAI VM Setup Guide

This guide will help you set up your FreqAI bot on your Linux VM, addressing the issues you've encountered.

## 1. Fix TA-Lib Installation

The first step is to install TA-Lib properly with all required dependencies, including CMake:

```bash
# Connect to your VM
ssh henry@65.21.187.63

# Make the installation script executable and run it
chmod +x ~/install_talib_complete.sh
sudo ~/install_talib_complete.sh
```

This script will:
- Install all system dependencies (including CMake)
- Download and install the TA-Lib C library
- Try multiple methods to install the Python TA-Lib wrapper
- Install the remaining Python requirements

## 2. Fix Port Conflict and Timeout Issues

Next, we'll modify the FreqTrade configuration to use a different port and increase API timeouts:

```bash
# Make the script executable and run it
chmod +x ~/modify_config.sh
~/modify_config.sh
```

This script will:
- Change the API server port from 8081 to 8082
- Increase API timeouts to 120 seconds to prevent connection issues with Binance

## 3. Run the Bot

Now you can run the bot with our improved startup script:

```bash
# Make the script executable
chmod +x ~/run_bot.sh

# Start a screen session
screen -S freqai

# Run the bot
~/run_bot.sh
```

To detach from the screen session (without stopping the bot):
- Press `Ctrl+A`, then press `D`

To reattach to the screen session later:
```bash
screen -r freqai
```

## 4. Monitoring the Bot

To check if your bot is running properly:

```bash
# Check for running FreqTrade processes
ps aux | grep freqtrade | grep -v grep

# View the log file
tail -f ~/freqai_optimization/freqai_strategy.log

# Check if the API server is running on the new port
curl -s -u freqai:optimization http://localhost:8082/api/v1/status | python -m json.tool
```

## 5. Accessing the Web UI

To access the FreqTrade web UI from your local machine:

```bash
# On your local machine, create an SSH tunnel
ssh -L 8082:localhost:8082 henry@65.21.187.63
```

Then open your browser and go to:
```
http://localhost:8082
```

Use the credentials from your config file (default: freqai/optimization).

## Troubleshooting

### If the bot still fails to start:

1. Check for processes using port 8082:
```bash
sudo lsof -i :8082
```

2. Kill any processes using that port:
```bash
sudo kill -9 <PID>
```

3. Check for FreqTrade processes that might be running:
```bash
ps aux | grep freqtrade
```

4. Kill any existing FreqTrade processes:
```bash
sudo kill -9 <PID>
```

5. Check the log file for specific errors:
```bash
tail -n 100 ~/freqai_optimization/freqai_strategy.log
```

### If you encounter network timeouts:

The `modify_config.sh` script increases the timeout to 120 seconds, but if you still experience timeouts, you can try:

1. Using a VPN on your VM to improve connectivity to Binance
2. Further increasing the timeout in the config file
3. Adding retry logic to your strategy 