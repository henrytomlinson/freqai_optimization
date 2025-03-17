# FreqTrade Installation Fix Guide (Updated)

This guide will help you fix the "no module named 'freqtrade'" error and properly set up your FreqAI bot on your VM.

## Problem

The error "no module named 'freqtrade'" indicates that the FreqTrade package is not properly installed in your virtual environment. Additionally, when running the installation script with `sudo`, there was a path resolution issue where `~` was resolving to `/root/` instead of `/home/henry/`.

## Solution

We've created updated scripts that use absolute paths instead of the `~` shortcut. Follow these steps in order:

### Step 1: Connect to your VM

```bash
ssh henry@65.21.187.63
```

### Step 2: Install FreqTrade

This script will install FreqTrade and all its dependencies, including TA-Lib:

```bash
# Make the script executable
chmod +x ~/install_talib_complete_fixed.sh

# Run the script
sudo ~/install_talib_complete_fixed.sh
```

This script will:
- Use absolute paths instead of `~` to avoid path resolution issues
- Install system dependencies
- Install TA-Lib C library if not already installed
- Install FreqTrade directly in your virtual environment
- Install the TA-Lib Python wrapper
- Install any remaining requirements

### Step 3: Fix Project Structure

This script will check and fix your project structure:

```bash
# Make the script executable
chmod +x ~/fix_project_structure_fixed.sh

# Run the script
~/fix_project_structure_fixed.sh
```

This script will:
- Create necessary directories if they don't exist
- Create a basic FreqAI strategy file if it doesn't exist
- Create a basic FreqTrade config file if it doesn't exist
- Create a basic run script if it doesn't exist

### Step 4: Run the Bot

Now you can run the bot with our improved startup script:

```bash
# Make the script executable
chmod +x ~/run_bot_fixed.sh

# Start a screen session
screen -S freqai

# Run the bot
~/run_bot_fixed.sh
```

To detach from the screen session (without stopping the bot):
- Press `Ctrl+A`, then press `D`

To reattach to the screen session later:
```bash
screen -r freqai
```

## Troubleshooting

### If you still get "no module named 'freqtrade'" error:

1. Make sure you're activating the virtual environment:
```bash
cd /home/henry/freqai_optimization
source /home/henry/freqai_optimization/venv/bin/activate
```

2. Check if FreqTrade is installed:
```bash
pip list | grep freqtrade
```

3. If not installed, install it manually:
```bash
pip install -U freqtrade
```

4. Try installing from GitHub:
```bash
pip install git+https://github.com/freqtrade/freqtrade.git
```

5. Check your Python path:
```bash
python -c "import sys; print(sys.path)"
```

### If the bot fails to start:

1. Check for processes using port 8082:
```bash
sudo lsof -i :8082
```

2. Kill any processes using that port:
```bash
sudo kill -9 <PID>
```

3. Check the log file for specific errors:
```bash
tail -n 100 /home/henry/freqai_optimization/freqai_strategy.log
```

## Verifying Installation

To verify that FreqTrade is properly installed:

```bash
cd /home/henry/freqai_optimization
source /home/henry/freqai_optimization/venv/bin/activate
python -c "import freqtrade; print('FreqTrade version:', freqtrade.__version__)"
```

If this command runs without errors and displays the FreqTrade version, the installation is successful. 