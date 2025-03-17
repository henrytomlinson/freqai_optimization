# FreqAI VM Setup Instructions

Follow these steps to set up and run your FreqAI bot on your Linux VM.

## Step 1: Connect to your VM

```bash
ssh henry@65.21.187.63
```

## Step 2: Run the setup script

The setup script will install all necessary dependencies, extract your project archive, set up a Python virtual environment, and install all required packages.

```bash
bash ~/vm_setup.sh
```

This script will:
- Install system dependencies (you'll need to enter your password for sudo)
- Create the project directory
- Extract your project archive (if it exists)
- Set up a Python virtual environment
- Install all required packages
- Create a script to run the bot

## Step 3: Run the bot in a screen session

After the setup is complete, you can run your bot in a screen session to keep it running even when you disconnect:

```bash
# Create a new screen session
screen -S freqai

# Run the bot
./run_bot.sh

# Detach from the screen session (without stopping the bot)
# Press Ctrl+A, then press D
```

## Step 4: Managing your bot

### Reconnect to your bot's screen session

```bash
screen -r freqai
```

### List all screen sessions

```bash
screen -ls
```

### Stop the bot

To stop the bot, reattach to the screen session and press Ctrl+C.

### Automatically start the bot on system reboot (optional)

If you want the bot to start automatically when the VM reboots, you can set up a crontab entry:

```bash
crontab -e
```

Add this line:

```
@reboot screen -dmS freqai ~/freqai_optimization/run_bot.sh
```

## Troubleshooting

### If TA-Lib installation fails

If you encounter issues with TA-Lib installation, you may need to install it from source:

```bash
cd ~
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ~/freqai_optimization
source venv/bin/activate
pip install ta-lib
```

### If port 8081 is already in use

If you see an error about port 8081 being in use, you can find and kill the process:

```bash
sudo lsof -i :8081
sudo kill -9 <PID>
```

### Checking logs

To view the bot's logs:

```bash
tail -f ~/freqai_optimization/freqai_strategy.log
``` 