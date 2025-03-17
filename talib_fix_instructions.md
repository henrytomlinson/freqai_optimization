# Fixing TA-Lib Installation on Your VM

I've created a comprehensive script to properly install TA-Lib on your Ubuntu 24.04 VM. Follow these steps to fix the installation issue:

## Step 1: Connect to your VM

```bash
ssh henry@65.21.187.63
```

## Step 2: Run the TA-Lib installation script

```bash
bash ~/install_talib.sh
```

This script will:
1. Install all necessary build dependencies
2. Download the TA-Lib source code
3. Compile and install the TA-Lib C library
4. Update the shared library cache
5. Install the Python TA-Lib wrapper in your virtual environment

## Step 3: Verify the installation

After the script completes, verify that TA-Lib is installed correctly:

```bash
cd ~/freqai_optimization
source venv/bin/activate
python -c "import talib; print(talib.__version__)"
```

If this command prints a version number (like "0.4.24"), the installation was successful.

## Step 4: Install the remaining requirements

Now that TA-Lib is installed, you can install the remaining requirements:

```bash
cd ~/freqai_optimization
source venv/bin/activate
pip install -r requirements.txt
```

## Step 5: Run your FreqAI bot

After all dependencies are installed, you can run your bot:

```bash
cd ~/freqai_optimization
screen -S freqai
./run_bot.sh
```

To detach from the screen session (without stopping the bot), press:
- `Ctrl+A`, then press `D`

## Troubleshooting

If you still encounter issues with TA-Lib installation, try these additional steps:

### Check if the TA-Lib library files exist

```bash
ls -la /usr/lib/libta_lib*
```

If these files don't exist, there was an issue with the C library installation.

### Check for compilation errors

If the script fails during compilation, you might need to install additional dependencies:

```bash
sudo apt-get install -y libssl-dev libffi-dev
```

### Alternative installation method

If all else fails, you can try installing TA-Lib using the system package manager:

```bash
sudo apt-get install -y ta-lib python3-ta-lib
cd ~/freqai_optimization
source venv/bin/activate
pip install ta-lib
```

### Check for port conflicts

If you encounter issues with port 8081 being in use when starting the bot, find and kill the process:

```bash
sudo lsof -i :8081
sudo kill -9 <PID>
``` 