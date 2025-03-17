# Fresh VM Setup Guide for FreqAI Bot

This guide will walk you through setting up your FreqAI bot on a fresh Ubuntu VM. This approach will help you avoid the issues you've been experiencing with port conflicts, TA-Lib installation problems, and other configuration issues.

## Step 1: Connect to Your Fresh VM

Connect to your VM using SSH:

```bash
ssh henry@65.21.187.63
```

## Step 2: Transfer the Setup Script

From your local machine, transfer the setup script to your VM:

```bash
scp fresh_vm_setup.sh henry@65.21.187.63:~/
```

## Step 3: Run the Setup Script

On your VM, make the script executable and run it:

```bash
chmod +x ~/fresh_vm_setup.sh
~/fresh_vm_setup.sh
```

This script will:

1. Update your system and install all necessary dependencies
2. Install TA-Lib from source
3. Create a Python virtual environment
4. Install FreqTrade and the TA-Lib Python wrapper
5. Set up the project structure
6. Create configuration files
7. Create a systemd service for running the bot
8. Verify the installation

The script includes multiple fallback methods for installing TA-Lib, so it should work even if one method fails.

## Step 4: Running the Bot

After the setup is complete, you can run the bot using one of these methods:

### Method 1: Using Screen (Recommended for Testing)

```bash
screen -S freqai
cd ~/freqai_optimization
./run_bot.sh
```

To detach from the screen session (without stopping the bot):
- Press `Ctrl+A`, then press `D`

To reattach to the screen session later:
```bash
screen -r freqai
```

### Method 2: Using Systemd Service (Recommended for Production)

```bash
sudo cp ~/freqai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable freqai.service
sudo systemctl start freqai.service
```

To check the status:
```bash
sudo systemctl status freqai.service
```

To view logs:
```bash
journalctl -u freqai.service -f
```

## Key Improvements in This Setup

1. **Port Configuration**: The bot is configured to use port 8082 instead of 8081 to avoid conflicts.
2. **Increased Timeouts**: API timeouts are set to 120 seconds to prevent connection issues with Binance.
3. **Proper Path Handling**: All paths use absolute references to avoid issues with `sudo` and the tilde (`~`) shortcut.
4. **Multiple TA-Lib Installation Methods**: The script tries multiple methods to install TA-Lib to ensure success.
5. **Systemd Service**: A proper systemd service is created for running the bot as a background service.

## Troubleshooting

If you encounter any issues:

1. Check the logs:
   ```bash
   tail -f ~/freqai_optimization/freqai_strategy.log
   ```

2. Verify that FreqTrade is installed correctly:
   ```bash
   cd ~/freqai_optimization
   source venv/bin/activate
   python -c "import freqtrade; print('FreqTrade version:', freqtrade.__version__)"
   ```

3. Check if any processes are using port 8082:
   ```bash
   sudo lsof -i :8082
   ```

4. If you need to kill a process:
   ```bash
   sudo kill -9 <PID>
   ```

## Accessing the Web UI

To access the FreqTrade web UI from your local machine:

1. Create an SSH tunnel:
   ```bash
   ssh -L 8082:localhost:8082 henry@65.21.187.63
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8082
   ```

3. Log in with the default credentials:
   - Username: freqai
   - Password: optimization 