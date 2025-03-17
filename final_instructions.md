# Final Instructions for Setting Up FreqAI on Your VM

I've prepared everything you need to set up and run your FreqAI bot on your Linux VM. Here's a summary of what's been done and what you need to do next:

## What's Been Done

1. Created a setup script (`vm_setup.sh`) that will install all necessary dependencies and set up your environment
2. Created a run script (`run_bot.sh`) that will start your FreqAI bot
3. Prepared detailed instructions (`vm_instructions.md`) for setting up and managing your bot
4. Transferred all these files to your VM

## What You Need to Do

1. SSH into your VM:
   ```bash
   ssh henry@65.21.187.63
   ```

2. Run the setup script:
   ```bash
   bash ~/vm_setup.sh
   ```
   - This will install system dependencies (you'll need to enter your password for sudo)
   - Set up a Python virtual environment
   - Install all required packages

3. After the setup is complete, run your bot in a screen session:
   ```bash
   cd ~/freqai_optimization
   screen -S freqai
   ./run_bot.sh
   ```

4. Detach from the screen session (without stopping the bot) by pressing:
   - `Ctrl+A`, then press `D`

5. You can reconnect to your bot's screen session later with:
   ```bash
   screen -r freqai
   ```

## Troubleshooting

If you encounter any issues, refer to the detailed instructions in `vm_instructions.md` on your VM:
```bash
cat ~/vm_instructions.md
```

The most common issues are:

1. **TA-Lib installation failure**: Follow the instructions in the troubleshooting section to install it from source
2. **Port 8081 already in use**: Follow the instructions to find and kill the process
3. **Missing dependencies**: Make sure you run the setup script completely

## Monitoring Your Bot

To check if your bot is running properly:
```bash
tail -f ~/freqai_optimization/freqai_strategy.log
```

## Keeping Your VM Running

Your VM should stay running as long as it's powered on. The screen session will keep your bot running even when you disconnect from SSH.

If you want the bot to start automatically when the VM reboots, follow the instructions in the "Automatically start the bot on system reboot" section of `vm_instructions.md`.

Good luck with your trading bot! 