#!/bin/bash

# Step 1: Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-pip python3-venv libta-lib-dev

# Step 2: Extract the archive if not already done
echo "Setting up project directory..."
mkdir -p ~/freqai_optimization
cd ~/freqai_optimization

# Step 3: Extract the archive if it exists and hasn't been extracted
if [ -f ~/freqai_project.tar.gz ] && [ ! -f ./config/freqtrade_config.json ]; then
    echo "Extracting project archive..."
    tar -xzvf ~/freqai_project.tar.gz -C .
fi

# Step 4: Create and activate virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 5: Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install wheel
pip install numpy pandas matplotlib

# Try to install TA-Lib first (it's often problematic)
echo "Installing TA-Lib..."
pip install ta-lib

# Install the rest of the requirements
echo "Installing remaining requirements..."
pip install -r requirements.txt

# Step 6: Create a script to run the bot
echo "Creating bot runner script..."
cat > run_bot.sh << 'EOF'
#!/bin/bash
cd ~/freqai_optimization
source venv/bin/activate
./scripts/run_freqai_strategy.py --mode dry_run
EOF

chmod +x run_bot.sh

echo "====================================="
echo "Setup complete!"
echo "To run the bot in a screen session:"
echo "1. Run: screen -S freqai"
echo "2. Execute: ./run_bot.sh"
echo "3. Detach with: Ctrl+A followed by D"
echo "4. Reattach later with: screen -r freqai"
echo "=====================================" 