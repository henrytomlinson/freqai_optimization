#!/bin/bash

# Exit on error
set -e

echo "=== FreqTrade Installation Script ==="
echo "This script will install FreqTrade and all its dependencies"

# Activate virtual environment
cd ~/freqai_optimization
source venv/bin/activate || {
  echo "Creating new virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
}

# Install system dependencies
echo "Step 1: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential git curl python3-dev python3-pip python3-venv cmake

# Install TA-Lib dependencies first
echo "Step 2: Installing TA-Lib dependencies..."
pip install --upgrade pip
pip install wheel setuptools numpy cython

# Check if TA-Lib C library is installed
if [ ! -f "/usr/lib/libta_lib.so" ]; then
  echo "TA-Lib C library not found. Installing..."
  cd ~
  wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
  tar -xzf ta-lib-0.4.0-src.tar.gz
  cd ta-lib/
  ./configure --prefix=/usr
  make
  sudo make install
  sudo ldconfig
  cd ~/freqai_optimization
else
  echo "TA-Lib C library already installed."
fi

# Install FreqTrade
echo "Step 3: Installing FreqTrade..."
pip install -U freqtrade

# Verify FreqTrade installation
echo "Step 4: Verifying FreqTrade installation..."
python -c "import freqtrade; print('FreqTrade version:', freqtrade.__version__)" || {
  echo "FreqTrade installation failed. Trying alternative method..."
  pip install git+https://github.com/freqtrade/freqtrade.git
  python -c "import freqtrade; print('FreqTrade version:', freqtrade.__version__)"
}

# Install TA-Lib Python wrapper
echo "Step 5: Installing TA-Lib Python wrapper..."
pip install ta-lib || {
  echo "Standard installation failed, trying alternative method..."
  pip install --no-binary :all: ta-lib || {
    echo "Alternative method failed, trying from GitHub..."
    pip install git+https://github.com/mrjbq7/ta-lib.git
  }
}

# Install remaining requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
  echo "Step 6: Installing remaining requirements..."
  pip install -r requirements.txt
fi

echo "=== FreqTrade installation complete ==="
echo "You can now run your FreqAI bot with:"
echo "cd ~/freqai_optimization"
echo "screen -S freqai"
echo "./run_bot.sh" 