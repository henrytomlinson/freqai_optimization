#!/bin/bash

# Exit on error
set -e

echo "=== Complete TA-Lib Installation Script ==="
echo "This script will install all dependencies and TA-Lib from source"

# Install system dependencies
echo "Step 1: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential wget curl python3-dev python3-pip python3-venv cmake

# Download and install TA-Lib C library
echo "Step 2: Downloading TA-Lib source..."
cd ~
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz

echo "Step 3: Building and installing TA-Lib C library..."
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Update the shared library cache
echo "Step 4: Updating shared library cache..."
sudo ldconfig

# Install Python wrapper
echo "Step 5: Installing Python TA-Lib wrapper..."
cd ~/freqai_optimization
source venv/bin/activate
pip install --upgrade pip
pip install wheel setuptools numpy cython

# Try different installation methods for TA-Lib Python wrapper
echo "Step 6: Trying different installation methods for TA-Lib Python wrapper..."

# Method 1: Standard installation
echo "Method 1: Standard installation..."
pip install ta-lib || {
  # Method 2: Installation with specific compiler flags
  echo "Method 1 failed. Trying Method 2: Installation with specific compiler flags..."
  pip install --no-binary :all: ta-lib || {
    # Method 3: Installation from GitHub source
    echo "Method 2 failed. Trying Method 3: Installation from GitHub source..."
    pip install git+https://github.com/mrjbq7/ta-lib.git || {
      # Method 4: Manual installation
      echo "Method 3 failed. Trying Method 4: Manual installation..."
      cd ~
      git clone https://github.com/mrjbq7/ta-lib.git
      cd ta-lib
      python setup.py install
    }
  }
}

echo "=== TA-Lib installation complete ==="
echo "Verifying installation..."
python -c "import talib; print('TA-Lib version:', talib.__version__)" || echo "Installation failed. Please check the error messages above."

# Install remaining requirements
echo "Step 7: Installing remaining requirements..."
cd ~/freqai_optimization
pip install -r requirements.txt

echo "=== All installations complete ==="
echo "You can now run your FreqAI bot with:"
echo "cd ~/freqai_optimization"
echo "screen -S freqai"
echo "./scripts/run_freqai_strategy.py --mode dry_run" 