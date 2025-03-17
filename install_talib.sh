#!/bin/bash

# Exit on error
set -e

echo "=== Installing TA-Lib on Ubuntu 24.04 ==="
echo "This script will install TA-Lib from source and all required dependencies"

# Install build dependencies
echo "Step 1: Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential wget curl python3-dev python3-pip python3-venv

# Download and install TA-Lib
echo "Step 2: Downloading TA-Lib source..."
cd ~
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz

echo "Step 3: Building and installing TA-Lib..."
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
pip install numpy  # Install numpy first as it's a dependency
pip install --no-binary :all: ta-lib

echo "=== TA-Lib installation complete ==="
echo "If you encounter any issues, please check that the library was installed correctly:"
echo "  - Run 'ls -la /usr/lib/libta_lib*' to verify the library files exist"
echo "  - Run 'python -c \"import talib; print(talib.__version__)\"' to verify the Python wrapper works" 