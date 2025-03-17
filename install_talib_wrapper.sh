#!/bin/bash

# Exit on error
set -e

echo "=== Installing TA-Lib Python wrapper ==="
echo "This script will install the TA-Lib Python wrapper in your virtual environment"

# Activate virtual environment
cd ~/freqai_optimization
source venv/bin/activate

# Install dependencies
echo "Step 1: Installing dependencies..."
pip install --upgrade pip
pip install wheel setuptools numpy cython

# Install TA-Lib Python wrapper
echo "Step 2: Installing TA-Lib Python wrapper..."
# Try the standard installation first
pip install ta-lib || {
  echo "Standard installation failed, trying alternative method..."
  # If that fails, try installing from source with specific compiler flags
  pip install --no-binary :all: ta-lib
}

echo "=== TA-Lib Python wrapper installation complete ==="
echo "Verifying installation..."
python -c "import talib; print('TA-Lib version:', talib.__version__)" || echo "Installation failed. Please check the error messages above." 