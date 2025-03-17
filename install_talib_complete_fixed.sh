#!/bin/bash

# Exit on error
set -e

# Define the user's home directory and project directory explicitly
USER_HOME="/home/henry"
PROJECT_DIR="${USER_HOME}/freqai_optimization"

echo "=== Complete TA-Lib Installation Script ==="
echo "This script will install all dependencies and TA-Lib from source"
echo "Using project directory: ${PROJECT_DIR}"

# Install system dependencies
echo "Step 1: Installing system dependencies..."
apt-get update
apt-get install -y build-essential wget curl python3-dev python3-pip python3-venv cmake

# Download and install TA-Lib C library
echo "Step 2: Downloading TA-Lib source..."
cd "${USER_HOME}"
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz

echo "Step 3: Building and installing TA-Lib C library..."
cd "${USER_HOME}/ta-lib/"
./configure --prefix=/usr
make
make install

# Update the shared library cache
echo "Step 4: Updating shared library cache..."
ldconfig

# Install Python wrapper
echo "Step 5: Installing Python TA-Lib wrapper..."
cd "${PROJECT_DIR}"

# Check if virtual environment exists, if not create it
if [ ! -d "${PROJECT_DIR}/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${PROJECT_DIR}/venv"
fi

# Activate virtual environment (using source with full path)
source "${PROJECT_DIR}/venv/bin/activate"
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
      cd "${USER_HOME}"
      if [ ! -d "${USER_HOME}/ta-lib-python" ]; then
        git clone https://github.com/mrjbq7/ta-lib.git ta-lib-python
      fi
      cd ta-lib-python
      python setup.py install
    }
  }
}

# Install FreqTrade
echo "Step 7: Installing FreqTrade..."
pip install -U freqtrade

# Verify FreqTrade installation
echo "Step 8: Verifying FreqTrade installation..."
python -c "import freqtrade; print('FreqTrade version:', freqtrade.__version__)" || {
  echo "FreqTrade installation failed. Trying alternative method..."
  pip install git+https://github.com/freqtrade/freqtrade.git
  python -c "import freqtrade; print('FreqTrade version:', freqtrade.__version__)"
}

echo "=== TA-Lib installation complete ==="
echo "Verifying installation..."
python -c "import talib; print('TA-Lib version:', talib.__version__)" || echo "Installation failed. Please check the error messages above."

# Install remaining requirements
echo "Step 9: Installing remaining requirements..."
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
  cd "${PROJECT_DIR}"
  pip install -r requirements.txt
fi

echo "=== All installations complete ==="
echo "You can now run your FreqAI bot with:"
echo "cd ${PROJECT_DIR}"
echo "screen -S freqai"
echo "./run_bot.sh" 