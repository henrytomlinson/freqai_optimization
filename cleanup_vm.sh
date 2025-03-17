#!/bin/bash

# Define colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== FreqAI Cleanup Script ===${NC}"
echo "This script will remove all FreqAI-related files and installations"

# Check if script is run with sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run this script with sudo:${NC}"
    echo "sudo $0"
    exit 1
fi

# Get the actual user who ran sudo
REAL_USER=$(logname || who am i | awk '{print $1}')
REAL_HOME=$(eval echo ~${REAL_USER})

# Step 1: Stop any running FreqAI processes
echo -e "\n${YELLOW}Step 1: Stopping FreqAI processes...${NC}"
# Kill any screen sessions running FreqAI
su - ${REAL_USER} -c "screen -ls | grep freqai | cut -d. -f1 | awk '{print \$1}' | xargs -I % screen -X -S % quit" || true
# Kill any processes using ports 8081 or 8082
lsof -t -i:8081 | xargs -r kill -9 || true
lsof -t -i:8082 | xargs -r kill -9 || true
# Kill any running freqtrade processes
pkill -f freqtrade || true

# Step 2: Remove systemd service if it exists
echo -e "\n${YELLOW}Step 2: Removing systemd service...${NC}"
if [ -f "/etc/systemd/system/freqai.service" ]; then
    echo "Stopping and removing FreqAI service..."
    systemctl stop freqai.service || true
    systemctl disable freqai.service || true
    rm -f /etc/systemd/system/freqai.service
    systemctl daemon-reload
fi

# Step 3: Remove FreqAI directories and files
echo -e "\n${YELLOW}Step 3: Removing FreqAI directories and files...${NC}"
# Remove project directory
su - ${REAL_USER} -c "rm -rf ${REAL_HOME}/freqai_optimization"
# Remove any TA-Lib source files
su - ${REAL_USER} -c "rm -rf ${REAL_HOME}/ta-lib"
su - ${REAL_USER} -c "rm -rf ${REAL_HOME}/ta-lib-python"
su - ${REAL_USER} -c "rm -f ${REAL_HOME}/ta-lib-0.4.0-src.tar.gz"

# Step 4: Remove Python packages and virtual environments
echo -e "\n${YELLOW}Step 4: Removing Python packages and virtual environments...${NC}"
# Remove virtual environments and pip cache
su - ${REAL_USER} -c "rm -rf ${REAL_HOME}/.local/share/virtualenvs/freqai*"
su - ${REAL_USER} -c "rm -rf ${REAL_HOME}/*/venv"
su - ${REAL_USER} -c "rm -rf ${REAL_HOME}/.cache/pip"

# Step 5: Remove TA-Lib system installation
echo -e "\n${YELLOW}Step 5: Removing TA-Lib system installation...${NC}"
if [ -f "/usr/lib/libta_lib.so.0" ]; then
    echo "Removing TA-Lib library..."
    rm -f /usr/lib/libta_lib.so*
    rm -f /usr/lib/libta_lib.a
    rm -f /usr/lib/libta_lib.la
    rm -rf /usr/include/ta-lib
    ldconfig
fi

# Step 6: Clean up any remaining configuration files
echo -e "\n${YELLOW}Step 6: Cleaning up configuration files...${NC}"
su - ${REAL_USER} -c "rm -rf ${REAL_HOME}/.freqtrade"
su - ${REAL_USER} -c "rm -f ${REAL_HOME}/.freqtrade.sqlite"
su - ${REAL_USER} -c "rm -rf ${REAL_HOME}/.cache/freqtrade"
su - ${REAL_USER} -c "rm -f ${REAL_HOME}/freqtrade.log"
su - ${REAL_USER} -c "rm -f ${REAL_HOME}/freqai_strategy.log"

# Step 7: Remove setup and installation files
echo -e "\n${YELLOW}Step 7: Removing setup and installation files...${NC}"
su - ${REAL_USER} -c "rm -f ${REAL_HOME}/install_talib*.sh"
su - ${REAL_USER} -c "rm -f ${REAL_HOME}/run_bot*.sh"
su - ${REAL_USER} -c "rm -f ${REAL_HOME}/fix_project_structure*.sh"
su - ${REAL_USER} -c "rm -f ${REAL_HOME}/fresh_vm_setup*.sh"
su - ${REAL_USER} -c "rm -f ${REAL_HOME}/freqtrade_fix_guide*.md"
su - ${REAL_USER} -c "rm -f ${REAL_HOME}/additional_requirements.txt"

# Step 8: Clean up system packages (optional)
echo -e "\n${YELLOW}Step 8: Do you want to remove system packages? (y/n)${NC}"
read -p "This will remove cmake, screen, and other build tools: " remove_packages

if [ "$remove_packages" = "y" ]; then
    echo "Removing system packages..."
    apt-get remove -y cmake build-essential wget curl git python3-dev python3-pip python3-venv || true
    apt-get autoremove -y
    apt-get clean
fi

echo -e "\n${GREEN}=== Cleanup Complete ===${NC}"
echo "Your system has been cleaned of FreqAI-related files and installations."
echo -e "${YELLOW}Note: If you want to reinstall FreqAI, you can now use the fresh_vm_setup.sh script.${NC}"

# Print system status
echo -e "\n${YELLOW}System Status:${NC}"
echo "Checking for any remaining FreqAI processes..."
ps aux | grep -i "freqtrade\|freqai" | grep -v grep || echo "No FreqAI processes found"
echo -e "\nChecking ports 8081 and 8082..."
lsof -i:8081 || echo "Port 8081 is free"
lsof -i:8082 || echo "Port 8082 is free" 