#!/bin/bash

# Define colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${RED}=== WARNING: This script will delete most files in your home directory ===${NC}"
echo -e "${YELLOW}Essential system files and directories will be preserved${NC}"
echo
echo -e "${RED}Are you absolutely sure you want to continue? (yes/no)${NC}"
read -p "Type 'yes' to continue: " confirmation

if [ "$confirmation" != "yes" ]; then
    echo "Operation cancelled."
    exit 1
fi

# Get the home directory
HOME_DIR="/home/henry"

echo -e "\n${YELLOW}Cleaning home directory at ${HOME_DIR}...${NC}"

# List of directories and files to preserve
PRESERVE=(
    ".ssh"
    ".bashrc"
    ".profile"
    ".bash_profile"
    ".bash_logout"
    ".config"
    ".local/share/keyrings"
)

# Create the delete command with exclusions
DELETE_CMD="cd \"$HOME_DIR\" && find . -mindepth 1"

# Add exclusions to the command
for item in "${PRESERVE[@]}"; do
    DELETE_CMD="$DELETE_CMD -not -path \"./$item\" -not -path \"./$item/*\""
done

# Add the delete operation
DELETE_CMD="$DELETE_CMD -delete"

# Execute the command
echo -e "${YELLOW}Deleting files...${NC}"
eval $DELETE_CMD

echo -e "\n${GREEN}Cleanup complete!${NC}"
echo -e "${YELLOW}Note: Essential system files and SSH keys have been preserved.${NC}" 