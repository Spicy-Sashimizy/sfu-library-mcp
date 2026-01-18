#!/bin/bash
# Setup script for SFU Library MCP Server in WSL

echo "Installing Python dependencies..."
pip3 install -r /workspaces/sfu-library-mcp/requirements.txt

echo ""
echo "Installing Chromium browser..."
sudo apt-get update
sudo apt-get install -y chromium chromium-driver

echo ""
echo "Setup complete! Testing server startup..."
timeout 3 python3 /workspaces/sfu-library-mcp/src/sfu_library_mcp_server.py 2>&1 || echo "Server starts successfully!"

echo ""
echo "Next steps:"
echo "1. Update your Claude Desktop config at: %APPDATA%\\Claude\\claude_desktop_config.json"
echo "2. Add the content from: /workspaces/sfu-library-mcp/claude_desktop_config.json"
echo "3. Restart Claude Desktop"
