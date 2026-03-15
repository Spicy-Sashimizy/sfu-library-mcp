# SFU Library MCP Server - Desktop Setup Guide

## Problem Fixed

The original error was caused by:
1. **Docker container not running** - The MCP server was trying to exec into a stopped container
2. **Wrong Python path** - Config used `python3.10` but container has `/usr/local/bin/python`
3. **Docker errors breaking JSON parsing** - Plain text Docker errors like "OCI runtime error..." were breaking the JSON-RPC protocol

## Solution

A PowerShell wrapper script (`start_sfu_library_mcp.ps1`) that:
1. Checks if Docker is running
2. Checks if the container is running
3. Starts the container automatically if needed
4. Filters out non-JSON output from Docker
5. Provides proper JSON-RPC error messages when things fail

## Installation Steps

### 1. Copy the Files to Your Windows Desktop Location

Make sure these files are in the correct location on your Windows machine:

```
<USERPROFILE>/OneDrive\Desktop\random ass scripts\sfu_library_mcp\
├── start_sfu_library_mcp.ps1        (NEW - wrapper script)
├── start_sfu_library_mcp.bat        (backup batch script)
└── claude_desktop_config.json       (updated config)
```

### 2. Update Claude Desktop Config

Copy the contents of `claude_desktop_config.json` to your Claude Desktop config file:

**Windows Claude Desktop Config Location:**
```
<USERPROFILE>/AppData\Roaming\Claude\claude_desktop_config.json
```

Or if you're using Claude AI (web):
```
<USERPROFILE>/AppData\Local\AnthropicClaude\app-1.0.*\resources\app.asar\.vite\build\
```

### 3. Allow PowerShell Script Execution (First Time Only)

Open PowerShell as Administrator and run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or you can unblock the specific script:
```powershell
Unblock-File -Path "<USERPROFILE>/OneDrive\Desktop\random ass scripts\sfu_library_mcp\start_sfu_library_mcp.ps1"
```

### 4. Ensure Docker Desktop is Running

The script will automatically start the container if Docker is running, but Docker Desktop itself must be started.

### 5. Test the MCP Server

Restart Claude Desktop and try using the SFU Library tools.

## Troubleshooting

### Error: "Docker is not running or not installed"

**Solution:** Start Docker Desktop on Windows.

### Error: "Failed to start Docker container"

**Solution:** The docker-compose file may not be in the expected location. Check the path in `start_sfu_library_mcp.ps1`:

```powershell
$COMPOSE_FILE = "$PROJECT_ROOT\.devcontainer\docker-compose.yml"
```

Update it to point to your actual docker-compose.yml location.

### Error: "Python not found in container"

**Solution:** The Python path in the script may be wrong. Check the path in `start_sfu_library_mcp.ps1`:

```powershell
$PYTHON_PATH = "/usr/local/bin/python"
```

You can find the correct Python path by running:
```bash
docker exec -it claudebox-sfu-library-mcp-app which python3
```

## Manual Container Start (Fallback)

If the automatic startup doesn't work, you can manually start the container:

```bash
cd "<USERPROFILE>/OneDrive\Desktop\random ass scripts\sfu_library_mcp\.devcontainer"
docker-compose up -d app
```

## Alternative: Direct Docker Config (No Wrapper)

If you prefer not to use the wrapper script, you can use the direct Docker command. However, this requires the container to already be running:

```json
{
  "mcpServers": {
    "sfu-library": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "claudebox-sfu-library-mcp-app",
        "/usr/bin/python3",
        "/workspaces/sfu-library-mcp/src/sfu_library_mcp_server.py"
      ]
    }
  }
}
```

Note: You'll need to start the container manually each time you restart Docker.
