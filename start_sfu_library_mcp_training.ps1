# SFU Library MCP Server (Training) Launcher
# Bridges Claude Desktop (Windows host) -> stdio MCP server inside the
# claudebox-sfu-library-mcp-training-app devcontainer.
#
# Uses the project venv python (system python3 lacks numpy/torch/etc).
# Passes stdin/stdout straight through to `docker exec -i` so the JSON-RPC
# stream stays interactive (do NOT capture output into a variable — that
# buffers until process exit and breaks the stdio transport).
# Version: 1.0

$ErrorActionPreference = "Stop"

$CONTAINER_NAME = "claudebox-sfu-library-mcp-training-app"
$PYTHON_PATH    = "/workspaces/sfu-library-mcp-training/.venv/bin/python3"
$SCRIPT_PATH    = "/workspaces/sfu-library-mcp-training/src/sfu_library_mcp_server.py"

# Verify Docker is available.
try {
    $null = docker version 2>&1 | Out-Null
} catch {
    Write-Output '{"jsonrpc":"2.0","id":null,"error":{"code":-32001,"message":"Docker is not running or not installed. Please start Docker Desktop."}}'
    exit 1
}

function Test-ContainerRunning {
    param([string]$Name)
    $result = docker ps --format "{{.Names}}" 2>$null | Select-String -Pattern "^$Name$"
    return ($null -ne $result)
}

if (-not (Test-ContainerRunning -Name $CONTAINER_NAME)) {
    Write-Output "{`"jsonrpc`":`"2.0`",`"id`":null,`"error`":{`"code`":-32002,`"message`":`"Container $CONTAINER_NAME is not running. Open the sfu-library-mcp-training devcontainer in VS Code first.`"}}"
    exit 1
}

# Direct passthrough: stdin -> container, container stdout -> Claude Desktop.
docker exec -i $CONTAINER_NAME $PYTHON_PATH $SCRIPT_PATH
