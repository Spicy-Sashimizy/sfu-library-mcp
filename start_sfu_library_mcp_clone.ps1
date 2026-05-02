# SFU Library MCP Server (Clone) Launcher
# Targets the sfu-library-mcp-clone devcontainer — distinct from the mainline server.
# Version: 1.0

$ErrorActionPreference = "Stop"

$CONTAINER_NAME = "claudebox-sfu-library-mcp-clone-app"
$PYTHON_PATH = "/workspaces/sfu-library-mcp-clone/.venv/bin/python3"
$SCRIPT_PATH = "/workspaces/sfu-library-mcp-clone/src/sfu_library_mcp_server.py"

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
    Write-Output "{`"jsonrpc`":`"2.0`",`"id`":null,`"error`":{`"code`":-32002,`"message`":`"Container $CONTAINER_NAME is not running. Open the sfu-library-mcp-clone devcontainer in VS Code first.`"}}"
    exit 1
}

docker exec -i $CONTAINER_NAME $PYTHON_PATH $SCRIPT_PATH
