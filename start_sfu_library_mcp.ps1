# SFU Library MCP Server Launcher
# This script ensures the Docker container is running and starts the MCP server
# Version: 1.1

$ErrorActionPreference = "Stop"

# Configuration
$CONTAINER_NAME = "claudebox-sfu-library-mcp-app"
$PYTHON_PATH = "/usr/bin/python3"
$SCRIPT_PATH = "/workspaces/sfu-library-mcp/src/sfu_library_mcp_server.py"
$PROJECT_ROOT = "C:\Users\gordo\OneDrive\Desktop\random ass scripts\sfu_library_mcp"
$COMPOSE_FILE = "$PROJECT_ROOT\.devcontainer\docker-compose.yml"

# Test if Docker is available
try {
    $null = docker version 2>&1 | Out-Null
} catch {
    # Docker not available, output error
    Write-Output '{"jsonrpc":"2.0","id":null,"error":{"code":-32001,"message":"Docker is not running or not installed. Please start Docker Desktop."}}'
    exit 1
}

function Test-ContainerRunning {
    param([string]$Name)
    $result = docker ps --format "{{.Names}}" 2>$null | Select-String -Pattern $Name
    return ($null -ne $result)
}

function Start-ContainerIfNeeded {
    param([string]$Name, [string]$ComposeFile)

    if (Test-ContainerRunning -Name $Name) {
        return $true
    }

    # Container not running, try to start it
    Write-Error "Container $Name is not running. Attempting to start..." 2>$null

    if (Test-Path $ComposeFile) {
        $composeDir = Split-Path $ComposeFile
        Push-Location $composeDir
        try {
            docker-compose up -d app 2>&1 | Out-Null
        } catch {
            # Fallback to docker compose (v2 command)
            docker compose up -d app 2>&1 | Out-Null
        }
        Pop-Location
        Start-Sleep -Seconds 5  # Give container time to start
    } else {
        docker start $Name 2>$null | Out-Null
        Start-Sleep -Seconds 3
    }

    # Check again after starting
    $maxAttempts = 10
    $attempt = 0
    while (-not (Test-ContainerRunning -Name $Name) -and $attempt -lt $maxAttempts) {
        Start-Sleep -Seconds 1
        $attempt++
    }

    return Test-ContainerRunning -Name $Name
}

function Invoke-MCPServer {
    param([string]$ContainerName, [string]$PythonPath, [string]$ScriptPath)

    # Execute the MCP server and capture all output
    $output = docker exec -i $ContainerName $PythonPath $ScriptPath 2>&1

    # Filter out any non-JSON output (Docker error messages, etc.)
    # Valid JSON-RPC messages start with { or contain "jsonrpc"
    foreach ($line in $output) {
        $trimmed = $line.Trim()
        if ($trimmed -match '^\s*\{' -or $trimmed -match '^\s*\[' -or $trimmed -match '"jsonrpc"') {
            Write-Output $trimmed
        }
    }
}

# Main execution
try {
    if (Start-ContainerIfNeeded -Name $CONTAINER_NAME -ComposeFile $COMPOSE_FILE) {
        Invoke-MCPServer -ContainerName $CONTAINER_NAME -PythonPath $PYTHON_PATH -ScriptPath $SCRIPT_PATH
    } else {
        # Container failed to start
        Write-Output '{"jsonrpc":"2.0","id":null,"error":{"code":-32002,"message":"Failed to start Docker container. Please check Docker and try again."}}'
    }
} catch {
    # Unexpected error - still try to provide valid JSON-RPC error
    Write-Output '{"jsonrpc":"2.0","id":null,"error":{"code":-32000,"message":"Unexpected error starting MCP server. Check logs for details."}}'
}
