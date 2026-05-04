# Training management script for SFU embedding model
# Runs commands inside the devcontainer via docker exec.
#
# Usage (from anywhere in the project):
#   .\scripts\run_training.ps1 status
#   .\scripts\run_training.ps1 start
#   .\scripts\run_training.ps1 resume
#   .\scripts\run_training.ps1 stop
#   .\scripts\run_training.ps1 logs
#   .\scripts\run_training.ps1 generate
#   .\scripts\run_training.ps1 setup

param(
    [Parameter(Position=0)]
    [string]$Command = "status"
)

$CONTAINER = "claudebox-sfu-library-mcp-clone-app"
$BASH_SCRIPT = "/workspaces/sfu-library-mcp-clone/scripts/run_training.sh"

try { $null = docker version 2>&1 } catch {
    Write-Error "Docker is not running. Please start Docker Desktop."
    exit 1
}

$running = docker ps --format "{{.Names}}" 2>$null | Select-String -Pattern "^$CONTAINER$"
if (-not $running) {
    Write-Error "Container '$CONTAINER' is not running. Open the devcontainer in VS Code first."
    exit 1
}

docker exec -it $CONTAINER bash $BASH_SCRIPT $Command
