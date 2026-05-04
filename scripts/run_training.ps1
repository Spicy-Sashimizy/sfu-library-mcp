# Training management script for SFU embedding model (PowerShell / Windows)
# Delegates to run_training.sh inside WSL.
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

# Resolve the WSL path to this script's project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WslProjectRoot = wsl wslpath -u "$($ProjectRoot.Replace('\','/'))"

# Run the bash script inside WSL
wsl bash "$WslProjectRoot/scripts/run_training.sh" $Command
