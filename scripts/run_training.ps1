# Training management script for SFU embedding model (PowerShell)
# Works from both Windows PowerShell (via WSL) and pwsh inside the Linux container.
#
# Usage (from the scripts/ directory or project root):
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

$ScriptDir  = $PSScriptRoot
$BashScript = Join-Path $ScriptDir "run_training.sh"

if ($env:OS -eq 'Windows_NT' -or [System.IO.Path]::DirectorySeparatorChar -eq '\') {
    # Running on Windows — convert path and delegate into WSL
    $WslScript = (wsl wslpath -u $BashScript.Replace('\', '/'))
    wsl bash $WslScript $Command
} else {
    # Running inside Linux container — call bash directly
    & /usr/bin/bash $BashScript $Command
}
