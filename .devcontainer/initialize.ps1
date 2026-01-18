# DevContainer initialization script - runs on HOST before container starts
# PowerShell version for Windows compatibility

$ErrorActionPreference = "SilentlyContinue"

$SHARED_NETWORK = "claudebox-shared-net"
$PROJECT_NAME = "claudebox-project"
if ($env:PROJECT_NAME) { $PROJECT_NAME = $env:PROJECT_NAME }

$SHARED_VOLUME = "claudebox-shared-config"

$CLAUDEBOX_DIR = "$env:USERPROFILE\.claudebox"
if ($env:CLAUDEBOX_DIR) { $CLAUDEBOX_DIR = $env:CLAUDEBOX_DIR }

Write-Host "ClaudeBox DevContainer Initialize"
Write-Host "=================================="

# 1. Ensure shared network exists
$existingNetworks = docker network ls --format "{{.Name}}" 2>$null
$networkExists = $false
if ($existingNetworks) {
    foreach ($net in $existingNetworks) {
        if ($net -eq $SHARED_NETWORK) {
            $networkExists = $true
            break
        }
    }
}

if ($networkExists) {
    Write-Host "[OK] Shared network exists: $SHARED_NETWORK"
} else {
    Write-Host "Creating shared network: $SHARED_NETWORK"
    docker network create --driver bridge --label "com.claudebox.managed=true" --label "com.claudebox.type=shared" $SHARED_NETWORK 2>$null
}

# 2. Copy credentials file to shared volume
$credentialsFile = Join-Path $CLAUDEBOX_DIR "credentials.json"
if (Test-Path $credentialsFile) {
    Write-Host "Copying credentials to shared volume..."
    docker run --rm -v "${SHARED_VOLUME}:/target" -v "${credentialsFile}:/source:ro" alpine:latest sh -c "cp /source /target/credentials.json 2>/dev/null || true" 2>$null
} else {
    Write-Host "[WARN] No credentials file found at $credentialsFile"
}

# 3. Check for stale containers
Write-Host "Checking for stale containers..."
$projectContainers = docker ps -a --filter "label=com.claudebox.project=$PROJECT_NAME" --format "{{.ID}}" 2>$null

if ($projectContainers) {
    Write-Host "Found existing project containers, checking network health..."
    $containerList = $projectContainers -split "`n"
    foreach ($containerId in $containerList) {
        $containerId = $containerId.Trim()
        if (-not $containerId) { continue }

        $containerName = docker inspect --format "{{.Name}}" $containerId 2>$null
        if ($containerName) {
            $containerName = $containerName.TrimStart('/')
        }

        $networkCheck = docker inspect $containerId --format "{{range .NetworkSettings.Networks}}{{.NetworkID}}{{end}}" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [WARN] Container $containerName has stale network references"
            Write-Host "  Removing stale container: $containerName"
            docker rm -f $containerId 2>$null
        }
    }
} else {
    Write-Host "[OK] No existing project containers found"
}

Write-Host "[OK] Initialization complete"
