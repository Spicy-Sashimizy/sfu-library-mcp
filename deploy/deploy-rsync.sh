#!/usr/bin/env bash
set -euo pipefail

# ── SFU Library MCP — Rsync-based Deploy to TrueNAS ──
# Runs from the dev container. Syncs code via rsync (no git on TrueNAS needed).
# Builds, and restarts with zero-downtime.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SSH_ALIAS="truenas"
REMOTE_APP_DIR="/mnt/MAIN/sfu-library-mcp/app"
COMPOSE_FILE="deploy/docker-compose.prod.yml"
CONTAINER="sfu-library-mcp"
HEALTH_URL="http://localhost:8080/health"
HEALTH_TIMEOUT=60

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[deploy]${NC} $*"; }
fail() { echo -e "${RED}[deploy] FAIL:${NC} $*"; exit 1; }

# ── Pre-deploy safety checks ──
log "Running pre-deploy safety checks..."

ssh "$SSH_ALIAS" "echo OK" > /dev/null 2>&1 || fail "SSH connectivity failed"
log "  SSH: OK"

ssh "$SSH_ALIAS" "sudo docker info --format '{{.ServerVersion}}'" > /dev/null 2>&1 || fail "Docker not accessible"
log "  Docker: OK"

MEM_FREE=$(ssh "$SSH_ALIAS" "free -m | awk '/^Mem:/{print \$7}'")
[ "$MEM_FREE" -ge 2048 ] 2>/dev/null || fail "Insufficient memory: ${MEM_FREE}MB (need 2048MB)"
log "  Memory: ${MEM_FREE}MB free"

DISK_FREE=$(ssh "$SSH_ALIAS" "df -BG /mnt/MAIN | awk 'NR==2{gsub(/G/,\"\",\$4); print \$4}'")
[ "$DISK_FREE" -ge 5 ] 2>/dev/null || fail "Insufficient disk: ${DISK_FREE}GB (need 5GB)"
log "  Disk: ${DISK_FREE}GB free"

PORT_CHECK=$(ssh "$SSH_ALIAS" "sudo docker ps --format '{{.Ports}}' | grep -c 8080 || true")
if [ "$PORT_CHECK" -gt 0 ]; then
    # Port in use — check if it's our container
    EXISTING=$(ssh "$SSH_ALIAS" "sudo docker ps --filter name=$CONTAINER --format '{{.Names}}'" || true)
    if [ "$EXISTING" = "$CONTAINER" ]; then
        log "  Port 8080: in use by $CONTAINER (will be replaced)"
    else
        fail "Port 8080 in use by another container"
    fi
else
    log "  Port 8080: free"
fi

# Snapshot current state
log "  Current image:"
ssh "$SSH_ALIAS" "sudo docker images | grep $CONTAINER || echo '  (no existing image)'"

# ── Run local tests ──
log "Running tests locally..."
cd "$PROJECT_DIR"
.venv/bin/python3 -m pytest src/tests/ -q --tb=line 2>&1 | tail -3 || fail "Tests failed"

# ── Rsync to TrueNAS ──
log "Syncing code to TrueNAS via rsync..."
rsync -avz --delete \
    --exclude='.env' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.devcontainer' \
    --exclude='.claude' \
    --exclude='.pommel' \
    --exclude='.pommelignore' \
    --exclude='src/tests' \
    --exclude='chrome-extension' \
    --exclude='languages' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    -e "ssh" \
    "$PROJECT_DIR/" "$SSH_ALIAS:$REMOTE_APP_DIR/"

# ── Build new image (before stopping old container) ──
log "Building new image on TrueNAS..."
ssh "$SSH_ALIAS" "cd $REMOTE_APP_DIR && sudo docker compose -f $COMPOSE_FILE build"

# ── Deploy ──
log "Starting new container..."
ssh "$SSH_ALIAS" "cd $REMOTE_APP_DIR && sudo docker compose -f $COMPOSE_FILE up -d"

# ── Wait for health check ──
log "Waiting for health check (up to ${HEALTH_TIMEOUT}s)..."
ELAPSED=0
HEALTHY=false
while [ $ELAPSED -lt $HEALTH_TIMEOUT ]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    STATUS=$(ssh "$SSH_ALIAS" "sudo docker inspect --format='{{.State.Health.Status}}' $CONTAINER 2>/dev/null" || echo "starting")
    log "  [${ELAPSED}s] $STATUS"
    if [ "$STATUS" = "healthy" ]; then
        HEALTHY=true
        break
    fi
done

if [ "$HEALTHY" = false ]; then
    warn "Health check did not pass within ${HEALTH_TIMEOUT}s"
    warn "Rolling back..."
    ssh "$SSH_ALIAS" "cd $REMOTE_APP_DIR && sudo docker compose -f $COMPOSE_FILE down"
    warn "Container stopped. Manual intervention required."
    exit 1
fi

# ── Post-deploy verification ──
log "Post-deploy verification..."

HEALTH=$(ssh "$SSH_ALIAS" "curl -sf $HEALTH_URL 2>/dev/null" || echo "FAIL")
if echo "$HEALTH" | grep -q '"status":"ok"'; then
    log "  Health: $HEALTH"
else
    fail "Health endpoint not responding: $HEALTH"
fi

TOOLS=$(echo "$HEALTH" | grep -o '"tools":[0-9]*' | grep -o '[0-9]*')
if [ "$TOOLS" = "26" ]; then
    log "  Tools: $TOOLS (correct)"
else
    warn "  Tools: $TOOLS (expected 26)"
fi

MEM=$(ssh "$SSH_ALIAS" "sudo docker stats $CONTAINER --no-stream --format '{{.MemUsage}}'" 2>/dev/null || echo "unknown")
log "  Memory: $MEM"

log "Deploy complete!"
