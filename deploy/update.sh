#!/usr/bin/env bash
set -euo pipefail

# ── SFU Library MCP — Git-based Update on TrueNAS ──
# Runs from the dev container. SSH to TrueNAS via ProxyJump.
# Pulls latest code, rebuilds, and restarts with zero-downtime.

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

log() { echo -e "${GREEN}[update]${NC} $*"; }
warn() { echo -e "${YELLOW}[update]${NC} $*"; }
fail() { echo -e "${RED}[update] FAIL:${NC} $*"; exit 1; }

# ── Pre-deploy safety checks ──
log "Running pre-deploy safety checks..."

ssh "$SSH_ALIAS" "echo OK" > /dev/null 2>&1 || fail "SSH connectivity failed"
log "  SSH: OK"

MEM_FREE=$(ssh "$SSH_ALIAS" "free -m | awk '/^Mem:/{print \$7}'")
[ "$MEM_FREE" -ge 2048 ] 2>/dev/null || fail "Insufficient memory: ${MEM_FREE}MB (need 2048MB)"
log "  Memory: ${MEM_FREE}MB free"

DISK_FREE=$(ssh "$SSH_ALIAS" "df -BG /mnt/MAIN | awk 'NR==2{gsub(/G/,\"\",\$4); print \$4}'")
[ "$DISK_FREE" -ge 5 ] 2>/dev/null || fail "Insufficient disk: ${DISK_FREE}GB (need 5GB)"
log "  Disk: ${DISK_FREE}GB free"

# Snapshot current state for rollback reference
log "  Current image:"
ssh "$SSH_ALIAS" "sudo docker images | grep $CONTAINER || echo '  (no existing image)'"

# ── Pull latest code ──
log "Pulling latest code on TrueNAS..."
ssh "$SSH_ALIAS" "cd $REMOTE_APP_DIR && git pull"

# ── Build new image (before stopping old container) ──
log "Building new image..."
ssh "$SSH_ALIAS" "cd $REMOTE_APP_DIR && sudo docker compose -f $COMPOSE_FILE build"

# ── Deploy ──
log "Starting new container..."
ssh "$SSH_ALIAS" "cd $REMOTE_APP_DIR && sudo docker compose -f $COMPOSE_FILE up -d"

# ── Wait for health check ──
log "Waiting for health check (up to ${HEALTH_TIMEOUT}s)..."
ELAPSED=0
while [ $ELAPSED -lt $HEALTH_TIMEOUT ]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    STATUS=$(ssh "$SSH_ALIAS" "sudo docker inspect --format='{{.State.Health.Status}}' $CONTAINER 2>/dev/null" || echo "starting")
    log "  [${ELAPSED}s] $STATUS"
    if [ "$STATUS" = "healthy" ]; then
        break
    fi
done

# ── Post-deploy verification ──
log "Post-deploy verification..."

HEALTH=$(ssh "$SSH_ALIAS" "curl -sf $HEALTH_URL 2>/dev/null" || echo "FAIL")
if echo "$HEALTH" | grep -q '"status":"ok"'; then
    log "  Health: $HEALTH"
else
    warn "  Health check failed: $HEALTH"
    warn "  Rollback command:"
    warn "    ssh $SSH_ALIAS \"cd $REMOTE_APP_DIR && sudo docker compose -f $COMPOSE_FILE down && sudo docker compose -f $COMPOSE_FILE up -d\""
    exit 1
fi

# Verify tools count
TOOLS=$(echo "$HEALTH" | grep -o '"tools":[0-9]*' | grep -o '[0-9]*')
if [ "$TOOLS" = "26" ]; then
    log "  Tools: $TOOLS (correct)"
else
    warn "  Tools: $TOOLS (expected 26)"
fi

MEM=$(ssh "$SSH_ALIAS" "sudo docker stats $CONTAINER --no-stream --format '{{.MemUsage}}'" 2>/dev/null || echo "unknown")
log "  Memory: $MEM"

log "Update complete!"
