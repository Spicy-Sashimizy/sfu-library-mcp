#!/usr/bin/env bash
set -euo pipefail

# ── SFU Library MCP — Deploy to TrueNAS ──
# Runs from inside the ClaudeBox dev container.
# SSH reaches TrueNAS via ProxyJump through Windows host.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SSH_ALIAS="truenas"
REMOTE_APP_DIR="/mnt/MAIN/sfu-library-mcp/app"
COMPOSE_FILE="deploy/docker-compose.prod.yml"

echo "═══ SFU Library MCP — Deploy to TrueNAS ═══"

# 1. Pre-deploy safety checks
echo "→ Running pre-deploy safety checks..."

echo -n "  SSH connectivity... "
ssh "$SSH_ALIAS" "echo OK" || { echo "FAIL"; exit 1; }

echo -n "  Docker access... "
ssh "$SSH_ALIAS" "sudo docker info --format '{{.ServerVersion}}'" || { echo "FAIL"; exit 1; }

echo -n "  Available memory... "
ssh "$SSH_ALIAS" "free -m | awk '/^Mem:/{if(\$7 < 2048) { print \"FAIL: only \" \$7 \"MB free\"; exit 1 } else print \"OK: \" \$7 \"MB available\"}'" || exit 1

echo -n "  Available disk... "
ssh "$SSH_ALIAS" "df -BG /mnt/MAIN | awk 'NR==2{gsub(/G/,\"\",\$4); if(\$4 < 5) { print \"FAIL: only \" \$4 \"GB free\"; exit 1 } else print \"OK: \" \$4 \"GB free\"}'" || exit 1

echo -n "  Port 8080... "
ssh "$SSH_ALIAS" "sudo docker ps --format '{{.Ports}}' | grep -q 8080 && echo 'FAIL: Port 8080 in use' && exit 1 || echo 'OK: free'" || exit 1

# 2. Run tests locally
echo "→ Running tests..."
cd "$PROJECT_DIR"
.venv/bin/python3 -m pytest src/tests/ -q || { echo "Tests failed — aborting deploy"; exit 1; }

# 3. Sync code to TrueNAS
echo "→ Syncing code to TrueNAS..."
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
    -e "ssh" \
    "$PROJECT_DIR/" "$SSH_ALIAS:$REMOTE_APP_DIR/"

# 4. Build new image BEFORE stopping old container (zero-downtime)
echo "→ Building production image on TrueNAS..."
ssh "$SSH_ALIAS" "cd $REMOTE_APP_DIR && sudo docker compose -f $COMPOSE_FILE build"

# 5. Restart with new image
echo "→ Starting new container..."
ssh "$SSH_ALIAS" "cd $REMOTE_APP_DIR && sudo docker compose -f $COMPOSE_FILE up -d"

# 6. Wait for health check
echo "→ Waiting for health check (up to 60s)..."
for i in $(seq 1 12); do
    sleep 5
    STATUS=$(ssh "$SSH_ALIAS" "sudo docker inspect --format='{{.State.Health.Status}}' sfu-library-mcp 2>/dev/null" || echo "starting")
    echo "  [$((i*5))s] Status: $STATUS"
    if [ "$STATUS" = "healthy" ]; then
        break
    fi
done

# 7. Post-deploy verification
echo "→ Post-deploy verification..."
ssh "$SSH_ALIAS" "sudo docker ps | grep sfu-library-mcp"
ssh "$SSH_ALIAS" "curl -sf http://localhost:8080/health" && echo ""

echo "═══ Deploy complete ═══"
