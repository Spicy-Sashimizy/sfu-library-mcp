#!/usr/bin/env bash
set -euo pipefail

# ── SFU Library MCP — Comprehensive Health Check ──
# Run from dev container to check production deployment on TrueNAS.
# Exit code 0 = healthy, exit code 1 = unhealthy.
#
# Usage:
#   ./deploy/healthcheck.sh

SSH_ALIAS="truenas"
CONTAINER="sfu-library-mcp"
HEALTH_URL="http://localhost:8080/health"
LOG_DIR="/mnt/MAIN/sfu-library-mcp/logs"
LOG_FILE="$LOG_DIR/sfu-library-mcp.log"
EXPECTED_TOOLS=26

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

check_pass() { echo -e "  [${GREEN} OK ${NC}] $1"; PASS=$((PASS + 1)); }
check_fail() { echo -e "  [${RED}FAIL${NC}] $1"; FAIL=$((FAIL + 1)); }
check_warn() { echo -e "  [${YELLOW}WARN${NC}] $1"; WARN=$((WARN + 1)); }

echo ""
echo "============================================================"
echo "  SFU Library MCP — Health Check"
echo "============================================================"
echo ""

# 1. SSH connectivity
if ssh "$SSH_ALIAS" "echo OK" > /dev/null 2>&1; then
    check_pass "SSH connectivity"
else
    check_fail "SSH connectivity — cannot reach TrueNAS"
    echo ""
    echo "Cannot continue without SSH. Aborting."
    exit 1
fi

# 2. Container running
CONTAINER_STATUS=$(ssh "$SSH_ALIAS" "sudo docker ps --filter name=$CONTAINER --format '{{.Status}}'" 2>/dev/null || echo "")
if [ -n "$CONTAINER_STATUS" ]; then
    check_pass "Container running: $CONTAINER_STATUS"
else
    check_fail "Container not running"
fi

# 3. Container healthy
HEALTH_STATUS=$(ssh "$SSH_ALIAS" "sudo docker inspect --format='{{.State.Health.Status}}' $CONTAINER 2>/dev/null" || echo "unknown")
if [ "$HEALTH_STATUS" = "healthy" ]; then
    check_pass "Container healthy"
elif [ "$HEALTH_STATUS" = "starting" ]; then
    check_warn "Container starting (health check not yet passed)"
else
    check_fail "Container health: $HEALTH_STATUS"
fi

# 4. HTTP responsive
HEALTH_RESPONSE=$(ssh "$SSH_ALIAS" "curl -sf $HEALTH_URL 2>/dev/null" || echo "")
if echo "$HEALTH_RESPONSE" | grep -q '"status":"ok"'; then
    check_pass "HTTP health endpoint"
else
    check_fail "HTTP health endpoint not responding"
fi

# 5. Tools available
if [ -n "$HEALTH_RESPONSE" ]; then
    TOOL_COUNT=$(echo "$HEALTH_RESPONSE" | grep -o '"tools":[0-9]*' | grep -o '[0-9]*' || echo "0")
    if [ "$TOOL_COUNT" = "$EXPECTED_TOOLS" ]; then
        check_pass "Tools: $TOOL_COUNT available"
    else
        check_warn "Tools: $TOOL_COUNT (expected $EXPECTED_TOOLS)"
    fi
fi

# 6. Memory usage
MEM_USAGE=$(ssh "$SSH_ALIAS" "sudo docker stats $CONTAINER --no-stream --format '{{.MemPerc}}'" 2>/dev/null || echo "unknown")
if [ "$MEM_USAGE" != "unknown" ]; then
    # Extract percentage number
    MEM_PCT=$(echo "$MEM_USAGE" | tr -d '%' | cut -d'.' -f1)
    if [ "$MEM_PCT" -lt 90 ] 2>/dev/null; then
        check_pass "Memory usage: $MEM_USAGE"
    else
        check_warn "Memory usage high: $MEM_USAGE"
    fi
else
    check_warn "Cannot check memory usage"
fi

# 7. Disk usage
DISK_FREE=$(ssh "$SSH_ALIAS" "df -BG /mnt/MAIN | awk 'NR==2{gsub(/G/,\"\",\$4); print \$4}'" 2>/dev/null || echo "0")
if [ "$DISK_FREE" -ge 5 ] 2>/dev/null; then
    check_pass "Disk: ${DISK_FREE}GB free on /mnt/MAIN"
elif [ "$DISK_FREE" -ge 2 ] 2>/dev/null; then
    check_warn "Disk: ${DISK_FREE}GB free (low)"
else
    check_fail "Disk: ${DISK_FREE}GB free (critical)"
fi

# 8. Log freshness
LOG_AGE=$(ssh "$SSH_ALIAS" "stat -c %Y $LOG_FILE 2>/dev/null || echo 0")
if [ "$LOG_AGE" != "0" ]; then
    NOW=$(ssh "$SSH_ALIAS" "date +%s")
    AGE_SECS=$((NOW - LOG_AGE))
    AGE_MINS=$((AGE_SECS / 60))
    if [ "$AGE_MINS" -lt 60 ]; then
        check_pass "Log freshness: last written ${AGE_MINS}m ago"
    elif [ "$AGE_MINS" -lt 360 ]; then
        check_warn "Log freshness: last written ${AGE_MINS}m ago (stale)"
    else
        check_fail "Log freshness: last written ${AGE_MINS}m ago (server may be frozen)"
    fi
else
    check_warn "Log file not found: $LOG_FILE"
fi

# 9. Secret integrity
SECRETS=$(ssh "$SSH_ALIAS" "sudo docker exec $CONTAINER ls /run/secrets/ 2>/dev/null" || echo "")
EXPECTED_SECRETS="sfu_username sfu_password sfu_mfa_secret sfu_mfa_device_name zotero_api_key zotero_user_id"
MISSING=""
for s in $EXPECTED_SECRETS; do
    if ! echo "$SECRETS" | grep -q "$s"; then
        MISSING="$MISSING $s"
    fi
done
if [ -z "$MISSING" ]; then
    check_pass "Secrets: all 6 mounted"
else
    check_fail "Secrets missing:$MISSING"
fi

# ── Summary ──
TOTAL=$((PASS + FAIL + WARN))
echo ""
echo "============================================================"
echo -e "  ${GREEN}$PASS${NC}/$TOTAL ok, ${YELLOW}$WARN${NC} warnings, ${RED}$FAIL${NC} failures"
echo "============================================================"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "Recommended actions:"
    if [ -z "$CONTAINER_STATUS" ]; then
        echo "  - Start container: ssh $SSH_ALIAS 'cd /mnt/MAIN/sfu-library-mcp/app && sudo docker compose -f deploy/docker-compose.prod.yml up -d'"
    fi
    if [ -n "$MISSING" ]; then
        echo "  - Create missing secrets in /mnt/MAIN/sfu-library-mcp/secrets/"
    fi
    exit 1
fi

exit 0
