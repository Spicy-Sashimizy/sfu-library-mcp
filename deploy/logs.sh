#!/usr/bin/env bash
set -euo pipefail

# ── SFU Library MCP — Log Access from Dev Container ──
# Usage:
#   ./deploy/logs.sh tail [N]       Last N lines of app log (default 50)
#   ./deploy/logs.sh follow         Real-time stream (tail -f)
#   ./deploy/logs.sh errors [N]     Last N ERROR/WARNING lines (default 20)
#   ./deploy/logs.sh docker [N]     Docker container logs (default 100)
#   ./deploy/logs.sh health         Run health check

SSH_ALIAS="truenas"
CONTAINER="sfu-library-mcp"
LOG_DIR="/mnt/MAIN/sfu-library-mcp/logs"
LOG_FILE="$LOG_DIR/sfu-library-mcp.log"

CMD="${1:-tail}"
COUNT="${2:-50}"

case "$CMD" in
    tail)
        echo "=== Last $COUNT lines of $LOG_FILE ==="
        ssh "$SSH_ALIAS" "tail -n $COUNT $LOG_FILE 2>/dev/null || echo 'Log file not found'"
        ;;
    follow|f)
        echo "=== Following $LOG_FILE (Ctrl+C to stop) ==="
        ssh "$SSH_ALIAS" "tail -f $LOG_FILE 2>/dev/null || echo 'Log file not found'"
        ;;
    errors|err)
        echo "=== Last $COUNT ERROR/WARNING lines ==="
        ssh "$SSH_ALIAS" "grep -E ' (ERROR|WARNING) ' $LOG_FILE 2>/dev/null | tail -n $COUNT || echo 'No errors found'"
        ;;
    docker|d)
        echo "=== Docker container logs (last $COUNT lines) ==="
        ssh "$SSH_ALIAS" "sudo docker logs $CONTAINER --tail $COUNT 2>&1"
        ;;
    health|h)
        echo "=== Container Health ==="
        STATUS=$(ssh "$SSH_ALIAS" "sudo docker inspect --format='{{.State.Health.Status}}' $CONTAINER 2>/dev/null" || echo "unknown")
        echo "  Docker health: $STATUS"

        RUNNING=$(ssh "$SSH_ALIAS" "sudo docker ps --filter name=$CONTAINER --format '{{.Status}}'" 2>/dev/null || echo "not running")
        echo "  Container: $RUNNING"

        HEALTH=$(ssh "$SSH_ALIAS" "curl -sf http://localhost:8080/health 2>/dev/null" || echo "not responding")
        echo "  HTTP health: $HEALTH"

        MEM=$(ssh "$SSH_ALIAS" "sudo docker stats $CONTAINER --no-stream --format '{{.MemUsage}}'" 2>/dev/null || echo "unknown")
        echo "  Memory: $MEM"
        ;;
    *)
        echo "Usage: $0 {tail|follow|errors|docker|health} [count]"
        echo ""
        echo "Commands:"
        echo "  tail [N]     Last N lines of application log (default 50)"
        echo "  follow       Real-time log stream"
        echo "  errors [N]   Last N ERROR/WARNING lines (default 20)"
        echo "  docker [N]   Docker container logs (default 100)"
        echo "  health       Container and HTTP health status"
        exit 1
        ;;
esac
