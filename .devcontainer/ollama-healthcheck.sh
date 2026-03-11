#!/bin/bash
# Ollama Proxy Health Check - runs periodically via supervisord
# Monitors socat proxy health, detects zombie connections, auto-restarts on failure
# Writes JSON metadata for dashboard API consumption

PROJECT_NAME="${PROJECT_NAME:-project}"
META_DIR="/home/vscode/.claudebox/ollama"
META_FILE="$META_DIR/${PROJECT_NAME}.json"
CHECK_INTERVAL="${OLLAMA_HEALTHCHECK_INTERVAL:-120}"  # 2 minutes default
ZOMBIE_THRESHOLD=50  # restart if more than this many socat children
MAX_CONSECUTIVE_FAILURES=2  # restart after this many consecutive HTTP failures

mkdir -p "$META_DIR"

consecutive_failures=0
last_restart=""

while true; do
    SOCAT_PARENT_RUNNING=false
    SOCAT_CHILD_COUNT=0
    OLLAMA_REACHABLE=false
    OLLAMA_VERSION=""
    ACTION_TAKEN=""
    HEALTHY=true

    # ── Check 1: Is the socat parent process running? ──
    if pgrep -f "socat.*TCP-LISTEN.*11434" > /dev/null 2>&1; then
        SOCAT_PARENT_RUNNING=true
    else
        HEALTHY=false
        echo "[OLLAMA-HEALTH] WARNING: socat parent process not running"
    fi

    # ── Check 2: Count socat child processes (zombie detection) ──
    SOCAT_CHILD_COUNT=$(pgrep -c -f "socat.*TCP.*host.docker.internal" 2>/dev/null || echo "0")
    # Subtract 1 for the parent listener if it's running
    if [ "$SOCAT_PARENT_RUNNING" = true ] && [ "$SOCAT_CHILD_COUNT" -gt 0 ]; then
        SOCAT_CHILD_COUNT=$((SOCAT_CHILD_COUNT - 1))
    fi

    if [ "$SOCAT_CHILD_COUNT" -gt "$ZOMBIE_THRESHOLD" ]; then
        HEALTHY=false
        echo "[OLLAMA-HEALTH] WARNING: $SOCAT_CHILD_COUNT socat children exceed threshold ($ZOMBIE_THRESHOLD) — restarting proxy"
        if supervisorctl restart ollama-proxy 2>/dev/null; then
            ACTION_TAKEN="restarted_zombie_threshold"
            last_restart=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            consecutive_failures=0
            SOCAT_CHILD_COUNT=0
        else
            ACTION_TAKEN="restart_failed"
            echo "[OLLAMA-HEALTH] ERROR: Failed to restart ollama-proxy via supervisorctl"
        fi
    fi

    # ── Check 3: Can we actually reach Ollama via HTTP? ──
    OLLAMA_RESPONSE=$(curl --connect-timeout 5 --max-time 10 -s http://localhost:11434/api/version 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$OLLAMA_RESPONSE" ]; then
        OLLAMA_REACHABLE=true
        OLLAMA_VERSION=$(echo "$OLLAMA_RESPONSE" | jq -r '.version // empty' 2>/dev/null || echo "")
        consecutive_failures=0
    else
        HEALTHY=false
        consecutive_failures=$((consecutive_failures + 1))
        echo "[OLLAMA-HEALTH] WARNING: Ollama HTTP check failed (consecutive: $consecutive_failures)"

        # Auto-remediate after consecutive failures
        if [ "$consecutive_failures" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
            echo "[OLLAMA-HEALTH] $consecutive_failures consecutive failures — restarting proxy"
            if supervisorctl restart ollama-proxy 2>/dev/null; then
                ACTION_TAKEN="restarted_http_failure"
                last_restart=$(date -u +%Y-%m-%dT%H:%M:%SZ)
                # Don't reset consecutive_failures — let the next check verify recovery
            else
                ACTION_TAKEN="restart_failed"
                echo "[OLLAMA-HEALTH] ERROR: Failed to restart ollama-proxy via supervisorctl"
            fi
        fi
    fi

    # ── Write metadata JSON (atomic tmp+mv) ──
    if command -v jq >/dev/null 2>&1; then
        jq -n \
            --arg project "$PROJECT_NAME" \
            --arg checkedAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --argjson healthy "$( [ "$HEALTHY" = true ] && echo true || echo false )" \
            --argjson socatRunning "$( [ "$SOCAT_PARENT_RUNNING" = true ] && echo true || echo false )" \
            --argjson socatChildren "$SOCAT_CHILD_COUNT" \
            --argjson ollamaReachable "$( [ "$OLLAMA_REACHABLE" = true ] && echo true || echo false )" \
            --arg ollamaVersion "${OLLAMA_VERSION:-}" \
            --argjson consecutiveFailures "$consecutive_failures" \
            --arg lastRestart "${last_restart:-}" \
            --arg actionTaken "${ACTION_TAKEN:-}" \
            '{
                project: $project,
                healthy: $healthy,
                socatRunning: $socatRunning,
                socatChildren: $socatChildren,
                ollamaReachable: $ollamaReachable,
                ollamaVersion: (if $ollamaVersion == "" then null else $ollamaVersion end),
                consecutiveFailures: $consecutiveFailures,
                lastRestart: (if $lastRestart == "" then null else $lastRestart end),
                actionTaken: (if $actionTaken == "" then null else $actionTaken end),
                checkedAt: $checkedAt
            }' > "$META_FILE.tmp" && mv "$META_FILE.tmp" "$META_FILE"
    fi

    sleep "$CHECK_INTERVAL"
done
