#!/bin/bash
# LSP Health Check - runs periodically via supervisord
# Checks which language servers are running and updates metadata JSON
# Does NOT start/stop servers - Claude Code manages their lifecycle

PROJECT_NAME="${PROJECT_NAME:-project}"
META_DIR="/home/vscode/.claudebox/lsp"
META_FILE="$META_DIR/${PROJECT_NAME}.json"
CHECK_INTERVAL="${LSP_HEALTHCHECK_INTERVAL:-300}"  # 5 minutes default

mkdir -p "$META_DIR"

# LSP server process patterns (what to look for in process list)
declare -A LSP_PROCESSES=(
    ["pyright"]="pyright-langserver"
    ["typescript-lsp"]="typescript-language-server"
    ["gopls"]="gopls"
    ["rust-analyzer"]="rust-analyzer"
    ["clangd"]="clangd"
)

while true; do
    RUNNING_SERVERS=""
    INSTALLED_SERVERS=""
    SERVER_DETAILS="[]"

    for server in pyright typescript-lsp gopls rust-analyzer clangd; do
        PROCESS="${LSP_PROCESSES[$server]}"
        INSTALLED=false
        RUNNING=false
        PID=""

        # Check if binary is installed
        case "$server" in
            "pyright") command -v pyright >/dev/null 2>&1 && INSTALLED=true ;;
            "typescript-lsp") command -v typescript-language-server >/dev/null 2>&1 && INSTALLED=true ;;
            "gopls") command -v gopls >/dev/null 2>&1 && INSTALLED=true ;;
            "rust-analyzer") command -v rust-analyzer >/dev/null 2>&1 && INSTALLED=true ;;
            "clangd") command -v clangd >/dev/null 2>&1 && INSTALLED=true ;;
        esac

        if [ "$INSTALLED" = true ]; then
            INSTALLED_SERVERS="$INSTALLED_SERVERS $server"
        fi

        # Check if process is running
        # Use pgrep -x (exact binary name) for native binaries to avoid
        # false positives from VS Code's --install-extension arguments
        case "$server" in
            "rust-analyzer"|"gopls"|"clangd")
                PID=$(pgrep -x "$PROCESS" 2>/dev/null | head -1) ;;
            *)
                PID=$(pgrep -f "$PROCESS" 2>/dev/null | head -1) ;;
        esac
        if [ -n "$PID" ]; then
            RUNNING=true
            RUNNING_SERVERS="$RUNNING_SERVERS $server"
        fi

        # Build JSON entry for this server
        SERVER_DETAILS=$(echo "$SERVER_DETAILS" | jq --arg name "$server" \
            --arg installed "$INSTALLED" --arg running "$RUNNING" --arg pid "${PID:-}" \
            '. + [{"name": $name, "installed": ($installed == "true"), "running": ($running == "true"), "pid": (if $pid == "" then null else ($pid | tonumber) end)}]' 2>/dev/null || echo "$SERVER_DETAILS")
    done

    # Check memory pressure
    MEM_TOTAL=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}')
    MEM_USED=$(free -m 2>/dev/null | awk '/^Mem:/{print $3}')
    MEM_WARNING=false
    if [ -n "$MEM_TOTAL" ] && [ "$MEM_TOTAL" -gt 0 ]; then
        MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))
        if [ "$MEM_PCT" -gt 85 ]; then
            MEM_WARNING=true
            echo "[LSP-HEALTH] WARNING: Memory at ${MEM_PCT}% (${MEM_USED}MB/${MEM_TOTAL}MB) - language servers may be affected"
        fi
    fi

    # Write metadata JSON
    if command -v jq >/dev/null 2>&1; then
        jq -n \
            --arg project "$PROJECT_NAME" \
            --arg checkedAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --argjson servers "$SERVER_DETAILS" \
            --arg memWarning "$MEM_WARNING" \
            --arg memPct "${MEM_PCT:-0}" \
            '{
                project: $project,
                servers: $servers,
                memoryWarning: ($memWarning == "true"),
                memoryPercent: ($memPct | tonumber),
                checkedAt: $checkedAt
            }' > "$META_FILE.tmp" && mv "$META_FILE.tmp" "$META_FILE"
    fi

    sleep "$CHECK_INTERVAL"
done
