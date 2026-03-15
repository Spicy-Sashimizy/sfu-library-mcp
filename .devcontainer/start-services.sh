#!/bin/bash
# Start background services for ClaudeBox container

# Verify we're on expected networks
echo "Verifying network connectivity..."
if ip addr show eth0 > /dev/null 2>&1; then
    echo "Network eth0 is up"
else
    echo "WARNING: eth0 not found - container may have network issues"
fi

# Start supervisor in background (manages socat proxy and pommeld)
# NOTE: docker-in-docker feature may have already started supervisord as root
# during container init. We use pgrep (not supervisorctl) to detect this
# because the socket may be root-owned and inaccessible to vscode user.
if command -v supervisord &> /dev/null; then
    # Clean stale supervisor files from unclean shutdown
    if [ -f /tmp/supervisord.pid ]; then
        OLD_PID=$(cat /tmp/supervisord.pid 2>/dev/null)
        if [ -n "$OLD_PID" ] && ! kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Cleaning stale supervisor files (PID $OLD_PID dead)..."
            rm -f /tmp/supervisor.sock /tmp/supervisord.pid
        fi
    fi

    if pgrep -x supervisord > /dev/null 2>&1; then
        echo "Supervisor already running (started by container init)"
        supervisorctl -c /etc/supervisor/supervisord.conf status 2>/dev/null && true
    else
        echo "Starting supervisor..."
        supervisord -c /etc/supervisor/supervisord.conf
        sleep 2

        if supervisorctl -c /etc/supervisor/supervisord.conf status > /dev/null 2>&1; then
            echo "Supervisor services:"
            supervisorctl -c /etc/supervisor/supervisord.conf status
        fi
    fi

    # Spawn watchdog to restart supervisord if it dies
    if ! pgrep -f "supervisord-watchdog" > /dev/null 2>&1; then
        (
            exec -a supervisord-watchdog bash -c '
                while true; do
                    sleep 60
                    if ! pgrep -x supervisord > /dev/null 2>&1; then
                        echo "[watchdog] $(date): supervisord died, restarting..."
                        rm -f /tmp/supervisor.sock /tmp/supervisord.pid
                        supervisord -c /etc/supervisor/supervisord.conf 2>&1 || true
                    fi
                done
            '
        ) >> /var/log/supervisor/watchdog.log 2>&1 &
        disown
    fi
else
    echo "Supervisor not installed, starting services manually..."

    # Start socat proxy manually (hardened — matches supervisord config)
    if ! pgrep -f "socat.*11434" > /dev/null 2>&1; then
        nohup socat -T30 TCP-LISTEN:11434,fork,reuseaddr,keepalive,keepidle=30,keepintvl=10,keepcnt=3 TCP:host.docker.internal:11434,connect-timeout=10,keepalive,keepidle=30,keepintvl=10,keepcnt=3 > /var/log/socat-ollama.log 2>&1 &
        echo "Started Ollama proxy (hardened)"
    fi

    # Start pommeld manually if installed
    if command -v pommeld &> /dev/null && [ -d "/workspaces/${PROJECT_NAME}/.pommel" ]; then
        cd "/workspaces/${PROJECT_NAME}"
        nohup pommeld -project "/workspaces/${PROJECT_NAME}" > /var/log/pommeld.log 2>&1 &
        echo "Started Pommel daemon"
    fi
fi

echo "Background services started"

# ==========================================
# Ensure LSP servers are installed (runs every start)
# ==========================================
# post-create.sh installs these once, but if it failed or the container
# predates LSP integration, this ensures servers are ready before the
# user starts Claude Code (avoiding the "install then restart" problem).
echo "Verifying LSP language servers..."

# Source PATH additions from devcontainer features (Go, Rust)
export PATH="/usr/local/go/bin:${GOPATH:-/home/vscode/go}/bin:/home/vscode/.cargo/bin:${PATH}"
for f in /etc/profile.d/*.sh; do
    [ -r "$f" ] && . "$f" 2>/dev/null || true
done

LSP_MISSING=0

# Python: pyright
if ! command -v pyright >/dev/null 2>&1; then
    echo "  Installing pyright..."
    npm install -g pyright 2>/dev/null && echo "  [OK] pyright" || echo "  [WARN] pyright failed"
    LSP_MISSING=1
fi

# TypeScript: typescript-language-server
if ! command -v typescript-language-server >/dev/null 2>&1; then
    echo "  Installing typescript-language-server..."
    npm install -g typescript-language-server typescript 2>/dev/null && echo "  [OK] typescript-language-server" || echo "  [WARN] typescript-lsp failed"
    LSP_MISSING=1
fi

# Go: gopls
if ! command -v gopls >/dev/null 2>&1; then
    if command -v go >/dev/null 2>&1; then
        echo "  Installing gopls..."
        go install golang.org/x/tools/gopls@latest 2>/dev/null && echo "  [OK] gopls" || echo "  [WARN] gopls failed"
        LSP_MISSING=1
    fi
fi

# Rust: rust-analyzer
if ! command -v rust-analyzer >/dev/null 2>&1; then
    if command -v rustup >/dev/null 2>&1; then
        echo "  Installing rust-analyzer..."
        rustup component add rust-analyzer 2>/dev/null && echo "  [OK] rust-analyzer" || echo "  [WARN] rust-analyzer failed"
        LSP_MISSING=1
    fi
fi

# C/C++: clangd
if ! command -v clangd >/dev/null 2>&1; then
    echo "  Installing clangd..."
    sudo apt-get update -qq && sudo apt-get install -y -qq clangd 2>/dev/null && echo "  [OK] clangd" || echo "  [WARN] clangd failed"
    LSP_MISSING=1
fi

# Ensure LSP MCP server is registered in settings.json (idempotent)
SETTINGS_FILE="/home/vscode/.claude/settings.json"
MCP_SERVER_DIR="/home/vscode/.claudebox/mcp"
if [ -f "$SETTINGS_FILE" ] && command -v jq >/dev/null 2>&1; then
    if ! jq -e '.mcpServers.lsp' "$SETTINGS_FILE" >/dev/null 2>&1; then
        echo "  Registering LSP MCP server in settings.json..."
        LOCK="/home/vscode/.claude/.settings.lock"
        (
          flock -w 10 200 || exit 0
          jq --arg dir "$MCP_SERVER_DIR" --arg proj "${PROJECT_NAME}" \
            '.mcpServers = (.mcpServers // {}) + {
                "lsp": {
                    "command": "python3",
                    "args": [$dir + "/lsp-mcp-server.py"],
                    "env": {"PROJECT_NAME": $proj, "WORKSPACE_PATH": "/workspaces/" + $proj}
                }
            }' "$SETTINGS_FILE" > "${SETTINGS_FILE}.tmp" && mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"
        ) 200>"$LOCK"
        LSP_MISSING=1
    fi

    # Clean up legacy no-op entries if present
    if jq -e '.env.ENABLE_LSP_TOOL' "$SETTINGS_FILE" >/dev/null 2>&1; then
        LOCK="/home/vscode/.claude/.settings.lock"
        (
          flock -w 10 200 || exit 0
          jq 'if .env then .env |= del(.ENABLE_LSP_TOOL) else . end | del(.enabledPlugins)' \
            "$SETTINGS_FILE" > "${SETTINGS_FILE}.tmp" && mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"
        ) 200>"$LOCK"
        echo "  Cleaned up legacy ENABLE_LSP_TOOL entries"
    fi
fi

if [ "$LSP_MISSING" -eq 0 ]; then
    echo "  All LSP servers verified"
else
    echo "  LSP servers installed/configured (were missing)"
fi

# Trigger library auto-updates in background (non-blocking)
# Log file may be root-owned if docker-in-docker init ran first.
# Fall back to /tmp if /var/log/claudebox is not writable.
if [ -x /usr/local/bin/update-libraries.sh ]; then
    LIB_LOG="/var/log/claudebox/library-updates.log"
    # If file exists but isn't writable, remove it (1777 dir allows this)
    if [ -f "$LIB_LOG" ] && [ ! -w "$LIB_LOG" ]; then
        rm -f "$LIB_LOG" 2>/dev/null || LIB_LOG="/tmp/library-updates.log"
    fi
    touch "$LIB_LOG" 2>/dev/null || LIB_LOG="/tmp/library-updates.log"
    echo "Starting background library updates..."
    nohup /usr/local/bin/update-libraries.sh > "$LIB_LOG" 2>&1 &
fi
