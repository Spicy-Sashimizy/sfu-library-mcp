#!/bin/bash
# ClaudeBox Container Entrypoint Script
# This script runs on EVERY container start (not just first creation)
# It ensures SSH keys and Git authentication work properly

set -e

# ========================================
# Clear Invalid Environment Variables
# ========================================
# Remove invalid GITHUB_TOKEN that conflicts with gh CLI
# The valid OAuth token is in git-credentials file, not env vars
clear_invalid_tokens() {
    # Check if GITHUB_TOKEN is set and starts with ghp_ (Personal Access Token)
    # PATs often expire; OAuth tokens (gho_) in git-credentials are preferred
    if [ -n "$GITHUB_TOKEN" ]; then
        # Test if the token is valid
        if ! curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user 2>/dev/null | grep -q "200"; then
            echo "[entrypoint] Clearing invalid GITHUB_TOKEN from environment..."
            unset GITHUB_TOKEN
            # Also clear for vscode user's shell
            echo "unset GITHUB_TOKEN" >> /home/vscode/.bashrc 2>/dev/null || true
        fi
    fi
}

# ========================================
# SSH Key Permission Setup
# ========================================
# SSH keys mounted from Windows have wrong permissions
# Fix them on every start to ensure Git operations work

SSH_DIR="/home/vscode/.ssh"

fix_ssh_permissions() {
    if [ -d "$SSH_DIR" ]; then
        echo "[entrypoint] Fixing SSH key permissions..."

        # Fix ownership (run as root)
        chown -R vscode:vscode "$SSH_DIR" 2>/dev/null || true

        # Fix directory permission
        chmod 700 "$SSH_DIR" 2>/dev/null || true

        # Fix private key permissions
        chmod 600 "$SSH_DIR"/id_* 2>/dev/null || true
        chmod 600 "$SSH_DIR"/id_ed25519 2>/dev/null || true
        chmod 600 "$SSH_DIR"/id_rsa 2>/dev/null || true

        # Fix public key permissions
        chmod 644 "$SSH_DIR"/*.pub 2>/dev/null || true

        # Fix SSH config
        if [ -f "$SSH_DIR/config" ]; then
            chmod 600 "$SSH_DIR/config" 2>/dev/null || true
        fi

        # Fix known_hosts
        if [ -f "$SSH_DIR/known_hosts" ]; then
            chmod 644 "$SSH_DIR/known_hosts" 2>/dev/null || true
        fi

        # Add GitHub to known_hosts if not present (prevents host key verification prompts)
        if ! grep -q "github.com" "$SSH_DIR/known_hosts" 2>/dev/null; then
            echo "[entrypoint] Adding GitHub to known_hosts..."
            ssh-keyscan -t ed25519,rsa github.com >> "$SSH_DIR/known_hosts" 2>/dev/null || true
            chown vscode:vscode "$SSH_DIR/known_hosts" 2>/dev/null || true
        fi

        echo "[entrypoint] SSH permissions fixed"
    else
        echo "[entrypoint] WARNING: SSH directory not found"
    fi
}

# ========================================
# Git Configuration
# ========================================
configure_git() {
    echo "[entrypoint] Configuring Git..."

    # Always add safe directory for the workspace (for both root and vscode)
    if [ -n "$PROJECT_NAME" ]; then
        # Configure for root (entrypoint runs as root)
        git config --global --add safe.directory "/workspaces/$PROJECT_NAME" 2>/dev/null || true

        # Configure for vscode user
        su - vscode -c "git config --global --add safe.directory '/workspaces/$PROJECT_NAME'" 2>/dev/null || true
    fi

    # Fix git-crypt filter paths (Windows paths don't work in Linux container)
    git config --global filter.git-crypt.clean "git-crypt clean" 2>/dev/null || true
    git config --global filter.git-crypt.smudge "git-crypt smudge" 2>/dev/null || true
    git config --global filter.git-crypt.required false 2>/dev/null || true
    su - vscode -c "git config --global filter.git-crypt.clean 'git-crypt clean'" 2>/dev/null || true
    su - vscode -c "git config --global filter.git-crypt.smudge 'git-crypt smudge'" 2>/dev/null || true
    su - vscode -c "git config --global filter.git-crypt.required false" 2>/dev/null || true

    # LOCAL override: local .git/config beats global, so fix there too
    if [ -n "$PROJECT_NAME" ] && [ -d "/workspaces/$PROJECT_NAME/.git" ]; then
        local smudge_val
        smudge_val=$(git -C "/workspaces/$PROJECT_NAME" config --local --get filter.git-crypt.smudge 2>/dev/null || true)
        if echo "$smudge_val" | grep -q 'git-crypt\.exe' 2>/dev/null; then
            echo "[entrypoint] Fixing Windows git-crypt paths in local config..."
            git -C "/workspaces/$PROJECT_NAME" config --local filter.git-crypt.clean "git-crypt clean" 2>/dev/null || true
            git -C "/workspaces/$PROJECT_NAME" config --local filter.git-crypt.smudge "git-crypt smudge" 2>/dev/null || true
            git -C "/workspaces/$PROJECT_NAME" config --local filter.git-crypt.required false 2>/dev/null || true
            local diff_val
            diff_val=$(git -C "/workspaces/$PROJECT_NAME" config --local --get diff.git-crypt.textconv 2>/dev/null || true)
            if echo "$diff_val" | grep -q 'git-crypt\.exe' 2>/dev/null; then
                git -C "/workspaces/$PROJECT_NAME" config --local diff.git-crypt.textconv "git-crypt diff" 2>/dev/null || true
            fi
        fi

        # Enforce filter.required=true locally for repos that have git-crypt initialized
        # This prevents accidental plaintext commits of encrypted files
        if [ -d "/workspaces/$PROJECT_NAME/.git-crypt" ]; then
            git -C "/workspaces/$PROJECT_NAME" config --local filter.git-crypt.required true 2>/dev/null || true
            echo "[entrypoint] git-crypt filter.required=true (repo has .git-crypt)"
        fi
    fi

    # NOTE: We do NOT set URL rewriting (git@github.com instead of https)
    # HTTPS with OAuth credentials is the preferred method
    # SSH can still be used by explicitly using git@github.com URLs

    echo "[entrypoint] Git configuration complete"
}

# ========================================
# GitHub OAuth/HTTPS Credentials Setup
# ========================================
configure_git_credentials() {
    OAUTH_CREDS="/mnt/claudebox-git-creds/.git-credentials"

    if [ -f "$OAUTH_CREDS" ]; then
        echo "[entrypoint] Setting up Git HTTPS credentials..."

        # Copy to vscode user's home directory
        cp "$OAUTH_CREDS" /home/vscode/.git-credentials 2>/dev/null || true
        chown vscode:vscode /home/vscode/.git-credentials 2>/dev/null || true
        chmod 600 /home/vscode/.git-credentials 2>/dev/null || true

        # Configure git credential helper for BOTH root and vscode users
        git config --global credential.helper "store --file=/home/vscode/.git-credentials" 2>/dev/null || true
        su - vscode -c "git config --global credential.helper 'store --file=/home/vscode/.git-credentials'" 2>/dev/null || true

        # Verify the credentials file was copied
        if [ -f "/home/vscode/.git-credentials" ]; then
            echo "[entrypoint] Git HTTPS credentials configured successfully"
        else
            echo "[entrypoint] WARNING: Failed to copy credentials file"
        fi
    else
        echo "[entrypoint] No OAuth credentials found at $OAUTH_CREDS"
        echo "[entrypoint] Git will use SSH keys for authentication"
    fi
}

# ========================================
# GitHub CLI (gh) Configuration
# ========================================
configure_gh_cli() {
    OAUTH_CREDS="/mnt/claudebox-git-creds/.git-credentials"
    GH_CONFIG_DIR="/home/vscode/.config/gh"

    if [ -f "$OAUTH_CREDS" ]; then
        # Extract OAuth token from git-credentials
        TOKEN=$(grep -o 'gho_[a-zA-Z0-9]*' "$OAUTH_CREDS" 2>/dev/null || true)
        USERNAME=$(grep -oP '(?<=https://)[^:]+(?=:)' "$OAUTH_CREDS" 2>/dev/null || echo "unknown")

        if [ -n "$TOKEN" ] && [ "$TOKEN" != "" ]; then
            echo "[entrypoint] Configuring GitHub CLI (gh)..."

            # Create gh config directory
            mkdir -p "$GH_CONFIG_DIR"

            # Check if hosts.yml already has valid token
            if [ -f "$GH_CONFIG_DIR/hosts.yml" ]; then
                EXISTING_TOKEN=$(grep -o 'gho_[a-zA-Z0-9]*' "$GH_CONFIG_DIR/hosts.yml" 2>/dev/null || true)
                if [ "$EXISTING_TOKEN" = "$TOKEN" ]; then
                    echo "[entrypoint] gh CLI already configured with current token"
                    return
                fi
            fi

            # Write gh hosts.yml with OAuth token
            cat > "$GH_CONFIG_DIR/hosts.yml" << EOF
github.com:
    oauth_token: $TOKEN
    user: $USERNAME
    git_protocol: https
EOF

            # Write gh config.yml
            cat > "$GH_CONFIG_DIR/config.yml" << EOF
version: 1
git_protocol: https
editor:
prompt: enabled
pager:
EOF

            # Fix permissions
            chown -R vscode:vscode "$GH_CONFIG_DIR" 2>/dev/null || true
            chmod 700 "$GH_CONFIG_DIR" 2>/dev/null || true
            chmod 600 "$GH_CONFIG_DIR/hosts.yml" 2>/dev/null || true
            chmod 600 "$GH_CONFIG_DIR/config.yml" 2>/dev/null || true

            echo "[entrypoint] gh CLI configured successfully"
        else
            echo "[entrypoint] No OAuth token found for gh CLI"
        fi
    fi
}

# ========================================
# Test GitHub Connection
# ========================================
test_github_connection() {
    echo "[entrypoint] Testing GitHub connections..."

    # Test HTTPS first (preferred)
    if [ -f "/home/vscode/.git-credentials" ]; then
        echo "[entrypoint] Testing HTTPS authentication..."
        TOKEN=$(grep -o 'gho_[a-zA-Z0-9]*' /home/vscode/.git-credentials 2>/dev/null || true)
        if [ -n "$TOKEN" ]; then
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: token $TOKEN" https://api.github.com/user 2>/dev/null || echo "000")
            if [ "$HTTP_CODE" = "200" ]; then
                echo "[entrypoint] ✓ HTTPS authentication working"
            else
                echo "[entrypoint] ⚠ HTTPS token may be expired (HTTP $HTTP_CODE)"
            fi
        fi
    fi

    # Test SSH as fallback
    if [ -f "$SSH_DIR/id_ed25519" ] || [ -f "$SSH_DIR/id_rsa" ]; then
        echo "[entrypoint] Testing SSH authentication..."
        if timeout 10 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
            echo "[entrypoint] ✓ SSH authentication working"
        else
            echo "[entrypoint] ⚠ SSH authentication may need setup"
        fi
    fi

    # Test gh CLI
    if command -v gh >/dev/null 2>&1; then
        echo "[entrypoint] Testing gh CLI..."
        if su - vscode -c "gh auth status" >/dev/null 2>&1; then
            echo "[entrypoint] ✓ gh CLI authentication working"
        else
            echo "[entrypoint] ⚠ gh CLI may need setup"
        fi
    fi
}

# ========================================
# Claude Code Native Installer
# ========================================
# Installs or updates Claude Code using the native installer
# which supports built-in auto-updates. The binary is persisted
# in the claude-local volume (~/.local/bin/claude).
install_or_update_claude_code() {
    local CLAUDE_BIN="/home/vscode/.local/bin/claude"
    local VERSION_JSON="/usr/local/share/claude-version.json"

    # Ensure ~/.local/bin exists and is owned by vscode
    mkdir -p /home/vscode/.local/bin
    chown -R vscode:vscode /home/vscode/.local

    # Ensure ~/.claude is owned by vscode (shared volume may be root-owned)
    if [ -d /home/vscode/.claude ]; then
        chown -R vscode:vscode /home/vscode/.claude
    fi

    if [ -x "$CLAUDE_BIN" ]; then
        # Claude Code already installed (persisted from volume)
        local current_version
        current_version=$(su - vscode -c "$CLAUDE_BIN --version" 2>/dev/null || echo "unknown")
        echo "[entrypoint] Claude Code already installed: $current_version (native, auto-update enabled)"

        # Update version JSON
        cat > "$VERSION_JSON" << VEOF
{"version":"$current_version","installedAt":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","installerType":"native","autoUpdate":true}
VEOF
    else
        # Install via native installer as vscode user
        echo "[entrypoint] Installing Claude Code via native installer..."

        if su - vscode -c "curl -fsSL https://claude.ai/install.sh | bash" 2>&1; then
            local installed_version
            installed_version=$(su - vscode -c "$CLAUDE_BIN --version" 2>/dev/null || echo "unknown")
            echo "[entrypoint] Claude Code installed: $installed_version (native installer)"

            cat > "$VERSION_JSON" << VEOF
{"version":"$installed_version","installedAt":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","installerType":"native","autoUpdate":true}
VEOF
        else
            echo "[entrypoint] Native installer failed, falling back to npm..."
            npm install -g @anthropic-ai/claude-code 2>&1 || echo "[entrypoint] npm install also failed"

            local npm_version
            npm_version=$(claude --version 2>/dev/null || echo "unknown")
            cat > "$VERSION_JSON" << VEOF
{"version":"$npm_version","installedAt":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","installerType":"npm","autoUpdate":false}
VEOF
            echo "[entrypoint] Claude Code installed via npm fallback: $npm_version"
        fi
    fi

    # Ensure ~/.local/bin is in PATH for vscode user
    if ! grep -q '\.local/bin' /home/vscode/.bashrc 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/vscode/.bashrc
        chown vscode:vscode /home/vscode/.bashrc
    fi

    # Also add to profile for login shells
    if ! grep -q '\.local/bin' /home/vscode/.profile 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/vscode/.profile 2>/dev/null || true
        chown vscode:vscode /home/vscode/.profile 2>/dev/null || true
    fi
}

# ========================================
# Socat Ollama Proxy (for Pommel)
# ========================================
start_ollama_proxy() {
    # Ollama proxy lifecycle is managed by supervisord (program:ollama-proxy)
    # with hardened timeouts (-T30, keepalive) and health monitoring.
    # This function is a no-op; kept for compatibility.
    echo "[entrypoint] Ollama proxy managed by supervisord (skipping manual start)"
}

# ========================================
# Public Branch Scripts CRLF Check
# ========================================
# Volume-mounted scripts from Windows host may have CRLF line endings.
# Source scripts should already be LF (.gitattributes enforces this),
# but this warns if CRLF is somehow present on read-only mounts.
fix_public_branch_scripts() {
    for script in generate-public-branch.sh public-branch-helper.py public-branch-prompts.sh; do
        if [ -f "/usr/local/bin/$script" ] && head -1 "/usr/local/bin/$script" | grep -qP '\r$'; then
            echo "[entrypoint] WARNING: /usr/local/bin/$script has CRLF line endings — generation will fail"
            echo "[entrypoint] WARNING: Fix the source file on the Windows host: sed -i 's/\\r\$//' <file>"
        fi
    done
}

# ========================================
# Start Supervisord
# ========================================
# Ensures background services run even without VS Code (postStartCommand).
# start-services.sh has "already running" guards, so this is safe to call first.
start_supervisord() {
    if ! command -v supervisord &>/dev/null; then
        echo "[entrypoint] supervisord not installed, skipping"
        return
    fi

    # Clean stale files from unclean shutdown
    if [ -f /tmp/supervisord.pid ]; then
        OLD_PID=$(cat /tmp/supervisord.pid 2>/dev/null)
        if [ -n "$OLD_PID" ] && ! kill -0 "$OLD_PID" 2>/dev/null; then
            echo "[entrypoint] Cleaning stale supervisor files (PID $OLD_PID dead)..."
            rm -f /tmp/supervisor.sock /tmp/supervisord.pid
        fi
    fi

    if pgrep -x supervisord > /dev/null 2>&1; then
        echo "[entrypoint] supervisord already running"
    else
        echo "[entrypoint] Starting supervisord..."
        supervisord -c /etc/supervisor/supervisord.conf 2>&1 || true
        sleep 1
        if pgrep -x supervisord > /dev/null 2>&1; then
            echo "[entrypoint] supervisord started successfully"
        else
            echo "[entrypoint] WARNING: supervisord failed to start"
        fi
    fi
}

# ========================================
# Run All Setup Steps
# ========================================
main() {
    echo "[entrypoint] ClaudeBox container starting..."
    echo "[entrypoint] Project: ${PROJECT_NAME:-unknown}"

    # Check for CRLF in public branch scripts early
    fix_public_branch_scripts

    # Clear invalid tokens first (before any git operations)
    clear_invalid_tokens

    # Run as root for permission fixes
    fix_ssh_permissions
    configure_git
    configure_git_credentials
    configure_gh_cli
    install_or_update_claude_code
    start_ollama_proxy

    # Test connection (non-blocking)
    test_github_connection || true

    # Start supervisord if not already running
    # This ensures background services (LSP healthcheck, Pommel, ollama proxy)
    # work even when the container starts without VS Code (e.g., docker compose up).
    # start-services.sh (postStartCommand) has guards for "already running" so
    # starting here won't conflict when VS Code attaches later.
    start_supervisord

    echo "[entrypoint] Startup complete"
    echo ""

    # Execute the original command (sleep infinity or whatever)
    exec "$@"
}

main "$@"
