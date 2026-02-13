#!/bin/bash
# ClaudeBox Library Auto-Updater
# Runs once on container start to apply safe patch-only updates
# Logs to /var/log/claudebox/library-updates.log

set -euo pipefail

LOG_DIR="/var/log/claudebox"
LOG_FILE="$LOG_DIR/library-updates.log"
LOCK_FILE="/tmp/library-update.lock"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1" | tee -a "$LOG_FILE"
}

# Prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "Another library update is already running (PID $pid), skipping"
        exit 0
    fi
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

log "=== Library Auto-Update Starting ==="
log "Project: ${PROJECT_NAME:-unknown}"

UPDATES_APPLIED=0

# ========================================
# Python Updates (patch-only)
# ========================================
update_python() {
    if ! command -v pip3 &>/dev/null && ! command -v pip &>/dev/null; then
        return
    fi

    local PIP_CMD="pip3"
    command -v pip3 &>/dev/null || PIP_CMD="pip"

    log "[Python] Checking for patch-level updates..."

    # Get outdated packages as JSON
    local outdated
    outdated=$($PIP_CMD list --outdated --format=json 2>/dev/null || echo "[]")

    if [ "$outdated" = "[]" ] || [ -z "$outdated" ]; then
        log "[Python] All packages up to date"
        return
    fi

    # Parse JSON and apply patch-only updates
    echo "$outdated" | python3 -c "
import json, sys
packages = json.load(sys.stdin)
for pkg in packages:
    name = pkg['name']
    current = pkg['version']
    latest = pkg['latest_version']
    # Parse versions
    cur_parts = current.split('.')
    lat_parts = latest.split('.')
    if len(cur_parts) >= 3 and len(lat_parts) >= 3:
        # Only update if major.minor match (patch-only)
        if cur_parts[0] == lat_parts[0] and cur_parts[1] == lat_parts[1]:
            print(f'{name}=={latest}')
" 2>/dev/null | while read -r pkg_spec; do
        if [ -n "$pkg_spec" ]; then
            log "[Python] Updating: $pkg_spec"
            $PIP_CMD install --quiet --break-system-packages "$pkg_spec" 2>/dev/null && UPDATES_APPLIED=$((UPDATES_APPLIED + 1)) || log "[Python] Failed to update $pkg_spec"
        fi
    done

    log "[Python] Patch updates complete"
}

# ========================================
# Node.js Updates (patch-only)
# ========================================
update_node() {
    if ! command -v npm &>/dev/null; then
        return
    fi

    # Only update if package.json exists in workspace
    local workspace="/workspaces/${PROJECT_NAME:-}"
    if [ ! -f "$workspace/package.json" ]; then
        return
    fi

    log "[Node] Checking for patch-level updates in $workspace..."

    cd "$workspace" || return

    # Get outdated packages
    local outdated
    outdated=$(npm outdated --json 2>/dev/null || echo "{}")

    if [ "$outdated" = "{}" ] || [ -z "$outdated" ]; then
        log "[Node] All packages up to date"
        return
    fi

    # Parse and apply patch-only updates
    echo "$outdated" | node -e "
const data = require('fs').readFileSync('/dev/stdin', 'utf8');
const packages = JSON.parse(data);
for (const [name, info] of Object.entries(packages)) {
    const current = (info.current || '').split('.');
    const wanted = (info.wanted || '').split('.');
    if (current.length >= 3 && wanted.length >= 3) {
        if (current[0] === wanted[0] && current[1] === wanted[1] && current[2] !== wanted[2]) {
            console.log(name);
        }
    }
}
" 2>/dev/null | while read -r pkg_name; do
        if [ -n "$pkg_name" ]; then
            log "[Node] Updating: $pkg_name"
            npm update "$pkg_name" --save 2>/dev/null && UPDATES_APPLIED=$((UPDATES_APPLIED + 1)) || log "[Node] Failed to update $pkg_name"
        fi
    done

    log "[Node] Patch updates complete"
}

# ========================================
# Go Updates (patch-only)
# ========================================
update_go() {
    if ! command -v go &>/dev/null; then
        return
    fi

    local workspace="/workspaces/${PROJECT_NAME:-}"
    if [ ! -f "$workspace/go.mod" ]; then
        return
    fi

    log "[Go] Checking for patch-level updates..."

    cd "$workspace" || return

    go get -u=patch ./... 2>/dev/null && log "[Go] Patch updates applied" || log "[Go] No updates or update failed"
    go mod tidy 2>/dev/null || true

    log "[Go] Updates complete"
}

# ========================================
# Rust Updates (compatible)
# ========================================
update_rust() {
    if ! command -v cargo &>/dev/null; then
        return
    fi

    local workspace="/workspaces/${PROJECT_NAME:-}"
    if [ ! -f "$workspace/Cargo.toml" ]; then
        return
    fi

    log "[Rust] Checking for compatible updates..."

    cd "$workspace" || return

    cargo update 2>/dev/null && log "[Rust] Lock file updated" || log "[Rust] No updates or update failed"

    log "[Rust] Updates complete"
}

# Run all updaters
update_python
update_node
update_go
update_rust

log "=== Library Auto-Update Complete (${UPDATES_APPLIED} updates applied) ==="
