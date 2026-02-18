#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──
# This script runs from inside the ClaudeBox dev container.
# SSH reaches TrueNAS via the host network.
WORKSPACE="/workspaces/vansedataadstackshit"
TRUENAS_HOST="${TRUENAS_HOST:-192.168.1.100}"
TRUENAS_USER="${TRUENAS_USER:-vanse}"
TRUENAS_PATH="/mnt/tank/vanse/app"
SSH_KEY="${SSH_KEY:-${WORKSPACE}/.ssh/id_ed25519}"

# Verify SSH key exists
if [ ! -f "${SSH_KEY}" ]; then
    echo "ERROR: SSH key not found at ${SSH_KEY}"
    echo "Run: ssh-keygen -t ed25519 -f ${SSH_KEY} -N ''"
    echo "Then: ssh-copy-id -i ${SSH_KEY}.pub ${TRUENAS_USER}@${TRUENAS_HOST}"
    exit 1
fi

echo "═══ Vanse Data Stack Deploy (from dev container) ═══"

# 1. Run tests inside dev container first
echo "→ Running tests..."
cd "${WORKSPACE}"
make test-unit
make lint

# 2. Sync project files to TrueNAS over LAN
echo "→ Syncing to TrueNAS at ${TRUENAS_HOST}..."
rsync -avz --delete \
    --exclude='.env' \
    --exclude='.ssh' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='docker-compose.dev.yml' \
    --exclude='vanse-dev-*' \
    --exclude='.pommel' \
    --exclude='.pommelignore' \
    --exclude='CLAUDEw.md' \
    --exclude='.claude' \
    --exclude='.devcontainer' \
    -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    "${WORKSPACE}/" "${TRUENAS_USER}@${TRUENAS_HOST}:${TRUENAS_PATH}/"

# 3. Check .env exists on remote
ssh -i "${SSH_KEY}" "${TRUENAS_USER}@${TRUENAS_HOST}" \
    "test -f ${TRUENAS_PATH}/.env || echo 'WARNING: .env missing on TrueNAS — copy .env.prod manually'"

# 4. Rebuild and restart containers on TrueNAS
echo "→ Rebuilding containers..."
ssh -i "${SSH_KEY}" "${TRUENAS_USER}@${TRUENAS_HOST}" \
    "cd ${TRUENAS_PATH} && docker compose -f docker-compose.prod.yml build --no-cache vanse_scrapers"

echo "→ Restarting services..."
ssh -i "${SSH_KEY}" "${TRUENAS_USER}@${TRUENAS_HOST}" \
    "cd ${TRUENAS_PATH} && docker compose -f docker-compose.prod.yml up -d"

# 5. Run migrations
echo "→ Running database migrations..."
ssh -i "${SSH_KEY}" "${TRUENAS_USER}@${TRUENAS_HOST}" \
    "cd ${TRUENAS_PATH} && docker compose -f docker-compose.prod.yml exec vanse_scrapers python -m database.migrate"

# 6. Health check
echo "→ Checking service health..."
ssh -i "${SSH_KEY}" "${TRUENAS_USER}@${TRUENAS_HOST}" \
    "cd ${TRUENAS_PATH} && docker compose -f docker-compose.prod.yml ps"

echo "═══ Deploy complete ═══"
