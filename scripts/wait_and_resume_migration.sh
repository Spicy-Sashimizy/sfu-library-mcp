#!/usr/bin/env bash
# Polls the source OpenSearch cluster and launches the 150M thin-client
# migration as soon as openalex_works is fully back (count stable >= 150M
# on two consecutive checks, so we don't export from a still-recovering index).
#
# Detached usage:  setsid nohup scripts/wait_and_resume_migration.sh \
#                    > logs/migration_watcher.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

SOURCE="${SFU_MIGRATION_SOURCE:-http://host.docker.internal:9200}"
INDEX="${SFU_MIGRATION_INDEX:-openalex_works}"
MIN_DOCS=150000000
POLL_SECS=60

if pgrep -f "build_thinclient_index.py" >/dev/null; then
    echo "$(date -Is) migration already running — exiting"
    exit 0
fi

prev=-1
while true; do
    count=$(curl -s -m 10 "$SOURCE/$INDEX/_count" | sed -n 's/.*"count":\([0-9]*\).*/\1/p')
    count=${count:-0}
    echo "$(date -Is) $SOURCE/$INDEX count=$count (prev=$prev)"
    if [ "$count" -ge "$MIN_DOCS" ] && [ "$count" -eq "$prev" ]; then
        break
    fi
    prev=$count
    sleep "$POLL_SECS"
done

echo "$(date -Is) source ready ($count docs) — launching migration"
SFU_MIGRATION_SOURCE="$SOURCE" nohup .venv/bin/python3 \
    scripts/build_thinclient_index.py --persona political_science \
    --slices 8 --workers 4 --build-workers 3 \
    --index-root data/thinclient_index > logs/migration_150m.log 2>&1 &
echo "$(date -Is) migration launched (pid $!) — tail logs/migration_150m.log"
