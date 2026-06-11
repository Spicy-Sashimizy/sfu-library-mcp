#!/usr/bin/env bash
# Restart the running 150M migration at the export->build boundary so the
# BUILD phase runs builder code committed AFTER the migration launched
# (meta v2 numeric schema, Cyrillic abstract dict). Export output is
# unaffected by those changes, so nothing already exported is wasted.
#
# Detached usage:  setsid nohup scripts/restart_at_build_phase.sh \
#                    > logs/migration_build_restart.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

STATUS=data/thinclient_index/build_status.json

while true; do
    phase=$(sed -n 's/.*"phase": "\([a-z]*\)".*/\1/p' "$STATUS" 2>/dev/null | head -1)
    if [ -n "$phase" ] && [ "$phase" != "export" ]; then
        break
    fi
    sleep 60
done

echo "$(date -Is) phase=$phase — restarting migration to load new builder code"
pid=$(pgrep -of "scripts/build_thinclient_inde[x]" || true)
if [ -n "$pid" ]; then
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    kill -TERM -"$pgid" 2>/dev/null
    sleep 10
    pkill -KILL -f "scripts/build_thinclient_inde[x]" 2>/dev/null || true
    sleep 2
fi

SFU_MIGRATION_SOURCE=http://host.docker.internal:9200 nohup .venv/bin/python3 \
    scripts/build_thinclient_index.py --persona political_science \
    --slices 8 --workers 4 --build-workers 3 \
    --index-root data/thinclient_index >> logs/migration_150m.log 2>&1 &
echo "$(date -Is) migration relaunched (pid $!) — resumes at phase=$phase"
