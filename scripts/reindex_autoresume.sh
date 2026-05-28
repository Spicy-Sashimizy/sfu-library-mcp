#!/usr/bin/env bash
# ============================================================================
# reindex_autoresume.sh — keep the local SPLADE reindex alive.
#
# Relaunches scripts/run_splade_pipeline_local.sh whenever the indexer is NOT
# running AND the reindex stage is not yet "done". Runs under supervisor
# (autostart + autorestart) so it survives container restarts and indexer
# crashes. Exits 0 once reindex is done (supervisor: autorestart=unexpected,
# so a clean exit is not restarted).
#
# Install (already done):
#   sudo cp deploy/reindex_autoresume.supervisor.conf /etc/supervisor/conf.d/
#   sudo supervisorctl reread && sudo supervisorctl update
# ============================================================================
set -uo pipefail
REPO=/workspaces/sfu-library-mcp-training
STATE="$REPO/data/training/splade_pipeline_state.json"
OS_URL="${SFU_OPENSEARCH_URL:-http://claudebox-sfu-library-mcp-training-opensearch:9200}"
LOG(){ echo "$(date -u +%FT%TZ) autoresume: $*"; }

reindex_done(){
  python3 - "$STATE" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)   # no state yet -> not done
sys.exit(0 if d.get("stages",{}).get("reindex",{}).get("status")=="done" else 1)
PY
}
indexer_running(){ pgrep -f "splade_indexer.py --model" >/dev/null 2>&1; }
pipeline_running(){ pgrep -f 'run_splade_pipeline_local.sh' >/dev/null 2>&1; }
os_ok(){ curl -s --max-time 8 "$OS_URL/_cluster/health" >/dev/null 2>&1; }

LOG "watchdog started (repo=$REPO)"
while true; do
  if reindex_done; then LOG "reindex stage = done — nothing to keep alive, exiting."; exit 0; fi
  if indexer_running || pipeline_running; then
    :  # healthy — already encoding
  elif os_ok; then
    LOG "indexer not running + reindex not done -> launching pipeline"
    ( cd "$REPO" && setsid nohup bash scripts/run_splade_pipeline_local.sh >> logs/pipeline_stdout.log 2>&1 & )
    sleep 60   # let it acquire the PID lock before re-checking
  else
    LOG "indexer down but OpenSearch unreachable -> waiting (no fail-loop)"
  fi
  sleep 120
done
