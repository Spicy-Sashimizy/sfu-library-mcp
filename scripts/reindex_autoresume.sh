#!/usr/bin/env bash
# ============================================================================
# reindex_autoresume.sh — keep the PARALLEL local SPLADE reindex alive.
#
# (Re)launches any missing/crashed workers via run_parallel_reindex.sh whenever
# they're not running and their subset isn't done. Runs under supervisor
# (autostart + autorestart=unexpected) so it survives container restarts and
# worker crashes. Exits 0 once ALL worker subsets are complete.
# ============================================================================
set -uo pipefail
REPO=/workspaces/sfu-library-mcp-training
OS_URL="${SFU_OPENSEARCH_URL:-http://claudebox-sfu-library-mcp-training-opensearch:9200}"
LOG(){ echo "$(date -u +%FT%TZ) autoresume: $*"; }
os_ok(){ curl -s --max-time 8 "$OS_URL/_cluster/health" >/dev/null 2>&1; }

LOG "parallel watchdog started (workers=${SFU_PARALLEL_WORKERS:-2})"
while true; do
  if bash "$REPO/scripts/run_parallel_reindex.sh" --check-done 2>/dev/null; then
    LOG "all worker subsets complete — exiting."; exit 0
  fi
  if os_ok; then
    bash "$REPO/scripts/run_parallel_reindex.sh" 2>&1 | sed 's/^/  /'
  else
    LOG "OpenSearch unreachable — waiting (no launch this cycle)"
  fi
  sleep 120
done
