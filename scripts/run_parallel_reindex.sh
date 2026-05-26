#!/usr/bin/env bash
# ============================================================================
# run_parallel_reindex.sh — launch/maintain N GPU-sharing SPLADE reindex workers
# over the REMAINING shards (the single-worker run is single-thread-CPU-bound, so
# 2 workers saturate the GPU). Each worker gets a disjoint round-robin subset in
# its own dir (own checkpoint + PID lock); all upsert to the same index by _id
# (idempotent). Idempotent: skips workers already running or already complete.
#
#   bash scripts/run_parallel_reindex.sh              # launch/ensure workers
#   bash scripts/run_parallel_reindex.sh --check-done # exit 0 iff ALL subsets done
# ============================================================================
set -uo pipefail
REPO=/workspaces/sfu-library-mcp-training
SNAP="$REPO/data/openalex_snapshot"
MODEL="$REPO/models/sfu-splade-v1"
OS="${SFU_OPENSEARCH_URL:-http://claudebox-sfu-library-mcp-training-opensearch:9200}"
PY="$REPO/.venv/bin/python3"
N="${SFU_PARALLEL_WORKERS:-2}"
FIRST_REMAINING="${SFU_FIRST_REMAINING:-47}"   # files numbered < this are already done by the single run
LOGD="$REPO/logs"; mkdir -p "$LOGD"

# (re)build per-worker subset symlink dirs from the remaining top-level shards.
for ((w=0; w<N; w++)); do mkdir -p "$SNAP/pw_$w"; done
i=0
for f in $(ls "$SNAP"/works_part_*.jsonl.gz 2>/dev/null | sort); do
  base=$(basename "$f"); num=${base//[!0-9]/}; num=$((10#$num))
  (( num < FIRST_REMAINING )) && continue
  w=$(( i % N )); ln -sf "$f" "$SNAP/pw_$w/$base"; i=$((i+1))
done

subset_count(){ ls "$SNAP/pw_$1"/works_part_*.jsonl.gz 2>/dev/null | wc -l; }
subset_done(){
  local cp="$SNAP/pw_$1/indexer_checkpoint.json" cnt; cnt=$(subset_count "$1")
  [ "$cnt" -gt 0 ] || return 0          # empty subset = nothing to do = "done"
  [ -f "$cp" ] || return 1
  CNT="$cnt" python3 -c "import json,os,sys;d=json.load(open('$cp'));sys.exit(0 if d.get('file_index',0)>=int(os.environ['CNT']) else 1)" 2>/dev/null
}
worker_running(){ pgrep -f "splade_indexer.py --input $SNAP/pw_$1 " >/dev/null 2>&1; }

# --check-done: exit 0 only if every worker's subset is complete
if [[ "${1:-}" == "--check-done" ]]; then
  for ((w=0; w<N; w++)); do subset_done "$w" || exit 1; done
  exit 0
fi

for ((w=0; w<N; w++)); do
  cnt=$(subset_count "$w")
  (( cnt == 0 )) && { echo "worker $w: no files"; continue; }
  worker_running "$w" && { echo "worker $w: already running ($cnt files)"; continue; }
  subset_done  "$w" && { echo "worker $w: subset complete ($cnt files)"; continue; }
  echo "worker $w: launching on $cnt files"
  SFU_PROGRESS_PUSH=off setsid nohup "$PY" "$REPO/scripts/splade_indexer.py" \
    --input "$SNAP/pw_$w" --model "$MODEL" --resume --device cuda --backend auto \
    --batch-size 64 --opensearch-url "$OS" --index openalex_works \
    >> "$LOGD/reindex_worker_$w.log" 2>&1 &
  sleep 8   # stagger model/TRT-engine load
done
