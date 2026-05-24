#!/usr/bin/env bash
# ============================================================================
# run_splade_pipeline_local.sh — fully-LOCAL, hardened, resumable orchestrator
# for the end-to-end SPLADE fine-tune pipeline on an RTX 4070 Ti SUPER (16 GB)
# + local OpenSearch. NO cloud, no doctl, no Azure.
#
# Stages (each resumable + skip-if-already-done via an atomic JSON state file):
#   1. mine       Mine hard negatives (splade + bm25f legs) -> merge      ~5-10m
#   2. triplets   Build query-doc triplets (atomic write)                 ~2-3m
#   3. finetune   SPLADE fine-tune, 16 GB-safe config + CUDA-OOM backoff  ~1-2h
#   4. reindex    Re-encode the 150M index in place (upsert by _id)       ~2.4h
#   5. benchmark  NDCG@10 new SPLADE leg vs old on the judged set         ~few min
#
# Hardening:
#   * Stage-level resume   — scripts/pipeline_state.py (atomic temp->os.replace)
#   * 10 h wall-clock budget watchdog — graceful stop between/within stages
#   * Per-stage verification gates — fail loud, never silently advance
#   * Heartbeat/progress logging to logs/ (atomic-append)
#   * --dry-run prints the full plan (stages, configs, timings, budget)
#
# Usage:
#   bash scripts/run_splade_pipeline_local.sh --dry-run
#   bash scripts/run_splade_pipeline_local.sh                 # full run
#   bash scripts/run_splade_pipeline_local.sh --max-hours 9.5 # custom budget
#   bash scripts/run_splade_pipeline_local.sh --from finetune # force-start stage
#
# Exit codes:
#   0   pipeline complete (or dry-run)
#   2   wall-clock budget hit — checkpointed; re-run to resume
#   3+  a stage or verification gate failed (fail loud)
# ============================================================================
set -uo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
STATE_HELPER="$SCRIPT_DIR/pipeline_state.py"
TRAIN_DIR="$REPO_ROOT/data/training"
STATE_FILE="$TRAIN_DIR/splade_pipeline_state.json"
LOG_DIR="$REPO_ROOT/logs"
RUN_LOG="$LOG_DIR/splade_pipeline_run.log"
DOC_BASELINE="$TRAIN_DIR/reindex_baseline_doc.json"

# ── Config / data files ───────────────────────────────────────────────────--
QUERIES="$REPO_ROOT/data/sfu_eval_queries.json"
JUDGE_CACHE="$REPO_ROOT/data/eval_results/llm_judge_cache.json"
NEG_SPLADE="$TRAIN_DIR/hard_negatives_splade.jsonl"
NEG_BM25="$TRAIN_DIR/hard_negatives_bm25.jsonl"
NEG_POOL="$TRAIN_DIR/hard_negatives_rrf_pool.jsonl"
TRIPLETS="$TRAIN_DIR/hard_negatives_triplets.jsonl"
MODEL_OUT="$REPO_ROOT/models/sfu-splade-v1"
BASE_MODEL="naver/splade-cocondenser-ensembledistil"

OPENSEARCH_URL="${SFU_OPENSEARCH_URL:-http://claudebox-sfu-library-mcp-training-opensearch:9200}"
INDEX="${SFU_OPENSEARCH_INDEX:-openalex_works}"
MIN_DOC_COUNT="${MIN_DOC_COUNT:-150000000}"   # 150M; index has ~150.4M

# ── Defaults (overridable by flags) ──────────────────────────────────────────
MAX_HOURS="9.5"        # soft budget; ceiling 10
DRY_RUN=0
FORCE_FROM=""          # mine|triplets|finetune|reindex|benchmark
FRESH_HOURS="24"       # treat a stage output newer than this as "fresh" -> skip

# Fine-tune 16 GB-safe config
FT_BATCH=2
FT_ACCUM=16
FT_SEQLEN=128
FT_EPOCHS=3
FT_CKPT_EVERY=15       # minutes

# Offline HF
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SFU_OPENSEARCH_URL="$OPENSEARCH_URL"
export SFU_OPENSEARCH_INDEX="$INDEX"
# Helps the allocator avoid fragmentation OOMs on a tight 16 GB card.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

START_EPOCH=$(date +%s)

# ── Arg parsing ───────────────────────────────────────────────────────────---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)     DRY_RUN=1; shift ;;
    --max-hours)   MAX_HOURS="$2"; shift 2 ;;
    --from)        FORCE_FROM="$2"; shift 2 ;;
    --fresh-hours) FRESH_HOURS="$2"; shift 2 ;;
    --model-out)   MODEL_OUT="$2"; shift 2 ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 64 ;;
  esac
done

# Clamp the budget to a hard 10 h ceiling.
MAX_SECONDS=$(awk -v h="$MAX_HOURS" 'BEGIN{ s=h*3600; c=10*3600; print (s>c)?c:s }')

mkdir -p "$LOG_DIR" "$TRAIN_DIR"

# ── Heartbeat / progress logging (atomic-append) ──────────────────────────────
hb() {
  # Atomic-append: a single >> write of one line is atomic for small lines on
  # local fs. Prefix with elapsed so progress is readable after a crash.
  local now elapsed
  now=$(date +%s); elapsed=$(( now - START_EPOCH ))
  printf '%s [+%02d:%02d:%02d] %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) \
    "$*" >> "$RUN_LOG"
  echo ">> $*"
}

# ── Wall-clock budget watchdog ────────────────────────────────────────────────
budget_left() { echo $(( MAX_SECONDS - ( $(date +%s) - START_EPOCH ) )); }

# Check BEFORE starting a stage: if the est. stage time won't fit, stop cleanly.
# We never kill a stage mid-flight; finetune/reindex have their own internal
# checkpoint+resume, and we additionally pass finetune a soft --max-runtime-min
# clamped to the remaining budget so it checkpoints and exits before we'd cut it.
check_budget_or_stop() {
  local stage="$1" est_min="$2" left
  left=$(budget_left)
  if (( left <= 0 )); then
    hb "BUDGET EXHAUSTED before stage '$stage' (${MAX_HOURS}h). Checkpointed state; re-run to resume."
    exit 2
  fi
  local left_min=$(( left / 60 ))
  if (( left_min < est_min )); then
    hb "BUDGET: only ${left_min}m left, stage '$stage' needs ~${est_min}m. Stopping cleanly; re-run to resume from '$stage'."
    exit 2
  fi
  hb "BUDGET: ${left_min}m remaining; proceeding with stage '$stage' (~${est_min}m est)."
}

# ── Freshness helpers ─────────────────────────────────────────────────────────
file_fresh() {
  # $1 path, $2 hours. True if file exists and mtime within N hours.
  local f="$1" hrs="$2"
  [[ -s "$f" ]] || return 1
  local age_s now mt
  now=$(date +%s); mt=$(stat -c %Y "$f" 2>/dev/null || echo 0)
  age_s=$(( now - mt ))
  (( age_s <= hrs * 3600 ))
}

stage_done() { "$PY" "$STATE_HELPER" --state "$STATE_FILE" is-done "$1" >/dev/null 2>&1; }
mark()       { "$PY" "$STATE_HELPER" --state "$STATE_FILE" set "$1" "$2" ${3:+--config "$3"} >/dev/null; }

# should_run STAGE  -> 0 (run) / 1 (skip). Respects --from override.
should_run() {
  local stage="$1"
  if [[ -n "$FORCE_FROM" ]]; then
    # Run this stage and every stage at/after FORCE_FROM in pipeline order.
    local order=(mine triplets finetune reindex benchmark) i from_i this_i
    for i in "${!order[@]}"; do
      [[ "${order[$i]}" == "$FORCE_FROM" ]] && from_i=$i
      [[ "${order[$i]}" == "$stage" ]] && this_i=$i
    done
    (( this_i >= from_i )) && return 0 || return 1
  fi
  stage_done "$stage" && return 1 || return 0
}

# ============================================================================
# DRY RUN
# ============================================================================
print_plan() {
  cat <<EOF
════════════════════════════════════════════════════════════════════════════
 LOCAL SPLADE pipeline — DRY RUN plan
════════════════════════════════════════════════════════════════════════════
 venv         : $PY
 state file   : $STATE_FILE  (atomic temp->os.replace; stage-level resume)
 run log      : $RUN_LOG  (heartbeat/progress, atomic-append)
 OpenSearch   : $OPENSEARCH_URL / $INDEX  (min sane count: $MIN_DOC_COUNT)
 model out    : $MODEL_OUT
 base model   : $BASE_MODEL
 wall budget  : ${MAX_HOURS}h  (hard ceiling 10h; watchdog stops cleanly + resumes)
 offline      : HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE
 alloc conf   : PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
────────────────────────────────────────────────────────────────────────────
 STAGES (skip if state=done OR output fresh < ${FRESH_HOURS}h):
   1. mine       splade+bm25f legs -> merge_negatives -> $NEG_POOL          ~5-10 min
   2. triplets   build_ce_triplets -> $TRIPLETS  (atomic write)            ~2-3  min
   3. finetune   finetune_splade.py --batch-size $FT_BATCH --grad-accum $FT_ACCUM \\
                   --max-seq-length $FT_SEQLEN --grad-checkpoint --oom-backoff
                   --epochs $FT_EPOCHS  -> $MODEL_OUT                       ~60-120 min
                 GATE: model loads + emits a valid sparse vector.
   4. reindex    splade_indexer.py --model $MODEL_OUT --resume (upsert by _id,
                   IN-PLACE; no staging copy — disk can't hold a 2nd index)  ~140 min
                 GATE: doc count >= $MIN_DOC_COUNT AND a spot-doc's
                       sparse_field changed vs a pre-reindex baseline.
   5. benchmark  ndcg_splade_eval.py -> data/eval_results/                  ~3-5  min
────────────────────────────────────────────────────────────────────────────
 EST TOTAL (cold): ~10+3+90+140+5 = ~248 min  (~4.1 h)  < 10 h budget. OK.
 Worst plausible (finetune 120m, reindex 160m): ~300 min (~5.0 h). < 10 h. OK.
════════════════════════════════════════════════════════════════════════════
EOF
  echo "── current state ──"
  "$PY" "$STATE_HELPER" --state "$STATE_FILE" show 2>/dev/null || echo "(no state file yet)"
  echo
  echo "── finetune stage --dry-run (sub-plan) ──"
  "$PY" "$SCRIPT_DIR/finetune_splade.py" --dry-run \
      --base-model "$BASE_MODEL" --train-data "$TRIPLETS" --output "$MODEL_OUT" \
      --batch-size "$FT_BATCH" --grad-accum "$FT_ACCUM" \
      --max-seq-length "$FT_SEQLEN" --grad-checkpoint --epochs "$FT_EPOCHS" 2>&1 \
    | sed 's/^/   /'
}

if (( DRY_RUN )); then
  print_plan
  exit 0
fi

# ============================================================================
# REAL RUN
# ============================================================================
"$PY" "$STATE_HELPER" --state "$STATE_FILE" init >/dev/null
hb "=== LOCAL SPLADE pipeline START (budget ${MAX_HOURS}h, ceiling 10h) ==="
hb "state=$STATE_FILE  model_out=$MODEL_OUT  os=$OPENSEARCH_URL/$INDEX"

# ── Stage 1: mine hard negatives ──────────────────────────────────────────────
if should_run mine; then
  if file_fresh "$NEG_POOL" "$FRESH_HOURS" && file_fresh "$NEG_SPLADE" "$FRESH_HOURS" \
     && file_fresh "$NEG_BM25" "$FRESH_HOURS"; then
    hb "STAGE 1 mine: outputs fresh (< ${FRESH_HOURS}h) — skipping; marking done."
    mark mine done
  else
    check_budget_or_stop mine 15
    mark mine running
    hb "STAGE 1 mine: SPLADE leg..."
    "$PY" "$SCRIPT_DIR/mine_hard_negatives.py" --queries "$QUERIES" \
        --retriever splade --top-k 50 --output "$NEG_SPLADE" \
        >>"$RUN_LOG" 2>&1 || { hb "STAGE 1 FAIL (splade leg)"; mark mine failed; exit 3; }
    hb "STAGE 1 mine: BM25F leg..."
    "$PY" "$SCRIPT_DIR/mine_hard_negatives.py" --queries "$QUERIES" \
        --retriever bm25f --top-k 50 --output "$NEG_BM25" \
        >>"$RUN_LOG" 2>&1 || { hb "STAGE 1 FAIL (bm25f leg)"; mark mine failed; exit 3; }
    hb "STAGE 1 mine: merge legs -> $NEG_POOL"
    "$PY" "$SCRIPT_DIR/merge_negatives.py" \
        --inputs "$NEG_SPLADE" "$NEG_BM25" \
        --judge-cache "$JUDGE_CACHE" --min-judge-score 2 \
        --output "$NEG_POOL" \
        >>"$RUN_LOG" 2>&1 || { hb "STAGE 1 FAIL (merge)"; mark mine failed; exit 3; }
    [[ -s "$NEG_POOL" ]] || { hb "STAGE 1 GATE FAIL: empty $NEG_POOL"; mark mine failed; exit 3; }
    mark mine done
    hb "STAGE 1 mine: DONE ($(wc -l < "$NEG_POOL") pooled negatives)."
  fi
else
  hb "STAGE 1 mine: already done — skip."
fi

# ── Stage 2: build triplets ───────────────────────────────────────────────────
if should_run triplets; then
  if file_fresh "$TRIPLETS" "$FRESH_HOURS"; then
    hb "STAGE 2 triplets: $TRIPLETS fresh — skipping; marking done."
    mark triplets done
  else
    check_budget_or_stop triplets 5
    mark triplets running
    hb "STAGE 2 triplets: build_ce_triplets -> $TRIPLETS"
    "$PY" "$SCRIPT_DIR/build_ce_triplets.py" \
        --judge-cache "$JUDGE_CACHE" --neg-pool "$NEG_POOL" \
        --output "$TRIPLETS" --max-neg-per-pos 5 \
        >>"$RUN_LOG" 2>&1 || { hb "STAGE 2 FAIL"; mark triplets failed; exit 4; }
    [[ -s "$TRIPLETS" ]] || { hb "STAGE 2 GATE FAIL: empty $TRIPLETS"; mark triplets failed; exit 4; }
    mark triplets done
    hb "STAGE 2 triplets: DONE ($(wc -l < "$TRIPLETS") triplets)."
  fi
else
  hb "STAGE 2 triplets: already done — skip."
fi

# ── Stage 3: SPLADE fine-tune (the risky 16 GB stage) ─────────────────────────
if should_run finetune; then
  check_budget_or_stop finetune 60
  mark finetune running
  # Clamp finetune's OWN soft runtime budget to whatever wall-clock is left
  # (minus a 25-min cushion for re-encode handoff), so it checkpoints+exits
  # before the orchestrator's hard deadline rather than being cut mid-step.
  LEFT_MIN=$(( $(budget_left) / 60 ))
  FT_RUNTIME=$(( LEFT_MIN - 25 )); (( FT_RUNTIME < 5 )) && FT_RUNTIME=5
  RESUME_ARGS=()
  if [[ -d "$MODEL_OUT/checkpoints/latest" || -f "$MODEL_OUT/checkpoints/latest.txt" ]]; then
    hb "STAGE 3 finetune: found checkpoint — resuming."
    RESUME_ARGS=(--resume-from "$MODEL_OUT/checkpoints/latest")
  fi
  hb "STAGE 3 finetune: batch=$FT_BATCH accum=$FT_ACCUM seq=$FT_SEQLEN grad-ckpt oom-backoff; soft budget ${FT_RUNTIME}m"
  "$PY" "$SCRIPT_DIR/finetune_splade.py" \
      --base-model "$BASE_MODEL" --train-data "$TRIPLETS" --output "$MODEL_OUT" \
      --batch-size "$FT_BATCH" --grad-accum "$FT_ACCUM" \
      --max-seq-length "$FT_SEQLEN" --grad-checkpoint --oom-backoff \
      --epochs "$FT_EPOCHS" --checkpoint-every-min "$FT_CKPT_EVERY" \
      --max-runtime-min "$FT_RUNTIME" "${RESUME_ARGS[@]}" \
      >>"$RUN_LOG" 2>&1
  FT_RC=$?
  if (( FT_RC != 0 )); then
    hb "STAGE 3 finetune: exited rc=$FT_RC (likely budget/error) — checkpoint left; re-run to resume."
    mark finetune running
    exit 2
  fi
  # GATE: model loads + emits a valid sparse vector.
  hb "STAGE 3 GATE: verifying $MODEL_OUT loads + emits a sparse vector..."
  if ! "$PY" "$STATE_HELPER" --state "$STATE_FILE" verify-finetune "$MODEL_OUT" >>"$RUN_LOG" 2>&1; then
    hb "STAGE 3 GATE FAIL: model did not verify. Stopping (fail loud)."
    mark finetune failed
    exit 5
  fi
  CFG=$([[ -f "$MODEL_OUT/oom_backoff_result.json" ]] && cat "$MODEL_OUT/oom_backoff_result.json" || echo '{}')
  "$PY" "$STATE_HELPER" --state "$STATE_FILE" set-model "$MODEL_OUT" >/dev/null
  mark finetune done "$CFG"
  hb "STAGE 3 finetune: DONE + verified. config=$CFG"
else
  hb "STAGE 3 finetune: already done — skip."
fi

# ── Stage 4: re-encode the 150M index IN PLACE ────────────────────────────────
if should_run reindex; then
  check_budget_or_stop reindex 140
  # Capture a pre-reindex baseline doc so the gate can prove sparse_field changed.
  if [[ ! -s "$DOC_BASELINE" ]]; then
    hb "STAGE 4 reindex: snapshotting a baseline doc's sparse_field for the gate..."
    "$PY" "$STATE_HELPER" --state "$STATE_FILE" snapshot-doc \
        --url "$OPENSEARCH_URL" --index "$INDEX" --out "$DOC_BASELINE" \
        >>"$RUN_LOG" 2>&1 || hb "STAGE 4 WARN: baseline snapshot failed (gate will skip change-check)."
  fi
  mark reindex running
  # CRITICAL: splade_indexer caches the ONNX/TRT engine at a FIXED path
  # (models/splade_onnx_fp16, models/trt_engine_cache) keyed only by a sentinel,
  # NOT by model name. A cache built for the OLD model would be silently reused
  # and the re-encode would write OLD weights. We must invalidate those caches
  # so the indexer re-exports the NEW model (keeps the fast TRT path; the
  # stage-4 GATE would catch a no-op re-encode anyway, but this avoids wasting
  # ~2.4h on it). Only clear once per reindex (guard so a --resume mid-reindex
  # doesn't blow away the engine the running pass already rebuilt).
  REENC_GUARD="$TRAIN_DIR/.reindex_cache_cleared"
  SNAP_DIR="$REPO_ROOT/data/openalex_snapshot"
  IDX_CKPT="$SNAP_DIR/indexer_checkpoint.json"
  if [[ ! -f "$REENC_GUARD" ]]; then
    hb "STAGE 4 reindex: invalidating stale ONNX/TRT caches so the NEW model is exported."
    rm -rf "$REPO_ROOT/models/splade_onnx_fp16" "$REPO_ROOT/models/trt_engine_cache" \
           "$REPO_ROOT/models/splade_onnx" 2>/dev/null || true
    # CRITICAL: the prior full index left indexer_checkpoint.json at
    # file_index=304 (state=completed). With --resume that means "nothing to
    # do" — it would NOT re-encode anything. Archive it so the NEW model
    # re-processes all 304 snapshot files from scratch. (Idempotent upsert-by-id
    # means the re-encode overwrites each doc's sparse_field in place.)
    if [[ -f "$IDX_CKPT" ]]; then
      hb "STAGE 4 reindex: archiving stale indexer checkpoint (was state=completed) so all files re-encode."
      mv "$IDX_CKPT" "$IDX_CKPT.preReindex.$(date +%Y%m%d-%H%M%S)" 2>/dev/null || rm -f "$IDX_CKPT"
    fi
    : > "$REENC_GUARD"
  fi
  hb "STAGE 4 reindex: splade_indexer --model $MODEL_OUT --resume (IN-PLACE upsert-by-id)"
  # splade_indexer has its own PID lock + checkpoint/resume + SIGTERM handler.
  "$PY" "$SCRIPT_DIR/splade_indexer.py" \
      --model "$MODEL_OUT" --resume \
      --opensearch-url "$OPENSEARCH_URL" --index "$INDEX" \
      >>"$RUN_LOG" 2>&1
  IDX_RC=$?
  if (( IDX_RC == 130 )); then
    hb "STAGE 4 reindex: interrupted (rc=130) — checkpoint saved; re-run to resume."
    mark reindex running
    exit 2
  elif (( IDX_RC != 0 )); then
    hb "STAGE 4 reindex: FAIL rc=$IDX_RC — checkpoint left; re-run to resume."
    mark reindex running
    exit 6
  fi
  # GATE: doc count sane + spot-doc sparse_field changed.
  hb "STAGE 4 GATE: verifying doc count + sparse_field change..."
  if ! "$PY" "$STATE_HELPER" --state "$STATE_FILE" verify-reindex \
        --url "$OPENSEARCH_URL" --index "$INDEX" \
        --min-count "$MIN_DOC_COUNT" --baseline-file "$DOC_BASELINE" \
        >>"$RUN_LOG" 2>&1; then
    hb "STAGE 4 GATE FAIL: count/change check failed. Stopping (fail loud)."
    mark reindex failed
    exit 6
  fi
  rm -f "$REENC_GUARD" 2>/dev/null || true   # reset guard for a future re-encode
  mark reindex done
  hb "STAGE 4 reindex: DONE + verified."
else
  hb "STAGE 4 reindex: already done — skip."
fi

# ── Stage 5: benchmark ────────────────────────────────────────────────────────
if should_run benchmark; then
  check_budget_or_stop benchmark 5
  mark benchmark running
  BENCH_OUT="$REPO_ROOT/data/eval_results/ndcg_splade_eval_$(date +%Y%m%d_%H%M).json"
  hb "STAGE 5 benchmark: ndcg_splade_eval -> $BENCH_OUT"
  "$PY" "$SCRIPT_DIR/ndcg_splade_eval.py" \
      --queries "$QUERIES" --k 10 --output "$BENCH_OUT" \
      >>"$RUN_LOG" 2>&1 || { hb "STAGE 5 FAIL"; mark benchmark failed; exit 7; }
  mark benchmark done
  hb "STAGE 5 benchmark: DONE -> $BENCH_OUT"
else
  hb "STAGE 5 benchmark: already done — skip."
fi

ELAPSED=$(( $(date +%s) - START_EPOCH ))
hb "=== PIPELINE COMPLETE in $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m. model=$MODEL_OUT ==="
"$PY" "$STATE_HELPER" --state "$STATE_FILE" show
exit 0
