#!/usr/bin/env bash
# OOM-safe 150M thin-client vs 150M OpenSearch parity run.
#
# This script runs each engine in its OWN process (the retriever has no close()
# hook, so only a fresh process frees its working set), dropping page cache
# between phases. At no instant are both engines serving — peak RAM ~= one
# engine, never the sum.
#
# ⚠ WARNING (measured 2026-06-17): at 150M this is necessary but NOT sufficient.
# The thin-client SPLADE leg is NOT mmap'd — bmp.Searcher loads each *.bmp into
# anonymous RAM at ~3.07x on disk, so the full set is ~211 GB RESIDENT. On the
# 24 GB host `record-tc` OOMs inside _load() before it serves a query. The
# retriever now aborts cleanly (RuntimeError, gate SFU_LOAD_MEM_FLOOR_GB) instead
# of being SIGKILLed. This single-process path is BLOCKED until a host can hold
# one whole engine (~232 GB — an artifact of BMP's load-into-memory design, NOT a
# serving recommendation; the mmap fix / hot-cold residency are the real answers).
# For 150M parity numbers on a small host, use scripts/eval_parity_section_waves.py.
#
# Usage:
#   scripts/run_parity_safe.sh [QUERIES] [INDEX_ROOT] [BASELINE_URL]
#   PAUSE_OS=1 scripts/run_parity_safe.sh   # also `docker pause` OS during phase A
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python3"
EVAL="${REPO_ROOT}/scripts/eval_thinclient_parity.py"

QUERIES="${1:-40}"
INDEX_ROOT="${2:-${REPO_ROOT}/data/thinclient_index}"
BASELINE_URL="${3:-http://host.docker.internal:9200}"
MEM_FLOOR_GB="${MEM_FLOOR_GB:-2.0}"
PREFLIGHT_FLOOR_GB="${PREFLIGHT_FLOOR_GB:-4.0}"
OS_CONTAINER="${OS_CONTAINER:-claudebox-sfu-library-mcp-opensearch}"

TS="$(date +%Y%m%d_%H%M)"
OUT_DIR="${REPO_ROOT}/data/eval_results"
TC_REC="${OUT_DIR}/parity_record_tc_${TS}.json"
OS_REC="${OUT_DIR}/parity_record_os_${TS}.json"
SUMMARY="${OUT_DIR}/thinclient_parity_${TS}.json"

mem_avail_gb() { awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo; }

preflight() {
  local avail; avail="$(mem_avail_gb)"
  echo ">> [$1] MemAvailable ${avail} GB"
  if awk "BEGIN{exit !(${avail} < ${PREFLIGHT_FLOOR_GB})}"; then
    echo "!! [$1] MemAvailable ${avail} GB < preflight floor ${PREFLIGHT_FLOOR_GB} GB — aborting" >&2
    exit 1
  fi
}

drop_caches() {
  sync || true
  if [ -w /proc/sys/vm/drop_caches ]; then
    echo 1 > /proc/sys/vm/drop_caches && echo ">> dropped page cache"
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null && echo ">> dropped page cache (sudo)"
  else
    echo ">> (cannot drop_caches without root — relying on reclaimable mmap; continuing)"
  fi
}

echo "== OOM-safe parity run ${TS} =="
echo "   queries=${QUERIES} index=${INDEX_ROOT} baseline=${BASELINE_URL}"

# ── Phase A: thin-client only ────────────────────────────────────────────────
preflight "phase-A/tc"
if [ "${PAUSE_OS:-0}" = "1" ]; then
  docker pause "${OS_CONTAINER}" 2>/dev/null && echo ">> paused ${OS_CONTAINER}" || \
    echo ">> (could not pause ${OS_CONTAINER}; continuing)"
fi
SFU_DENSE_WARMCACHE="${SFU_DENSE_WARMCACHE:-0}" \
  "${PY}" "${EVAL}" record-tc --index-root "${INDEX_ROOT}" \
    --queries "${QUERIES}" --mem-floor-gb "${MEM_FLOOR_GB}" --output "${TC_REC}"
if [ "${PAUSE_OS:-0}" = "1" ]; then
  docker unpause "${OS_CONTAINER}" 2>/dev/null && echo ">> unpaused ${OS_CONTAINER}" || true
fi

drop_caches

# ── Phase B: OpenSearch only (thin-client process has exited) ─────────────────
preflight "phase-B/os"
"${PY}" "${EVAL}" record-os --baseline-url "${BASELINE_URL}" \
  --queries "${QUERIES}" --mem-floor-gb "${MEM_FLOOR_GB}" --output "${OS_REC}"

drop_caches

# ── Phase C: offline join (few MB) ───────────────────────────────────────────
"${PY}" "${EVAL}" compare --tc-record "${TC_REC}" --os-record "${OS_REC}" \
  --output "${SUMMARY}"

echo "== done =="
echo "   tc record:  ${TC_REC}"
echo "   os record:  ${OS_REC}"
echo "   summary:    ${SUMMARY}"
