#!/usr/bin/env bash
# ============================================================================
# cloud_init.sh — ENCODE-ONLY on a cheap DO GPU droplet (user_data; values baked
# in by provision_build.sh as an exported preamble).
#
# Reads snapshot shards + the fine-tuned model over the NPM proxy, encodes SPLADE
# sparse vectors, and PUSHes bulk-ready NDJSON shards back to TrueNAS via the proxy.
# NO OpenSearch, NO volume — the 412GB index is built later from these shards.
# Dead-man timer armed FIRST so the droplet can never bill idle.
#
# MEASUREMENT MODE: SFU_ENCODE_MAX_SHARDS>0 limits how many shards to process
# (lets a cheap, short run measure real throughput / GPU / CPU before the full job).
# Validated only by a real run; --dry-run validates the plan, not this script.
# ============================================================================
set -uo pipefail
exec > /var/log/sfu_build.log 2>&1
ts(){ date -u +%H:%M:%S; }
echo "=== SFU encode-only start $(date -u) ==="

DEADMAN_HOURS="${SFU_DEADMAN_HOURS:-3}"
PROXY_URL="${SFU_DATA_PROXY_URL:-}"; PROXY_AUTH="${SFU_DATA_PROXY_AUTH:-}"
PROXY_SNAPS="${SFU_DATA_PROXY_SNAPSHOTS:-/snapshots}"
PROXY_MODELS="${SFU_DATA_PROXY_MODELS:-/models}"
PROXY_CODE="${SFU_DATA_PROXY_CODE:-/code}"
ENCODE_OUT_PATH="${SFU_ENCODE_OUT_PATH:-/upload/encoded}"
MODEL_REF="${SFU_MODEL_REF:-sfu-splade-v1}"
BATCH_SIZE="${SFU_BATCH_SIZE:-128}"
MAX_SHARDS="${SFU_ENCODE_MAX_SHARDS:-0}"

# (1) DEAD-MAN
( sleep "$(( DEADMAN_HOURS*3600 ))"; echo "DEAD-MAN ${DEADMAN_HOURS}h -> shutdown"; shutdown -h now ) &
echo ">> dead-man armed: ${DEADMAN_HOURS}h"
finish(){ echo "=== encode exit rc=$1 $(date -u) ==="; sync; shutdown -h now; }

UA=(); [[ -n "$PROXY_AUTH" ]] && UA=(--user "${PROXY_AUTH%%:*}" --password "${PROXY_AUTH#*:}")
getf(){ wget -q "${UA[@]}" -O "$2" "${PROXY_URL%/}$1"; }
putf(){ wget -q "${UA[@]}" --method=PUT --body-file="$2" -O /dev/null "${PROXY_URL%/}$1"; }
WORK=/opt/sfu; mkdir -p "$WORK/in"; cd "$WORK"

# (2) deps — NVIDIA AI/ML image ships CUDA + python. Install torch + onnxruntime-gpu
#     (+ tensorrt so the fast TRT EP can load) + transformers. No OpenSearch.
echo ">> [$(ts)] installing deps..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3-venv python3-pip >/dev/null 2>&1 || true
python3 -m venv venv; PY="$WORK/venv/bin/python"; PIP="$WORK/venv/bin/pip"
$PIP install -q --upgrade pip
$PIP install -q torch --index-url https://download.pytorch.org/whl/cu124 || $PIP install -q torch
$PIP install -q transformers onnxruntime-gpu "optimum[onnxruntime-gpu]" tensorrt numpy requests tqdm sentence-transformers || true
echo ">> [$(ts)] gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) | vCPUs=$(nproc)"

# (3) fetch code + model over the proxy
echo ">> [$(ts)] fetching code + model..."
getf "$PROXY_CODE/sfu-encode-code.tar.gz" code.tar.gz && tar -xzf code.tar.gz || { echo "code fetch failed"; finish 70; }
mkdir -p "models/$MODEL_REF"
wget -q "${UA[@]}" -r -np -nH --cut-dirs=2 -R "index.html*" -P "models/$MODEL_REF" "${PROXY_URL%/}$PROXY_MODELS/$MODEL_REF/"
[ -f "models/$MODEL_REF/model.safetensors" ] || { echo "model fetch failed"; ls -R models | head; finish 71; }

# (4) list shards (optionally limited for a measurement run), fetch them
shards=$(wget -q "${UA[@]}" -O - "${PROXY_URL%/}$PROXY_SNAPS/" | grep -oE "works_part_[0-9]+\.jsonl\.gz" | sort -u)
if [ "${MAX_SHARDS:-0}" -gt 0 ] 2>/dev/null; then shards=$(echo "$shards" | head -n "$MAX_SHARDS"); fi
total=$(echo "$shards" | grep -c . )
echo ">> [$(ts)] encoding $total shard(s) (MAX_SHARDS=$MAX_SHARDS, batch=$BATCH_SIZE)"
for s in $shards; do echo ">> [$(ts)] fetch $s"; getf "$PROXY_SNAPS/$s" "in/$s" || { echo "fetch $s failed"; finish 72; }; done

# (5) encode ALL selected shards in ONE run (single model load) -> one gz shard
OUT="$WORK/encoded_$(date +%s).ndjson.gz"
echo ">> [$(ts)] ENCODE START -> $(basename "$OUT")"
SFU_ENCODE_OUT="$OUT" SFU_PROGRESS_PUSH=off \
  $PY scripts/splade_indexer.py --input "$WORK/in" --model "$WORK/models/$MODEL_REF" \
  --batch-size "$BATCH_SIZE" --async-workers 2 --device cuda --backend auto || { echo "ENCODE FAILED"; finish 73; }
echo ">> [$(ts)] ENCODE DONE; uploading $(basename "$OUT") ($(du -h "$OUT" 2>/dev/null | cut -f1))"

# (6) push the encoded shard back to TrueNAS via the proxy
putf "$ENCODE_OUT_PATH/$(basename "$OUT")" "$OUT" || { echo "upload failed"; finish 74; }
echo ">> [$(ts)] uploaded. encode-run complete."
finish 0
