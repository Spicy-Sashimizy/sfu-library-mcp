#!/usr/bin/env bash
# ============================================================================
# cloud_init.sh — runs ON the DO GPU droplet (passed as user_data, with
# ${SFU_*} substituted by provision_build.sh via envsubst).
#
# Order matters: the DEAD-MAN TIMER is armed FIRST so the droplet self-destructs
# even if everything below hangs or the orchestrator dies — it can never bill idle.
#
# Flow: arm dead-man → mount index volume → deps → fetch model+code+snapshots over
# the NPM HTTPS proxy → OpenSearch on the volume (bulk-tuned) → splade_indexer in
# FRESH-INSERT mode (~17.4k docs/s, NOT the slow in-place upsert) → snapshot the
# finished index to DO Spaces → power off (orchestrator sees status=off → teardown).
#
# Validated end-to-end only by a real (paid) run — `provision_build.sh --dry-run`
# validates the PLAN, not this script. Review before the first provision.
# ============================================================================
set -uo pipefail
exec > /var/log/sfu_build.log 2>&1
echo "=== SFU fresh-build cloud-init start $(date -u) ==="

# Substituted by provision_build.sh (envsubst). Defaults are harmless fallbacks.
DEADMAN_HOURS="${SFU_DEADMAN_HOURS:-5}"
VOL_FS="${SFU_VOLUME_FS:-ext4}"
VOL_MNT="${SFU_VOLUME_MOUNT:-/mnt/index}"
DATA_SOURCE="${SFU_DATA_SOURCE:-proxy}"
PROXY_URL="${SFU_DATA_PROXY_URL:-}"
PROXY_AUTH="${SFU_DATA_PROXY_AUTH:-}"           # "user:pass" for basic auth (may be empty)
PROXY_SNAPS="${SFU_DATA_PROXY_SNAPSHOTS:-/snapshots}"
PROXY_MODELS="${SFU_DATA_PROXY_MODELS:-/models}"
PROXY_CODE="${SFU_DATA_PROXY_CODE:-/code}"
MODEL_REF="${SFU_MODEL_REF:-sfu-splade-v1}"
INDEX_NAME="${SFU_INDEX_NAME:-openalex_works}"
BATCH_SIZE="${SFU_BATCH_SIZE:-64}"
INDEX_SINK="${SFU_INDEX_SINK:-spaces}"
SPACES_BUCKET="${SFU_SPACES_BUCKET:-}"
SPACES_REGION="${SFU_SPACES_REGION:-}"
SPACES_ENDPOINT="${SFU_SPACES_ENDPOINT:-}"
SPACES_KEY="${SFU_SPACES_KEY:-}"
SPACES_SECRET="${SFU_SPACES_SECRET:-}"
SPACES_PREFIX="${SFU_SPACES_PREFIX:-openalex_works_snapshot}"

# (1) DEAD-MAN: hard self-shutdown after the budget, no matter what.
( sleep "$(( DEADMAN_HOURS * 3600 ))"; echo "DEAD-MAN ${DEADMAN_HOURS}h reached -> shutdown"; shutdown -h now ) &
echo ">> dead-man armed: ${DEADMAN_HOURS}h"
finish() { echo "=== build phase exit rc=$1 $(date -u) ==="; sync; shutdown -h now; }

wget_proxy() {  # $1=remote-path-under-proxy  $2=local-dest   (basic-auth aware)
  local url="${PROXY_URL%/}$1" auth=()
  [[ -n "$PROXY_AUTH" ]] && auth=(--user "${PROXY_AUTH%%:*}" --password "${PROXY_AUTH#*:}")
  wget -q "${auth[@]}" -O "$2" "$url"
}
mirror_proxy() { # recursively mirror an nginx-autoindex directory: $1=sub-path $2=dest
  local sub="$1" dest="$2" wauth=()
  [[ -n "$PROXY_AUTH" ]] && wauth=(--user "${PROXY_AUTH%%:*}" --password "${PROXY_AUTH#*:}")
  mkdir -p "$dest"
  wget -q -r -np -nH --cut-dirs=1 -R "index.html*" "${wauth[@]}" -P "$dest" "${PROXY_URL%/}$sub/"
}

# (2) Mount the attached block volume for the index (DO exposes it by-id).
echo ">> locating attached volume..."
VOL_DEV="$(ls /dev/disk/by-id/scsi-0DO_Volume_* 2>/dev/null | head -1 || true)"
if [[ -n "$VOL_DEV" ]]; then
  blkid "$VOL_DEV" >/dev/null 2>&1 || mkfs."$VOL_FS" -F "$VOL_DEV"
  mkdir -p "$VOL_MNT"; mount -o discard,defaults "$VOL_DEV" "$VOL_MNT"
  echo ">> mounted $VOL_DEV at $VOL_MNT ($(df -h "$VOL_MNT" | tail -1))"
else
  echo ">> no block volume found — using local NVMe at $VOL_MNT"; mkdir -p "$VOL_MNT"
fi
SNAP_DIR="$VOL_MNT/snapshots"; OS_DATA="$VOL_MNT/os-data"; mkdir -p "$SNAP_DIR" "$OS_DATA"
chmod 777 "$OS_DATA"   # opensearch container (uid 1000) writes here

# (3) Deps. The NVIDIA AI/ML Ready image ships CUDA + docker + python3.
echo ">> installing deps..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3-venv python3-pip awscli jq >/dev/null 2>&1 || true
python3 -m venv /opt/sfu/venv
PY=/opt/sfu/venv/bin/python; PIP=/opt/sfu/venv/bin/pip
$PIP install -q --upgrade pip
# encode stack (torch/onnxruntime-gpu come as CUDA wheels; indexer falls back to
# pytorch FP16 if the TRT EP is unavailable). requests drives OpenSearch.
$PIP install -q torch --index-url https://download.pytorch.org/whl/cu124 || $PIP install -q torch
$PIP install -q transformers onnxruntime-gpu "optimum[onnxruntime-gpu]" numpy requests tqdm sentence-transformers || true

# (4) Fetch model + code + snapshots over the NPM proxy.
echo ">> fetching code + model over proxy ($PROXY_URL)..."
mkdir -p /opt/sfu
wget_proxy "$PROXY_CODE/sfu-encode-code.tar.gz" /opt/sfu/code.tar.gz \
  && tar -xzf /opt/sfu/code.tar.gz -C /opt/sfu || { echo "code fetch FAILED"; finish 70; }
mirror_proxy "$PROXY_MODELS/$MODEL_REF" "/opt/sfu/models/$MODEL_REF"
echo ">> fetching ${INDEX_NAME} snapshots -> $SNAP_DIR (the bulk of the transfer)..."
mirror_proxy "$PROXY_SNAPS" "$SNAP_DIR"
SHARDS=$(ls "$SNAP_DIR"/works_part_*.jsonl.gz 2>/dev/null | wc -l)
echo ">> $SHARDS snapshot shards present"; (( SHARDS > 0 )) || { echo "no snapshots fetched"; finish 71; }

# (5) OpenSearch on the volume, tuned for a one-shot bulk load (security off; it's
#     ephemeral + localhost-only). Heap ~half RAM (cap 31g), single node, no replicas.
echo ">> starting OpenSearch (data on $OS_DATA)..."
HEAP_GB=$(( $(awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo) / 2 )); (( HEAP_GB>31 )) && HEAP_GB=31
docker run -d --name os --restart no \
  -p 127.0.0.1:9200:9200 \
  -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
  -e "OPENSEARCH_JAVA_OPTS=-Xms${HEAP_GB}g -Xmx${HEAP_GB}g" \
  -e "bootstrap.memory_lock=true" --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
  -v "$OS_DATA":/usr/share/opensearch/data \
  opensearchproject/opensearch:2 || { echo "opensearch start FAILED"; finish 72; }
echo ">> waiting for OpenSearch..."; for i in $(seq 1 60); do
  curl -s localhost:9200/_cluster/health >/dev/null 2>&1 && break; sleep 5; done
curl -s "localhost:9200/_cluster/health?pretty" || { echo "OS never came up"; finish 73; }

# Fresh index, bulk-tuned: disable refresh + replicas during the load.
curl -s -X PUT "localhost:9200/$INDEX_NAME" -H 'Content-Type: application/json' -d '{
  "settings": {"index": {"number_of_replicas": 0, "refresh_interval": "-1"}}}' >/dev/null

# (6) FRESH bulk insert (NO --resume → fast path). Indexer auto-uses CUDA/TRT.
echo ">> indexing (fresh bulk insert)..."
export SFU_PROGRESS_PUSH=off   # droplet can't reach the NAS push target; disable the mirror
$PY /opt/sfu/scripts/splade_indexer.py \
    --input "$SNAP_DIR" \
    --model "/opt/sfu/models/$MODEL_REF" \
    --index "$INDEX_NAME" \
    --opensearch-url "http://localhost:9200" \
    --batch-size "$BATCH_SIZE" \
    --async-workers 6 \
    --device cuda --backend auto
IDX_RC=$?; echo ">> indexer rc=$IDX_RC"; (( IDX_RC == 0 )) || { echo "INDEXING FAILED"; finish 74; }

# restore refresh + flush + one merge pass so the snapshot is compact
curl -s -X PUT "localhost:9200/$INDEX_NAME/_settings" -H 'Content-Type: application/json' \
     -d '{"index":{"refresh_interval":"1s"}}' >/dev/null
curl -s -X POST "localhost:9200/$INDEX_NAME/_forcemerge?max_num_segments=1" >/dev/null
echo ">> final count: $(curl -s "localhost:9200/$INDEX_NAME/_count")"

# (7) Snapshot the finished index out to the sink (DO Spaces = S3-compatible).
if [[ "$INDEX_SINK" == "spaces" && -n "$SPACES_BUCKET" && -n "$SPACES_KEY" ]]; then
  echo ">> registering Spaces snapshot repo + snapshotting..."
  curl -s -X PUT "localhost:9200/_snapshot/spaces" -H 'Content-Type: application/json' -d "{
    \"type\":\"s3\",\"settings\":{\"bucket\":\"$SPACES_BUCKET\",\"endpoint\":\"$SPACES_ENDPOINT\",
      \"region\":\"$SPACES_REGION\",\"base_path\":\"$SPACES_PREFIX\",
      \"access_key\":\"$SPACES_KEY\",\"secret_key\":\"$SPACES_SECRET\"}}" >/dev/null
  curl -s -X PUT "localhost:9200/_snapshot/spaces/build-$(date +%s)?wait_for_completion=true" \
       -H 'Content-Type: application/json' -d "{\"indices\":\"$INDEX_NAME\",\"include_global_state\":false}" \
    | tee /var/log/sfu_snapshot.json
  echo ">> snapshot to Spaces complete."
else
  echo ">> WARN: no Spaces sink configured — index NOT exported. Set SFU_SPACES_* in .env."
fi

echo ">> build done; powering off so nothing bills."
finish 0
