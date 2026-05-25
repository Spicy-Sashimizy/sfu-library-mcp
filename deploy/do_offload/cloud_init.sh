#!/usr/bin/env bash
# ============================================================================
# cloud_init.sh — runs ON the DO GPU droplet (passed as user_data, with
# ${SFU_*} substituted by provision_build.sh via envsubst).
#
# Order matters: the DEAD-MAN TIMER is armed FIRST so the droplet self-destructs
# even if everything below hangs or the orchestrator dies — it can never bill idle.
#
# NOTE: the build/fetch specifics below are intentionally marked TODO — they are
# fleshed out in P2 after a dry-run, and depend on the chosen data path
# (Spaces vs rsync-from-TrueNAS). This file is the safe skeleton + cost guard.
# ============================================================================
set -uo pipefail
exec > /var/log/sfu_build.log 2>&1
echo "=== SFU fresh-build cloud-init start $(date -u) ==="

# (1) DEAD-MAN: hard self-shutdown after the budget, no matter what.
DEADMAN_HOURS="${SFU_DEADMAN_HOURS:-5}"
( sleep "$(( DEADMAN_HOURS * 3600 ))"; echo "DEAD-MAN ${DEADMAN_HOURS}h reached -> shutdown"; shutdown -h now ) &
echo ">> dead-man armed: ${DEADMAN_HOURS}h"

finish() { echo "=== build phase exit rc=$1 $(date -u) ==="; shutdown -h now; }

# (2) deps (GPU base image has CUDA; add encode + opensearch deps)
# TODO(P2): pin versions; install torch/onnxruntime-gpu/tensorrt-cu12/opensearch,
#           matching the encoder used in scripts/splade_indexer.py.

# (3) fetch model + code + snapshots
case "${SFU_DATA_SOURCE:-spaces}" in
  spaces)   echo "TODO(P2): aws s3 sync from DO Spaces -> /mnt/data (fast, same-region)";;
  truenas)  echo "TODO(P2): rsync -e ssh from ${SFU_TRUENAS_SSH:-truenas}:${SFU_TRUENAS_SNAPSHOT_DIR} -> /mnt/data";;
  *)        echo "unknown SFU_DATA_SOURCE"; finish 64;;
esac

# (4) fresh bulk build on local NVMe (NOT upsert -> the fast ~17.4k docs/s path)
# TODO(P2): start OpenSearch on NVMe, run splade_indexer in FRESH-INSERT mode
#           with --model ${SFU_MODEL_REF} --index ${SFU_INDEX_NAME}.

# (5) snapshot the finished index out to the sink (Spaces/volume) for retrieval
# TODO(P2): create index snapshot -> ${SFU_INDEX_SINK}; write a DONE marker.

echo ">> (skeleton) build steps are TODO(P2); powering off now so nothing bills."
finish 0
