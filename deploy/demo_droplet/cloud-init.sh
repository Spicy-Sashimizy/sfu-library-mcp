#!/usr/bin/env bash
# DO demo droplet boot script (run as cloud-init user-data, or by hand on the
# pre-baked snapshot). Idempotent: safe to re-run.
#
# Assumes the snapshot already has: docker + compose, this repo at /opt/sfu, and
# models/ baked in. The persistent volume (seeded by scripts/seed_demo_volume.sh)
# carries thinclient_index/ + qdrant_storage/ and is attached as a DO block volume.
#
# Flow: mount volume -> compose up (qdrant + mcp) -> PRE-WARM (absorb the ~115s
# SPLADE cold-load + cross-encoder load so the first real user is fast) -> the
# MCP /health the NAS broker polls only passes once warm.
set -euo pipefail

REPO="${REPO:-/opt/sfu}"
IDX_VOL="${IDX_VOL:-/mnt/idxvol}"
VOL_LABEL="${VOL_LABEL:-sfu_idx}"          # DO volume filesystem label
SPLADE_COLLECTION="${SPLADE_COLLECTION:-splade_150m}"
export IDX_VOL SPLADE_COLLECTION

echo ">> [1/4] mount persistent volume -> $IDX_VOL"
mkdir -p "$IDX_VOL"
if ! mountpoint -q "$IDX_VOL"; then
  DEV="$(readlink -f /dev/disk/by-label/$VOL_LABEL 2>/dev/null || true)"
  DEV="${DEV:-$(lsblk -rno NAME,TYPE | awk '$2=="disk"{print "/dev/"$1}' | tail -1)}"
  echo "   mounting $DEV"
  mount -o defaults,noatime "$DEV" "$IDX_VOL"
fi
mkdir -p "$IDX_VOL/logs"
test -d "$IDX_VOL/qdrant_storage"   || { echo "!! qdrant_storage missing on volume — seed it first"; exit 1; }
test -d "$IDX_VOL/thinclient_index" || { echo "!! thinclient_index missing on volume — seed it first"; exit 1; }

echo ">> [2/4] bring up qdrant + thin-client MCP"
cd "$REPO/deploy/demo_droplet"
# no --build: the snapshot already carries the baked sfu-library-mcp:demo image, so
# a wake is just a container start (rebuilding here would add minutes to every wake).
docker compose -f docker-compose.demo.yml up -d

echo ">> [3/4] wait for Qdrant readiness, then MCP /health"
# Qdrant has no shell/curl; probe it from inside the MCP container (which has curl)
# over the compose network. The MCP would degrade to BM25F if we skipped this, but
# we want the SPLADE leg live before declaring ready.
for i in $(seq 1 40); do
  docker exec demo-sfu-mcp curl -sf http://qdrant:6333/readyz >/dev/null 2>&1 && { echo "   qdrant ready"; break; }
  docker exec demo-sfu-mcp curl -sf http://qdrant:6333/ >/dev/null 2>&1 && { echo "   qdrant up"; break; }
  sleep 5
done
for i in $(seq 1 60); do
  curl -sf http://localhost:8080/health >/dev/null 2>&1 && break
  sleep 5
done

echo ">> [4/4] PRE-WARM (absorb cold model load before the first user)"
# Warms the server process's SPLADE encoder + Qdrant page cache + cross-encoder by
# issuing one real query through the in-container retriever (same process the MCP
# server reuses as a warm singleton). Best-effort; finalize against the live droplet
# in the Phase-5 dry run.
docker exec demo-sfu-mcp python3 - <<'PYWARM' || echo "   (pre-warm skipped/failed; first user pays cold load)"
import os, time
os.environ.setdefault("SFU_SPLADE_BACKEND", "qdrant")
t0 = time.time()
try:
    from lib.thinclient.retriever import ThinClientRetriever
    r = ThinClientRetriever(index_root=os.environ.get("SFU_THINCLIENT_INDEX_ROOT", "/app/data/thinclient_index"))
    hits = r.search("academic library search warmup query", top_k=10)
    print(f"   warmed retriever: {len(hits)} hits in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"   warm error: {e}")
PYWARM

echo ">> demo droplet ready — broker can proxy http://<droplet>:8080/mcp"
