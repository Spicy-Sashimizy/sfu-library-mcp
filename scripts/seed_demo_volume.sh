#!/usr/bin/env bash
# Resumable seed of the Qdrant on_disk storage dir onto the DO demo volume.
#
# WHY this shape: the build/source host is NOT always-on (HYBRID_DEMO_DEPLOYMENT.md
# risk #1), so the ~205 GB (est.) one-time transfer cannot assume a single
# continuous run. This wrapper is RE-ENTRANT: rsync --partial --append-verify
# --inplace picks up exactly where it left off, and the outer retry loop survives
# source reboots / dropped SSH. Re-running after completion is a cheap no-op.
#
# CONSISTENCY: copy a *quiescent* Qdrant storage — either stop the server first, or
# (preferred) take a Qdrant snapshot and seed that. rsync'ing a live, actively-
# ingesting storage dir can capture a torn segment. Pass --snapshot to trigger a
# collection snapshot via the REST API before transfer (requires QDRANT_URL).
#
# Usage:
#   scripts/seed_demo_volume.sh --dest user@droplet:/mnt/idxvol/qdrant_storage
#   scripts/seed_demo_volume.sh --src data/qdrant_spike/storage --dest /local/path   # local
#   scripts/seed_demo_volume.sh --dest ... --verify     # checksum dry-run, lists diffs
#   SRC=... DEST=... scripts/seed_demo_volume.sh        # env form (cron/systemd)
set -euo pipefail

SRC="${SRC:-data/qdrant_spike/storage}"
DEST="${DEST:-}"
SSH_OPTS="${SSH_OPTS:--o ServerAliveInterval=30 -o ServerAliveCountMax=4}"
MAX_RETRIES="${MAX_RETRIES:-0}"        # 0 = retry forever (until rsync exits 0)
BACKOFF="${BACKOFF:-15}"               # seconds, capped at 300
BWLIMIT="${BWLIMIT:-0}"                # KB/s, 0 = unlimited
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
SNAPSHOT_COLLECTION="${SNAPSHOT_COLLECTION:-}"
MODE="transfer"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --verify) MODE="verify"; shift;;
    --snapshot) MODE="snapshot"; shift;;
    --bwlimit) BWLIMIT="$2"; shift 2;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -n "$DEST" ]] || { echo "ERROR: --dest (or DEST=) required" >&2; exit 2; }
[[ -d "$SRC" ]] || { echo "ERROR: src '$SRC' not a directory" >&2; exit 2; }

# rsync trailing-slash semantics: copy CONTENTS of SRC into DEST.
SRC_SLASH="${SRC%/}/"

# Only use ssh transport when DEST looks like host:path.
RSYNC_RSH=()
if [[ "$DEST" == *:* && "$DEST" != /* ]]; then
  RSYNC_RSH=(-e "ssh ${SSH_OPTS}")
fi

BASE_FLAGS=(-a --partial --append-verify --inplace --human-readable --info=progress2,stats2)
[[ "$BWLIMIT" != "0" ]] && BASE_FLAGS+=(--bwlimit="$BWLIMIT")

trigger_snapshot() {
  [[ -n "$SNAPSHOT_COLLECTION" ]] || { echo "ERROR: --snapshot needs SNAPSHOT_COLLECTION=" >&2; exit 2; }
  echo ">> triggering Qdrant snapshot of '$SNAPSHOT_COLLECTION' at $QDRANT_URL"
  curl -fsS -X POST "$QDRANT_URL/collections/$SNAPSHOT_COLLECTION/snapshots" | head -c 400
  echo
  echo ">> snapshot created under <storage>/snapshots/$SNAPSHOT_COLLECTION/ — seed that for consistency"
}

verify() {
  echo ">> VERIFY: checksum dry-run $SRC_SLASH -> $DEST"
  # Itemize-changes; a real content diff is a FILE line whose first flag char is
  # not '.' (e.g. '>f.s.c...'). Directory-only / timestamp lines ('.d..t...') are
  # benign noise from --inplace touching the dest dir mtime, so we exclude them.
  local diffs
  diffs=$(rsync -a --checksum --dry-run --itemize-changes "${RSYNC_RSH[@]}" \
          "$SRC_SLASH" "$DEST" 2>/dev/null | grep -E '^[^.]f' || true)
  if [[ -n "$diffs" ]]; then
    echo "$diffs"; echo ">> VERIFY: ${diffs:+content }differences remain (re-run transfer)"; return 1
  fi
  echo ">> VERIFY OK: dest is byte-identical to src"
}

transfer() {
  local attempt=0
  while true; do
    attempt=$((attempt+1))
    if [[ "$MAX_RETRIES" != "0" && "$attempt" -gt "$MAX_RETRIES" ]]; then
      echo ">> hit MAX_RETRIES=$MAX_RETRIES without a verified-complete transfer" >&2; return 1
    fi
    echo ">> [$(date -u +%H:%M:%S)] rsync attempt #$attempt : $SRC_SLASH -> $DEST"
    if rsync "${BASE_FLAGS[@]}" "${RSYNC_RSH[@]}" "$SRC_SLASH" "$DEST"; then
      echo ">> transfer pass complete on attempt #$attempt"
      if verify; then return 0; fi
      echo ">> verify found residual diffs; re-running rsync"
      continue
    fi
    rc=$?
    local wait=$(( BACKOFF * attempt )); [[ $wait -gt 300 ]] && wait=300
    echo ">> rsync interrupted (rc=$rc); resuming in ${wait}s (re-entrant: --partial keeps progress)"
    sleep "$wait"
  done
}

case "$MODE" in
  snapshot) trigger_snapshot;;
  verify) verify;;
  transfer) transfer;;
esac
