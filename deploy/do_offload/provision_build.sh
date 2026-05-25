#!/usr/bin/env bash
# ============================================================================
# provision_build.sh — GATED. Create an ephemeral DO GPU droplet, run the fresh
# index build on it, then DESTROY it. Cost is double-capped:
#   (1) orchestrator --max-hours -> calls teardown.sh
#   (2) droplet-side dead-man (cloud_init.sh) -> self `shutdown` if we die
# An EXIT trap always runs teardown, so the droplet cannot leak even on crash.
#
#   bash deploy/do_offload/provision_build.sh --dry-run   # print plan, create NOTHING
#   bash deploy/do_offload/provision_build.sh             # provision (SPENDS MONEY)
#
# DOES NOT RUN unless you pass no --dry-run AND set SFU_CONFIRM=I_UNDERSTAND_COST.
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1

# Config + token
[[ -f "$SCRIPT_DIR/config.env" ]] && source "$SCRIPT_DIR/config.env" || source "$SCRIPT_DIR/config.example.env"
TOKEN="${DIGITALOCEAN_ACCESS_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$REPO_ROOT/.env" ]]; then
  TOKEN="$(grep -E '^[[:space:]]*DIGITALOCEAN_ACCESS_TOKEN=' "$REPO_ROOT/.env" | head -1 \
           | sed -E 's/^[^=]+=//; s/[[:space:]]+#.*$//; s/^["'"'"']//; s/["'"'"']$//; s/[[:space:]]+$//')"
fi
API="https://api.digitalocean.com/v2"

plan() {
  cat <<EOF
──────────────────────────────────────────────────────────────────────────
 DO GPU fresh-build plan
   tag         : $SFU_DO_TAG   (teardown.sh scope)
   size/region : $SFU_DO_SIZE @ $SFU_DO_REGION   image: $SFU_DO_IMAGE
   cost caps   : orchestrator ${SFU_MAX_HOURS}h  +  droplet dead-man ${SFU_DEADMAN_HOURS}h
   data source : $SFU_DATA_SOURCE   model: $SFU_MODEL_REF   index: $SFU_INDEX_NAME
   index sink  : $SFU_INDEX_SINK
   token       : $([[ -n "$TOKEN" ]] && echo "present" || echo "MISSING")
 Estimated GPU cost: size-rate x ~$(awk "BEGIN{print $SFU_MAX_HOURS*1}")h ceiling.
──────────────────────────────────────────────────────────────────────────
EOF
}

if (( DRY )); then plan; echo "(--dry-run: created nothing)"; exit 0; fi
[[ -n "$TOKEN" ]] || { echo "ERROR: no DIGITALOCEAN_ACCESS_TOKEN" >&2; exit 1; }
[[ "${SFU_CONFIRM:-}" == "I_UNDERSTAND_COST" ]] || {
  echo "Refusing to provision without SFU_CONFIRM=I_UNDERSTAND_COST. Re-run with it set." >&2
  plan; exit 2; }

auth=(-H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json")

# ALWAYS tear down on exit (success, failure, or Ctrl-C) — droplet can't leak.
cleanup() { echo ">> EXIT: ensuring droplet is destroyed"; bash "$SCRIPT_DIR/teardown.sh" || true; }
trap cleanup EXIT INT TERM

# Render cloud-init with our config baked in (dead-man timer included).
USER_DATA="$(SFU_DEADMAN_HOURS="$SFU_DEADMAN_HOURS" SFU_DATA_SOURCE="$SFU_DATA_SOURCE" \
             SFU_MODEL_REF="$SFU_MODEL_REF" SFU_INDEX_NAME="$SFU_INDEX_NAME" \
             SFU_INDEX_SINK="$SFU_INDEX_SINK" envsubst < "$SCRIPT_DIR/cloud_init.sh")"

# Create droplet (tagged so teardown can find it).
echo ">> creating droplet ($SFU_DO_SIZE @ $SFU_DO_REGION, tag $SFU_DO_TAG)..."
payload="$(python3 - "$SFU_DO_SIZE" "$SFU_DO_REGION" "$SFU_DO_IMAGE" "$SFU_DO_TAG" <<'PY'
import json,sys,os
size,region,image,tag=sys.argv[1:5]
print(json.dumps({"name":f"{tag}-{os.getpid()}","region":region,"size":size,
 "image":image,"tags":[tag],"user_data":os.environ.get("USER_DATA","")}))
PY
)"
resp="$(USER_DATA="$USER_DATA" curl -s --max-time 60 "${auth[@]}" -d "$payload" "$API/droplets")"
DID="$(echo "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('droplet',{}).get('id',''))" 2>/dev/null)"
[[ -n "$DID" ]] || { echo "ERROR creating droplet: $(echo "$resp" | head -c 300)" >&2; exit 3; }
echo ">> droplet id $DID created. Build runs via cloud-init; polling with ${SFU_MAX_HOURS}h budget."

# Poll until done or budget hit; teardown fires via the EXIT trap regardless.
START=$(date +%s); DEADLINE=$(( START + SFU_MAX_HOURS*3600 ))
while :; do
  now=$(date +%s); (( now >= DEADLINE )) && { echo ">> BUDGET ${SFU_MAX_HOURS}h hit — tearing down."; exit 2; }
  status="$(curl -s --max-time 20 "${auth[@]}" "$API/droplets/$DID" \
            | python3 -c "import sys,json;print(json.load(sys.stdin).get('droplet',{}).get('status',''))" 2>/dev/null)"
  # The droplet powers off (status=off) when cloud_init finishes or the dead-man fires.
  echo "   [$(( (now-START)/60 ))m] droplet status=$status"
  [[ "$status" == "off" ]] && { echo ">> droplet powered off (build done or dead-man). Tearing down."; break; }
  sleep 60
done
# trap handles teardown on exit
