#!/usr/bin/env bash
# ============================================================================
# provision_build.sh — GATED. Create an ephemeral DO GPU droplet (+ a block
# volume for the ~412GB index), run the fresh build on it, then DESTROY both.
# Cost is double-capped:
#   (1) orchestrator --max-hours -> calls teardown.sh (droplet + volume)
#   (2) droplet-side dead-man (cloud_init.sh) -> self `shutdown` if we die
# An EXIT trap always runs teardown, so nothing can leak even on crash.
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

# Config
[[ -f "$SCRIPT_DIR/config.env" ]] && source "$SCRIPT_DIR/config.env" || source "$SCRIPT_DIR/config.example.env"
API="https://api.digitalocean.com/v2"

# ── secrets from .env (never printed) ─────────────────────────────────────────
env_val() { # $1=KEY -> value from repo .env, stripped of quotes/comments
  [[ -f "$REPO_ROOT/.env" ]] || return 0
  grep -E "^[[:space:]]*$1=" "$REPO_ROOT/.env" | head -1 \
    | sed -E "s/^[^=]+=//; s/[[:space:]]+#.*$//; s/^[\"']//; s/[\"']$//; s/[[:space:]]+$//"
}
TOKEN="${DIGITALOCEAN_ACCESS_TOKEN:-$(env_val DIGITALOCEAN_ACCESS_TOKEN)}"
SFU_DATA_PROXY_AUTH="${SFU_DATA_PROXY_AUTH:-$(env_val SFU_DATA_PROXY_AUTH)}"
SFU_SPACES_KEY="$(env_val DO_SPACES_KEY)";       SFU_SPACES_SECRET="$(env_val DO_SPACES_SECRET)"
SFU_SPACES_REGION="$(env_val DO_SPACES_REGION)"; SFU_SPACES_BUCKET="${SFU_SPACES_BUCKET:-$(env_val DO_SPACES_BUCKET)}"
SFU_SPACES_ENDPOINT="$(env_val DO_SPACES_ENDPOINT)"
[[ -z "$SFU_SPACES_ENDPOINT" && -n "$SFU_SPACES_REGION" ]] && SFU_SPACES_ENDPOINT="https://${SFU_SPACES_REGION}.digitaloceanspaces.com"

auth=(-H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json")
hourly() { case "$1" in gpu-4000adax1-20gb) echo 0.76;; gpu-l40sx1-48gb|gpu-6000adax1-48gb) echo 1.57;;
  gpu-h100x1-80gb) echo 3.39;; gpu-h200x1-141gb) echo 3.44;; *) echo "?";; esac; }

plan() {
  local rate; rate="$(hourly "$SFU_DO_SIZE")"
  cat <<EOF
──────────────────────────────────────────────────────────────────────────
 DO GPU fresh-build plan
   tag         : $SFU_DO_TAG   (teardown.sh scope: droplet + volume)
   size/region : $SFU_DO_SIZE @ $SFU_DO_REGION   image: $SFU_DO_IMAGE
   ssh key     : ${SFU_DO_SSH_KEY_NAME:-<none>}
   volume      : ${SFU_VOLUME_SIZE_GB}GB ${SFU_VOLUME_FS} -> ${SFU_VOLUME_MOUNT}
   cost caps   : orchestrator ${SFU_MAX_HOURS}h  +  droplet dead-man ${SFU_DEADMAN_HOURS}h
   data path   : $SFU_DATA_SOURCE  ${SFU_DATA_PROXY_URL:-}  (auth: $([[ -n "$SFU_DATA_PROXY_AUTH" ]] && echo set || echo MISSING))
   model/index : $SFU_MODEL_REF / $SFU_INDEX_NAME
   index sink  : $SFU_INDEX_SINK  -> push ${SFU_DATA_PROXY_URL}${SFU_UPLOAD_PATH:-/upload/openalex_works_snapshot}
   token       : $([[ -n "$TOKEN" ]] && echo present || echo MISSING)
   GPU cost ceiling : \$${rate}/hr × ${SFU_MAX_HOURS}h ≈ \$$(awk "BEGIN{r=\"$rate\"; print (r==\"?\")?\"?\":r*$SFU_MAX_HOURS}") (+ volume ~\$$(awk "BEGIN{print $SFU_VOLUME_SIZE_GB*0.00015*$SFU_MAX_HOURS}"))
──────────────────────────────────────────────────────────────────────────
EOF
}

if (( DRY )); then plan; echo "(--dry-run: created nothing)"; exit 0; fi
[[ -n "$TOKEN" ]] || { echo "ERROR: no DIGITALOCEAN_ACCESS_TOKEN" >&2; exit 1; }
[[ "${SFU_CONFIRM:-}" == "I_UNDERSTAND_COST" ]] || {
  echo "Refusing to provision without SFU_CONFIRM=I_UNDERSTAND_COST. Re-run with it set." >&2
  plan; exit 2; }

# ALWAYS tear down on exit (success, failure, or Ctrl-C) — droplet + volume can't leak.
cleanup() { echo ">> EXIT: ensuring droplet + volume are destroyed"; bash "$SCRIPT_DIR/teardown.sh" || true; }
trap cleanup EXIT INT TERM

# Resolve SSH key id (optional but recommended for debug access).
SSH_ID=""
if [[ -n "${SFU_DO_SSH_KEY_NAME:-}" ]]; then
  SSH_ID="$(curl -s --max-time 20 "${auth[@]}" "$API/account/keys?per_page=200" \
    | python3 -c "import sys,json,os;n=os.environ['K'];print(next((str(k['id']) for k in json.load(sys.stdin).get('ssh_keys',[]) if k['name']==n),''))" K="$SFU_DO_SSH_KEY_NAME" 2>/dev/null)"
  [[ -n "$SSH_ID" ]] || echo ">> WARN: SSH key '$SFU_DO_SSH_KEY_NAME' not found in DO account — booting without inbound SSH."
fi

# Create the index volume (tagged so teardown finds it), same region as the droplet.
VOL_ID=""
if [[ "${SFU_VOLUME_SIZE_GB:-0}" -gt 0 ]]; then
  echo ">> creating ${SFU_VOLUME_SIZE_GB}GB volume in $SFU_DO_REGION..."
  vresp="$(curl -s --max-time 60 "${auth[@]}" -d "$(python3 -c "import json,os;print(json.dumps({
    'name':os.environ['T']+'-vol','region':os.environ['R'],
    'size_gigabytes':int(os.environ['S']),'tags':[os.environ['T']]}))" \
    T="$SFU_DO_TAG" R="$SFU_DO_REGION" S="$SFU_VOLUME_SIZE_GB")" "$API/volumes")"
  VOL_ID="$(echo "$vresp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('volume',{}).get('id',''))" 2>/dev/null)"
  [[ -n "$VOL_ID" ]] || { echo "ERROR creating volume: $(echo "$vresp" | head -c 300)" >&2; exit 3; }
  echo ">> volume $VOL_ID created."
fi

# Bake config into cloud-init via a shell-safe exported preamble (robust vs envsubst).
preamble=""; ex() { preamble+="export $1=$(printf %q "${2:-}")"$'\n'; }
ex SFU_DEADMAN_HOURS "$SFU_DEADMAN_HOURS";   ex SFU_VOLUME_FS "$SFU_VOLUME_FS"
ex SFU_VOLUME_MOUNT "$SFU_VOLUME_MOUNT";     ex SFU_DATA_SOURCE "$SFU_DATA_SOURCE"
ex SFU_DATA_PROXY_URL "$SFU_DATA_PROXY_URL"; ex SFU_DATA_PROXY_AUTH "$SFU_DATA_PROXY_AUTH"
ex SFU_DATA_PROXY_SNAPSHOTS "${SFU_DATA_PROXY_SNAPSHOTS:-/snapshots}"
ex SFU_DATA_PROXY_MODELS "${SFU_DATA_PROXY_MODELS:-/models}"
ex SFU_UPLOAD_PATH "${SFU_UPLOAD_PATH:-/upload/openalex_works_snapshot}"
ex SFU_MODEL_REF "$SFU_MODEL_REF";           ex SFU_INDEX_NAME "$SFU_INDEX_NAME"
ex SFU_BATCH_SIZE "${SFU_BATCH_SIZE:-64}";   ex SFU_INDEX_SINK "$SFU_INDEX_SINK"
ex SFU_SPACES_BUCKET "$SFU_SPACES_BUCKET";   ex SFU_SPACES_REGION "$SFU_SPACES_REGION"
ex SFU_SPACES_ENDPOINT "$SFU_SPACES_ENDPOINT"; ex SFU_SPACES_KEY "$SFU_SPACES_KEY"
ex SFU_SPACES_SECRET "$SFU_SPACES_SECRET";   ex SFU_SPACES_PREFIX "${SFU_SPACES_PREFIX:-openalex_works_snapshot}"
USER_DATA="$(printf '#!/usr/bin/env bash\n%s\n%s\n' "$preamble" "$(tail -n +2 "$SCRIPT_DIR/cloud_init.sh")")"

# Create droplet (tagged; with ssh key + volume attached).
echo ">> creating droplet ($SFU_DO_SIZE @ $SFU_DO_REGION, tag $SFU_DO_TAG)..."
payload="$(USER_DATA="$USER_DATA" SSH_ID="$SSH_ID" VOL_ID="$VOL_ID" python3 - \
  "$SFU_DO_SIZE" "$SFU_DO_REGION" "$SFU_DO_IMAGE" "$SFU_DO_TAG" <<'PY'
import json,sys,os
size,region,image,tag=sys.argv[1:5]
d={"name":f"{tag}-{os.getpid()}","region":region,"size":size,"image":image,
   "tags":[tag],"user_data":os.environ.get("USER_DATA","")}
if os.environ.get("SSH_ID"): d["ssh_keys"]=[int(os.environ["SSH_ID"])]
if os.environ.get("VOL_ID"): d["volumes"]=[os.environ["VOL_ID"]]
print(json.dumps(d))
PY
)"
resp="$(curl -s --max-time 60 "${auth[@]}" -d "$payload" "$API/droplets")"
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
