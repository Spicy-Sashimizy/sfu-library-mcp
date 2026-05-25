#!/usr/bin/env bash
# ============================================================================
# teardown.sh — COST LYNCHPIN. Idempotently destroy the SFU encode GPU droplet.
#
# Safe to run ANYTIME (idempotent). It only ever destroys droplets carrying our
# tag (default: sfu-encode) — it will NEVER touch your other DigitalOcean
# droplets. Run this if anything goes sideways so a GPU droplet can't sit idle
# billing.
#
#   bash deploy/do_offload/teardown.sh            # destroy tagged droplets
#   bash deploy/do_offload/teardown.sh --dry-run  # list what WOULD be destroyed
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TAG="${SFU_DO_TAG:-sfu-encode}"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1

# Load token from env or .env (never printed).
TOKEN="${DIGITALOCEAN_ACCESS_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$REPO_ROOT/.env" ]]; then
  TOKEN="$(grep -E '^[[:space:]]*DIGITALOCEAN_ACCESS_TOKEN=' "$REPO_ROOT/.env" | head -1 \
           | sed -E 's/^[^=]+=//; s/[[:space:]]+#.*$//; s/^["'"'"']//; s/["'"'"']$//; s/[[:space:]]+$//')"
fi
[[ -n "$TOKEN" ]] || { echo "ERROR: no DIGITALOCEAN_ACCESS_TOKEN in env or .env" >&2; exit 1; }

API="https://api.digitalocean.com/v2"
auth=(-H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json")

# Find droplet IDs carrying our tag.
ids="$(curl -s --max-time 30 "${auth[@]}" "$API/droplets?tag_name=$TAG&per_page=200" \
       | python3 -c "import sys,json;print('\n'.join(str(d['id']) for d in json.load(sys.stdin).get('droplets',[])))" 2>/dev/null)"

if [[ -z "$ids" ]]; then
  echo "No droplets tagged '$TAG'. Nothing to destroy (already clean)."
  exit 0
fi

echo "Droplets tagged '$TAG':"; echo "$ids" | sed 's/^/  id /'
if (( DRY )); then echo "(--dry-run: not destroying)"; exit 0; fi

# Destroy by tag in one call (atomic, idempotent).
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X DELETE "${auth[@]}" \
        "$API/droplets?tag_name=$TAG")"
if [[ "$code" == "204" ]]; then
  echo "Destroyed all droplets tagged '$TAG' (HTTP 204)."
else
  echo "WARN: delete-by-tag returned HTTP $code; destroying individually..." >&2
  for id in $ids; do
    c="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X DELETE "${auth[@]}" "$API/droplets/$id")"
    echo "  droplet $id -> HTTP $c"
  done
fi
echo "Teardown complete."
