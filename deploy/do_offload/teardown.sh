#!/usr/bin/env bash
# ============================================================================
# teardown.sh — COST LYNCHPIN. Idempotently destroy the SFU encode GPU droplet
# AND its tagged block volume (an orphaned/unattached volume keeps billing).
#
# Safe to run ANYTIME (idempotent). It only ever touches resources carrying our
# tag (default: sfu-encode) — it will NEVER touch your other DigitalOcean
# droplets/volumes. Run this if anything goes sideways so nothing sits idle billing.
#
#   bash deploy/do_offload/teardown.sh            # destroy tagged droplet + volume
#   bash deploy/do_offload/teardown.sh --dry-run  # list what WOULD be destroyed
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[[ -f "$SCRIPT_DIR/config.env" ]] && source "$SCRIPT_DIR/config.env" 2>/dev/null || true
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

# ── droplets carrying our tag ─────────────────────────────────────────────────
ids="$(curl -s --max-time 30 "${auth[@]}" "$API/droplets?tag_name=$TAG&per_page=200" \
       | python3 -c "import sys,json;print('\n'.join(str(d['id']) for d in json.load(sys.stdin).get('droplets',[])))" 2>/dev/null)"
# ── volumes carrying our tag (region-scoped list, filter by tag) ──────────────
vols="$(curl -s --max-time 30 "${auth[@]}" "$API/volumes?per_page=200" \
       | python3 -c "import sys,json,os;
tag=os.environ.get('TAG');
print('\n'.join(f\"{v['id']} {v['name']}\" for v in json.load(sys.stdin).get('volumes',[]) if tag in (v.get('tags') or [])))" 2>/dev/null)"

[[ -n "$ids" ]] && { echo "Droplets tagged '$TAG':"; echo "$ids" | sed 's/^/  id /'; } || echo "No droplets tagged '$TAG'."
[[ -n "$vols" ]] && { echo "Volumes tagged '$TAG':"; echo "$vols" | sed 's/^/  vol /'; } || echo "No volumes tagged '$TAG'."

if [[ -z "$ids" && -z "$vols" ]]; then echo "Already clean."; exit 0; fi
if (( DRY )); then echo "(--dry-run: not destroying)"; exit 0; fi

# Destroy droplets by tag (atomic, idempotent).
if [[ -n "$ids" ]]; then
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X DELETE "${auth[@]}" "$API/droplets?tag_name=$TAG")"
  if [[ "$code" == "204" ]]; then echo "Destroyed droplets tagged '$TAG' (HTTP 204)."
  else echo "WARN: delete-by-tag returned HTTP $code; destroying individually..." >&2
    for id in $ids; do
      c="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X DELETE "${auth[@]}" "$API/droplets/$id")"
      echo "  droplet $id -> HTTP $c"; done
  fi
fi

# Delete volumes (must be detached; droplet destroy detaches, but allow a brief settle).
if [[ -n "$vols" ]]; then
  sleep 8
  while read -r vid vname; do
    [[ -z "$vid" ]] && continue
    c="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X DELETE "${auth[@]}" "$API/volumes/$vid")"
    if [[ "$c" != "204" ]]; then   # likely still attached/settling — retry once
      sleep 12
      c="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 -X DELETE "${auth[@]}" "$API/volumes/$vid")"
    fi
    echo "  volume $vname ($vid) -> HTTP $c"
  done <<< "$vols"
fi
echo "Teardown complete."
