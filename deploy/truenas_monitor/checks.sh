#!/usr/bin/env bash
# Read-only health probes. Emits a compact report to stdout and sets exit code
# 0 = all clear, 1 = anomalies found. NEVER mutates anything (no docker control,
# no host writes). Used by monitor_loop.sh; also safe to run by hand.
set -uo pipefail
PROXY="${DOCKER_PROXY:-http://docker-socket-proxy:2375}"
DISK_WARN_PCT="${DISK_WARN_PCT:-88}"
anom=0

echo "### check @ $(date -u +%FT%TZ)"

# 1) Container health via the READ-ONLY socket proxy (GET only).
echo "## containers"
if json="$(curl -s --max-time 15 "$PROXY/containers/json?all=1" 2>/dev/null)"; then
  # name | state | health(if any). Flag anything not running, or unhealthy.
  echo "$json" | jq -r '.[] | [(.Names[0]|ltrimstr("/")), .State, (.Status)] | @tsv' \
    | while IFS=$'\t' read -r name state status; do
        flag=""
        [[ "$state" != "running" ]] && flag="  <-- NOT RUNNING"
        [[ "$status" == *"unhealthy"* ]] && flag="  <-- UNHEALTHY"
        printf '  %-40s %-9s %s%s\n' "$name" "$state" "$status" "$flag"
      done
  # anomaly if any container not running or unhealthy
  if echo "$json" | jq -e 'any(.[]; .State!="running" or (.Status|test("unhealthy")))' >/dev/null 2>&1; then
    anom=1; echo "  ANOMALY: one or more containers not running / unhealthy"
  fi
else
  anom=1; echo "  ANOMALY: cannot reach docker-socket-proxy at $PROXY"
fi

# 2) Dataset capacity (read-only mounts).
echo "## capacity"
while read -r used pct mnt; do
  [[ -z "${pct:-}" ]] && continue
  p="${pct%\%}"
  printf '  %-40s %s used (%s)\n' "$mnt" "$pct" "$used"
  if [[ "$p" =~ ^[0-9]+$ ]] && (( p >= DISK_WARN_PCT )); then
    anom=1; echo "  ANOMALY: $mnt at $pct (>= ${DISK_WARN_PCT}%)"
  fi
done < <(df -h --output=used,pcent,target 2>/dev/null | tail -n +2 | grep -E '/mnt/MAIN' || true)

# 3) DigitalOcean cost watch — "are credits being used?" (read-only API).
#    Reports month-to-date usage + any active droplets (with age), emits their
#    ids for the loop's new-droplet detection, and flags a RUNAWAY droplet
#    (older than DO_MAX_DROPLET_HOURS) as an anomaly so it can't quietly bill.
echo "## digitalocean"
DO_TOKEN="$(cat /secrets/do_token 2>/dev/null || true)"
if [[ -n "$DO_TOKEN" ]]; then
  do_out="$(DO_TOKEN="$DO_TOKEN" DO_MAX_HOURS="${DO_MAX_DROPLET_HOURS:-6}" python3 - <<'PY'
import os, json, urllib.request, datetime
tok=os.environ["DO_TOKEN"]; mx=float(os.environ.get("DO_MAX_HOURS","6")); rc=0
def get(u):
    r=urllib.request.Request(u, headers={"Authorization":"Bearer "+tok})
    return json.load(urllib.request.urlopen(r, timeout=15))
try:
    b=get("https://api.digitalocean.com/v2/customers/my/balance")
    print("  month_to_date_usage=$%s  account_balance=$%s" % (b.get("month_to_date_usage","?"), b.get("account_balance","?")))
except Exception as e:
    print("  balance: unavailable (%s)" % str(e)[:60])
try:
    d=get("https://api.digitalocean.com/v2/droplets?per_page=200").get("droplets",[])
    now=datetime.datetime.now(datetime.timezone.utc)
    active=[x for x in d if x.get("status")=="active"]
    print("  active_droplets=%d" % len(active))
    for x in active:
        age="?"
        try:
            t=datetime.datetime.fromisoformat(x.get("created_at","").replace("Z","+00:00"))
            h=(now-t).total_seconds()/3600; age="%.1fh"%h
            if h>mx: rc=2; print("  RUNAWAY: droplet %s up %s (> %sh) — destroy it (teardown.sh)!" % (x.get("name"), age, mx))
        except Exception: pass
        print("    - %s %s id=%s age=%s" % (x.get("name"), x.get("size_slug"), x.get("id"), age))
    print("  active_ids=" + ",".join(str(x.get("id")) for x in active))
except Exception as e:
    print("  droplets: unavailable (%s)" % str(e)[:60])
raise SystemExit(rc)
PY
)"; do_rc=$?
  echo "$do_out"
  [[ "$do_rc" == "2" ]] && { anom=1; }
else
  echo "  (no /secrets/do_token — DO cost watch off)"
fi

# 4) Phase-2 host hooks (zpool status -x, smartctl -H) go here via a
#    command-restricted localhost SSH key. Intentionally absent in Phase 1.

[[ "$anom" -eq 0 ]] && echo "## result: ALL CLEAR" || echo "## result: ANOMALIES FOUND"
exit "$anom"
