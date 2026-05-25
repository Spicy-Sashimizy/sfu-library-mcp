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
# "bad" = unhealthy, OR not running but NOT a clean one-shot (Exited (0)).
# This ignores init/permissions sidecars (e.g. ix-*-permissions) that correctly
# exit 0 once at setup and stay stopped — those are normal, not anomalies.
JQ_BAD='def bad: (.Status|test("unhealthy")) or (.State!="running" and ((.Status|test("Exited \\(0\\)"))|not));'
if json="$(curl -s --max-time 15 "$PROXY/containers/json?all=1" 2>/dev/null)"; then
  echo "$json" | jq -r "$JQ_BAD"'
    .[] | [ (.Names[0]|ltrimstr("/")), .State, .Status, (if bad then "  <-- CHECK" else "" end) ] | @tsv' \
    | while IFS=$'\t' read -r name state status flag; do
        printf '  %-40s %-9s %s%s\n' "$name" "$state" "$status" "$flag"
      done
  if echo "$json" | jq -e "$JQ_BAD"' any(.[]; bad)' >/dev/null 2>&1; then
    anom=1; echo "  ANOMALY: a container is unhealthy or unexpectedly stopped"
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
  mx="${DO_MAX_DROPLET_HOURS:-6}"
  bal="$(curl -s --max-time 15 -H "Authorization: Bearer $DO_TOKEN" 'https://api.digitalocean.com/v2/customers/my/balance' 2>/dev/null)"
  echo "$bal" | jq -r '"  month_to_date_usage=$\(.month_to_date_usage // "?")  account_balance=$\(.account_balance // "?")"' 2>/dev/null || echo "  balance: unavailable"
  dj="$(curl -s --max-time 15 -H "Authorization: Bearer $DO_TOKEN" 'https://api.digitalocean.com/v2/droplets?per_page=200' 2>/dev/null)"
  echo "$dj" | jq -r --argjson mx "$mx" '
    ((.droplets // []) | map(select(.status=="active"))) as $a
    | "  active_droplets=\($a|length)",
      ($a[] | "    - \(.name) \(.size_slug) id=\(.id) age=\(((now-(.created_at|fromdateiso8601))/3600)|floor)h"),
      ($a[] | select((now-(.created_at|fromdateiso8601))/3600 > $mx) | "  RUNAWAY: \(.name) > \($mx)h — destroy it (teardown.sh)!"),
      "  active_ids=\($a|map(.id|tostring)|join(\",\"))"
  ' 2>/dev/null || echo "  droplets: unavailable"
  if echo "$dj" | jq -e --argjson mx "$mx" 'any((.droplets // [])[]; .status=="active" and ((now-(.created_at|fromdateiso8601))/3600 > $mx))' >/dev/null 2>&1; then
    anom=1
  fi
else
  echo "  (no /secrets/do_token — DO cost watch off)"
fi

# 4) Phase-2 host hooks (zpool status -x, smartctl -H) go here via a
#    command-restricted localhost SSH key. Intentionally absent in Phase 1.

[[ "$anom" -eq 0 ]] && echo "## result: ALL CLEAR" || echo "## result: ANOMALIES FOUND"
exit "$anom"
