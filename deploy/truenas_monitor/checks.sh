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

# 3) Phase-2 host hooks (zpool status -x, smartctl -H) go here via a
#    command-restricted localhost SSH key. Intentionally absent in Phase 1.

[[ "$anom" -eq 0 ]] && echo "## result: ALL CLEAR" || echo "## result: ANOMALIES FOUND"
exit "$anom"
