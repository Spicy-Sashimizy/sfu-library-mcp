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
  # Monitor's own RestartCount — lets a reader tell a REAL crash-loop (count
  # climbing) from a fresh redeploy (count 0, just low uptime). Informational.
  self_rc="$(curl -s --max-time 10 "$PROXY/containers/sfu-monitor/json" 2>/dev/null | jq -r '.RestartCount // "?"' 2>/dev/null)"
  echo "  monitor_self: RestartCount=${self_rc:-?} (a fresh deploy shows 0 with low uptime; a crash-loop keeps climbing)"
else
  anom=1; echo "  ANOMALY: cannot reach docker-socket-proxy at $PROXY"
fi

# 2) Dataset capacity (read-only mounts). Probe each mounted path EXPLICITLY —
#    grepping df's full table misses them, because inside the container the bind
#    mounts on the same ZFS pool get coalesced under one device line that isn't
#    labelled /mnt/MAIN. Querying the path directly reports its backing fs.
echo "## capacity"
got_cap=0
for path in /mnt/MAIN/sfu-library-mcp /agent/state; do
  [[ -e "$path" ]] || continue
  line="$(df -h --output=used,pcent,target "$path" 2>/dev/null | tail -n +2 | head -1)"
  [[ -z "$line" ]] && continue
  got_cap=1
  used="$(awk '{print $1}' <<<"$line")"; pct="$(awk '{print $2}' <<<"$line")"
  printf '  %-32s %s used (%s)\n' "$path" "$pct" "$used"
  p="${pct%\%}"
  if [[ "$p" =~ ^[0-9]+$ ]] && (( p >= DISK_WARN_PCT )); then
    anom=1; echo "  ANOMALY: $path at $pct (>= ${DISK_WARN_PCT}%)"
  fi
done
(( got_cap == 0 )) && { anom=1; echo "  ANOMALY: could not read any dataset capacity (df returned nothing)"; }

# 3) DigitalOcean cost watch — bulletproof "are credits being used?" (read-only API).
#    Design rule: a credit monitor that CAN'T SEE is worse than useless, so every
#    failure mode here is LOUD (sets anom=1), never silent — an unreachable API or
#    a bad token could be hiding a droplet that's billing. We check, in order:
#      - droplets: any active one (with age); RUNAWAY if older than DO_MAX_DROPLET_HOURS
#      - balance: month-to-date spend vs DO_BUDGET_USD cap
#      - orphaned billables: unattached volumes + reserved IPs (bill with no droplet)
#    Emits `active_ids=` for the loop's new-droplet detection and `do_check_ok=1`
#    ONLY when the droplet read truly succeeded (so the loop won't false-alarm on a blip).
echo "## digitalocean"
DO_TOKEN="$(cat /secrets/do_token 2>/dev/null || true)"
if [[ -z "$DO_TOKEN" ]]; then
  echo "  (no /secrets/do_token — DO cost watch off)"
else
  mx="${DO_MAX_DROPLET_HOURS:-6}"
  budget="${DO_BUDGET_USD:-20}"
  HTTP_CODE=""
  do_api() {  # $1=path -> prints body to stdout, sets HTTP_CODE, returns 0 only on 2xx (with retries)
    # Body goes to a temp file (-o) and the status to stdout (-w) so the JSON is
    # never contaminated by the status code — earlier in-string splitting left a
    # stray "200" that made jq choke. mktemp falls back to a pid path if absent.
    local path="$1" code attempt tmp
    tmp="$(mktemp 2>/dev/null || echo "/tmp/do_api.$$")"
    for attempt in 1 2 3; do
      code="$(curl -sS --max-time 20 -o "$tmp" -w '%{http_code}' \
              -H "Authorization: Bearer $DO_TOKEN" \
              "https://api.digitalocean.com${path}" 2>/dev/null)"
      HTTP_CODE="$code"
      if [[ "$code" =~ ^2[0-9][0-9]$ ]]; then cat "$tmp"; rm -f "$tmp"; return 0; fi
      sleep 2
    done
    cat "$tmp"; rm -f "$tmp"; return 1
  }
  do_ok=1

  # --- droplets: the thing that actually burns credits ---
  if dj="$(do_api '/v2/droplets?per_page=200')"; then
    echo "$dj" | jq -r --argjson mx "$mx" '
      ((.droplets // []) | map(select(.status=="active"))) as $a
      | "  active_droplets=\($a|length)",
        ($a[] | "    - \(.name) \(.size_slug) id=\(.id) age=\(((now-(.created_at|fromdateiso8601))/3600)|floor)h"),
        ($a[] | select((now-(.created_at|fromdateiso8601))/3600 > $mx) | "  RUNAWAY: \(.name) age>\($mx)h — destroy it (teardown.sh)!"),
        "  active_ids=\($a|map(.id|tostring)|join(","))"
    ' 2>/dev/null || { echo "  ANOMALY: DO droplets response unparseable"; do_ok=0; anom=1; }
    if echo "$dj" | jq -e --argjson mx "$mx" 'any((.droplets // [])[]; .status=="active" and ((now-(.created_at|fromdateiso8601))/3600 > $mx))' >/dev/null 2>&1; then
      anom=1
    fi
  else
    do_ok=0; anom=1
    echo "  ANOMALY: cannot read DO droplets API (HTTP ${HTTP_CODE:-none}) — can't confirm nothing is billing"
    [[ "$HTTP_CODE" == "401" ]] && echo "  (HTTP 401 — the do_token is invalid/expired; fix it or the cost watch is blind)"
  fi

  # --- balance / month-to-date spend vs budget cap ---
  if bal="$(do_api '/v2/customers/my/balance')"; then
    mtd="$(echo "$bal" | jq -r '.month_to_date_usage // empty' 2>/dev/null)"
    echo "  month_to_date_usage=\$${mtd:-?}  account_balance=\$$(echo "$bal" | jq -r '.account_balance // "?"' 2>/dev/null)"
    if [[ "$mtd" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
       && [[ "$(jq -n --argjson a "$mtd" --argjson b "$budget" '$a >= $b' 2>/dev/null)" == "true" ]]; then
      anom=1; echo "  ANOMALY: month-to-date DO usage \$$mtd >= budget \$$budget"
    fi
  else
    echo "  WARN: cannot read DO balance API (HTTP ${HTTP_CODE:-none}) — spend cap not verified this cycle"
  fi

  # --- orphaned billables: a destroyed droplet can leave a volume or reserved IP billing ---
  if vj="$(do_api '/v2/volumes?per_page=200')"; then
    nv="$(echo "$vj" | jq -r '(.volumes // []) | length' 2>/dev/null)"
    if [[ "$nv" =~ ^[0-9]+$ ]] && (( nv > 0 )); then
      echo "  volumes=$nv (billable):"
      echo "$vj" | jq -r '(.volumes // [])[] | "    - \(.name) \(.size_gigabytes)GB attached=\(((.droplet_ids // [])|length) > 0)"' 2>/dev/null
      if echo "$vj" | jq -e 'any((.volumes // [])[]; ((.droplet_ids // [])|length) == 0)' >/dev/null 2>&1; then
        anom=1; echo "  ANOMALY: an unattached DO volume is billing with no droplet"
      fi
    fi
  fi
  if rj="$(do_api '/v2/reserved_ips?per_page=200')"; then
    if echo "$rj" | jq -e 'any((.reserved_ips // [])[]; .droplet == null)' >/dev/null 2>&1; then
      anom=1; echo "  ANOMALY: an unattached DO reserved IP is billing"
    fi
  fi

  echo "  do_check_ok=$do_ok"
fi

# 4) Progress report (INFORMATIONAL — never sets an anomaly): summarize the
#    SPLADE snapshot + re-encode/index job from the status files the monitor can
#    read. These are mirrored from the not-always-on host, so we report HOW OLD
#    they are (and whether the heartbeat is live) rather than implying it's live.
#    Emits a single `progress_line=` the loop lifts into the heartbeat.
echo "## progress"
PROGRESS_DIR="${PROGRESS_DIR:-/mnt/MAIN/sfu-library-mcp/from-host/snapshots}"
now_s=$(date +%s); skept=0; prog_line=""
if [[ ! -d "$PROGRESS_DIR" ]]; then
  echo "  (no progress dir $PROGRESS_DIR)"; prog_line="no progress data (dir missing)"
else
  sf="$PROGRESS_DIR/snapshot_status.json"
  if [[ -f "$sf" ]]; then
    IFS=$'\t' read -r sstate sdone stot skept < <(jq -r '[.state,(.completed_parts//0),(.total_parts//0),(.total_kept//0)]|@tsv' "$sf" 2>/dev/null)
    printf '  snapshot: %s %s/%s parts (%sM docs)\n' "$sstate" "$sdone" "$stot" "$(awk -v k="${skept:-0}" 'BEGIN{printf "%.1f",k/1e6}')"
  fi
  xf="$PROGRESS_DIR/indexer_status.json"
  hb="$PROGRESS_DIR/indexer_heartbeat"
  if [[ -f "$xf" ]]; then
    # Freshness from the heartbeat EPOCH (tz-independent, bumped every few seconds
    # while encoding, and unaffected by rsync preserving the status file's mtime).
    # Fall back to the status file mtime only if there's no heartbeat.
    age_min=""
    if [[ -f "$hb" ]]; then
      hbts=$(cut -d. -f1 < "$hb" 2>/dev/null)
      [[ "$hbts" =~ ^[0-9]+$ ]] && age_min=$(( (now_s - hbts) / 60 ))
    fi
    [[ -z "$age_min" ]] && age_min=$(( (now_s - $(stat -c %Y "$xf" 2>/dev/null || echo "$now_s")) / 60 ))
    run_state=$([[ "$age_min" -le 5 ]] && echo RUNNING || echo 'idle/stale')
    IFS=$'\t' read -r istate fidx ftot indexed backend < <(jq -r '[.state,(.file_index//0),(.total_files//0),(.total_indexed//0),(.gpu.backend//"?")]|@tsv' "$xf" 2>/dev/null)
    pct=$(awk -v a="${indexed:-0}" -v b="${skept:-0}" 'BEGIN{if(b>0)printf "%.1f",a*100/b; else printf "?"}')
    fpct=$(awk -v a="${fidx:-0}" -v b="${ftot:-0}" 'BEGIN{if(b>0)printf "%.0f",a*100/b; else printf "?"}')
    idxM=$(awk -v a="${indexed:-0}" 'BEGIN{printf "%.2f",a/1e6}'); keptM=$(awk -v b="${skept:-0}" 'BEGIN{printf "%.1f",b/1e6}')
    printf '  indexer: %s [%s]  file %s/%s (%s%%)  docs %sM/%sM (%s%%)  backend %s  last beat %sm ago\n' \
      "$istate" "$run_state" "$fidx" "$ftot" "$fpct" "$idxM" "$keptM" "$pct" "$backend" "$age_min"
    prog_line="indexer ${istate} [${run_state}] ${idxM}M/${keptM}M docs (${pct}%), file ${fidx}/${ftot}, last beat ${age_min}m ago"
  else
    echo "  (no indexer_status.json)"; prog_line="no indexer status"
  fi
fi
echo "  progress_line=${prog_line}"

# 5) Phase-2 host hooks (zpool status -x, smartctl -H) go here via a
#    command-restricted localhost SSH key. Intentionally absent in Phase 1.

[[ "$anom" -eq 0 ]] && echo "## result: ALL CLEAR" || echo "## result: ANOMALIES FOUND"
exit "$anom"
