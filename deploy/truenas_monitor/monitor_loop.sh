#!/usr/bin/env bash
# Entrypoint for the TrueNAS monitor agent. Each cycle:
#   1) run read-only checks.sh
#   2) on anomaly (or periodic heartbeat) ask Claude Code to assess + write a
#      concise alert; notify via webhook + append to state/alerts.log
# Reactions are NOTIFY-ONLY unless REACTIONS_ENABLED=true AND an allowlist exists
# (Phase 2). This loop itself never stops/kills containers.
set -uo pipefail
STATE=/agent/state; mkdir -p "$STATE"
ALERTS="$STATE/alerts.log"
INTERVAL="${CHECK_INTERVAL:-300}"
HEARTBEAT_S=$(( ${HEARTBEAT_HOURS:-24} * 3600 ))

# Auth: prefer a subscription (Pro/Max) headless token from `claude setup-token`
# (CLAUDE_CODE_OAUTH_TOKEN). Fall back to an API key only if that's what's provided.
if [[ -s /secrets/claude_oauth_token ]]; then
  export CLAUDE_CODE_OAUTH_TOKEN="$(cat /secrets/claude_oauth_token)"
elif [[ -s /secrets/anthropic.key ]]; then
  export ANTHROPIC_API_KEY="$(cat /secrets/anthropic.key)"
fi
NOTIFY_WEBHOOK="${NOTIFY_WEBHOOK:-$(cat /secrets/notify_webhook 2>/dev/null || true)}"
# Bearer token for the self-hosted, auth-locked ntfy (exposed via Cloudflare->NPM).
NOTIFY_TOKEN="${NOTIFY_TOKEN:-$(cat /secrets/notify_token 2>/dev/null || true)}"

notify() {  # $1=severity $2=message
  local sev="$1"; shift; local msg="$*"
  local line; line="$(date -u +%FT%TZ) [$sev] $msg"
  echo "$line" | tee -a "$ALERTS"
  [[ -z "$NOTIFY_WEBHOOK" ]] && return 0
  local h=(-H 'Title: TrueNAS monitor' -H "Priority: $([[ $sev == CRIT ]] && echo urgent || echo default)")
  [[ -n "$NOTIFY_TOKEN" ]] && h+=(-H "Authorization: Bearer $NOTIFY_TOKEN")
  curl -s --max-time 15 "${h[@]}" -d "$line" "$NOTIFY_WEBHOOK" >/dev/null 2>&1 || true
}

assess() {  # feed check output to Claude for a concise human verdict (read-only tools only)
  local report="$1"
  # settings.json auto-loads from $HOME/.claude/settings.json (HOME=/agent in the image).
  if [[ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}${ANTHROPIC_API_KEY:-}" ]]; then echo "(no Claude auth; raw report follows) $report"; return; fi
  printf 'You are a read-only watchdog for a TrueNAS server and its DigitalOcean usage. From the health report below, reply in <=6 short lines:\n- overall severity: OK / WARN / CRIT\n- anything wrong or noteworthy: a container not running/unhealthy, a dataset near full, OR a DigitalOcean droplet consuming credits (give its name + age; mark CRIT if a RUNAWAY line is present)\n- PROGRESS: one line from the ## progress section — the SPLADE re-encode/index job state, docs indexed and %% done, and whether it is RUNNING vs idle/stale\n- the single safest human action, if any (never a destructive command)\nDo NOT call a container crash-loop from low uptime alone: a freshly (re)deployed container normally shows low uptime with monitor_self RestartCount=0 — only flag a crash-loop if RestartCount is elevated/climbing.\nIf all is well health-wise, say so in one line but STILL give the progress line. Report:\n\n%s\n' "$report" \
    | timeout 150 claude -p 2>/dev/null || echo "(assessment unavailable) $report"
}

SPEND_DELTA="${DO_SPEND_ALERT_DELTA:-1.0}"   # $ rise in month-to-date usage that (re)fires a "credits used" alert

notify INFO "monitor started (checks ${INTERVAL}s, progress every $(( HEARTBEAT_S/3600 ))h, reactions=${REACTIONS_ENABLED:-false})"
last_heartbeat=$(date +%s)   # don't fire a heartbeat immediately on boot
touch "$STATE/seen_droplets"
while true; do
  report="$(bash /agent/checks.sh 2>&1)"; rc=$?
  now=$(date +%s)

  # --- DigitalOcean credit alerts ---------------------------------------------
  # Only act on droplet/spend state when the DO read actually SUCCEEDED this cycle
  # (do_check_ok=1). A transient API/network blip must not wipe seen_droplets and
  # then re-fire "new droplet" for everything next cycle.
  if echo "$report" | grep -q 'do_check_ok=1'; then
    # (a) NEW droplet id(s) appeared -> a machine was just created, credits start now.
    cur_ids="$(echo "$report" | grep -oE 'active_ids=[^[:space:]]*' | cut -d= -f2 | tr ',' '\n' | grep -E '^[0-9]+$' | sort -u)"
    new_ids="$(comm -13 <(sort -u "$STATE/seen_droplets" 2>/dev/null) <(echo "$cur_ids") 2>/dev/null | grep -E '^[0-9]+$' || true)"
    echo "$cur_ids" > "$STATE/seen_droplets"
    if [[ -n "$new_ids" ]]; then
      notify WARN "DigitalOcean: new droplet id(s) $(echo $new_ids) — credits now IN USE. $(assess "$report" | tr '\n' ' ')"
    fi
    # (b) month-to-date spend rose -> credits are actively being consumed (covers
    #     non-droplet usage too). Fire on the first dollar from zero, or any jump
    #     >= DO_SPEND_ALERT_DELTA; reset the baseline if DO rolled the billing month.
    cur_mtd="$(echo "$report" | grep -oE 'month_to_date_usage=\$[0-9]+(\.[0-9]+)?' | head -1 | sed 's/.*\$//')"
    if [[ "$cur_mtd" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      prev_mtd="$(cat "$STATE/last_mtd" 2>/dev/null)"; [[ "$prev_mtd" =~ ^[0-9]+(\.[0-9]+)?$ ]] || prev_mtd=0
      if [[ "$(jq -n --argjson c "$cur_mtd" --argjson p "$prev_mtd" '$c < $p' 2>/dev/null)" == "true" ]]; then
        echo "$cur_mtd" > "$STATE/last_mtd"   # billing month rolled over; rebase quietly
      elif [[ "$(jq -n --argjson c "$cur_mtd" --argjson p "$prev_mtd" --argjson d "$SPEND_DELTA" '(($c - $p) >= $d) or ($p == 0 and $c > 0)' 2>/dev/null)" == "true" ]]; then
        notify WARN "DigitalOcean: credits used — month-to-date \$$prev_mtd -> \$$cur_mtd."
        echo "$cur_mtd" > "$STATE/last_mtd"
      fi
    fi
  fi

  # --- important events: container down / disk full / RUNAWAY droplet ---
  if (( rc != 0 )); then
    verdict="$(assess "$report")"
    sev=CRIT; echo "$verdict" | grep -qi 'WARN' && sev=WARN
    notify "$sev" "$(echo "$verdict" | tr '\n' ' ')"
    echo "$(date -u +%FT%TZ)"$'\n'"$report" >> "$STATE/last_anomaly.txt"
    # Phase-2 reaction hook (disabled by default):
    # [[ "${REACTIONS_ENABLED:-false}" == "true" ]] && bash /agent/react.sh "$report"

  # --- heartbeat every HEARTBEAT_S (default 5h): health + a concrete progress report ---
  elif (( now - last_heartbeat >= HEARTBEAT_S )); then
    # Lift the deterministic progress line straight from the report so the numbers
    # are always present even if the Claude prose rewords them.
    prog="$(echo "$report" | grep -oE 'progress_line=.*' | head -1 | sed 's/^progress_line=//')"
    notify INFO "$(( HEARTBEAT_S/3600 ))h heartbeat — PROGRESS: ${prog:-n/a} || HEALTH: $(assess "$report" | tr '\n' ' ')"
    last_heartbeat=$now
  fi
  sleep "$INTERVAL"
done
