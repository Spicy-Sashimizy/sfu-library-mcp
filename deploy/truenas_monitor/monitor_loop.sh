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
  printf 'You are a read-only watchdog for a TrueNAS server and its DigitalOcean usage. From the health report below, reply in <=5 short lines:\n- overall severity: OK / WARN / CRIT\n- anything wrong or noteworthy: a container not running/unhealthy, a dataset near full, OR a DigitalOcean droplet consuming credits (give its name + age; mark CRIT if a RUNAWAY line is present)\n- the single safest human action, if any (never a destructive command)\nIf all is well, say so in one line. Report:\n\n%s\n' "$report" \
    | timeout 150 claude -p 2>/dev/null || echo "(assessment unavailable) $report"
}

notify INFO "monitor started (checks ${INTERVAL}s, progress every $(( HEARTBEAT_S/3600 ))h, reactions=${REACTIONS_ENABLED:-false})"
last_heartbeat=$(date +%s)   # don't fire a heartbeat immediately on boot
touch "$STATE/seen_droplets"
while true; do
  report="$(bash /agent/checks.sh 2>&1)"; rc=$?
  now=$(date +%s)

  # --- DigitalOcean: alert the moment a NEW droplet appears (credits start) ---
  cur_ids="$(echo "$report" | grep -oE 'active_ids=[^[:space:]]*' | cut -d= -f2 | tr ',' '\n' | grep -E '^[0-9]+$' | sort -u)"
  new_ids="$(comm -13 <(sort -u "$STATE/seen_droplets" 2>/dev/null) <(echo "$cur_ids") 2>/dev/null | grep -E '^[0-9]+$' || true)"
  echo "$cur_ids" > "$STATE/seen_droplets"
  if [[ -n "$new_ids" ]]; then
    notify WARN "DigitalOcean: credits now IN USE — new droplet id(s) $(echo $new_ids). $(assess "$report" | tr '\n' ' ')"
  fi

  # --- important events: container down / disk full / RUNAWAY droplet ---
  if (( rc != 0 )); then
    verdict="$(assess "$report")"
    sev=CRIT; echo "$verdict" | grep -qi 'WARN' && sev=WARN
    notify "$sev" "$(echo "$verdict" | tr '\n' ' ')"
    echo "$(date -u +%FT%TZ)"$'\n'"$report" >> "$STATE/last_anomaly.txt"
    # Phase-2 reaction hook (disabled by default):
    # [[ "${REACTIONS_ENABLED:-false}" == "true" ]] && bash /agent/react.sh "$report"

  # --- progress update every HEARTBEAT_S (default 5h): a real status summary ---
  elif (( now - last_heartbeat >= HEARTBEAT_S )); then
    notify INFO "$(( HEARTBEAT_S/3600 ))h status — $(assess "$report" | tr '\n' ' ')"
    last_heartbeat=$now
  fi
  sleep "$INTERVAL"
done
