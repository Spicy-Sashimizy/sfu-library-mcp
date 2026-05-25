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

notify() {  # $1=severity $2=message
  local sev="$1"; shift; local msg="$*"
  local line; line="$(date -u +%FT%TZ) [$sev] $msg"
  echo "$line" | tee -a "$ALERTS"
  [[ -n "$NOTIFY_WEBHOOK" ]] && curl -s --max-time 15 -H 'Title: TrueNAS monitor' \
       -H "Priority: $([[ $sev == CRIT ]] && echo urgent || echo default)" \
       -d "$line" "$NOTIFY_WEBHOOK" >/dev/null 2>&1 || true
}

assess() {  # feed check output to Claude for a concise human verdict (read-only tools only)
  local report="$1"
  # settings.json auto-loads from $HOME/.claude/settings.json (HOME=/agent in the image).
  if [[ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}${ANTHROPIC_API_KEY:-}" ]]; then echo "(no Claude auth; raw report)"; return; fi
  printf 'You are a read-only TrueNAS watchdog. Given this health report, reply in <=4 lines: severity (OK/WARN/CRIT), what is wrong, and the single safest suggested human action. Do NOT propose destructive commands.\n\n%s\n' "$report" \
    | timeout 120 claude -p 2>/dev/null || echo "(assessment unavailable)"
}

notify INFO "monitor started (interval ${INTERVAL}s, reactions=${REACTIONS_ENABLED:-false})"
last_heartbeat=0
while true; do
  report="$(bash /agent/checks.sh 2>&1)"; rc=$?
  now=$(date +%s)
  if (( rc != 0 )); then
    verdict="$(assess "$report")"
    sev=CRIT; echo "$verdict" | grep -qi 'WARN' && sev=WARN
    notify "$sev" "$(echo "$verdict" | tr '\n' ' ')"
    echo "$report" >> "$STATE/last_anomaly.txt"
    # Phase-2 reaction hook (disabled by default):
    # [[ "${REACTIONS_ENABLED:-false}" == "true" ]] && bash /agent/react.sh "$report"
  elif (( now - last_heartbeat >= HEARTBEAT_S )); then
    notify INFO "heartbeat: all clear"
    last_heartbeat=$now
  fi
  sleep "$INTERVAL"
done
