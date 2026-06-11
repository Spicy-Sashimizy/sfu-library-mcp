#!/usr/bin/env bash
# Stop hook: enforces CLAUDE.md "Session-end doc sync".
# A SessionStart hook records HEAD in .claude/.session_start_head; this hook
# diffs that baseline against HEAD when Claude tries to stop. If the session's
# commits touched src/ or scripts/ but never docs/, it blocks the stop ONCE
# and feeds the doc-sync instruction back to Claude (stop_hook_active guards
# against loops, so an intentional "docs unaffected" stop goes through).
set -u
input=$(cat 2>/dev/null || true)

# Loop guard: this stop is already a continuation from a Stop hook — warn once, never loop.
if printf '%s' "$input" | jq -e '.stop_hook_active == true' >/dev/null 2>&1; then
  exit 0
fi

root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
base_file="$root/.claude/.session_start_head"
[ -r "$base_file" ] || exit 0
base=$(cat "$base_file")
git -C "$root" rev-parse --verify --quiet "${base}^{commit}" >/dev/null 2>&1 || exit 0

changed=$(git -C "$root" diff --name-only "$base"..HEAD 2>/dev/null || true)
[ -n "$changed" ] || exit 0

code_n=$(printf '%s\n' "$changed" | grep -cE '^(src|scripts)/' || true)
docs_n=$(printf '%s\n' "$changed" | grep -cE '^docs/' || true)

if [ "${code_n:-0}" -gt 0 ] && [ "${docs_n:-0}" -eq 0 ]; then
  jq -n --arg n "$code_n" --arg base "$base" '{
    decision: "block",
    reason: ("Doc-sync check (CLAUDE.md, Session-end doc sync): commits since session start (\($base[0:9])) touched \($n) file(s) under src/ or scripts/ but ZERO files under docs/. Before stopping: update docs/THIN_CLIENT_SWAP.md / docs/STORAGE_BUDGET_150M.md / docs/README.md if this session changed architecture, measured numbers, code defaults, or added modules — and record measured efficacy (numbers + eval script + results path + date) for anything new. If the docs are genuinely unaffected (e.g. pure bugfix or data-only change), state that briefly and stop again; this check fires only once per stop.")
  }'
  exit 0
fi
exit 0
