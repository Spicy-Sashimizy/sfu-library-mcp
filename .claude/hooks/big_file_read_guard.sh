#!/usr/bin/env bash
# PreToolUse(Read) hook: context-pollution guard for large files.
# Denies an UNBOUNDED Read of any file larger than CLAUDE_BIG_READ_MAX_BYTES
# (default 1 MB), regardless of where it lives — this generalizes the static
# permissions.deny globs to every current and future large file. A Read that
# passes an explicit `limit` is allowed (deliberate, bounded sampling).
set -u
input=$(cat 2>/dev/null || true)
fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
[ -n "$fp" ] && [ -f "$fp" ] || exit 0
limit=$(printf '%s' "$input" | jq -r '.tool_input.limit // empty' 2>/dev/null)
[ -n "$limit" ] && exit 0
max="${CLAUDE_BIG_READ_MAX_BYTES:-1048576}"
size=$(stat -c %s "$fp" 2>/dev/null || echo 0)
if [ "$size" -gt "$max" ]; then
  mb=$(awk "BEGIN{printf \"%.1f\", $size/1048576}")
  jq -n --arg fp "$fp" --arg mb "$mb" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: ("Blocked unbounded Read of \($fp) (\($mb) MB > 1 MB): dumping large data files pollutes context. Sample it instead — re-issue the Read with an explicit small limit/offset, or extract just the fields you need via jq / head / python json probing.")
    }
  }'
fi
exit 0
