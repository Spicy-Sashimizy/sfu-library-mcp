#!/bin/bash
# PostToolUse hook: auto-run LSP diagnostics after Edit/Write
# Queries the LSP MCP server's HTTP sidecar for type errors/warnings

FILE="$TOOL_FILE_PATH"
[ -z "$FILE" ] && exit 0
[ ! -f "$FILE" ] && exit 0

# Only supported file types
case "${FILE##*.}" in
    py|js|jsx|ts|tsx|go|rs|c|cpp|h|hpp|cc) ;;
    *) exit 0 ;;
esac

# Query LSP MCP server sidecar (up to 20s for cold start)
RESULT=$(curl -s --max-time 20 --unix-socket /tmp/lsp-mcp.sock \
    -X POST http://localhost/diagnostics \
    -H "Content-Type: application/json" \
    -d "{\"file_path\": \"$FILE\"}" 2>/dev/null)

[ -z "$RESULT" ] || [ "$RESULT" = "[]" ] && exit 0

echo "[LSP] Diagnostics for $(basename "$FILE"):"
echo "$RESULT" | python3 -c "
import json, sys
for d in json.load(sys.stdin):
    sev = {1:'ERROR',2:'WARN',3:'INFO',4:'HINT'}.get(d.get('severity',3),'INFO')
    print(f'  {sev} line {d[\"line\"]}: {d[\"message\"]}')
" 2>/dev/null || true
