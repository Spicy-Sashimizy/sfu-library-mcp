#!/bin/bash
# ==========================================================
# LSP-Pommel Symbol Bridge
# ==========================================================
# Extracts structured symbols (functions, classes, signatures)
# from source files and writes a JSON manifest for Pommel's
# agent to consume. Runs as a supervisord program on a
# configurable interval (default 300s / 5 min).
#
# Extraction tools:
#   Python (.py)  → python3 ast module (zero-dep)
#   Go (.go)      → gopls symbols (if available), else ctags
#   All others    → universal-ctags --output-format=json
#
# Outputs:
#   /home/vscode/.claudebox/pommel/symbols.json
#   /home/vscode/.claudebox/pommel/changed-files.json
# ==========================================================

set -o pipefail

# --- Configuration ---
PROJECT_NAME="${PROJECT_NAME:-project}"
INTERVAL="${SYMBOL_BRIDGE_INTERVAL:-300}"
WORKSPACE="/workspaces/${PROJECT_NAME}"
POMMEL_DIR="/home/vscode/.claudebox/pommel"
MANIFEST="${POMMEL_DIR}/symbols.json"
CHANGED_FILES="${POMMEL_DIR}/changed-files.json"
MTIME_CACHE="/tmp/lsp-bridge-mtimes"
MAX_FILE_SIZE=102400   # 100KB
MAX_FILES=200
LOG_PREFIX="[symbol-bridge]"

# Source extensions we care about
SOURCE_EXTS="py|go|js|jsx|ts|tsx|rs|c|cpp|h|hpp|cs|java|rb|php|swift|kt"

# --- Functions ---

log() { echo "${LOG_PREFIX} $(date '+%H:%M:%S') $*"; }

# Write JSON atomically: write to .tmp then mv
atomic_write() {
    local dest="$1"
    local content="$2"
    local tmp="${dest}.tmp"
    printf '%s' "$content" > "$tmp" && mv -f "$tmp" "$dest"
}

# Check if file should be ignored via .pommelignore
is_ignored() {
    local file="$1"
    local ignore_file="${WORKSPACE}/.pommelignore"
    if [ ! -f "$ignore_file" ]; then
        return 1  # not ignored
    fi
    # Simple glob matching — check each pattern
    while IFS= read -r pattern || [ -n "$pattern" ]; do
        pattern=$(echo "$pattern" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        [ -z "$pattern" ] && continue
        [[ "$pattern" == \#* ]] && continue
        # Use bash glob matching
        if [[ "$file" == $pattern ]]; then
            return 0  # ignored
        fi
    done < "$ignore_file"
    return 1
}

# Get file mtime as epoch seconds
get_mtime() {
    stat -c '%Y' "$1" 2>/dev/null || echo "0"
}

# Extract Python symbols using ast module
extract_python_symbols() {
    local file="$1"
    python3 -c "
import ast, json, sys

try:
    with open('$file', 'r', encoding='utf-8', errors='ignore') as f:
        tree = ast.parse(f.read())
except:
    sys.exit(1)

symbols = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        bases = ', '.join(
            getattr(b, 'id', getattr(b, 'attr', '?')) for b in node.bases
        )
        sig = f'class {node.name}({bases})' if bases else f'class {node.name}'
        symbols.append({
            'name': node.name,
            'kind': 'class',
            'line': node.lineno,
            'signature': sig
        })
    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
        args = []
        for a in node.args.args:
            ann = ''
            if a.annotation:
                if hasattr(a.annotation, 'id'):
                    ann = f': {a.annotation.id}'
                elif hasattr(a.annotation, 'attr'):
                    ann = f': {a.annotation.attr}'
            args.append(f'{a.arg}{ann}')
        ret = ''
        if node.returns:
            if hasattr(node.returns, 'id'):
                ret = f' -> {node.returns.id}'
            elif hasattr(node.returns, 'attr'):
                ret = f' -> {node.returns.attr}'
        prefix = 'async def' if isinstance(node, ast.AsyncFunctionDef) else 'def'
        sig = f'{prefix} {node.name}({", ".join(args)}){ret}'
        # Detect parent class
        parent = None
        for pnode in ast.walk(tree):
            if isinstance(pnode, ast.ClassDef):
                for item in pnode.body:
                    if item is node:
                        parent = pnode.name
                        break
        symbols.append({
            'name': node.name,
            'kind': 'function',
            'line': node.lineno,
            'signature': sig,
            'parent': parent
        })
print(json.dumps(symbols))
" 2>/dev/null
}

# Extract Go symbols using gopls or ctags
extract_go_symbols() {
    local file="$1"
    if command -v gopls &>/dev/null; then
        # gopls symbols outputs: "Name Kind Line:Col-Line:Col"
        gopls symbols "$file" 2>/dev/null | python3 -c "
import sys, json
symbols = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) >= 3:
        name = parts[0]
        kind = parts[1].lower()
        loc = parts[2]
        line_num = int(loc.split(':')[0]) if ':' in loc else 0
        symbols.append({'name': name, 'kind': kind, 'line': line_num})
print(json.dumps(symbols))
" 2>/dev/null && return 0
    fi
    # Fallback to ctags
    extract_ctags_symbols "$file"
}

# Extract symbols using universal-ctags
extract_ctags_symbols() {
    local file="$1"
    if ! command -v ctags &>/dev/null; then
        echo "[]"
        return 0
    fi
    ctags --output-format=json --fields=+nkS -f - "$file" 2>/dev/null | python3 -c "
import sys, json
symbols = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        sym = {
            'name': obj.get('name', ''),
            'kind': obj.get('kind', ''),
            'line': obj.get('line', 0)
        }
        if obj.get('signature'):
            sym['signature'] = obj['signature']
        if obj.get('scope'):
            sym['parent'] = obj['scope']
        if obj.get('scopeKind'):
            sym['parentKind'] = obj['scopeKind']
        symbols.append(sym)
    except json.JSONDecodeError:
        continue
print(json.dumps(symbols))
" 2>/dev/null
}

# Detect language from extension
detect_language() {
    local file="$1"
    case "${file##*.}" in
        py) echo "python" ;;
        go) echo "go" ;;
        js|jsx) echo "javascript" ;;
        ts|tsx) echo "typescript" ;;
        rs) echo "rust" ;;
        c|h) echo "c" ;;
        cpp|hpp|cc|cxx) echo "cpp" ;;
        cs) echo "csharp" ;;
        java) echo "java" ;;
        rb) echo "ruby" ;;
        php) echo "php" ;;
        swift) echo "swift" ;;
        kt) echo "kotlin" ;;
        *) echo "unknown" ;;
    esac
}

# Extract symbols for a file, choosing the best tool
extract_symbols() {
    local file="$1"
    local lang
    lang=$(detect_language "$file")

    case "$lang" in
        python)
            extract_python_symbols "$file"
            ;;
        go)
            extract_go_symbols "$file"
            ;;
        *)
            extract_ctags_symbols "$file"
            ;;
    esac
}

# --- Main loop ---

run_extraction() {
    if [ ! -d "$WORKSPACE" ]; then
        log "Workspace $WORKSPACE not found, waiting..."
        return
    fi

    cd "$WORKSPACE" || return

    mkdir -p "$POMMEL_DIR"

    # List source files (git-tracked if possible, else find)
    local file_list
    if git rev-parse --is-inside-work-tree &>/dev/null; then
        file_list=$(git ls-files 2>/dev/null | grep -E "\.(${SOURCE_EXTS})$" | head -n "$MAX_FILES")
    else
        file_list=$(find . -maxdepth 5 -type f \
            -regextype posix-extended -regex ".*\.(${SOURCE_EXTS})" \
            -not -path '*/node_modules/*' \
            -not -path '*/.venv/*' \
            -not -path '*/vendor/*' \
            -not -path '*/.git/*' \
            -not -path '*/dist/*' \
            -not -path '*/build/*' \
            2>/dev/null | sed 's|^\./||' | head -n "$MAX_FILES")
    fi

    if [ -z "$file_list" ]; then
        log "No source files found"
        return
    fi

    # Load previous mtimes
    declare -A prev_mtimes
    if [ -f "$MTIME_CACHE" ]; then
        while IFS='=' read -r key val; do
            prev_mtimes["$key"]="$val"
        done < "$MTIME_CACHE"
    fi

    local files_json="{}"
    local changed_list=""
    local total_files=0
    local total_symbols=0
    local methods_used=""

    while IFS= read -r file; do
        [ -z "$file" ] && continue
        local full_path="${WORKSPACE}/${file}"

        # Skip if doesn't exist or too large
        [ ! -f "$full_path" ] && continue
        local fsize
        fsize=$(stat -c '%s' "$full_path" 2>/dev/null || echo "0")
        [ "$fsize" -gt "$MAX_FILE_SIZE" ] && continue

        # Skip if in .pommelignore
        is_ignored "$file" && continue

        # Check mtime for incremental processing
        local mtime
        mtime=$(get_mtime "$full_path")
        if [ "${prev_mtimes[$file]:-0}" = "$mtime" ]; then
            # File unchanged — use existing manifest entry if available
            if [ -f "$MANIFEST" ]; then
                local existing
                existing=$(python3 -c "
import json, sys
try:
    with open('$MANIFEST') as f:
        m = json.load(f)
    e = m.get('files', {}).get('$file')
    if e:
        print(json.dumps(e))
    else:
        print('')
except:
    print('')
" 2>/dev/null)
                if [ -n "$existing" ] && [ "$existing" != "" ]; then
                    files_json=$(echo "$files_json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
d['$file'] = json.loads('$existing')
print(json.dumps(d))
" 2>/dev/null) || true
                    total_files=$((total_files + 1))
                    continue
                fi
            fi
        fi

        # Extract symbols
        local lang
        lang=$(detect_language "$file")
        local symbols
        symbols=$(extract_symbols "$full_path") || symbols="[]"

        # Validate JSON
        if ! echo "$symbols" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
            symbols="[]"
        fi

        local sym_count
        sym_count=$(echo "$symbols" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
        total_symbols=$((total_symbols + sym_count))
        total_files=$((total_files + 1))

        # Track extraction method
        case "$lang" in
            python) [[ "$methods_used" != *"ast"* ]] && methods_used="${methods_used:+$methods_used+}ast" ;;
            go)
                if command -v gopls &>/dev/null; then
                    [[ "$methods_used" != *"gopls"* ]] && methods_used="${methods_used:+$methods_used+}gopls"
                else
                    [[ "$methods_used" != *"ctags"* ]] && methods_used="${methods_used:+$methods_used+}ctags"
                fi
                ;;
            *) [[ "$methods_used" != *"ctags"* ]] && methods_used="${methods_used:+$methods_used+}ctags" ;;
        esac

        local now_iso
        now_iso=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

        # Add to files JSON
        files_json=$(python3 -c "
import json, sys
d = json.load(sys.stdin)
d['$file'] = {
    'language': '$lang',
    'symbols': json.loads('''$symbols'''),
    'extractedAt': '$now_iso'
}
print(json.dumps(d))
" <<< "$files_json" 2>/dev/null) || true

        # Track changed file
        changed_list="${changed_list:+$changed_list,}\"$file\""

        # Update mtime
        prev_mtimes["$file"]="$mtime"

    done <<< "$file_list"

    # Write mtime cache
    : > "$MTIME_CACHE"
    for key in "${!prev_mtimes[@]}"; do
        echo "${key}=${prev_mtimes[$key]}" >> "$MTIME_CACHE"
    done

    # Build final manifest
    local now_iso
    now_iso=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    local manifest_json
    manifest_json=$(python3 -c "
import json, sys
files = json.load(sys.stdin)
manifest = {
    'project': '$PROJECT_NAME',
    'generatedAt': '$now_iso',
    'version': '1.0.0',
    'files': files,
    'stats': {
        'totalFiles': $total_files,
        'totalSymbols': $total_symbols,
        'method': '${methods_used:-none}'
    }
}
print(json.dumps(manifest, indent=2))
" <<< "$files_json" 2>/dev/null)

    if [ -n "$manifest_json" ]; then
        atomic_write "$MANIFEST" "$manifest_json"
        chown vscode:vscode "$MANIFEST" 2>/dev/null || true
        log "Manifest updated: ${total_files} files, ${total_symbols} symbols (${methods_used:-none})"
    fi

    # Write changed-files.json if there were changes
    if [ -n "$changed_list" ]; then
        local changed_json
        changed_json=$(python3 -c "
import json
from datetime import datetime, timezone
print(json.dumps({
    'changedSince': '$(date -u '+%Y-%m-%dT%H:%M:%SZ')',
    'files': [$changed_list]
}))
" 2>/dev/null)
        if [ -n "$changed_json" ]; then
            atomic_write "$CHANGED_FILES" "$changed_json"
            chown vscode:vscode "$CHANGED_FILES" 2>/dev/null || true
            log "Changed files: $(echo "$changed_list" | tr ',' '\n' | wc -l | tr -d ' ')"
        fi
    fi
}

# --- Entry point ---

log "Starting (project=$PROJECT_NAME, interval=${INTERVAL}s)"

# Check tool availability
if command -v ctags &>/dev/null; then
    log "ctags: $(ctags --version 2>/dev/null | head -1)"
else
    log "[WARN] ctags not found — only Python ast extraction available"
fi

if command -v gopls &>/dev/null; then
    log "gopls: available"
fi

# Single-run mode (INTERVAL=0)
if [ "$INTERVAL" = "0" ]; then
    run_extraction
    exit 0
fi

# Continuous loop
while true; do
    run_extraction
    sleep "$INTERVAL"
done
