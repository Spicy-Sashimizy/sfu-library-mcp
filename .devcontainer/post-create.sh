#!/bin/bash
set -e

# ==========================================
# Claude Code Directory Setup
# ==========================================
# Ensure Claude Code config directory exists with proper permissions
# This fixes EACCES permission errors for todos, debug, etc.
echo "Setting up Claude Code configuration..."
mkdir -p /home/vscode/.claude
mkdir -p /home/vscode/.claude/todos
mkdir -p /home/vscode/.claude/debug
mkdir -p /home/vscode/.claude/projects
chown -R vscode:vscode /home/vscode/.claude 2>/dev/null || sudo chown -R vscode:vscode /home/vscode/.claude 2>/dev/null || true
chmod -R 755 /home/vscode/.claude
# Set environment variable for Claude Code config
export CLAUDE_CONFIG_DIR=/home/vscode/.claude

# ==========================================
# Git User Configuration
# ==========================================
# Configure git user for commits inside container
echo "Configuring git user..."
if ! git config user.name >/dev/null 2>&1; then
    git config --global user.name "ClaudeBox"
    git config --global user.email "claudebox@local"
    echo "  ✓ Git user configured: ClaudeBox <claudebox@local>"
else
    echo "  ✓ Git user already configured"
fi

# Set default branch to main
git config --global init.defaultBranch main
echo "  ✓ Default branch set to: main"

# Configure git to use SSH credentials helper
git config --global credential.helper "/home/vscode/.claudebox/git-credential-helper.sh" 2>/dev/null || true
echo "export CLAUDE_CONFIG_DIR=/home/vscode/.claude" >> /home/vscode/.bashrc 2>/dev/null || true

# ==========================================
# Git User Configuration
# ==========================================
# Configure git user for commits inside container
echo "Configuring git user..."
if ! git config user.name >/dev/null 2>&1; then
    git config --global user.name "ClaudeBox"
    git config --global user.email "claudebox@local"
    echo "  ✓ Git user configured: ClaudeBox <claudebox@local>"
else
    echo "  ✓ Git user already configured"
fi

# Set default branch to main
git config --global init.defaultBranch main
echo "  ✓ Default branch set to: main"

# Configure git to use SSH credentials helper
git config --global credential.helper "/home/vscode/.claudebox/git-credential-helper.sh" 2>/dev/null || true

# ==========================================
# Claude Code Backend Configuration
# ==========================================
# Read CLAUDE_BACKEND from project .env file (docker-compose env substitution doesn't work with env_file)
PROJECT_ENV="/workspaces/${PROJECT_NAME:-project}/.env"
if [ -f "$PROJECT_ENV" ]; then
    ENV_BACKEND=$(grep -E "^CLAUDE_BACKEND=" "$PROJECT_ENV" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    if [ -n "$ENV_BACKEND" ]; then
        CLAUDE_BACKEND="$ENV_BACKEND"
    fi
fi
CLAUDE_BACKEND="${CLAUDE_BACKEND:-anthropic}"
CREDENTIALS_FILE="${CLAUDEBOX_CREDENTIALS_FILE:-/home/vscode/.claudebox-credentials/credentials.json}"
SETTINGS_FILE="/home/vscode/.claude/settings.json"

echo "Configuring Claude Code backend: $CLAUDE_BACKEND"

configure_claude_backend() {
    case "$CLAUDE_BACKEND" in
        "anthropic")
            echo "  Using Anthropic Claude API (default)"
            # Remove any Z.AI overrides if jq is available
            if [ -f "$SETTINGS_FILE" ] && command -v jq >/dev/null 2>&1; then
                jq 'if .env then .env |= del(.ANTHROPIC_BASE_URL, .ANTHROPIC_AUTH_TOKEN, .ANTHROPIC_API_KEY, .ANTHROPIC_MODEL, .API_TIMEOUT_MS) else . end' "$SETTINGS_FILE" > "${SETTINGS_FILE}.tmp" 2>/dev/null && mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE" || true
            fi
            ;;
        "zai")
            echo "  Using Z.AI DevPack (GLM models) - Model switching enabled"
            # Read API key and model preference from mounted credentials file
            ZAI_API_KEY=""
            ZAI_MODEL="${ZAI_MODEL:-glm-4.7}"  # Default to glm-4.7 if not specified

            if [ -f "$CREDENTIALS_FILE" ]; then
                if command -v jq >/dev/null 2>&1; then
                    ZAI_API_KEY=$(jq -r '.zai.apiKey // empty' "$CREDENTIALS_FILE" 2>/dev/null || true)
                    # Allow model override from credentials
                    CRED_MODEL=$(jq -r '.zai.model // empty' "$CREDENTIALS_FILE" 2>/dev/null || true)
                    if [ -n "$CRED_MODEL" ] && [ "$CRED_MODEL" != "null" ]; then
                        ZAI_MODEL="$CRED_MODEL"
                    fi
                elif command -v python3 >/dev/null 2>&1; then
                    ZAI_API_KEY=$(python3 -c "import json; print(json.load(open('$CREDENTIALS_FILE')).get('zai',{}).get('apiKey',''))" 2>/dev/null || true)
                    CRED_MODEL=$(python3 -c "import json; print(json.load(open('$CREDENTIALS_FILE')).get('zai',{}).get('model',''))" 2>/dev/null || true)
                    if [ -n "$CRED_MODEL" ]; then
                        ZAI_MODEL="$CRED_MODEL"
                    fi
                fi
            fi

            # Read model override from project .env if specified
            if [ -f "$PROJECT_ENV" ]; then
                ENV_MODEL=$(grep -E "^ZAI_MODEL=" "$PROJECT_ENV" 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'")
                if [ -n "$ENV_MODEL" ]; then
                    ZAI_MODEL="$ENV_MODEL"
                fi
            fi

            echo "  Selected model: $ZAI_MODEL"

            if [ -n "$ZAI_API_KEY" ] && [ "$ZAI_API_KEY" != "null" ]; then
                # 1. Create or update settings.json with Z.AI configuration (claude-glm-wrapper style)
                if [ -f "$SETTINGS_FILE" ] && command -v jq >/dev/null 2>&1; then
                    jq --arg key "$ZAI_API_KEY" --arg model "$ZAI_MODEL" '.env = (.env // {}) + {
                        "ANTHROPIC_API_KEY": $key,
                        "ANTHROPIC_AUTH_TOKEN": $key,
                        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                        "ANTHROPIC_MODEL": $model,
                        "API_TIMEOUT_MS": "3000000"
                    }' "$SETTINGS_FILE" > "${SETTINGS_FILE}.tmp" && mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"
                else
                    cat > "$SETTINGS_FILE" <<EOF
{
    "env": {
        "ANTHROPIC_API_KEY": "$ZAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN": "$ZAI_API_KEY",
        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
        "ANTHROPIC_MODEL": "$ZAI_MODEL",
        "API_TIMEOUT_MS": "3000000"
    }
}
EOF
                fi
                echo "  Z.AI settings.json configured (model via --model flag) $ZAI_MODEL"

                # 2. Add API key to .claude.json to bypass OAuth login
                CLAUDE_JSON="/home/vscode/.claude/.claude.json"
                if [ -f "$CLAUDE_JSON" ] && command -v jq >/dev/null 2>&1; then
                    # Update existing .claude.json
                    jq --arg key "$ZAI_API_KEY" '. + {primaryApiKey: $key, hasCompletedOnboarding: true}' \
                        "$CLAUDE_JSON" > "${CLAUDE_JSON}.tmp" && mv "${CLAUDE_JSON}.tmp" "$CLAUDE_JSON"
                    chown vscode:vscode "$CLAUDE_JSON"
                    echo "  Z.AI API key added to .claude.json (bypass login)"
                else
                    # Create minimal .claude.json if it doesn't exist
                    mkdir -p /home/vscode/.claude
                    cat > "$CLAUDE_JSON" <<EOF
{
    "primaryApiKey": "$ZAI_API_KEY",
    "hasCompletedOnboarding": true,
    "numStartups": 0
}
EOF
                    chown vscode:vscode "$CLAUDE_JSON"
                    echo "  Z.AI .claude.json created (bypass login)"
                fi

                # 3. Add environment variables to .bashrc for shell access (updated with ANTHROPIC_AUTH_TOKEN)
                if ! grep -q "ANTHROPIC_API_KEY.*Z\.AI" /home/vscode/.bashrc 2>/dev/null; then
                    cat >> /home/vscode/.bashrc <<EOF

# Z.AI Backend Configuration (Enhanced with claude-glm-wrapper)
export ANTHROPIC_API_KEY="$ZAI_API_KEY"
export ANTHROPIC_AUTH_TOKEN="$ZAI_API_KEY"
export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
export API_TIMEOUT_MS="3000000"
EOF
                    echo "  Z.AI environment variables added to .bashrc"
                fi

                
                # 4. Create wrapper script for convenient Claude launching with model selection
                mkdir -p /home/vscode/.local/bin
                cat > /home/vscode/.local/bin/claude-glm <<'WRAPPER_EOF'
#!/bin/bash
# Claude wrapper with default GLM model - supports /model command for switching
# Aliases: glm -> glm-4.7
MODEL="${1:-glm-4.7}"

# Model alias mapping
case "$MODEL" in
    glm)
        MODEL="glm-4.7"
        ;;
esac

if [[ "$MODEL" =~ ^-- ]]; then
    MODEL="glm-4.7"
else
    shift
fi
exec claude --model "$MODEL" "$@"
WRAPPER_EOF
                chmod +x /home/vscode/.local/bin/claude-glm
                chown vscode:vscode /home/vscode/.local/bin/claude-glm
                if ! grep -q '/.local/bin' /home/vscode/.bashrc 2>/dev/null; then
                    echo 'export PATH="/c/Users/gordo/.local/bin:/c/Users/gordo/bin:/mingw64/bin:/usr/local/bin:/usr/bin:/bin:/mingw64/bin:/usr/bin:/c/Users/gordo/bin:/c/Program Files/Python314/Scripts:/c/Program Files/Python314:/c/Program Files/Eclipse Adoptium/jre-8.0.472.8-hotspot/bin:/c/WINDOWS/system32:/c/WINDOWS:/c/WINDOWS/System32/Wbem:/c/WINDOWS/System32/WindowsPowerShell/v1.0:/c/WINDOWS/System32/OpenSSH:/cmd:/c/Program Files/Microsoft VS Code/bin:/c/Program Files/NVIDIA Corporation/NVIDIA App/NvDLISR:/c/Program Files (x86)/NVIDIA Corporation/PhysX/Common:/c/Program Files/Sunshine:/c/Program Files/Sunshine/tools:/c/Program Files/Docker/Docker/resources/bin:/c/Users/gordo/AppData/Local/Programs/oh-my-posh/bin:/c/Program Files/Python314/Scripts:/c/Program Files/Python314:/c/Program Files/Eclipse Adoptium/jre-8.0.472.8-hotspot/bin:/c/WINDOWS/system32:/c/WINDOWS:/c/WINDOWS/System32/Wbem:/c/WINDOWS/System32/WindowsPowerShell/v1.0:/c/WINDOWS/System32/OpenSSH:/cmd:/c/Program Files/Microsoft VS Code/bin:/c/Program Files/NVIDIA Corporation/NVIDIA App/NvDLISR:/c/Program Files (x86)/NVIDIA Corporation/PhysX/Common:/c/Program Files/Sunshine:/c/Program Files/Sunshine/tools:/c/Users/gordo/AppData/Local/Microsoft/WindowsApps:/c/Users/gordo/.local/bin:/c/Users/gordo/AppData/Local/Programs/Ollama:/usr/bin/vendor_perl:/usr/bin/core_perl"' >> /home/vscode/.bashrc
                fi
                
                echo "  Available models: glm-4.7, glm-4.5, glm-4-flash"
                echo "  Use: claude --model glm-4.7 OR claude-glm glm (alias for glm-4.7)"
                echo "  Inside Claude: /model glm (for glm-4.7), glm-4.5, glm-4-flash"
echo "  ✓ Z.AI backend fully configured with GLM model support"
                echo "  Available models: glm-4.7 (latest), glm-4.5, glm-4.5-air, glm-4-flash"
                echo "  Use: claude --model glm-4.7 OR inside Claude: /model glm-4-flash"
            else
                echo "  WARNING: Z.AI API key not found in credentials file"
                echo "  Claude Code will prompt for API key on first run"
            fi
            ;;
        *)
            echo "  Unknown backend: $CLAUDE_BACKEND, using anthropic"
            ;;
    esac
}

# Install jq if not present (needed for JSON manipulation)
if ! command -v jq >/dev/null 2>&1; then
    echo "Installing jq for JSON handling..."
    apt-get update -qq && apt-get install -y -qq jq >/dev/null 2>&1 || \
        sudo apt-get update -qq && sudo apt-get install -y -qq jq >/dev/null 2>&1 || true
fi

configure_claude_backend

echo ""
echo "=========================================="
echo "  ClaudeBox Environment Ready!"
echo "=========================================="
echo ""
echo "Languages available:"
echo "  - Node.js $(node --version 2>/dev/null || echo 'installing...')"
echo "  - Python $(python3 --version 2>/dev/null | cut -d' ' -f2 || echo 'installing...')"
echo "  - Go $(go version 2>/dev/null | cut -d' ' -f3 || echo 'installing via feature...')"
echo "  - Rust $(rustc --version 2>/dev/null | cut -d' ' -f2 || echo 'installing via feature...')"
echo ""
echo "Services:"
echo "  - PostgreSQL: localhost:5432 (dev/dev)"
echo "  - Redis: localhost:6379"
echo ""

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
until pg_isready -h postgres -U dev -q 2>/dev/null; do
    sleep 1
done
echo "PostgreSQL is ready!"

# Wait for Redis
echo "Waiting for Redis..."
until redis-cli -h redis ping 2>/dev/null | grep -q PONG; do
    sleep 1
done
echo "Redis is ready!"

# Install project dependencies based on what exists
cd /workspaces/${PROJECT_NAME:-project}

if [ -f "package.json" ]; then
    echo ""
    echo "Installing Node.js dependencies..."
    npm install
fi

if [ -f "requirements.txt" ]; then
    echo ""
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

if [ -f "pyproject.toml" ]; then
    echo ""
    echo "Installing Python project..."
    pip install -e ".[dev]" 2>/dev/null || pip install -e .
fi

if [ -f "go.mod" ]; then
    echo ""
    echo "Downloading Go modules..."
    go mod download
fi

if [ -f "Cargo.toml" ]; then
    echo ""
    echo "Building Rust project..."
    cargo build
fi

echo ""
echo "=========================================="
echo "  Pommel Setup Starting..."
echo "=========================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ========================================
# Verify Host Ollama Connectivity
# ========================================
echo "Checking host Ollama connectivity..."

OLLAMA_HOST="${OLLAMA_HOST:-http://host.docker.internal:11434}"
MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        echo "✓ Connected to host Ollama at $OLLAMA_HOST"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "  Retry $RETRY_COUNT/$MAX_RETRIES..."
            sleep 2
        else
            echo "⚠ WARNING: Cannot reach host Ollama at $OLLAMA_HOST"
            echo "  Pommel will not work until Ollama is started."
            echo "  Run on host: C:\\Users\\gordo\\.claudebox\\scripts\\setup-host-ollama.ps1"
            break
        fi
    fi
done

# ========================================
# Install socat and set up Ollama proxy
# ========================================
echo ""
echo "Setting up Ollama proxy..."

# Install socat if not present
if ! command_exists socat; then
    echo "Installing socat..."
    apt-get update -qq && apt-get install -y -qq socat >/dev/null 2>&1 || \
        sudo apt-get update -qq && sudo apt-get install -y -qq socat >/dev/null 2>&1
fi

# Set up socat proxy to forward localhost:11434 to host Ollama
# This is a workaround for Pommel not reading the Ollama URL from config
if ! pgrep -f "socat.*11434" > /dev/null 2>&1; then
    echo "Starting Ollama proxy (localhost:11434 -> host.docker.internal:11434)..."
    nohup socat TCP-LISTEN:11434,fork,reuseaddr TCP:host.docker.internal:11434 > /tmp/socat-ollama.log 2>&1 &
    sleep 1
    echo "✓ Ollama proxy started"
else
    echo "✓ Ollama proxy already running"
fi

# ========================================
# Install Pommel CLI
# ========================================
echo ""
echo "Checking Pommel installation..."

if ! command_exists pm; then
    echo "Installing Pommel..."

    # Download and run Pommel installer with auto-accept for missing Ollama
    curl -fsSL https://raw.githubusercontent.com/dbinky/Pommel/main/scripts/install.sh -o /tmp/install-pommel.sh
    chmod +x /tmp/install-pommel.sh
    echo 'y' | /tmp/install-pommel.sh || true
    rm -f /tmp/install-pommel.sh

    # Add to PATH for current session
    export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

    # Add to bashrc for future sessions
    if ! grep -q "/.local/bin" ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"' >> ~/.bashrc
    fi

    echo "✓ Pommel installed"
else
    echo "✓ Pommel already installed"
fi

# ========================================
# Initialize Pommel for this project
# ========================================
WORKSPACE_DIR="/workspaces/${PROJECT_NAME:-project}"
POMMEL_DATA_DIR="${POMMEL_DATA_DIR:-/home/vscode/.pommel-data}"

echo ""
echo "Initializing Pommel for ${PROJECT_NAME}..."

cd "$WORKSPACE_DIR"

# Create .pommel directory in persistent volume
mkdir -p "$POMMEL_DATA_DIR"

# Create symlink from workspace to volume (if not exists)
if [ ! -L "$WORKSPACE_DIR/.pommel" ]; then
    ln -s "$POMMEL_DATA_DIR" "$WORKSPACE_DIR/.pommel"
fi

# Initialize Pommel if not already done
if [ ! -f "$POMMEL_DATA_DIR/config.yml" ]; then
    echo "Initializing new Pommel instance..."
    pm init

    # Configure Pommel
    pm config set ollama.host "$OLLAMA_HOST"
    pm config set daemon.host "0.0.0.0"  # Listen on all interfaces for cross-container access
    pm config set daemon.port 7420
    pm config set daemon.readonly true  # SECURITY: Read-only API (no write operations)

    echo "✓ Pommel initialized with read-only API"
else
    echo "✓ Pommel already initialized"
fi

# ========================================
# Start Pommel Daemon
# ========================================
echo ""
echo "Starting Pommel daemon..."

# Check if daemon is already running
if pm status > /dev/null 2>&1; then
    echo "✓ Pommel daemon already running"
else
    # Start daemon in background
    pm start

    # Wait for daemon to be ready
    sleep 3

    if pm status > /dev/null 2>&1; then
        echo "✓ Pommel daemon started successfully"
    else
        echo "⚠ WARNING: Pommel daemon failed to start"
        echo "  Check logs with: pm logs"
    fi
fi

# ========================================
# Setup .pommelignore
# ========================================
echo ""
echo "Configuring Pommel ignore patterns..."

POMMELIGNORE_FILE="$WORKSPACE_DIR/.pommelignore"

if [ ! -f "$POMMELIGNORE_FILE" ]; then
    cat > "$POMMELIGNORE_FILE" <<'POMMELIGNORE_EOF'
# Dependencies
node_modules/
venv/
.venv/
__pycache__/
*.pyc

# Build outputs
dist/
build/
*.egg-info/
.next/
out/

# Version control
.git/
.svn/

# IDE
.vscode/
.idea/

# Logs and temp files
*.log
.tmp/
.cache/

# Docker
.devcontainer/
Dockerfile
docker-compose.yml

# Large data files
*.db
*.sqlite

# OS files
.DS_Store
Thumbs.db
POMMELIGNORE_EOF

    echo "✓ Created .pommelignore file"
else
    echo "✓ .pommelignore already exists"
fi

# ========================================
# Setup Pommel Agent
# ========================================
echo ""
echo "Setting up Pommel AI Agent..."

CLAUDEBOX_DIR="/home/vscode/.claudebox/pommel"
mkdir -p "$CLAUDEBOX_DIR"

# Copy agent script if it doesn't exist
if [ ! -f "$CLAUDEBOX_DIR/agent.py" ]; then
    # Download from the host's claudebox directory (mounted or copied)
    cat > "$CLAUDEBOX_DIR/agent.py" << 'AGENT_EOF'
#!/usr/bin/env python3
"""Pommel AI Agent - Lightweight version for containers"""
import asyncio, json, os, subprocess, urllib.request
from pathlib import Path
from datetime import datetime

class PommelAgent:
    API_URL = "http://localhost:7420"

    def __init__(self, project_name, workspace_dir=None):
        self.project_name = project_name
        self.workspace_dir = workspace_dir or f"/workspaces/{project_name}"

    async def search(self, query, limit=5):
        try:
            data = json.dumps({"query": query, "limit": limit}).encode()
            req = urllib.request.Request(f"{self.API_URL}/search", data=data,
                                          headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode()).get('results', [])
        except Exception as e:
            return []

    async def get_suggestions(self):
        try:
            result = subprocess.run(["git", "diff", "--name-only"], capture_output=True,
                                    text=True, timeout=10, cwd=self.workspace_dir)
            files = [f for f in result.stdout.strip().split('\n') if f and f.endswith(('.py','.js','.ts','.go','.rs'))]
            suggestions = []
            for f in files[:3]:
                path = Path(self.workspace_dir) / f
                if path.exists():
                    with open(path, 'r', errors='ignore') as fp:
                        content = fp.read(2000)
                    query = ' '.join([w for w in content.split()[:10] if len(w) > 3])[:100]
                    if query:
                        results = await self.search(query)
                        if results:
                            suggestions.append({'file': f, 'matches': len(results)})
            return suggestions
        except:
            return []

async def main():
    project = os.environ.get('PROJECT_NAME', 'unknown')
    interval = int(os.environ.get('POMMEL_AGENT_INTERVAL', '60'))
    if project == 'unknown':
        cwd = os.getcwd()
        if '/workspaces/' in cwd:
            project = cwd.split('/workspaces/')[-1].split('/')[0]
    agent = PommelAgent(project)
    print(f"Pommel Agent started for {project}")
    while True:
        try:
            suggestions = await agent.get_suggestions()
            if suggestions:
                print(f"[{datetime.now()}] Found {len(suggestions)} suggestions")
            await asyncio.sleep(interval)
        except KeyboardInterrupt:
            break
        except Exception as e:
            await asyncio.sleep(interval)

if __name__ == "__main__":
    asyncio.run(main())
AGENT_EOF
    chmod +x "$CLAUDEBOX_DIR/agent.py"
    echo "Created Pommel agent"
fi

# ========================================
# Trigger Initial Indexing
# ========================================
echo ""
echo "Triggering initial code indexing..."
echo "(This runs in background and may take a few minutes)"

# Trigger indexing (non-blocking)
pm reindex --background || echo "  (Indexing will continue in background)"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Pommel semantic search available:"
echo "  - Local: pm search 'your query'"
echo "  - Dashboard: http://localhost:3001"
echo "  - AI Agent: Running in background"
echo ""
echo "Run 'claude --model <model>' to start Claude Code"
echo "Use '--dangerously-skip-permissions' for YOLO mode"
echo ""
