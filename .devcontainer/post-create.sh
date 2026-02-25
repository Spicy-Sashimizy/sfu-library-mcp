#!/bin/bash
set -e

# ==========================================
# Copy Credentials from Shared Volume
# ==========================================
# Copy credentials from shared volume to expected location
SHARED_CREDENTIALS="/home/vscode/.claude/credentials.json"
CREDENTIALS_DIR="/home/vscode/.claudebox-credentials"
CREDENTIALS_FILE="${CREDENTIALS_DIR}/credentials.json"

if [ -f "$SHARED_CREDENTIALS" ]; then
    echo "Setting up credentials file..."
    mkdir -p "$CREDENTIALS_DIR"
    cp "$SHARED_CREDENTIALS" "$CREDENTIALS_FILE"
    chown -R vscode:vscode "$CREDENTIALS_DIR" 2>/dev/null || true
    echo "âœ“ Credentials file copied from shared volume"
else
    echo "âš  No credentials file found in shared volume"
fi

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
echo "export CLAUDE_CONFIG_DIR=/home/vscode/.claude" >> /home/vscode/.bashrc 2>/dev/null || true

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
CREDENTIALS_FILE="${CREDENTIALS_FILE:-/home/vscode/.claudebox-credentials/credentials.json}"
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
                # 1. Create or update settings.json with Z.AI configuration
                # NOTE: NOT setting ANTHROPIC_MODEL to allow /model command to work
                if [ -f "$SETTINGS_FILE" ] && command -v jq >/dev/null 2>&1; then
                    jq --arg key "$ZAI_API_KEY" '.env = (.env // {}) + {
                        "ANTHROPIC_API_KEY": $key,
                        "ANTHROPIC_AUTH_TOKEN": $key,
                        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                        "API_TIMEOUT_MS": "3000000"
                    }' "$SETTINGS_FILE" > "${SETTINGS_FILE}.tmp" && mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"
                else
                    cat > "$SETTINGS_FILE" <<EOF
{
    "env": {
        "ANTHROPIC_API_KEY": "$ZAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN": "$ZAI_API_KEY",
        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
        "API_TIMEOUT_MS": "3000000"
    }
}
EOF
                fi
                echo "  Z.AI settings.json configured (model via --model flag)"

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

# Z.AI Backend Configuration (GLM models - use /model to switch)
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
                echo "  Z.AI backend fully configured with GLM model support"
                echo "  Available models: glm-4.7 (latest), glm-4.5, glm-4.5-air, glm-4-flash"
                echo "  To change model: Add ZAI_MODEL=<model-name> to your project .env file"
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


# ==========================================
# Configure MCP Servers for Claude Code in Container
# ==========================================
echo ""
echo "Configuring MCP servers for Claude Code..."

MCP_SERVER_DIR="/home/vscode/.claudebox/mcp"
mkdir -p "$MCP_SERVER_DIR"

# Copy Pommel MCP server script from centralized location (mounted from host)
if [ -f "/usr/local/bin/pommel-mcp-server.py" ]; then
    cp "/usr/local/bin/pommel-mcp-server.py" "$MCP_SERVER_DIR/pommel-mcp-server.py"
    chmod +x "$MCP_SERVER_DIR/pommel-mcp-server.py"
    echo "  - Copied Pommel MCP server script from shared location"
elif [ -f "/workspaces/${PROJECT_NAME}/.devcontainer/pommel-mcp-server.py" ]; then
    cp "/workspaces/${PROJECT_NAME}/.devcontainer/pommel-mcp-server.py" "$MCP_SERVER_DIR/pommel-mcp-server.py"
    chmod +x "$MCP_SERVER_DIR/pommel-mcp-server.py"
    echo "  - Copied Pommel MCP server script from project template"
fi

# Install MCP Python package (required for the server)
pip install mcp >/dev/null 2>&1 || pip install --user mcp >/dev/null 2>&1 || true
echo "  - Installed MCP Python package"

# Configure MCP server in Claude Code settings.json
# Claude Code uses settings.json with mcpServers key
if [ -f "$SETTINGS_FILE" ] && command -v jq >/dev/null 2>&1; then
    if ! jq -e '.mcpServers.pommel' "$SETTINGS_FILE" >/dev/null 2>&1; then
        jq --arg dir "$MCP_SERVER_DIR" --arg proj "${PROJECT_NAME}" --arg host "claudebox-${PROJECT_NAME}-pommel" '.mcpServers = (.mcpServers // {}) + {
            "pommel": {
                "command": "python3",
                "args": [$dir + "/pommel-mcp-server.py"],
                "env": {"PROJECT_NAME": $proj, "POMMEL_HOST": $host, "POMMEL_PORT": "7420"}
            }
        }' "$SETTINGS_FILE" > "${SETTINGS_FILE}.tmp" && mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"
        echo "  - Added Pommel MCP server to settings.json"
    else
        echo "  - Pommel MCP already configured in settings.json"
    fi
else
    # Create settings.json with MCP configuration if it doesn't exist
    cat > "$SETTINGS_FILE" <<EOF
{
    "mcpServers": {
        "pommel": {
            "command": "python3",
            "args": ["${MCP_SERVER_DIR}/pommel-mcp-server.py"],
            "env": {
                "PROJECT_NAME": "${PROJECT_NAME}",
                "POMMEL_HOST": "claudebox-${PROJECT_NAME}-pommel",
                "POMMEL_PORT": "7420"
            }
        }
    }
}
EOF
    echo "  - Created settings.json with Pommel MCP server"
fi

chown -R vscode:vscode "$MCP_SERVER_DIR" /home/vscode/.claude 2>/dev/null || true
echo "  MCP servers configured"
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

    # Also install into .venv if it exists (MCP server uses the venv)
    VENV_DIR="/workspaces/${PROJECT_NAME:-project}/.venv"
    if [ -d "$VENV_DIR" ]; then
        echo "Installing Python dependencies into .venv..."
        sudo "$VENV_DIR/bin/pip" install -r requirements.txt 2>/dev/null || \
            "$VENV_DIR/bin/pip" install -r requirements.txt 2>/dev/null || true
    else
        echo "Creating .venv and installing dependencies..."
        python3 -m venv "$VENV_DIR"
        sudo "$VENV_DIR/bin/pip" install -r requirements.txt 2>/dev/null || \
            "$VENV_DIR/bin/pip" install -r requirements.txt 2>/dev/null || true
    fi

    # Install src/requirements.txt if it exists (may have additional deps like pyzotero)
    if [ -f "src/requirements.txt" ]; then
        echo "Installing src/requirements.txt..."
        pip install -r src/requirements.txt 2>/dev/null || true
        if [ -d "$VENV_DIR" ]; then
            sudo "$VENV_DIR/bin/pip" install -r src/requirements.txt 2>/dev/null || \
                "$VENV_DIR/bin/pip" install -r src/requirements.txt 2>/dev/null || true
        fi
    fi
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
        echo "âœ“ Connected to host Ollama at $OLLAMA_HOST"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "  Retry $RETRY_COUNT/$MAX_RETRIES..."
            sleep 2
        else
            echo "âš  WARNING: Cannot reach host Ollama at $OLLAMA_HOST"
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
    echo "âœ“ Ollama proxy started"
else
    echo "âœ“ Ollama proxy already running"
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

    echo "âœ“ Pommel installed"
else
    echo "âœ“ Pommel already installed"
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
    if command_exists pm; then
        pm init 2>/dev/null || echo "  Note: Pommel init failed, will retry on next start"

        # Configure Pommel
        pm config set ollama.host "$OLLAMA_HOST" 2>/dev/null || true
        pm config set daemon.host "0.0.0.0" 2>/dev/null || true
        pm config set daemon.port 7420 2>/dev/null || true
        pm config set daemon.readonly true 2>/dev/null || true

        echo "âœ“ Pommel initialized with read-only API"
    else
        echo "âš  WARNING: pm command not found, skipping Pommel initialization"
        echo "  Pommel will be set up on next container start"
    fi
else
    echo "âœ“ Pommel already initialized"
fi

# ========================================
# Start Pommel Daemon
# ========================================
echo ""
echo "Starting Pommel daemon..."

if command_exists pm; then
    # Check if daemon is already running
    if pm status > /dev/null 2>&1; then
        echo "âœ“ Pommel daemon already running"
    else
        # Start daemon in background
        pm start 2>/dev/null || true

        # Wait for daemon to be ready
        sleep 3

        if pm status > /dev/null 2>&1; then
            echo "âœ“ Pommel daemon started successfully"
        else
            echo "âš  WARNING: Pommel daemon failed to start"
            echo "  Check logs with: pm logs"
        fi
    fi
else
    echo "âš  WARNING: pm command not found, skipping daemon start"
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

    echo "âœ“ Created .pommelignore file"
else
    echo "âœ“ .pommelignore already exists"
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

# ========================================
# Git-Crypt Secret Encryption Setup
# ========================================
echo ""
echo "Setting up git-crypt for secret encryption..."

WORKSPACE_DIR="/workspaces/${PROJECT_NAME:-project}"
GITCRYPT_KEY_DIR="/home/vscode/.claudebox-credentials/git-crypt-keys"
GITCRYPT_KEY_FILE="$GITCRYPT_KEY_DIR/${PROJECT_NAME:-project}.key"

cd "$WORKSPACE_DIR"

# Check if git-crypt is available
if command -v git-crypt >/dev/null 2>&1; then
    # Ensure git filter config uses Linux paths (not Windows paths)
    # This fixes issues where Windows git config is copied to container
    # Also make git-crypt non-blocking so it doesn't break OAuth git operations
    git config --global filter.git-crypt.clean "git-crypt clean" 2>/dev/null || true
    git config --global filter.git-crypt.smudge "git-crypt smudge" 2>/dev/null || true
    git config --global filter.git-crypt.required "false" 2>/dev/null || true

    # Create key storage directory
    mkdir -p "$GITCRYPT_KEY_DIR"

    # Check if this repo already has git-crypt initialized
    if [ -d ".git-crypt" ]; then
        echo "  Git-crypt already initialized"

        # Try to unlock if key exists
        if [ -f "$GITCRYPT_KEY_FILE" ]; then
            if git-crypt unlock "$GITCRYPT_KEY_FILE" 2>/dev/null; then
                echo "  Repository unlocked with existing key"
            else
                echo "  Note: Could not unlock (may already be unlocked or key mismatch)"
            fi
        fi
    else
        # Initialize git-crypt for new repo
        echo "  Initializing git-crypt..."
        git-crypt init

        # Export the symmetric key for backup/sharing
        git-crypt export-key "$GITCRYPT_KEY_FILE"
        chmod 600 "$GITCRYPT_KEY_FILE"

        echo "  Git-crypt key saved to: $GITCRYPT_KEY_FILE"
        echo "  IMPORTANT: Back up this key! Without it, encrypted files cannot be recovered."
    fi

    # Create .gitattributes if it doesn't exist
    GITATTRIBUTES_FILE="$WORKSPACE_DIR/.gitattributes"
    if [ ! -f "$GITATTRIBUTES_FILE" ]; then
        cat > "$GITATTRIBUTES_FILE" << 'GITATTRIBUTES_EOF'
# ============================================
# GIT-CRYPT ENCRYPTION PATTERNS
# ============================================
# Files matching these patterns are automatically encrypted on push
# and decrypted on pull. They appear as binary on GitHub.

# Environment files with secrets
.env filter=git-crypt diff=git-crypt
.env.* filter=git-crypt diff=git-crypt
!.env.example
!.env.template

# Credential files
credentials.json filter=git-crypt diff=git-crypt
*credentials*.json filter=git-crypt diff=git-crypt
*secrets*.json filter=git-crypt diff=git-crypt
*token*.json filter=git-crypt diff=git-crypt
oauth*.json filter=git-crypt diff=git-crypt
client_secret*.json filter=git-crypt diff=git-crypt
service_account*.json filter=git-crypt diff=git-crypt

# API keys and certificates
*.key filter=git-crypt diff=git-crypt
*.pem filter=git-crypt diff=git-crypt
*.p12 filter=git-crypt diff=git-crypt

# Secrets directory
secrets/** filter=git-crypt diff=git-crypt
.secrets/** filter=git-crypt diff=git-crypt

# Config files that might contain secrets
config.local.* filter=git-crypt diff=git-crypt
settings.local.* filter=git-crypt diff=git-crypt
GITATTRIBUTES_EOF
        echo "  Created .gitattributes with encryption patterns"

        # Stage the .gitattributes file
        git add .gitattributes 2>/dev/null || true
    else
        echo "  .gitattributes already exists"
    fi

    echo "  Git-crypt setup complete"
else
    echo "  WARNING: git-crypt not installed, skipping encryption setup"
fi

# ========================================
# SSH Key Permission Setup
# ========================================
echo ""
echo "Setting up SSH key permissions for Git..."

SSH_DIR="/home/vscode/.ssh"
if [ -d "$SSH_DIR" ]; then
    # Fix ownership of SSH keys (mounted from Windows host)
    # This is needed because Windows files are owned by root in the container
    if [ -w "$SSH_DIR" ]; then
        echo "  Fixing SSH key ownership and permissions..."
        chown -R vscode:vscode "$SSH_DIR" 2>/dev/null || true
        chmod 700 "$SSH_DIR" 2>/dev/null || true
        chmod 600 "$SSH_DIR"/id_* 2>/dev/null || true
        chmod 644 "$SSH_DIR"/*.pub 2>/dev/null || true

        # Fix SSH config if it exists
        if [ -f "$SSH_DIR/config" ]; then
            chmod 600 "$SSH_DIR/config" 2>/dev/null || true
        fi

        echo "  ✓ SSH key permissions fixed"
    else
        echo "  ⚠ SSH directory is read-only (mounted with :ro flag)"
        echo "    To fix: Remove :ro from docker-compose.yml SSH mount"
    fi

    # Test SSH connection to GitHub
    if command -v ssh >/dev/null 2>&1; then
        if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
            echo "  ✓ SSH authentication to GitHub working"
        else
            echo "  ⚠ SSH authentication to GitHub not working"
            echo "    You may need to: add SSH key to GitHub account"
        fi
    fi
else
    echo "  ⚠ No SSH directory found"
fi

# ========================================
# GitHub OAuth Authentication Setup
# ========================================
echo ""
echo "Setting up GitHub OAuth authentication..."

# Install GitHub CLI if not present
if ! command -v gh >/dev/null 2>&1; then
    echo "  Installing GitHub CLI (gh)..."

    # Add GitHub CLI repository
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq curl >/dev/null 2>&1

    # Download and install gh CLI
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg >/dev/null 2>&1
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y gh >/dev/null 2>&1

    if command -v gh >/dev/null 2>&1; then
        echo "  ✓ GitHub CLI installed"
    else
        echo "  ⚠ GitHub CLI installation failed"
        echo ""
        echo "  Git push will require manual authentication"
        echo ""
        # Skip OAuth setup
        exit 0
    fi
fi

# Check if already authenticated
if gh auth status >/dev/null 2>&1; then
    echo "  ✓ GitHub OAuth already configured"
    gh auth status
else
    echo ""
    echo "  ==========================================="
    echo "  GitHub OAuth Authentication"
    echo "  ==========================================="
    echo ""
    echo "  For automated container setup, GitHub CLI can be configured"
    echo "  to use an OAuth token. This enables git push/pull without"
    echo "  personal access tokens."
    echo ""
    echo "  Option 1: Skip OAuth setup (recommended for containers)"
    echo "    - Git operations will use the host's OAuth credentials"
    echo "    - The host machine (your Windows PC) handles authentication"
    echo ""
    echo "  Option 2: Set up OAuth in this container"
    echo "    - Requires interactive browser authentication"
    echo "    - Run: gh auth login"
    echo ""
    echo "  ✓ Skipping OAuth setup in container (host will handle git auth)"
fi

# ========================================
# Configure Git for OAuth Authentication
# ========================================
# Set up OAuth credentials from the mounted volume for automatic git operations
OAUTH_CREDS="/mnt/claudebox-git-creds/.git-credentials"

if [ -f "$OAUTH_CREDS" ]; then
    echo "  Configuring git to use OAuth credentials..."

    # Copy OAuth credentials to a writable location
    mkdir -p /home/vscode/.git-creds
    cp "$OAUTH_CREDS" /home/vscode/.git-credentials
    chmod 600 /home/vscode/.git-credentials

    # Configure git to use the OAuth credentials file
    git config --global credential.helper "store --file=/home/vscode/.git-credentials"
    git config --global credential.helper store

    echo "  ✓ OAuth credentials configured"
    echo "    Git push/pull will work automatically without authentication prompts"
else
    echo "  No OAuth credentials found in /mnt/claudebox-git-creds"
    echo "  Git operations will use SSH keys for authentication"
    echo "  If HTTPS remotes fail, switch to SSH: git remote set-url origin git@github.com:user/repo.git"
fi

# Configure git to prefer SSH over HTTPS when SSH keys are available
if [ -f "$SSH_DIR/id_ed25519" ] || [ -f "$SSH_DIR/id_rsa" ]; then
    echo ""
    echo "  SSH keys detected - configuring URL rewriting to use SSH..."
    # This automatically converts https://github.com/ to git@github.com: for git operations
    git config --global url."git@github.com:".insteadOf "https://github.com/"
    echo "  ✓ Git will use SSH for GitHub operations"
fi

echo ""

# ========================================
# Pre-commit Hook for Secret Detection
# ========================================
echo ""
echo "Installing pre-commit secret scanner..."

HOOKS_DIR="$WORKSPACE_DIR/.git/hooks"
PRE_COMMIT_HOOK="$HOOKS_DIR/pre-commit"

mkdir -p "$HOOKS_DIR"

# Create pre-commit hook (Gitleaks-based with regex fallback)
cat > "$PRE_COMMIT_HOOK" << 'PRECOMMIT_EOF'
#!/bin/bash
# ClaudeBox Pre-commit Secret Scanner
# Blocks commits containing potential secrets

set -e

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Scanning for secrets...${NC}"

# Patterns that indicate secrets
SECRET_PATTERNS=(
    'ghp_[a-zA-Z0-9]{36}'
    'gho_[a-zA-Z0-9]{36}'
    'sk-[a-zA-Z0-9]{48}'
    'sk-proj-[a-zA-Z0-9-_]{80,}'
    'AIza[0-9A-Za-z\\-_]{35}'
    'AKIA[0-9A-Z]{16}'
    '-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY'
    'xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*'
)

# Files to skip
SKIP_PATTERNS='\.env\.example$|\.env\.template$|\.gitattributes$|pre-commit$|\.md$'

STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null || true)
SECRETS_FOUND=0

for file in $STAGED_FILES; do
    # Skip excluded files
    if echo "$file" | grep -qE "$SKIP_PATTERNS"; then
        continue
    fi

    # Skip binary/encrypted files
    if file "$file" 2>/dev/null | grep -q "binary\|data"; then
        continue
    fi

    CONTENT=$(git show ":$file" 2>/dev/null || true)
    [ -z "$CONTENT" ] && continue

    for pattern in "${SECRET_PATTERNS[@]}"; do
        if echo "$CONTENT" | grep -qE "$pattern" 2>/dev/null; then
            SECRETS_FOUND=$((SECRETS_FOUND + 1))
            echo -e "${RED}POTENTIAL SECRET in ${file}${NC}"
            break
        fi
    done
done

if [ $SECRETS_FOUND -gt 0 ]; then
    echo -e "\n${RED}COMMIT BLOCKED: $SECRETS_FOUND potential secret(s) detected${NC}"
    echo -e "${YELLOW}Options:${NC}"
    echo "  1. Remove secrets and use .env files (gitignored)"
    echo "  2. Add files to .gitattributes for git-crypt encryption"
    echo "  3. Bypass: git commit --no-verify (use with caution)"
    exit 1
fi

echo -e "${GREEN}No secrets detected${NC}"
exit 0
PRECOMMIT_EOF

chmod +x "$PRE_COMMIT_HOOK"
echo "  Pre-commit hook installed"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Pommel semantic search:"
echo "  - Local: pm search 'your query'"
echo "  - Dashboard: http://localhost:3001"
echo ""
echo "Secret Protection:"
echo "  - Git-crypt: Encrypts .env, credentials.json on GitHub"
echo "  - Pre-commit hook: Blocks commits with detected secrets"
echo "  - Key location: ~/.claudebox-credentials/git-crypt-keys/"
echo ""
echo "Run 'claude --model <model>' to start Claude Code"
echo ""
