# Pommel Setup Guide for ClaudeBox Projects

This guide documents how to set up Pommel semantic code search for any ClaudeBox project.

## Prerequisites

- Docker and Docker Compose installed
- A working Pommel container to copy binaries from (e.g., `claudebox-sfu-auto-researcher-pommel`)
- Available port for Pommel API (check existing ports with `docker ps --filter "name=pommel"`)

## Step 1: Find Available Port

List all Pommel services and their ports:
```bash
docker ps --filter "name=pommel" --format "{{.Names}}\t{{.Ports}}"
```

Choose an unused port number. Current allocations:
- 7421: dashboard
- 7422: business-plan-cataloger
- 7423: ccpsandyass-cybermap
- 7424: sfu-library-mcp
- 7425: ccpsandyass-scripts
- 7426: ccpsandyass-separate
- etc.

## Step 2: Copy Pommel Binaries

From a working Pommel container, copy the binaries to your project:

```bash
# Set your project name and path
PROJECT_NAME="your-project-name"
PROJECT_PATH="<USERPROFILE>/claudebox-projects/$PROJECT_NAME"

# Create binaries directory
mkdir -p "$PROJECT_PATH/.pommel-bin"

# Copy binaries from working container
docker cp claudebox-sfu-auto-researcher-pommel:/usr/local/bin/pm "$PROJECT_PATH/.pommel-bin/"
docker cp claudebox-sfu-auto-researcher-pommel:/usr/local/bin/pommeld "$PROJECT_PATH/.pommel-bin/"

# Verify
ls -lh "$PROJECT_PATH/.pommel-bin/"
```

## Step 3: Copy Language Configuration Files

```bash
# Copy language files
docker cp claudebox-sfu-auto-researcher-pommel:/workspace/languages "$PROJECT_PATH/"

# Verify
ls "$PROJECT_PATH/languages/"
```

## Step 4: Create docker-compose.pommel.yml

Create a file at `$PROJECT_PATH/docker-compose.pommel.yml`:

```yaml
# Pommel service for <PROJECT_NAME>
# Provides semantic code search for the <PROJECT_NAME> codebase

name: claudebox-<PROJECT_NAME>-pommel

services:
  pommel:
    container_name: claudebox-<PROJECT_NAME>-pommel
    image: debian:bookworm-slim
    volumes:
      - .:/workspace
      - pommel-data:/pommel-data
      - ./.pommel-bin:/binaries:ro
    working_dir: /workspace
    entrypoint: ["/bin/bash", "-c"]
    command:
      - |
        set -e
        echo "Setting up Pommel for <PROJECT_NAME>..."

        # Copy pre-built binaries
        echo "Installing Pommel binaries..."
        cp /binaries/pm /usr/local/bin/pm
        cp /binaries/pommeld /usr/local/bin/pommeld
        chmod +x /usr/local/bin/pm /usr/local/bin/pommeld
        echo "Pommel binaries installed"

        # Install minimal dependencies (just socat for Ollama proxy)
        echo "Installing socat..."
        apt-get update -qq 2>/dev/null && apt-get install -y -qq socat 2>/dev/null || echo "Using existing packages"

        # Setup socat proxy for Ollama
        echo "Starting Ollama proxy..."
        socat TCP-LISTEN:11434,fork,reuseaddr TCP:host.docker.internal:11434 &
        sleep 2

        # Initialize Pommel in persistent data directory
        cd /workspace
        export HOME=/pommel-data
        mkdir -p /pommel-data/.pommel
        rm -rf /workspace/.pommel 2>/dev/null || true
        ln -sf /pommel-data/.pommel /workspace/.pommel 2>/dev/null || true

        # Only initialize if not already done
        if [ ! -f "/pommel-data/.pommel/config.yaml" ]; then
            echo "Initializing Pommel..."
            rm -f /pommel-data/.pommel/pommel.db 2>/dev/null || true
            pm init
            pm config set embedding.ollama_url http://localhost:11434
            pm config set daemon.host 0.0.0.0
            pm config set daemon.port 7420
        else
            echo "Pommel already initialized"
        fi

        # Create .pommelignore in workspace
        echo "node_modules/" > /workspace/.pommelignore
        echo ".git/" >> /workspace/.pommelignore
        echo "*.log" >> /workspace/.pommelignore
        echo "nul" >> /workspace/.pommelignore
        echo ".claude/" >> /workspace/.pommelignore
        echo "__pycache__/" >> /workspace/.pommelignore
        echo "*.pyc" >> /workspace/.pommelignore
        echo ".pommel/" >> /workspace/.pommelignore
        echo ".pommel-bin/" >> /workspace/.pommelignore
        echo "dist/" >> /workspace/.pommelignore
        echo "build/" >> /workspace/.pommelignore
        echo ".venv/" >> /workspace/.pommelignore
        echo "venv/" >> /workspace/.pommelignore

        echo "Triggering initial index..."
        pm reindex || echo "Reindex started..."

        echo "Pommel setup complete! Starting daemon..."
        exec pommeld -project /workspace
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
    extra_hosts:
      - "host.docker.internal:host-gateway"
    ports:
      - "<YOUR_PORT>:7420"  # e.g., "7424:7420"
    networks:
      - claudebox-shared
    labels:
      - "com.claudebox.managed=true"
      - "com.claudebox.service=pommel"
      - "com.claudebox.project=<PROJECT_NAME>"
    restart: unless-stopped

networks:
  claudebox-shared:
    external: true
    name: claudebox-shared-net

volumes:
  pommel-data:
    name: claudebox-<PROJECT_NAME>-pommel-data
    labels:
      com.claudebox.managed: "true"
      com.claudebox.project: "<PROJECT_NAME>"
```

**Important**: Replace all instances of `<PROJECT_NAME>` and `<YOUR_PORT>` with your actual values.

## Step 5: Start Pommel Container

```bash
cd "$PROJECT_PATH"
docker compose -f docker-compose.pommel.yml up -d

# Check logs
docker logs claudebox-<PROJECT_NAME>-pommel

# Verify it's running
docker ps --filter "name=claudebox-<PROJECT_NAME>-pommel"

# Test the API
curl -s http://localhost:<YOUR_PORT>/status
```

Expected output: `{"daemon":{"pid":1,"running":true},"index":{...}}`

## Step 6: Update MCP Server Configuration

Edit `<USERPROFILE>/.claudebox/mcp-servers/pommel/host-server-safe.py`:

Add your project to the `STANDALONE_POMMEL_SERVICES` dictionary:

```python
STANDALONE_POMMEL_SERVICES = {
    # ... existing projects ...
    "<PROJECT_NAME>": {
        "host": "localhost",
        "port": <YOUR_PORT>,  # e.g., 7424
        "description": "<PROJECT DESCRIPTION>"
    },
    # ... more projects ...
}
```

## Step 7: Restart Claude Code

The MCP server caches configuration at startup. Restart Claude Code to load the new project.

## Step 8: Verify in Claude Code

After restart, test the Pommel integration:

```python
# List available projects
mcp__pommel__pommel_list_projects()

# Search your project
mcp__pommel__pommel_search_project(
    project="<PROJECT_NAME>",
    query="your search query",
    limit=5
)
```

## Troubleshooting

### Container keeps restarting

Check logs: `docker logs claudebox-<PROJECT_NAME>-pommel`

Common issues:
- Network connectivity (socat failing to connect to Ollama)
- Missing binaries in `.pommel-bin/`
- Missing language files in `languages/`

### Port already in use

```bash
# Find what's using the port
netstat -ano | findstr :<YOUR_PORT>

# Choose a different port and update docker-compose.pommel.yml
```

### MCP not finding project

1. Verify the project name matches exactly in:
   - `docker-compose.pommel.yml` (labels)
   - `host-server-safe.py` (dictionary key)
2. Restart Claude Code
3. Check MCP server logs

### Low file/chunk count

Check `.pommelignore` - you may be excluding too much:

```bash
docker exec claudebox-<PROJECT_NAME>-pommel pm status
docker exec claudebox-<PROJECT_NAME>-pommel pm reindex
```

## Quick Reference Script

Save this as `setup-pommel.sh` for faster setup:

```bash
#!/bin/bash
PROJECT_NAME="$1"
PORT="$2"

if [ -z "$PROJECT_NAME" ] || [ -z "$PORT" ]; then
    echo "Usage: $0 <project-name> <port>"
    echo "Example: $0 my-project 7435"
    exit 1
fi

PROJECT_PATH="<USERPROFILE>/claudebox-projects/$PROJECT_NAME"

echo "Setting up Pommel for $PROJECT_NAME on port $PORT..."

# Create directories
mkdir -p "$PROJECT_PATH/.pommel-bin"

# Copy binaries
docker cp claudebox-sfu-auto-researcher-pommel:/usr/local/bin/pm "$PROJECT_PATH/.pommel-bin/"
docker cp claudebox-sfu-auto-researcher-pommel:/usr/local/bin/pommeld "$PROJECT_PATH/.pommel-bin/"

# Copy language files
docker cp claudebox-sfu-auto-researcher-pommel:/workspace/languages "$PROJECT_PATH/"

echo "Files copied. Now:"
echo "1. Create docker-compose.pommel.yml in $PROJECT_PATH"
echo "2. Replace <PROJECT_NAME> with: $PROJECT_NAME"
echo "3. Replace <YOUR_PORT> with: $PORT"
echo "4. Update <USERPROFILE>/.claudebox/mcp-servers/pommel/host-server-safe.py"
echo "5. Run: cd '$PROJECT_PATH' && docker compose -f docker-compose.pommel.yml up -d"
```

## Files Created

After setup, your project should have:
```
<project>/
├── .pommel-bin/
│   ├── pm
│   └── pommeld
├── languages/
│   ├── python.yaml
│   ├── javascript.yaml
│   └── ... (other language configs)
├── docker-compose.pommel.yml
└── .pommelignore (created by container)
```

## Maintenance

### Reindex after major changes
```bash
docker exec claudebox-<PROJECT_NAME>-pommel pm reindex
```

### Check indexing status
```bash
docker exec claudebox-<PROJECT_NAME>-pommel pm status
```

### Restart Pommel
```bash
cd "$PROJECT_PATH"
docker compose -f docker-compose.pommel.yml restart
```

### Stop Pommel
```bash
cd "$PROJECT_PATH"
docker compose -f docker-compose.pommel.yml down
```

### Update Pommel binaries
```bash
# Copy latest binaries from a working container
docker cp claudebox-sfu-auto-researcher-pommel:/usr/local/bin/pm "$PROJECT_PATH/.pommel-bin/"
docker cp claudebox-sfu-auto-researcher-pommel:/usr/local/bin/pommeld "$PROJECT_PATH/.pommel-bin/"

# Restart container
docker compose -f docker-compose.pommel.yml restart
```
