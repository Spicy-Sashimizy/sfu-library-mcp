#!/bin/bash
# Start background services for ClaudeBox container

# Verify we're on expected networks
echo "Verifying network connectivity..."
if ip addr show eth0 > /dev/null 2>&1; then
    echo "Network eth0 is up"
else
    echo "WARNING: eth0 not found - container may have network issues"
fi

# Start supervisor in background (manages socat proxy and pommeld)
if command -v supervisord &> /dev/null; then
    echo "Starting supervisor..."
    supervisord -c /etc/supervisor/supervisord.conf
    sleep 2

    # Check status
    if supervisorctl status > /dev/null 2>&1; then
        echo "Supervisor services:"
        supervisorctl status
    fi
else
    echo "Supervisor not installed, starting services manually..."

    # Start socat proxy manually
    if ! pgrep -f "socat.*11434" > /dev/null 2>&1; then
        nohup socat TCP-LISTEN:11434,fork,reuseaddr TCP:host.docker.internal:11434 > /var/log/socat-ollama.log 2>&1 &
        echo "Started Ollama proxy"
    fi

    # Start pommeld manually if installed
    if command -v pommeld &> /dev/null && [ -d "/workspaces/${PROJECT_NAME}/.pommel" ]; then
        cd "/workspaces/${PROJECT_NAME}"
        nohup pommeld -project "/workspaces/${PROJECT_NAME}" > /var/log/pommeld.log 2>&1 &
        echo "Started Pommel daemon"
    fi
fi

echo "Background services started"

# Trigger library auto-updates in background (non-blocking)
if [ -x /usr/local/bin/update-libraries.sh ]; then
    mkdir -p /var/log/claudebox
    touch /var/log/claudebox/library-updates.log
    chown -R vscode:vscode /var/log/claudebox
    echo "Starting background library updates..."
    nohup su - vscode -c "/usr/local/bin/update-libraries.sh" \
        > /var/log/claudebox/library-updates.log 2>&1 &
fi
