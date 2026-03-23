FROM mcr.microsoft.com/devcontainers/base:bookworm

# Install common development tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        git-crypt \
        wget \
        unzip \
        jq \
        postgresql-client \
        redis-tools \
        ca-certificates \
        gnupg \
        supervisor \
        socat \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Create supervisor config directory and log directories
# /var/log/supervisor and /var/log/claudebox need to be writable by BOTH root
# and vscode because: docker-in-docker feature starts supervisord as root
# during container init, but postStartCommand runs as vscode (remoteUser).
# Using 1777 (sticky + world-writable) prevents the permission race.
RUN mkdir -p /etc/supervisor/conf.d /var/log/supervisor /var/log/claudebox \
    && chmod 1777 /var/log/supervisor /var/log/claudebox

# Copy supervisor configuration and startup script
COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY start-services.sh /usr/local/bin/start-services.sh
COPY update-libraries.sh /usr/local/bin/update-libraries.sh
COPY lsp-healthcheck.sh /usr/local/bin/lsp-healthcheck.sh
COPY lsp-symbol-bridge.sh /usr/local/bin/lsp-symbol-bridge.sh
COPY ollama-healthcheck.sh /usr/local/bin/ollama-healthcheck.sh
COPY lsp-mcp-server.py /usr/local/bin/lsp-mcp-server.py
COPY lsp-diagnostics-hook.sh /usr/local/bin/lsp-diagnostics-hook.sh
RUN chmod +x /usr/local/bin/start-services.sh /usr/local/bin/update-libraries.sh /usr/local/bin/lsp-healthcheck.sh /usr/local/bin/lsp-symbol-bridge.sh /usr/local/bin/ollama-healthcheck.sh /usr/local/bin/lsp-mcp-server.py /usr/local/bin/lsp-diagnostics-hook.sh \
    && sed -i 's/\r$//' /usr/local/bin/start-services.sh /usr/local/bin/update-libraries.sh /usr/local/bin/lsp-healthcheck.sh /usr/local/bin/lsp-symbol-bridge.sh /usr/local/bin/ollama-healthcheck.sh /usr/local/bin/lsp-mcp-server.py /usr/local/bin/lsp-diagnostics-hook.sh

# Install Node.js 20 (for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@10.9.4

# Claude Code is installed at container start via native installer (supports auto-updates)
# See entrypoint.sh install_or_update_claude_code() function
# Initialize version JSON placeholder (updated at runtime by entrypoint)
RUN echo "{\"version\":\"pending\",\"installedAt\":\"pending\",\"installerType\":\"native\",\"autoUpdate\":true}" > /usr/local/share/claude-version.json

# Install Python and quality tools (CodePro features)
# These enable the dashboard quality-status and quality-check APIs
RUN apt-get update && apt-get install -y python3-full python3-pip python3-venv \
    && python3 -m pip install --break-system-packages ruff mypy \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# ─── Chromium + Playwright browsers (PERMANENT FIX) ───────────────────────
# DO NOT REMOVE — this has been lost 6 times in container rebuilds.
# Required by: SFU auth module (Selenium) + download tier 2 (Playwright).
# Three layers ensure persistence:
#   1. This Dockerfile step (bakes into image)
#   2. Docker volume playwright_browsers (survives rebuilds)
#   3. entrypoint.sh safety-net check (catches edge cases)
# See git log: 410fd21, ee1b2a0, 893ce39, 5cc25a8, eab190a, 2fe8b0e
RUN python3 -m pip install --break-system-packages playwright \
    && python3 -m playwright install chromium --with-deps \
    && mkdir -p /home/vscode/.cache \
    && cp -r /root/.cache/ms-playwright /home/vscode/.cache/ms-playwright \
    && chown -R 1000:1000 /home/vscode/.cache/ms-playwright

# ─── poppler-utils / pdftotext (PERMANENT FIX) ────────────────────────
# DO NOT REMOVE — lost once already in commit 237beb5 when the Chromium
# section was rewritten. Required by: read_article text extraction pipeline.
# Two layers ensure persistence:
#   1. This Dockerfile step (bakes into image)
#   2. entrypoint.sh safety-net check (catches edge cases)
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install MCP Python package (required for LSP MCP server)
RUN python3 -m pip install --break-system-packages mcp

# Language Servers for Claude Code LSP integration
# Covers all languages with runtimes in this container
# Go (gopls) and Rust (rust-analyzer) installed in post-create.sh
# because their runtimes are added by devcontainer features after Dockerfile
RUN npm install -g pyright typescript-language-server typescript 2>/dev/null || \
    echo "[WARN] npm LSP server install failed, will retry at container start"

# C/C++ language server (clangd) - build-essential already provides gcc/g++
RUN apt-get update && apt-get install -y --no-install-recommends clangd \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* \
    || echo "[WARN] clangd install failed"

# universal-ctags for multi-language symbol extraction (LSP-Pommel synergy)
RUN apt-get update && apt-get install -y --no-install-recommends universal-ctags \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* \
    || echo "[WARN] ctags install failed"

# Install GitHub CLI for OAuth authentication
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y gh \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install Gitleaks for pre-commit secret scanning (industry standard, 800+ rules)
# SHA256 verified against official checksums.txt from GitHub releases
RUN GITLEAKS_VERSION="8.30.0" && \
    GITLEAKS_SHA256="79a3ab579b53f71efd634f3aaf7e04a0fa0cf206b7ed434638d1547a2470a66e" && \
    curl -sSfL -o /tmp/gitleaks.tar.gz \
      "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" && \
    echo "${GITLEAKS_SHA256}  /tmp/gitleaks.tar.gz" | sha256sum -c - && \
    tar xz -C /usr/local/bin gitleaks < /tmp/gitleaks.tar.gz && \
    chmod +x /usr/local/bin/gitleaks && \
    rm /tmp/gitleaks.tar.gz

# Add health check endpoint for version monitoring
# Checks native installer path first, then falls back to npm global
RUN echo '#!/bin/bash\nif [ -x "$HOME/.local/bin/claude" ]; then\n  VERSION=$("$HOME/.local/bin/claude" --version 2>/dev/null || echo "unknown")\nelif command -v claude &>/dev/null; then\n  VERSION=$(claude --version 2>/dev/null || echo "unknown")\nelse\n  VERSION="not-installed"\nfi\necho "Content-Type: application/json"\necho ""\ncat /usr/local/share/claude-version.json' > /usr/local/bin/claude-version && \
    chmod +x /usr/local/bin/claude-version

# Create workspace directory
WORKDIR /workspaces

# Set default shell to bash
SHELL ["/bin/bash", "-c"]
