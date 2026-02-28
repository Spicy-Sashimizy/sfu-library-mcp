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

# Create supervisor config directory
RUN mkdir -p /etc/supervisor/conf.d /var/log/supervisor

# Copy supervisor configuration and startup script
COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY start-services.sh /usr/local/bin/start-services.sh
COPY update-libraries.sh /usr/local/bin/update-libraries.sh
RUN chmod +x /usr/local/bin/start-services.sh /usr/local/bin/update-libraries.sh \
    && sed -i 's/\r$//' /usr/local/bin/start-services.sh /usr/local/bin/update-libraries.sh

# Install Node.js 20 (for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@10.9.4

# Claude Code is installed at container start via native installer (supports auto-updates)
# See entrypoint.sh install_or_update_claude_code() function
# Initialize version JSON placeholder (updated at runtime by entrypoint)
RUN echo "{\"version\":\"pending\",\"installedAt\":\"pending\",\"installerType\":\"native\",\"autoUpdate\":true}" > /usr/local/share/claude-version.json

# Install Chrome + ChromeDriver for Selenium auth (SFU Library MCP)
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Python and quality tools (CodePro features)
# These enable the dashboard quality-status and quality-check APIs
RUN apt-get update && apt-get install -y python3-full python3-pip python3-venv \
    && python3 -m pip install --break-system-packages ruff mypy \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

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
