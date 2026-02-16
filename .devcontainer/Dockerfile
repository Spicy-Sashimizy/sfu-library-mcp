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
    && npm install -g npm@latest

# Claude Code is installed at container start via native installer (supports auto-updates)
# See entrypoint.sh install_or_update_claude_code() function
# Initialize version JSON placeholder (updated at runtime by entrypoint)
RUN echo "{\"version\":\"pending\",\"installedAt\":\"pending\",\"installerType\":\"native\",\"autoUpdate\":true}" > /usr/local/share/claude-version.json

# Install Google Chrome + ChromeDriver (required for Selenium-based SFU CAS/Duo MFA auth)
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+\.\d+') \
    && wget -q "https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VERSION}/linux64/chromedriver-linux64.zip" -O /tmp/chromedriver.zip \
    && unzip -o /tmp/chromedriver.zip -d /tmp \
    && mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver \
    && chmod +x /usr/local/bin/chromedriver \
    && rm -rf /tmp/chromedriver* \
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

# Add health check endpoint for version monitoring
# Checks native installer path first, then falls back to npm global
RUN echo '#!/bin/bash\nif [ -x "$HOME/.local/bin/claude" ]; then\n  VERSION=$("$HOME/.local/bin/claude" --version 2>/dev/null || echo "unknown")\nelif command -v claude &>/dev/null; then\n  VERSION=$(claude --version 2>/dev/null || echo "unknown")\nelse\n  VERSION="not-installed"\nfi\necho "Content-Type: application/json"\necho ""\ncat /usr/local/share/claude-version.json' > /usr/local/bin/claude-version && \
    chmod +x /usr/local/bin/claude-version

# Create workspace directory
WORKDIR /workspaces

# Set default shell to bash
SHELL ["/bin/bash", "-c"]
