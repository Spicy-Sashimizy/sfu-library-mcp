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
        python3 \
        python3-pip \
        python3-venv \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install Node.js 20 (for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code || echo "Claude Code install skipped"

# Install GitHub CLI for OAuth authentication
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y gh \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
# Install Chrome and ChromeDriver for Selenium
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O /tmp/chrome.deb \
    && apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        /tmp/chrome.deb \
        ca-certificates \
        fonts-liberation \
        libasound2 \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libatspi2.0-0 \
        libcups2 \
        libdbus-1-3 \
        libdrm2 \
        libgbm1 \
        libgtk-3-0 \
        libnspr4 \
        libnss3 \
        libwayland-client0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxkbcommon0 \
        libxrandr2 \
        xdg-utils \
    && rm -f /tmp/chrome.deb \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver
RUN CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+\.\d+') \
    && wget -q "https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VERSION}/linux64/chromedriver-linux64.zip" -O /tmp/chromedriver.zip \
    && unzip -q -o /tmp/chromedriver.zip -d /tmp/ \
    && mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm -rf /tmp/chromedriver*

# Install MCP dependencies
RUN python -m pip install --no-cache-dir --upgrade --break-system-packages pip \
    && python -m pip install --no-cache-dir --break-system-packages \
    mcp>=1.0.0 \
    selenium>=4.15.0 \
    requests>=2.31.0 \
    pyotp>=2.9.0

# Create workspace directory
WORKDIR /workspaces

# Set default shell to bash
SHELL ["/bin/bash", "-c"]
