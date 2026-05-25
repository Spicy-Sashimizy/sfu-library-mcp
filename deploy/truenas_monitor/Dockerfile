# Tiny image for the read-only TrueNAS monitor agent (Claude Code + curl/jq).
FROM node:22-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl jq ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Claude Code CLI (the "intelligent" assessment layer). Pinned-ish; bump deliberately.
RUN npm install -g @anthropic-ai/claude-code
WORKDIR /agent
ENV HOME=/agent
# settings.json / checks.sh / monitor_loop.sh are bind-mounted by compose (read-only),
# so the image stays generic and the policy is reviewable in-repo.
CMD ["/agent/monitor_loop.sh"]
