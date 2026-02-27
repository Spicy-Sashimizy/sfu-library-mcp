# SFU Library MCP Server — TrueNAS Production Deployment Plan

## Context

The SFU Library MCP server currently runs inside a ClaudeBox dev container using stdio transport. It needs to be deployed to a TrueNAS SCALE 25.04.1 server at 192.168.1.142 as a lean production container — without dev tools — while keeping all logging, securing credentials on the shared server, enabling remote access from Claude Desktop, and providing update/log access mechanisms.

Key problems this solves:
- Hardcoded credentials in config.py (SFU password, MFA secret, Zotero key) — security risk on shared server
- No production Docker image — dev container is bloated with Node.js, Claude Code, ruff, mypy, etc.
- stdio-only transport — can't access MCP remotely from Claude Desktop
- 4 modules lack logging — blind spots for production debugging
- No deployment pipeline for TrueNAS

---

## Pre-Deployment Audit (2026-02-21)

### Verified Infrastructure

| Component | Details |
|-----------|---------|
| TrueNAS version | 25.04.1 (Linux 6.12) |
| Docker | v27.5.0 + Compose v2.32.3 |
| ZFS pool | `MAIN` — 6.3TB total, 2.2TB free |
| SSH user | `<SSH_USER>` (key-based auth, ProxyJump via Windows host) |
| Sudo scope | **Restricted to `/usr/bin/docker` only** — no ZFS, no rm, no bash |
| SSH path | Container → `gordo@host.docker.internal` (Windows) → `gordoz@192.168.1.142` (TrueNAS) |
| SSH alias | `ssh truenas` (configured in `~/.ssh/config`) |
| RAM | 23GB total, **5.6GB available** (16 containers already running, no swap) |
| CPU | 4 cores |
| Docker images | 65GB used, **54.85GB reclaimable** (recommend cleanup) |
| Port 8080 | **Confirmed free** — no conflicts |
| Test suite | **420/420 tests passing** |
| Python | 3.11.2 in venv |
| MCP version | 1.26.0 |

### Known Issues Found During Audit

| # | Issue | Severity | File(s) | Action |
|---|-------|----------|---------|--------|
| 1 | Hardcoded credentials as fallback defaults | CRITICAL | `config.py:138-141,159-160` | Phase 1 fixes |
| 2 | MCP HTTP class is `StreamableHTTPServerTransport` (NOT `StreamableHTTPSessionManager`) | CRITICAL | Plan correction | Phase 3 uses correct class |
| 3 | `tools.py:2219` hardcodes `/tmp/sfu-library-mcp.log` ignoring config | HIGH | `tools.py` | Phase 2 fixes |
| 4 | `devcontainer.json:46` has broken JSON syntax on forwardPorts | HIGH | `devcontainer.json` | Phase 0 fixes |
| 5 | Stale references to `vansedataadstackshit` and `192.168.1.100` | HIGH | `deploy/health_check.py`, `deploy/truenas_setup.md` | Phase 0 cleans up |
| 6 | `client.py` has ~6 silent `except Exception:` blocks (no logging) | MEDIUM | `client.py:159,280,303,307,402` | Phase 2 adds logging |
| 7 | `client.py:787` has hardcoded email in User-Agent | MEDIUM | `client.py` | Phase 1 makes configurable |
| 8 | Only 5.6GB RAM available on TrueNAS (no swap) | HIGH | TrueNAS config | Memory limits in compose |
| 9 | 54.85GB reclaimable Docker images on TrueNAS | MEDIUM | TrueNAS | Recommend cleanup pre-deploy |
| 10 | `/mnt/MAIN/sfu-library-mcp/` doesn't exist yet | INFO | TrueNAS | Manual step in Phase 5 |
| 11 | `requirements.txt` mixes dev and prod dependencies | MEDIUM | `requirements.txt` | Phase 4 creates `requirements.prod.txt` |
| 12 | Profane text in config.py comment (line 23) | LOW | `config.py` | Phase 1 cleans up |

### SSH Config (already set up in dev container)

```
Host truenas
    HostName 192.168.1.142
    User gordoz
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump windows-host
    StrictHostKeyChecking no

Host windows-host
    HostName host.docker.internal
    User gordo
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
```

### Security Constraints

- `<SSH_USER>` can ONLY run `sudo docker *` commands
- ZFS dataset creation, secret file permissions, and other system admin tasks must be done **manually via TrueNAS Web Shell** or by a user with full sudo
- All deploy scripts must use `sudo docker` (not bare `docker`)

---

## Port Requirements

| Port | Direction | Purpose |
|------|-----------|---------|
| 22 | Dev → Windows → TrueNAS | SSH for deploy scripts, log tailing, rsync (ProxyJump) |
| 8080 | Inside TrueNAS Docker | MCP Streamable HTTP endpoint |
| 8080 | LAN → TrueNAS | Claude Desktop direct LAN access to MCP |
| 443 (outbound) | TrueNAS → Cloudflare | Cloudflare Tunnel (no inbound port forwarding) |

Port 8080 is confirmed free on TrueNAS — no existing container uses it.

---

## Safety Checkpoints

Every phase has a **CHECKPOINT** gate. Deployment does NOT proceed to the next phase unless the checkpoint passes. If a checkpoint fails, the phase is rolled back and the issue is fixed before retrying.

### Checkpoint Protocol

```
┌─────────────┐     ┌──────────┐     ┌────────────┐     ┌──────────┐
│ Phase N      │────▶│CHECKPOINT│────▶│ Phase N+1  │────▶│CHECKPOINT│──▶ ...
│ (implement)  │     │ (verify) │     │ (implement)│     │ (verify) │
└─────────────┘     └──────────┘     └────────────┘     └──────────┘
                         │                                    │
                    FAIL ▼                               FAIL ▼
                    ┌──────────┐                         ┌──────────┐
                    │ ROLLBACK │                         │ ROLLBACK │
                    │ & FIX    │                         │ & FIX    │
                    └──────────┘                         └──────────┘
```

### MCP Server Operational Guarantee

**The dev container's MCP server (stdio) must remain working at ALL times.**
After every phase that modifies `src/` files:

1. Run `pytest src/tests/ -v` — all 420+ tests must pass
2. Dry-run the server: `timeout 5 python3 src/sfu_library_mcp_server.py 2>&1 || true` — must not crash on import
3. If either fails: **revert changes immediately** (`git checkout -- src/`) and fix before retrying

---

## Phase 0: Pre-Deployment Cleanup

Fix existing issues that would complicate later phases.

### 0.1 Fix `devcontainer.json` syntax error (line 46)

```json
// BROKEN:
"forwardPorts": [3000, 5050, 5432, 6379, 8000, 8080], 7420],

// FIXED:
"forwardPorts": [3000, 5050, 5432, 6379, 8000, 8080, 7420],
```

### 0.2 Clean up stale deploy files

- Remove or rewrite `deploy/truenas_setup.md` (references wrong project `vansedataadstackshit` and wrong IP `192.168.1.100`)
- Remove or rewrite `deploy/health_check.py` (same stale references)
- Remove or rewrite `deploy/deploy.sh` (points to wrong project)

### 0.3 Recommend TrueNAS Docker cleanup (manual, optional)

```bash
# Run from TrueNAS Web Shell — frees ~54GB of unused images
# CAUTION: only prune images not used by running containers
sudo docker image prune -a --filter "until=720h"
```

### CHECKPOINT 0
- [ ] `devcontainer.json` is valid JSON (parse with `python3 -m json.tool`)
- [ ] No references to `vansedataadstackshit` in `deploy/` directory
- [ ] All 420 tests still pass
- [ ] MCP server dry-run: no import errors

---

## Phase 1: Security — Remove Hardcoded Credentials

**Priority: CRITICAL — do this first**

### 1.1 Modify `src/lib/config.py`

- Remove hardcoded credential fallbacks from `load_config()` (lines 138-141, 159-160)
- Replace with empty string defaults: `os.environ.get("SFU_USERNAME", "")` etc.
- Add `_read_secret()` helper to read Docker secrets from `/run/secrets/` with env var fallback
- Add `import logging` and `logger = logging.getLogger("sfu_library_mcp")`
- Log when secrets are loaded from files vs env vars (debug level)
- Clean up profane comment on line 23

Credentials to remove from source:
- `"REDACTED_SFU_USERNAME"` (SFU username)
- `"REDACTED_SFU_PASSWORD"` (SFU password)
- `"REDACTED_MFA_SECRET"` (MFA secret)
- `"mfa-device"` (MFA device name)
- `"REDACTED_ZOTERO_API_KEY"` (Zotero API key)
- `"16321308"` (Zotero user ID)

Secret loading priority: Docker secret file (`/run/secrets/X`) → env var (`SFU_X`) → empty string

### 1.2 Set credentials as environment variables in dev container

After removing hardcoded defaults, the dev container needs the credentials injected via env vars so the MCP server continues to work locally. Add to `.devcontainer/docker-compose.yml` or a `.env` file (gitignored).

### CHECKPOINT 1
- [ ] `grep -rn "REDACTED_SFU_USERNAME\|REDACTED_PW_PREFIX\|REDACTED_MFA_SECRET\|REDACTED_ZOTERO_API_KEY" src/` returns **nothing**
- [ ] `_read_secret()` function exists and has a unit test
- [ ] All 420+ tests pass (mock_config fixture provides explicit values)
- [ ] MCP server dry-run: no import errors
- [ ] Dev container MCP server still works with env vars set

---

## Phase 2: Add Logging to All Modules

### 2.1 Modules to add logging to (4 files):

- `src/lib/formatters.py` — Add logger, log result count in `format_search_results()`, log record ID in `format_item_details()`
- `src/lib/proxy_utils.py` — Add logger, log URL transformations in `make_proxied_url()`
- `src/lib/stealth.py` — Add logger, log stealth application in `apply_stealth()` and `get_stealth_context_options()`
- `src/lib/config.py` — (already covered in Phase 1) Add logger, log config loading and validation

All use: `logger = logging.getLogger("sfu_library_mcp")`

### 2.2 Fix hardcoded log path in `tools.py`

- Line 2219: Change `log_path = "/tmp/sfu-library-mcp.log"` to read from config
- Ensure the `get_server_logs` tool uses the configured `log_file` path

### 2.3 Add logging to silent exception handlers in `client.py`

Add `logger.warning()` or `logger.debug()` to the ~6 bare `except Exception:` blocks that currently swallow errors silently (lines 159, 280, 303, 307, 402).

### CHECKPOINT 2
- [ ] Every `.py` file in `src/lib/` has `logging.getLogger` (verify with grep)
- [ ] `tools.py` no longer hardcodes `/tmp/sfu-library-mcp.log`
- [ ] All 420+ tests pass
- [ ] MCP server dry-run: no import errors
- [ ] Logging output visible when running server with `SFU_LOG_LEVEL=DEBUG`

---

## Phase 3: Add Streamable HTTP Transport

### 3.1 Create `src/sfu_library_mcp_http.py`

New entry point that runs the same MCP server over Streamable HTTP instead of stdio. Reuses:
- Same `Server("sfu-library")` instance
- Same `TOOL_DEFINITIONS` and `handle_tool_call()` from `tools.py`
- Same `SFULibraryClient` and config

Uses `mcp.server.streamable_http.StreamableHTTPServerTransport` (confirmed available in `mcp==1.26.0`) + `uvicorn` + `starlette`.

**NOTE:** The class is `StreamableHTTPServerTransport`, NOT `StreamableHTTPSessionManager`. Verified by import test against installed mcp 1.26.0.

- Listens on `MCP_HTTP_HOST` (default `0.0.0.0`) port `MCP_HTTP_PORT` (default `8080`)
- Endpoint: `POST /mcp`
- Stateless mode (no session persistence needed)
- Health endpoint: `GET /health` — returns 200 with `{"status": "ok", "tools": <count>}`

### 3.2 The stdio entry point (`sfu_library_mcp_server.py`) remains unchanged

- Dev container continues to use stdio as before
- Production container uses HTTP by default, can override CMD for stdio

### CHECKPOINT 3
- [ ] `python3 src/sfu_library_mcp_http.py` starts without errors (test with timeout)
- [ ] `curl -X POST http://localhost:8080/mcp` with JSON-RPC `initialize` returns a response
- [ ] `curl http://localhost:8080/health` returns `{"status": "ok"}`
- [ ] stdio entry point still works (unchanged)
- [ ] All 420+ tests pass
- [ ] HTTP server shuts down cleanly on SIGTERM

---

## Phase 4: Production Docker Image

### 4.1 Create `requirements.prod.txt`

Production dependencies only (no pytest, pytest-asyncio, ruff, mypy):
`mcp>=1.26.0`, `selenium`, `requests`, `pyotp`, `pyzotero`, `curl_cffi`, `playwright`, `uvicorn`, `starlette`

### 4.2 Create `deploy/Dockerfile.prod`

Lean image based on `python:3.11-slim-bookworm`:
- **Includes:** Chrome + ChromeDriver (auth), Playwright Chromium (downloads), `poppler-utils` (pdftotext)
- **Excludes:** Node.js, Claude Code, ruff, mypy, pytest, gh CLI, redis-tools, postgresql-client, supervisor, socat, git-crypt, build-essential
- Runs as non-root user `mcp`
- Default CMD: `python3 src/sfu_library_mcp_http.py`
- Exposes port 8080

### 4.3 Create `deploy/docker-compose.prod.yml`

- Service `sfu-library-mcp` built from `Dockerfile.prod`
- **Container name:** `sfu-library-mcp`
- Port mapping: `8080:8080`
- Docker secrets for all 6 credential files
- Persistent volumes (bind mounts to ZFS datasets):
  - `/mnt/MAIN/sfu-library-mcp/logs` → `/app/logs`
  - `/mnt/MAIN/sfu-library-mcp/data` → `/app/data` (token cache)
  - `/mnt/MAIN/sfu-library-mcp/downloads` → `/app/downloads`
- **Memory limit: `2g`** (protect TrueNAS from OOM — only 5.6GB free with 16 other containers)
- **Memory reservation: `512m`** (guaranteed minimum)
- Health check: `curl -sf http://localhost:8080/health || exit 1` (every 30s, 3 retries, 10s timeout)
- `restart: unless-stopped`
- JSON-file logging driver with 10MB rotation, 3 files max
- **Stop grace period: 30s** (allow in-flight requests to complete)

### 4.4 Create `deploy/.dockerignore`

Exclude: `.git`, `.venv`, `.devcontainer`, `.claude`, `src/tests`, `chrome-extension`, `languages`, `__pycache__`, dev files

### CHECKPOINT 4
- [ ] `docker build -f deploy/Dockerfile.prod -t sfu-library-mcp:test .` succeeds in dev container
- [ ] Container starts: `docker run --rm -p 8080:8080 sfu-library-mcp:test`
- [ ] Health endpoint responds: `curl http://localhost:8080/health`
- [ ] Container respects memory limit (check with `docker stats`)
- [ ] Container shuts down cleanly on `docker stop` (within 30s)
- [ ] No credentials baked into the image: `docker run --rm sfu-library-mcp:test env | grep -i sfu` returns nothing
- [ ] Image size is reasonable (target: <1.5GB vs dev container)

---

## Phase 5: TrueNAS Setup

### 5.1 Pre-deploy safety checks (automated)

**Run these BEFORE deploying to TrueNAS.** Built into deploy scripts.

```bash
# 1. Verify SSH connectivity
ssh truenas "echo 'SSH OK'"

# 2. Verify Docker access
ssh truenas "sudo docker info --format '{{.ServerVersion}}'"

# 3. Check available memory (FAIL if <2GB free)
ssh truenas "free -m | awk '/^Mem:/{if(\$7 < 2048) exit 1; else print \"RAM OK: \" \$7 \"MB available\"}'"

# 4. Check available disk (FAIL if <5GB free on MAIN)
ssh truenas "df -BG /mnt/MAIN | awk 'NR==2{gsub(/G/,\"\",\$4); if(\$4 < 5) exit 1; else print \"Disk OK: \" \$4 \"GB free\"}'"

# 5. Check port 8080 not already bound
ssh truenas "sudo docker ps --format '{{.Ports}}' | grep -q 8080 && echo 'FAIL: Port 8080 in use' && exit 1 || echo 'Port 8080 OK'"
```

**If any check fails: ABORT deployment and report which check failed.**

### 5.2 ZFS dataset setup (manual — run from TrueNAS Web Shell)

**The `<SSH_USER>` user cannot create ZFS datasets (sudo restricted to docker only).
These commands must be run by an admin user via the TrueNAS Web Shell or SSH as `truenas_admin`.**

```bash
# Run from TrueNAS Web Shell (System Settings → Shell) or as truenas_admin
zfs create MAIN/sfu-library-mcp
mkdir -p /mnt/MAIN/sfu-library-mcp/{app,secrets,logs,data,downloads}
chown -R gordoz:gordoz /mnt/MAIN/sfu-library-mcp
chmod 700 /mnt/MAIN/sfu-library-mcp/secrets
```

Resulting layout:
```
/mnt/MAIN/sfu-library-mcp/
├── app/          # Git clone or rsync target
├── secrets/      # chmod 700 credential files
├── logs/         # Persistent log volume
├── data/         # Token cache
└── downloads/    # Downloaded PDFs
```

### 5.3 Create secret files on TrueNAS (manual — admin task)

**Run from TrueNAS Web Shell** — 6 files in `/mnt/MAIN/sfu-library-mcp/secrets/`:

```bash
# Run as admin in TrueNAS Web Shell
echo -n "YOUR_SFU_USERNAME" > /mnt/MAIN/sfu-library-mcp/secrets/sfu_username
echo -n "YOUR_SFU_PASSWORD" > /mnt/MAIN/sfu-library-mcp/secrets/sfu_password
echo -n "YOUR_MFA_SECRET" > /mnt/MAIN/sfu-library-mcp/secrets/sfu_mfa_secret
echo -n "YOUR_MFA_DEVICE_NAME" > /mnt/MAIN/sfu-library-mcp/secrets/sfu_mfa_device_name
echo -n "YOUR_ZOTERO_API_KEY" > /mnt/MAIN/sfu-library-mcp/secrets/zotero_api_key
echo -n "YOUR_ZOTERO_USER_ID" > /mnt/MAIN/sfu-library-mcp/secrets/zotero_user_id

chown gordoz:gordoz /mnt/MAIN/sfu-library-mcp/secrets/*
chmod 600 /mnt/MAIN/sfu-library-mcp/secrets/*
```

### CHECKPOINT 5 (manual steps)
- [ ] ZFS dataset exists: `zfs list MAIN/sfu-library-mcp`
- [ ] Directories created: `ls /mnt/MAIN/sfu-library-mcp/{app,secrets,logs,data,downloads}`
- [ ] Ownership correct: `stat -c '%U:%G' /mnt/MAIN/sfu-library-mcp/` shows `gordoz:gordoz`
- [ ] Secret files exist and are not world-readable: `ls -la /mnt/MAIN/sfu-library-mcp/secrets/`
- [ ] All 6 secret files contain values (not empty)

### 5.4 Deploy (automated — runs from dev container via `<SSH_USER>`)

```bash
# From dev container:
ssh truenas "cd /mnt/MAIN/sfu-library-mcp/app && git clone https://github.com/Spicy-Sashimizy/sfu-library-mcp.git ."
ssh truenas "cd /mnt/MAIN/sfu-library-mcp/app && sudo docker compose -f deploy/docker-compose.prod.yml build"
ssh truenas "cd /mnt/MAIN/sfu-library-mcp/app && sudo docker compose -f deploy/docker-compose.prod.yml up -d"
```

### 5.5 Post-deploy verification

```bash
# 1. Container running
ssh truenas "sudo docker ps | grep sfu-library-mcp"

# 2. Health check passing
ssh truenas "sudo docker inspect --format='{{.State.Health.Status}}' sfu-library-mcp"

# 3. Logs writing
ssh truenas "ls -la /mnt/MAIN/sfu-library-mcp/logs/"

# 4. Secrets mounted
ssh truenas "sudo docker exec sfu-library-mcp ls /run/secrets/"

# 5. HTTP responding (JSON-RPC initialize)
ssh truenas "curl -sf -X POST http://localhost:8080/mcp \
  -H 'Content-Type: application/json' \
  -d '{\"jsonrpc\":\"2.0\",\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1.0\"}},\"id\":1}'"

# 6. Health endpoint
ssh truenas "curl -sf http://localhost:8080/health"

# 7. Memory usage within limits
ssh truenas "sudo docker stats sfu-library-mcp --no-stream --format '{{.MemUsage}}'"
```

### CHECKPOINT 5-DEPLOY
- [ ] Container status: `running`
- [ ] Health status: `healthy`
- [ ] All 6 secrets accessible in container
- [ ] HTTP endpoint returns valid JSON-RPC response
- [ ] Health endpoint returns `{"status": "ok"}`
- [ ] Memory usage < 2GB (container limit)
- [ ] Log file exists in `/mnt/MAIN/sfu-library-mcp/logs/`

---

## Phase 6: Update Mechanism

### 6.1 Create `deploy/update.sh` — Git-based updates

SSH to TrueNAS (via `ssh truenas`) → `git pull` → `sudo docker compose build` → `sudo docker compose up -d`

### 6.2 Create `deploy/deploy-rsync.sh` — Detached from git

rsync from dev container → TrueNAS, excludes dev files, then rebuild container. For when you don't want git on the TrueNAS server.

### 6.3 Both scripts include:

**Pre-deploy safety:**
- Verify SSH connectivity
- Verify available memory (>2GB) and disk (>5GB)
- Verify port 8080 not stolen by another container
- Snapshot current state: `sudo docker images | grep sfu-library-mcp` (for rollback reference)

**Zero-downtime deploy:**
- Build new image BEFORE stopping old container
- Only stop old container after new image is ready
- Start new container
- Wait for health check (up to 60s)

**Post-deploy validation:**
- Container running and healthy
- HTTP endpoint responds to JSON-RPC `initialize`
- Health endpoint returns `{"status": "ok"}`
- Verify all 26 tools are listed via `tools/list`

**Rollback on failure:**
- If health check fails after 60s: stop new container, restart previous image
- Print rollback command: `sudo docker compose -f deploy/docker-compose.prod.yml down && sudo docker compose -f deploy/docker-compose.prod.yml up -d`

**All docker commands prefixed with `sudo`.
SSH via `truenas` alias (ProxyJump through Windows host).**

### CHECKPOINT 6
- [ ] `deploy/update.sh` runs end-to-end on TrueNAS
- [ ] Pre-deploy checks catch simulated failures (low memory, port conflict)
- [ ] Post-deploy validation confirms all 26 tools available
- [ ] Rollback scenario tested: manually break deploy, verify rollback works

---

## Phase 7: Remote Access from Claude Desktop

### 7.1 LAN Access (same network)

Claude Desktop config:
```json
{
  "mcpServers": {
    "sfu-library": {
      "command": "npx",
      "args": ["mcp-remote", "http://192.168.1.142:8080/mcp"]
    }
  }
}
```
`mcp-remote` (npm package) bridges Claude Desktop's stdio to the remote HTTP endpoint.

### 7.2 Cloudflare Tunnel (off-network, no port forwarding)

- Add `cloudflared` service to `docker-compose.prod.yml`
- Configure tunnel in Cloudflare Zero Trust dashboard: `sfu-mcp.yourdomain.com` → `http://localhost:8080`
- Claude Desktop config uses: `"https://sfu-mcp.yourdomain.com/mcp"`
- Requires: Cloudflare account, domain, tunnel token
- Ports: None to forward (outbound only)

### 7.3 Tailscale VPN (off-network, mesh network)

- Install Tailscale app on TrueNAS (available in app catalog)
- TrueNAS gets a Tailscale IP (e.g., `100.x.y.z`)
- Claude Desktop config uses: `"http://100.x.y.z:8080/mcp"`
- Requires: Tailscale account on both devices
- Ports: None to forward

### 7.4 SSH Tunnel (fallback, zero server changes)

Claude Desktop can SSH into TrueNAS and run stdio MCP directly:
```json
{
  "mcpServers": {
    "sfu-library": {
      "command": "ssh",
      "args": ["truenas",
               "sudo docker exec -i sfu-library-mcp python3 src/sfu_library_mcp_server.py"]
    }
  }
}
```
No HTTP transport needed — uses stdio over SSH. Works on LAN or VPN.
Note: Requires the `truenas` SSH alias configured on the machine running Claude Desktop.

### CHECKPOINT 7
- [ ] Claude Desktop can connect via at least one method
- [ ] `tools/list` returns all 26 tools
- [ ] A test `search_library` call returns results
- [ ] Connection survives for >5 minutes without dropping

---

## Phase 8: Log Access from Dev Container

### 8.1 Create `deploy/logs.sh`

Convenience script with subcommands:
- `./deploy/logs.sh tail 100` — last 100 lines of application log
- `./deploy/logs.sh follow` — real-time stream (`tail -f`)
- `./deploy/logs.sh errors 50` — grep ERROR lines
- `./deploy/logs.sh docker 100` — Docker container logs (`sudo docker logs`)
- `./deploy/logs.sh health` — run health check and show status

All via `ssh truenas`, reading from `/mnt/MAIN/sfu-library-mcp/logs/`.

### 8.2 Docker logging

Production compose configured with `json-file` driver, 10MB max, 3 rotations. Application-level logs use the existing `RotatingFileHandler` (5MB, 3 backups) writing to the persistent volume.

### CHECKPOINT 8
- [ ] `./deploy/logs.sh tail 10` shows recent log lines
- [ ] `./deploy/logs.sh errors 5` filters ERROR-level messages
- [ ] `./deploy/logs.sh health` reports container status

---

## Phase 9: Operational Health Monitoring

### 9.1 Create `deploy/healthcheck.sh`

Comprehensive health check script that can be run on-demand or scheduled:

```bash
./deploy/healthcheck.sh
```

Checks:
1. **SSH connectivity** — can reach TrueNAS
2. **Container running** — `sudo docker ps | grep sfu-library-mcp`
3. **Container healthy** — Docker health check status is `healthy`
4. **HTTP responsive** — `curl` to `/health` endpoint returns 200
5. **Tools available** — JSON-RPC `tools/list` returns 26 tools
6. **Memory usage** — container under 2GB limit
7. **Disk usage** — `/mnt/MAIN` has >5GB free
8. **Log freshness** — last log line is <1 hour old (server isn't frozen)
9. **Secret integrity** — all 6 secrets mounted in container

Output: Pass/fail for each check, overall status, and recommended actions for failures.

### 9.2 Automatic recovery in `docker-compose.prod.yml`

- `restart: unless-stopped` — Docker restarts container if it crashes
- Health check with `start_period: 60s` — gives container time to initialize Chrome/Playwright
- `--retries 3` — marks unhealthy after 3 consecutive failures
- Memory limit `2g` — prevents OOM from killing other TrueNAS services

### 9.3 Alerting (optional, future)

If desired, a cron job on TrueNAS can run `healthcheck.sh` and email/notify on failure. Not implemented in initial deployment but the health check script is designed to support it (exit code 0 = healthy, exit code 1 = unhealthy).

---

## Phase 10: Testing Plan

### 10.1 Pre-deployment (run in dev container)

1. `pytest src/tests/ -v` — all 420+ tests pass
2. Verify no hardcoded credentials: `grep -rn "REDACTED_SFU_USERNAME\|REDACTED_PW_PREFIX" src/` returns nothing
3. Verify all modules have logging: `grep -rn "getLogger" src/lib/*.py` — every file has it
4. Test HTTP entry point starts locally: `python3 src/sfu_library_mcp_http.py` + curl
5. `docker build -f deploy/Dockerfile.prod -t sfu-library-mcp:test .` succeeds
6. Existing test suite still passes after config.py refactor (tests use `mock_config` fixture)
7. Container starts and responds to health check

### 10.2 Post-deployment (on TrueNAS)

1. Container running and healthy
2. HTTP endpoint responds to JSON-RPC `initialize` and `tools/list`
3. Secrets accessible inside container
4. Log file being written to persistent volume
5. Claude Desktop connects via `mcp-remote` and can list tools
6. Run a test search: `search_library` with a simple query
7. Memory stays under 2GB after test search
8. Container survives 10-minute soak test (stays healthy)

### 10.3 Test changes to existing tests

- `src/tests/test_config.py` — update tests that rely on hardcoded defaults to provide env vars or mock secrets
- `src/tests/conftest.py` `mock_config` fixture — already provides explicit values, should work as-is
- Add new test for `_read_secret()` function

### 10.4 Rollback test

- Deliberately break the container (e.g., remove a secret)
- Verify health check detects the failure
- Verify `restart: unless-stopped` restarts the container
- Verify deploy script rollback works

---

## Files Summary

| Action | File | Phase |
|--------|------|-------|
| Fix | `.devcontainer/devcontainer.json` | 0 |
| Clean up | `deploy/truenas_setup.md` | 0 |
| Clean up | `deploy/health_check.py` | 0 |
| Clean up | `deploy/deploy.sh` | 0 |
| Modify | `src/lib/config.py` | 1, 2 |
| Modify | `src/lib/tools.py` (line 2219) | 2 |
| Modify | `src/lib/client.py` (silent exceptions) | 2 |
| Modify | `src/lib/formatters.py` | 2 |
| Modify | `src/lib/proxy_utils.py` | 2 |
| Modify | `src/lib/stealth.py` | 2 |
| Create | `src/sfu_library_mcp_http.py` | 3 |
| Create | `requirements.prod.txt` | 4 |
| Create | `deploy/Dockerfile.prod` | 4 |
| Create | `deploy/docker-compose.prod.yml` | 4 |
| Create | `deploy/.dockerignore` | 4 |
| Create | `deploy/update.sh` | 6 |
| Create | `deploy/deploy-rsync.sh` | 6 |
| Create | `deploy/logs.sh` | 8 |
| Create | `deploy/healthcheck.sh` | 9 |
| Modify | `src/tests/test_config.py` | 10 |

---

## Implementation Order

```
Phase 0  (cleanup)          → CHECKPOINT 0
Phase 1  (security fix)     → CHECKPOINT 1
Phase 2  (logging)          → CHECKPOINT 2
Phase 10.3 (fix tests)      → verify 420+ tests pass
Phase 3  (HTTP transport)   → CHECKPOINT 3
Phase 4  (Docker image)     → CHECKPOINT 4
--- MANUAL: User creates ZFS datasets + secrets on TrueNAS ---
Phase 5  (TrueNAS deploy)   → CHECKPOINT 5 + CHECKPOINT 5-DEPLOY
Phase 6  (update scripts)   → CHECKPOINT 6
Phase 7  (remote access)    → CHECKPOINT 7
Phase 8  (log access)       → CHECKPOINT 8
Phase 9  (health monitoring) → verify healthcheck.sh passes
Phase 10.1-10.2 (validation) → full system test
Phase 10.4 (rollback test)  → verify recovery works
```

**Manual steps required (TrueNAS Web Shell / admin user):**
- Phase 5.2: ZFS dataset creation
- Phase 5.3: Secret file creation
- Optional: Docker image cleanup (Phase 0.3)

---

## Verification (Final Acceptance)

After full implementation, ALL of these must be true:

1. All 420+ tests pass in dev container
2. No credentials visible in source code: `grep -rn "REDACTED_SFU_USERNAME\|REDACTED_PW_PREFIX\|REDACTED_MFA_SECRET\|REDACTED_ZOTERO_API_KEY" src/` returns nothing
3. Production container builds and starts on TrueNAS
4. Container memory < 2GB, restarts automatically on crash
5. Claude Desktop connects via LAN (`mcp-remote` to `192.168.1.142:8080/mcp`)
6. All 26 MCP tools are listed and a test search works
7. Logs visible from dev container via `deploy/logs.sh follow`
8. `deploy/healthcheck.sh` reports all checks passing
9. `<SSH_USER>` sudo restricted to docker only — no system-level risk
10. Dev container MCP server (stdio) still works exactly as before
