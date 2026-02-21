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

## Verified Infrastructure

Connectivity tested and confirmed on 2026-02-21:

| Component | Details |
|-----------|---------|
| TrueNAS version | 25.04.1 (Linux 6.12) |
| Docker | v27.5.0 + Compose v2.32.3 |
| ZFS pool | `MAIN` — 6.3TB total, 2.2TB free |
| SSH user | `gordoz` (key-based auth, ProxyJump via Windows host) |
| Sudo scope | **Restricted to `/usr/bin/docker` only** — no ZFS, no rm, no bash |
| SSH path | Container → `gordo@host.docker.internal` (Windows) → `gordoz@192.168.1.142` (TrueNAS) |
| SSH alias | `ssh truenas` (configured in `~/.ssh/config`) |

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

- `gordoz` can ONLY run `sudo docker *` commands
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

For off-network access: Cloudflare Tunnel or Tailscale require zero port forwarding. Traditional approach requires forwarding 8080 on the router.

---

## Phase 1: Security — Remove Hardcoded Credentials

**Priority: CRITICAL — do this first**

### 1.1 Modify `src/lib/config.py`

- Remove hardcoded credential fallbacks from `load_config()` (lines 138-141, 159-160)
- Replace with empty string defaults: `os.environ.get("SFU_USERNAME", "")` etc.
- Add `_read_secret()` helper to read Docker secrets from `/run/secrets/` with env var fallback
- Add `import logging` and `logger = logging.getLogger("sfu_library_mcp")`
- Log when secrets are loaded from files vs env vars (debug level)

Credentials to remove from source:
- `"REDACTED_SFU_USERNAME"` (SFU username)
- `"REDACTED_SFU_PASSWORD"` (SFU password)
- `"REDACTED_MFA_SECRET"` (MFA secret)
- `"mfa-device"` (MFA device name)
- `"REDACTED_ZOTERO_API_KEY"` (Zotero API key)
- `"16321308"` (Zotero user ID)

Secret loading priority: Docker secret file (`/run/secrets/X`) → env var (`SFU_X`) → empty string

---

## Phase 2: Add Logging to All Modules

### 2.1 Modules to add logging to (4 files):

- `src/lib/formatters.py` — Add logger, log result count in `format_search_results()`, log record ID in `format_item_details()`
- `src/lib/proxy_utils.py` — Add logger, log URL transformations in `make_proxied_url()`
- `src/lib/stealth.py` — Add logger, log stealth application in `apply_stealth()` and `get_stealth_context_options()`
- `src/lib/config.py` — (already covered in Phase 1) Add logger, log config loading and validation

All use: `logger = logging.getLogger("sfu_library_mcp")`

---

## Phase 3: Add Streamable HTTP Transport

### 3.1 Create `src/sfu_library_mcp_http.py`

New entry point that runs the same MCP server over Streamable HTTP instead of stdio. Reuses:
- Same `Server("sfu-library")` instance
- Same `TOOL_DEFINITIONS` and `handle_tool_call()` from `tools.py`
- Same `SFULibraryClient` and config

Uses `mcp.server.streamable_http_manager.StreamableHTTPSessionManager` (available in `mcp>=1.26.0`, already installed) + `uvicorn` + `starlette`.

- Listens on `MCP_HTTP_HOST` (default `0.0.0.0`) port `MCP_HTTP_PORT` (default `8080`)
- Endpoint: `POST /mcp`
- Stateless mode (no session persistence needed)

### 3.2 The stdio entry point (`sfu_library_mcp_server.py`) remains unchanged

- Dev container continues to use stdio as before
- Production container uses HTTP by default, can override CMD for stdio

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

- Service `mcp-server` built from `Dockerfile.prod`
- Port mapping: `8080:8080`
- Docker secrets for all 6 credential files
- Persistent volumes (bind mounts to ZFS datasets):
  - `/mnt/MAIN/sfu-library-mcp/logs` → `/app/logs`
  - `/mnt/MAIN/sfu-library-mcp/data` → `/app/data` (token cache)
  - `/mnt/MAIN/sfu-library-mcp/downloads` → `/app/downloads`
- Health check: HTTP POST to `/mcp` endpoint
- `restart: unless-stopped`
- JSON-file logging driver with 10MB rotation

### 4.4 Create `deploy/.dockerignore`

Exclude: `.git`, `.venv`, `.devcontainer`, `.claude`, `src/tests`, `chrome-extension`, `languages`, `__pycache__`, dev files

---

## Phase 5: TrueNAS Setup

### 5.1 ZFS dataset setup (manual — run from TrueNAS Web Shell)

**The `gordoz` user cannot create ZFS datasets (sudo restricted to docker only).
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

### 5.2 Create secret files on TrueNAS (manual — admin task)

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

### 5.3 Deploy (automated — runs from dev container via `gordoz`)

```bash
# From dev container:
ssh truenas "cd /mnt/MAIN/sfu-library-mcp/app && git clone https://github.com/Spicy-Sashimizy/sfu-library-mcp.git ."
ssh truenas "cd /mnt/MAIN/sfu-library-mcp/app && sudo docker compose -f deploy/docker-compose.prod.yml build"
ssh truenas "cd /mnt/MAIN/sfu-library-mcp/app && sudo docker compose -f deploy/docker-compose.prod.yml up -d"
```

### 5.4 Verify

```bash
# Container running
ssh truenas "sudo docker ps | grep sfu-library-mcp"

# Health check passing
ssh truenas "sudo docker inspect --format='{{.State.Health.Status}}' sfu-library-mcp"

# Logs writing
ssh truenas "ls -la /mnt/MAIN/sfu-library-mcp/logs/"

# Secrets mounted
ssh truenas "sudo docker exec sfu-library-mcp ls /run/secrets/"

# HTTP responding
ssh truenas "curl -s -X POST http://localhost:8080/mcp -H 'Content-Type: application/json' -d '{\"jsonrpc\":\"2.0\",\"method\":\"initialize\",\"id\":1}'"
```

---

## Phase 6: Update Mechanism

### 6.1 Create `deploy/update.sh` — Git-based updates

SSH to TrueNAS (via `ssh truenas`) → `git pull` → `sudo docker compose build` → `sudo docker compose up -d`

### 6.2 Create `deploy/deploy-rsync.sh` — Detached from git

rsync from dev container → TrueNAS, excludes dev files, then rebuild container. For when you don't want git on the TrueNAS server.

### 6.3 Both scripts include:

- Pre-deploy health check
- Post-deploy validation (container running, HTTP responding)
- Rollback hint (previous image tag)
- All docker commands prefixed with `sudo`
- SSH via `truenas` alias (ProxyJump through Windows host)

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

---

## Phase 8: Log Access from Dev Container

### 8.1 Create `deploy/logs.sh`

Convenience script with subcommands:
- `./deploy/logs.sh tail 100` — last 100 lines of application log
- `./deploy/logs.sh follow` — real-time stream (`tail -f`)
- `./deploy/logs.sh errors 50` — grep ERROR lines
- `./deploy/logs.sh docker 100` — Docker container logs (`sudo docker logs`)

All via `ssh truenas`, reading from `/mnt/MAIN/sfu-library-mcp/logs/`.

### 8.2 Docker logging

Production compose configured with `json-file` driver, 10MB max, 3 rotations. Application-level logs use the existing `RotatingFileHandler` (5MB, 3 backups) writing to the persistent volume.

---

## Phase 9: Testing Plan

### 9.1 Pre-deployment (run in dev container)

1. `pytest src/tests/ -v` — all existing tests pass
2. Verify no hardcoded credentials: `grep -rn "REDACTED_SFU_USERNAME\|REDACTED_PW_PREFIX" src/lib/config.py` returns nothing
3. Verify all modules have logging: check each `.py` file has `getLogger`
4. Test HTTP entry point starts locally: `python3 src/sfu_library_mcp_http.py` + curl
5. `sudo docker build -f deploy/Dockerfile.prod -t sfu-library-mcp:test .` succeeds
6. Existing test suite still passes after config.py refactor (tests use `mock_config` fixture)

### 9.2 Post-deployment (on TrueNAS)

1. Container running and healthy
2. HTTP endpoint responds to JSON-RPC `initialize` and `tools/list`
3. Secrets accessible inside container
4. Log file being written to persistent volume
5. Claude Desktop connects via `mcp-remote` and can list tools
6. Run a test search: `search_library` with a simple query

### 9.3 Test changes to existing tests

- `src/tests/test_config.py` — update tests that rely on hardcoded defaults to provide env vars or mock secrets
- `src/tests/conftest.py` `mock_config` fixture — already provides explicit values, should work as-is
- Add new test for `_read_secret()` function

---

## Files Summary

| Action | File | Phase |
|--------|------|-------|
| Modify | `src/lib/config.py` | 1, 2 |
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
| Modify | `src/tests/test_config.py` | 9 |

---

## Implementation Order

1. **Phase 1** (security fix) → 2. **Phase 2** (logging) → 3. **Phase 9.3** (fix tests) → 4. **Phase 3** (HTTP transport) → 5. **Phase 4** (Docker image) → 6. **Phase 5** (TrueNAS deploy) → 7. **Phase 6** (update scripts) → 8. **Phase 7** (remote access) → 9. **Phase 8** (log access) → 10. **Phase 9.1-9.2** (validation)

**Manual steps required (TrueNAS Web Shell / admin user):**
- Phase 5.1: ZFS dataset creation
- Phase 5.2: Secret file creation

---

## Verification

After full implementation:
1. All tests pass in dev container
2. Production container builds and starts on TrueNAS
3. Claude Desktop connects via LAN (`mcp-remote` to `192.168.1.142:8080/mcp`)
4. All 26 MCP tools are listed and a test search works
5. Logs visible from dev container via `deploy/logs.sh follow`
6. No credentials visible in source code, Docker image, or container env vars
7. `gordoz` sudo restricted to docker only — no system-level risk
