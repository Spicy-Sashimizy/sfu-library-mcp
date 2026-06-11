# Connecting Claude to the SFU Library MCP server

Two access paths, both pointing at the **same MCP toolset** (`sfu-library`,
Streamable HTTP transport, `/mcp` endpoint):

| Path | Use from | Server it hits | Always on? |
|------|----------|----------------|------------|
| A. Local stdio bridge | Claude Desktop on the dev box | dev devcontainer (`:8080`) | only while the dev box + devcontainer are up |
| B. Public HTTPS connector | Claude **mobile / web / desktop, anywhere** | **NAS** prod container (`:8080`) | yes — NAS is always on (`restart: unless-stopped`) |

For off-network / phone use, **use Path B** — Claude reaches custom connectors
from Anthropic's *cloud*, not from your device, so the endpoint must be public
HTTPS. The NAS already hosts the server, a `cloudflared` tunnel, and NPM.

---

## Path A — local Claude Desktop (already wired)

Merge `claude_desktop_config.training.json` into
`%APPDATA%\Claude\claude_desktop_config.json`. It runs an in-container
`mcp-remote` bridge over `docker exec` to the persistent HTTP server
(`deploy/mcp_http.supervisor.conf`), so the heavy models stay warm
(cold first query ~134s → warm ~5s).

---

## Path B — public connector via NAS + Cloudflare (for mobile / off-network)

### What is already running on the NAS (no changes needed)
- `sfu-library-mcp` container — Streamable HTTP MCP, published `:8080`,
  `restart: unless-stopped`. Verify: `curl http://127.0.0.1:8080/health` → `{"status":"ok",...}`.
- `cloudflared` — **token-managed** tunnel (routing configured in the Cloudflare
  Zero Trust dashboard, NOT in a file on the NAS).
- `nginx-proxy-manager` — not required for this path; the tunnel can hit the MCP
  port directly.

> **Automated:** `scripts/setup_cloudflare_mcp.sh` performs Steps 1–2 idempotently
> from `.env` (needs `CLOUDFLARE_API_TOKEN` with `Cloudflare Tunnel:Edit`,
> `DNS:Edit`, `WAF:Edit`, `Account:Read`). It merges the tunnel ingress
> (preserving existing hostnames), creates the DNS CNAME, and creates the WAF
> allowlist — auto-creating the `http_request_firewall_custom` entrypoint ruleset
> if the zone has never had a custom rule. The manual dashboard steps below are
> the equivalent if you prefer clicking.

### Step 1 — add a tunnel public hostname (Cloudflare Zero Trust dashboard)
`Networks → Tunnels → (your tunnel) → Public Hostname → Add a public hostname`
- **Subdomain:** `mcp`   **Domain:** `<YOURDOMAIN>`
- **Service:** `HTTP`  →  `192.168.1.142:8080`  *(NAS LAN IP : MCP port; bypasses NPM)*

Saving auto-creates the DNS record. Quick check from anywhere:
`curl https://mcp.<YOURDOMAIN>/health` → `{"status":"ok","tools":26}`.

### Step 2 — lock it to Anthropic's IPs (Cloudflare → Security → WAF → Custom rules)
The connector traffic only ever originates from Anthropic's published range, so
deny everything else on this hostname:
- **Expression:** `(http.host eq "mcp.<YOURDOMAIN>" and not ip.src in {160.79.104.0/21})`
- **Action:** `Block`

`160.79.104.0/21` is Anthropic's **outbound** range (where MCP connector calls
come from). Source: <https://platform.claude.com/docs/en/api/ip-addresses>.

### Step 3 — add the connector in Claude
`Settings → Connectors → Add custom connector`
- **URL:** `https://mcp.<YOURDOMAIN>/mcp`

---

## ⚠️ Security caveat (important)
The Anthropic-IP allowlist stops random internet scanning, **but Anthropic's
`/21` is shared by all Anthropic connector traffic** — so another Claude user who
learns your URL could still reach it (their connector egresses from the same
IPs). Therefore:
- **Treat `mcp.<YOURDOMAIN>` as a secret.** (This is why it isn't committed here.)
- The tools can spend your OpenAlex budget and **write to your Zotero**
  (`save_to_zotero`, `batch_save_to_zotero`).
- For real protection add a **bearer token** (a Cloudflare WAF rule additionally
  requiring a secret header, if your Claude client lets you set one) or proper
  **OAuth 2.1**. The server has no native auth today — ask to add this.
