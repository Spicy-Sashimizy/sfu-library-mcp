# SFU Library demo broker (NAS-side, always-on)

Always-on FastAPI gateway that fronts an **on-demand DigitalOcean droplet** running
the thin-client MCP server, with a **hot 09:00–17:00 weekday / cold after-hours**
schedule to protect the $200 credit. Companion build for
`docs/infrastructure/HYBRID_DEMO_DEPLOYMENT.md` §0/§2A.

**Engine-agnostic** — it brokers a droplet, not a search engine, so it was built
during (and is unaffected by) the BMP→Qdrant `on_disk` migration.

## Status: scaffold, DRY-RUN by default

No DigitalOcean API call is made until `DEMO_BROKER_LIVE=1`. In dry-run, boots /
destroys are **logged, not sent**, so the full control plane (schedule, single-flight
wake, waiting room, proxy, reaper, killswitch) can be exercised without spend.

The budget-safety decision core (`scheduler.py`) is pure and **unit-tested**:

```bash
python3 scripts/demo_broker/test_scheduler.py   # 13 tests, no deps
```

## Files

| File | Role |
|---|---|
| `scheduler.py` | **Pure** hot/cold + idle-reaper + killswitch decision core (no I/O) |
| `state.py` | sqlite state: single-flight wake lock + per-day runtime accounting |
| `do_client.py` | DigitalOcean v2 wrapper (boot-from-snapshot / attach-volume / destroy); dry-run default |
| `config.py` | env-driven config + `validate()` preflight |
| `tenants.py` | per-person invites + sessions + durable events + rate caps (sqlite) |
| `notify.py` | push/webhook notifier (ntfy/Slack/Discord/generic); pure `build_payload` |
| `mint_invite.py` | CLI to mint/list/revoke per-person invite links |
| `app.py` | FastAPI: `/start` (invite+access info), `/status`, `/mcp` (per-session proxy), `/admin/report`, `/health`, scheduler thread |
| `test_scheduler.py` / `test_tenants.py` / `test_app.py` | decision core / tenancy / end-to-end wiring |

## Run (dry-run, locally)

```bash
sudo /workspaces/sfu-library-thinclient/.venv/bin/python3 -m pip install fastapi uvicorn  # NAS: into the venv
cd scripts/demo_broker
DEMO_LINK_TOKEN=test uvicorn app:app --host 0.0.0.0 --port 8088
# then: curl 'http://localhost:8088/start?t=test'   (logs a [DRY-RUN] boot)
#       curl  http://localhost:8088/health
```

## Schedule policy

- **Weekday 09:00–17:00 (local `DEMO_TZ`)**: droplet kept HOT; idle reaper suppressed.
  Pre-warm `DEMO_PREWARM_MINUTES` (5) before the open so 09:00 is already serving.
- **After hours / weekends**: scale-to-zero. A valid `/start?t=` click wakes it
  on demand; the idle reaper destroys it after `DEMO_IDLE_MINUTES` (20).
- **Hard daily killswitch**: `DEMO_MAX_RUNTIME_HOURS_PER_DAY` (10) force-destroys
  regardless of state — the budget backstop (≤ ~$124 worst case on the 64 GB tier).

## Env vars (all optional; safe defaults)

| Var | Default | Meaning |
|---|---|---|
| `DEMO_BROKER_LIVE` | `0` | `1` = make real DO API calls (spend) |
| `DIGITALOCEAN_ACCESS_TOKEN` | — | DO API token (already in repo `.env`) |
| `DEMO_DO_SIZE` | `g-16vcpu-64gb` | droplet tier (re-sized for Qdrant; see §0) |
| `DEMO_DO_REGION` | `tor1` | DO region |
| `DEMO_DO_SNAPSHOT_ID` | — | pre-baked image id (Phase 2) |
| `DEMO_DO_VOLUME_ID` | — | persistent index volume id (Phase 1) |
| `DEMO_DO_FIREWALL_ID` | — | firewall locking `:8080` to the NAS IP |
| `DEMO_LINK_TOKEN` | — | required token on `/start` (protects credit) |
| `DEMO_BEARER_TOKEN` | — | bearer required on `/mcp` proxy |
| `DEMO_TZ` | `America/Vancouver` | schedule timezone |
| `DEMO_BUSINESS_START_HOUR` / `_END_HOUR` | `9` / `17` | hot window |
| `DEMO_BUSINESS_DAYS` | `0,1,2,3,4` | Mon–Fri (0=Mon) |
| `DEMO_PREWARM_MINUTES` | `5` | boot lead before window open |
| `DEMO_IDLE_MINUTES` | `20` | after-hours idle reaper threshold |
| `DEMO_MAX_RUNTIME_HOURS_PER_DAY` | `10` | hard killswitch cap |
| `DEMO_POLL_SECONDS` | `60` | scheduler tick interval |
| `DEMO_PUBLIC_BASE` | `https://demo.example.org` | public base URL for minted invite links |
| `DEMO_ADMIN_TOKEN` | — | token gating `GET /admin/report` |
| `DEMO_NOTIFY_WEBHOOK` | — | push URL (ntfy/Slack/Discord/Pushover/generic) |
| `DEMO_NOTIFY_KIND` | `ntfy` | `ntfy` \| `slack` \| `discord` \| `json` |
| `DEMO_NOTIFY_ON_QUERY` | `0` | `1` = notify on every query (default: session-start + errors only) |
| `DEMO_SESSION_RATE_MAX` | `30` | max requests per session per window |
| `DEMO_SESSION_RATE_WINDOW_S` | `60` | rate window (seconds) |
| `DEMO_SESSION_DAILY_MAX` | `500` | max requests per session per 24h |
| `DEMO_BLOCKED_TOOLS` | `save_to_zotero,batch_save_to_zotero` | MCP tools blocked for visitors |
| `DEMO_LOG_PATH` | `data/demo_broker/broker.jsonl` | durable rotating JSONL event log |

## Multi-tenant access (per-person, account-free)

Each invitee gets an unguessable token baked into their email link — no sign-ups,
**and no OpenAlex key**: OpenAlex is keyless, so the droplet uses one shared
server-side polite-pool identity and per-session rate caps stop any one visitor
exhausting it. Clicking a link mints/reuses that person's isolated session with its
own `/mcp` bearer, so usage is attributed and isolated per person.

```bash
# mint a personal link (email the printed URL to the invitee)
DEMO_PUBLIC_BASE=https://demo.<domain> python3 mint_invite.py --name "Prof Jane Doe"
python3 mint_invite.py --name "Workshop seat" --count 20   # batch
python3 mint_invite.py --list
python3 mint_invite.py --revoke inv_xxx
```

**Flow:** click `/start?t=<token>` → validates invite → (boots droplet if down) →
returns the personal connector URL + `Bearer` token → **notifies the owner** (push
webhook) → all `/mcp` calls are bearer-gated, rate-capped, visitor-tool-blocked,
and recorded.

### Notifications
Set `DEMO_NOTIFY_WEBHOOK` to any push sink. Owner is notified on **session start**
and **errors** by default (`DEMO_NOTIFY_ON_QUERY=1` for every query). ntfy example:
`DEMO_NOTIFY_WEBHOOK=https://ntfy.sh/your-secret-topic DEMO_NOTIFY_KIND=ntfy`.

### Durable tracking + debug + reports
- **sqlite** (`invites`/`sessions`/`events` tables, in the broker state db) survives
  NAS reboots — per-person query/error/blocked counts + every event with latency +
  upstream status.
- **JSONL** mirror (`DEMO_LOG_PATH`, rotating) for raw debug grepping.
- **`GET /admin/report?since_hours=24`** (header `x-admin-token: <DEMO_ADMIN_TOKEN>`):
  events-by-kind, per-person usage, recent errors, droplet runtime + est. cost today.

### Safety caps (enforced at the broker, no droplet change)
- **Zotero writes blocked** for visitors (`save_to_zotero`/`batch_save_to_zotero`)
  by inspecting the JSON-RPC `tools/call` name → 403.
- **Per-session rate + daily caps** → 429, protecting the shared OpenAlex pool.

### Tests
```bash
python3 test_scheduler.py   # 13 — hot/cold decision core
python3 test_tenants.py     # 13 — invites/sessions/rate caps/events/notify
.venv/bin/python3 test_app.py   # 1  — full /start->/mcp->/admin wiring (dry-run)
```

## Not yet done (needs the migration / Phase 1–2 artifacts)

- Real snapshot + volume IDs (`DEMO_DO_SNAPSHOT_ID` / `_VOLUME_ID`) — Phase 1/2.
- Cloudflare hostname `demo.<domain>` → this gateway (`scripts/setup_cloudflare_mcp.sh`
  pattern) — Phase 3.
- Live dry-run → real boot timing measurement — Phase 5.
- Wiring the droplet's MCP server to the Qdrant sparse leg (post-migration GO).
