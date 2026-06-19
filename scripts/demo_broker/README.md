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
| `app.py` | FastAPI: `/start` (token+waiting room), `/status`, `/mcp` proxy, `/health`, scheduler thread |
| `test_scheduler.py` | exhaustive tests for the decision core |

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

## Not yet done (needs the migration / Phase 1–2 artifacts)

- Real snapshot + volume IDs (`DEMO_DO_SNAPSHOT_ID` / `_VOLUME_ID`) — Phase 1/2.
- Cloudflare hostname `demo.<domain>` → this gateway (`scripts/setup_cloudflare_mcp.sh`
  pattern) — Phase 3.
- Live dry-run → real boot timing measurement — Phase 5.
- Wiring the droplet's MCP server to the Qdrant sparse leg (post-migration GO).
