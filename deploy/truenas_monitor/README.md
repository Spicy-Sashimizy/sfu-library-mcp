# TrueNAS Monitor (Claude Code watchdog) — GATED deploy

A tiny, **resource-capped, read-only-by-default** agent that watches the TrueNAS
box and notifies on trouble. It is built to **not impede** the 18 production
containers or the box (≤768 MB / 1.0 CPU; box has ~7 GB free RAM).

## Safety posture (what makes it safe)
- **Cannot control containers.** It reads status via a `docker-socket-proxy`
  with `POST=0` — start/stop/delete are impossible through it. The proxy mounts
  the docker socket **read-only**.
- **Resource-capped** so it can't starve the media stack / HA / etc.
- **Notify-only** (`REACTIONS_ENABLED=false`). No auto-remediation in Phase 1.
- **Own dataset only.** Writes solely to `/mnt/MAIN/sfu-library-mcp/monitor/state`.
  Other mounts are `:ro`. Secrets are `:ro`.
- **Permission allowlist** (`settings.json`) denies every mutating command as
  defense-in-depth.

## Required inputs before it can run
1. **Claude auth — subscription, headless (no API key).** On any machine where
   you can log in interactively, run `claude setup-token` (OAuth flow → prints a
   long-lived token), then place it on the NAS:
   `echo -n "<token from setup-token>" | sudo tee /mnt/MAIN/sfu-library-training/secrets/claude_oauth_token`
   Uses your Pro/Max subscription. The agent only calls Claude **on anomalies**
   (heartbeats are static), so subscription usage stays minimal.
   *(An API key still works instead — put it in `…/secrets/anthropic.key`.)*
2. **Notify via ntfy (easiest).** Pick an unguessable topic, install the ntfy
   app + subscribe to it, then:
   `echo -n "https://ntfy.sh/sfu-truenas-<random>" | sudo tee /mnt/MAIN/sfu-library-training/secrets/notify_webhook`
   No account/key needed; urgent alerts push to your phone. Without it, alerts go
   to `monitor/state/alerts.log` only.

## Deploy (run ON TrueNAS — docker requires sudo there)
```bash
sudo mkdir -p /mnt/MAIN/sfu-library-mcp/monitor/state \
              /mnt/MAIN/sfu-library-training/secrets
# copy this folder to the NAS, e.g.:
#   rsync -a deploy/truenas_monitor/ truenas:/mnt/MAIN/sfu-library-mcp/monitor/
cd /mnt/MAIN/sfu-library-mcp/monitor
sudo docker compose up -d --build      # builds the tiny agent image, starts capped
```

## Verify (it's behaving + capped)
```bash
sudo docker ps --filter name=sfu-monitor                 # 2 containers, healthy
sudo docker stats --no-stream sfu-monitor                # well under 768M / 1 CPU
tail -f /mnt/MAIN/sfu-library-mcp/monitor/state/alerts.log
```

## Stop / remove (instant, clean)
```bash
cd /mnt/MAIN/sfu-library-mcp/monitor && sudo docker compose down
```

## What it watches (Phase 1)
- Every container's running/health state (via the read-only proxy).
- `/mnt/MAIN` dataset capacity (alerts ≥ `DISK_WARN_PCT`, default 88%).
- Periodic heartbeat ("all clear") every `HEARTBEAT_HOURS`.

## Phase 2 (opt-in, NOT enabled)
- Host hooks for `zpool status -x` / `smartctl -H` via a **command-restricted**
  localhost SSH key (read-only).
- A narrow auto-remediation allowlist (e.g. `docker restart <explicit set>`),
  enabled only by setting `REACTIONS_ENABLED=true` and providing `react.sh` with
  an explicit per-container allowlist. Every action logged + notified.

See `docs/CLOUD_OFFLOAD_AND_MONITOR_PLAN.md` for the full architecture + the DO
offload pipeline this monitor also watches.
