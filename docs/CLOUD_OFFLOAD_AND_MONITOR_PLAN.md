# Cloud Offload Pipeline + TrueNAS Monitor — Plan & Safety Contract

**Status:** scaffolding (artifacts in repo; NOTHING deployed/provisioned yet)
**Created:** 2026-05-25
**Owner intent:** host (4070 Ti + local OpenSearch) can't stay on 24/7; TrueNAS can.
Move the heavy re-encode/index *build* to an ephemeral DO GPU droplet, keep
TrueNAS as the always-on data origin + cold store, and run a small Claude Code
agent on TrueNAS to watch the box and react to issues.

---

## 0. NON-NEGOTIABLE SAFETY CONTRACT

These bind every script and deployment in this plan.

1. **Do NOT impede current TrueNAS operations.** The box runs 18 production
   containers (Jellyfin, *arr stack, Home Assistant-adjacent, NPM reverse proxy,
   WireGuard, Portainer, the live `sfu-library-mcp`) and has only **~7 GB free
   RAM** / an i5-6500 (4c4t). Anything we add is **hard resource-capped**
   (monitor ≤ 768 MB RAM, ≤ 1.0 CPU) and lives only under the dedicated dataset
   `/mnt/MAIN/sfu-library-mcp` (+ `…/sfu-library-training`). It never modifies,
   stops, or competes with the existing apps.
2. **The monitor is READ-ONLY BY DEFAULT.** It observes via a **read-only
   docker-socket proxy** (cannot start/stop/delete containers) and read-only
   host queries. "React" = **notify first**. Any auto-remediation is an explicit,
   narrow allowlist (per-container restart only) that is **disabled by default**
   and must be turned on knowingly.
3. **Cost is hard-capped.** Every DO GPU droplet has (a) an orchestrator
   `--max-hours` budget that destroys it, AND (b) a droplet-side dead-man timer
   that `shutdown`s itself if the orchestrator dies. A droplet must never sit
   idle billing. `teardown.sh` is idempotent and safe to run anytime.
4. **No secrets in git.** Tokens/keys are read from `.env` (gitignored) or
   `/mnt/MAIN/sfu-library-training/secrets` on the NAS. Scripts redact on output.
5. **Nothing consequential runs without an explicit go.** Provisioning a droplet
   (spends money) and deploying to the NAS (touches the box) are gated. Building
   the artifacts here is free and side-effect-free.

---

## 1. Why this shape (the bottleneck truths)

- The local re-encode is slow (~1k docs/s → ~40h) because it's an **in-place
  upsert into a 410 GB index**, NOT because of the GPU (which sits ~50% idle).
- The fix is a **fresh bulk build on NVMe** (the original empty-index path was
  ~17.4k docs/s ≈ 2.4h). DO droplets have fast local NVMe.
- **Network is no longer a constraint:** TELUS 3 Gbps symmetric + 2.5 GbE NIC on
  the NAS ≈ 312 MB/s. 55 GB snapshots move in ~3–6 min; 410 GB index back ~22 min.
  GPU encode appetite (~6 MB/s compressed) is fed 25–50× over → GPU never starves.
- **HDD caveat:** TrueNAS spinning disks are fine for *sequential* snapshot reads
  but **must not host the live index** (random-write + merges). Index lives on
  DO NVMe; TrueNAS holds source snapshots + cold artifacts only.
- The TrueNAS GTX 1050 Ti (4 GB) is **too weak to compute** — origin/storage role only.

## 2. Architecture

```
 TrueNAS (24/7, 2.5GbE)                 DO ephemeral GPU droplet (NVMe)
 ┌───────────────────────┐  snapshots  ┌────────────────────────────────┐
 │ /mnt/MAIN/sfu-library-*│ ──~3-6min──▶│ encode (TRT) → bulk-build FRESH │
 │  • works_part_*.jsonl  │             │  OpenSearch index on local NVMe │
 │  • cold artifacts      │◀──index────│  → snapshot index out           │
 │  • Claude Code monitor │  ~22min     │  → self-destruct (cost cap)     │
 └───────────────────────┘             └────────────────────────────────┘
        ▲ reverse proxy (NPM) / Cloudflare / WireGuard for transport
```

Serving home for the always-on index is a separate decision (§5).

## 3. Repo layout (artifacts)

```
deploy/
  do_offload/
    provision_build.sh   # create GPU droplet + cloud-init build + cost cap (GATED)
    cloud_init.sh        # runs ON the droplet: deps, pull data, fresh-build, sync out
    teardown.sh          # idempotent droplet destroy (cost lynchpin)
    config.example.env   # region/size/image/budget knobs (no secrets)
  truenas_monitor/
    docker-compose.yml   # capped Claude Code agent + read-only docker-socket-proxy
    monitor_loop.sh      # entrypoint: periodic read-only checks -> Claude -> alert
    checks.sh            # read-only health probes (containers, datasets, pool*, smart*)
    settings.json        # Claude Code permission allowlist (read-only + notify)
    README.md            # gated deploy steps + safety posture
docs/CLOUD_OFFLOAD_AND_MONITOR_PLAN.md   # this file (authoritative plan/tracker)
```

## 4. TrueNAS monitor design

- **Container**, resource-capped (`mem_limit: 768m`, `cpus: 1.0`), on its own,
  under `/mnt/MAIN/sfu-library-mcp/monitor`. Restart `unless-stopped`.
- **Observability inputs (read-only):**
  - Container health: `tecnativa/docker-socket-proxy` with ONLY `CONTAINERS=1`,
    `INFO=1` (no POST) → the agent can read status, never control.
  - Dataset capacity: bind-mount target datasets read-only; `df`.
  - Pool/SMART/temps: a tiny read-only host hook (`zpool status -x`, `smartctl -H`)
    invoked over localhost SSH with a **command-restricted** key (`command=` in
    authorized_keys) — never general shell. (Phase 2; Phase 1 uses TrueNAS's own
    alerts + container/df checks.)
- **The agent:** Claude Code (`claude -p`) invoked each cycle with the check
  output as context; it summarizes health, decides if anything is anomalous, and
  **notifies**. Permissions (`settings.json`) allow only read-only commands +
  the notify webhook; all mutating/destructive commands denied.
- **Reactions:** Phase 1 = **notify only**. Phase 2 (opt-in) = a narrow allowlist
  (`docker restart <one of an explicit set>`) via a write-scoped action helper,
  with every action logged + notified.
- **Alerts:** pluggable `NOTIFY_WEBHOOK` (ntfy/Discord/Pushover/Gmail). Default
  writes structured alerts to `monitor/alerts.log` + POSTs the webhook if set.

## 5. Open decisions / inputs needed (to finish the decision-dependent parts)

1. **Anthropic API key for the NAS agent** — provide one for
   `/mnt/MAIN/sfu-library-training/secrets/anthropic.key` (the agent needs its own;
   keep it separate from any dev key). REQUIRED before the monitor can run.
2. **Notification channel** — ntfy topic? Discord webhook? Pushover? Gmail? Sets
   `NOTIFY_WEBHOOK`. (Default until then: alerts to a logfile only.)
3. **Reaction authority** — keep Phase-1 notify-only, or enable the Phase-2
   restart allowlist (and for which containers)? Default: notify-only.
4. **DO provisioning timing** — build scripts now; provision the GPU droplet
   only on your explicit go (spends ~$5–10/run).
5. **Serving home for the always-on index** — cloud droplet+volume (fast, ~$/mo)
   vs TrueNAS (free, HDD-slow unless SSD-cached) vs local-with-interruptions.

## 6. Phased plan

- **P0 (now, free):** scaffold artifacts in repo; read-only NAS recon. ✅
- **P1 (gated):** deploy the read-only monitor (notify-only) — needs decisions #1,#2.
- **P2 (gated):** DO fresh-build pipeline dry-run, then a real provisioned run — needs #4.
- **P3 (gated):** pick + stand up the serving home — needs #5.
- **P4 (opt-in):** enable narrow monitor auto-remediation — needs #3.
