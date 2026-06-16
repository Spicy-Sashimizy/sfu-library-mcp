# Hybrid demo deployment — NAS broker + DO scale-to-zero compute

**Date:** 2026-06-16 · **Status:** PLAN (nothing built yet) · **Goal:** host a public
tech demo of the **full 150M thin-client stack** for a **5–15 concurrent, self-serving**
audience, CPU-only, on **$200 DigitalOcean credit that must last ≥3 weeks**
(Student Pack credit expires 2026-07-31).

This is the companion build-plan to the analysis in `THIN_CLIENT_SWAP.md` (engines)
and `STORAGE_BUDGET_150M.md` (footprint). It supersedes the DO-only sketch discussed
in chat.

> **GATE (2026-06-16): do not start building any phase yet.** Per owner: wait until
> the index build **fully finishes** before standing up the demo. Current state is
> `build_status.json` `phase:build` — all 5 subject sections built, but **`dense_done:
> false` and `sections_packed:[]`**, so the dense + pack phases are still pending.
> Completion signal to watch: `sudo supervisorctl status migration-150m` exits 0
> (DONE, no restart) **and** `build_status.json` shows the dense/pack phases done.
> **Also:** the build/source host is **not always-on** — the build runs intermittently
> and the later 280 GB upload (§4.1) must be **resumable/chunked**, never a single
> assumed-continuous run.

---

## 1. Why this shape (the constraints that forced it)

Measured facts that drive the design:

- **NAS cannot serve compute.** TrueNAS box (`192.168.1.142`) measured 2026-06-16:
  Intel **i5-6500, 4 cores**, **23 GB RAM (~7 GB free)**, MAIN pool **435 GB free**.
  Disk is fine for the 280 GB index; RAM/CPU are far too small to serve full 150M
  (wants ~128 GB to page-cache; cross-encoder rerank needs real CPU). So the NAS is a
  **broker**, not a compute node. The "NAS serves normal load + DO bursts" hybrid is
  **dead**.
- **Always-on DO for 3 weeks is impossible on $200.** 21 d × 24 h = 504 h; the
  smallest serving-capable tier (64 GB @ $0.50/h) = $252; 128 GB @ $1/h = $504. The
  budget only survives if the droplet runs **only during actual demo use**
  → **scale-to-zero**, which needs an always-on thing to wake it and reap it. That
  thing is the NAS (free; already always-on with a Cloudflare tunnel).
- **Index is 280 GB, CPU-only serving.** mmap stack (tantivy + BMP + usearch); rerank
  is a CPU cross-encoder (~5 s warm/query, ~134 s cold model load).

Result architecture:

```
email link ──> https://demo.<domain>/start  (Cloudflare tunnel ──> NAS gateway, always-on)
                     │  validate link token
                     ├─ if compute DOWN: DO API → create droplet, attach persistent volume, boot
                     ├─ waiting-room page polls droplet /health  (~2–4 min)
                     └─ when healthy: show the MCP connector URL (proxied via the tunnel)
                                          │
   NAS idle-reaper watchdog: no queries for N min  ──> DO API destroy droplet (keep volume)
   NAS hard killswitch:      max daily runtime cap  ──> force destroy (budget backstop)
```

---

## 2. Components

### A. NAS gateway (always-on broker) — build new
Small Python (FastAPI) service on the NAS, fronted by the **existing** cloudflared tunnel
(add one public hostname `demo.<domain>` → NAS gateway port). Responsibilities:

1. **Landing page** `/start?t=<token>` — the email link target. Validates a static
   **link token** (rejects randoms; protects the credit from drive-by wakes).
2. **Wake**: on valid click with compute down, call DO API to create the droplet from
   the pre-baked **snapshot** + attach the **persistent volume**. **Single-flight lock**
   (a state file / DB row) so 15 simultaneous clicks create **one** droplet.
3. **Waiting room**: HTML that polls the droplet `/health`; on healthy, reveal the MCP
   connector URL + "Add custom connector" instructions.
4. **Reverse-proxy** the stable public URL `demo.<domain>/mcp` → current droplet `:8080`
   (so the connector URL is stable across droplet IP churn).
5. **Idle reaper** (background loop): poll the droplet's query log / last-activity; after
   `IDLE_MINUTES` (default 20) of no queries → **destroy** the droplet, keep the volume.
6. **Hard killswitch** (cron): enforce a `MAX_RUNTIME_HOURS_PER_DAY` cap — force-destroy
   even if "active", as a budget backstop against a stuck reaper.

Footprint on the NAS: tiny (<256 MB) — fits the 7 GB free headroom easily.

### B. DO compute droplet (on-demand) — provision via API
- Size: **`m-16vcpu-128gb`** (128 GB, $1.00/h) — sweet spot. (`m-24vcpu-192gb`, $1.50/h,
  if 5–15 concurrent thrashes in the dry run.)
- Boots from a **pre-baked snapshot**: Ubuntu + thin-client code + models
  (`models/sfu-cross-encoder-v1`, `models/splade_onnx*`) + cloud-init that:
  mounts the volume, starts the thin-client MCP container
  (`SFU_SEARCH_BACKEND=thinclient`, `SFU_DENSE_WARMCACHE=0`), runs the **model
  pre-warm**, then flips `/health` to ready.
- DO firewall: `:8080` reachable only from the **NAS IP** (+ Anthropic range only if MCP
  connects directly rather than via the NAS proxy).

### C. Persistent block-storage volume (always allocated) — create once
- **~300 GB**, region same as the droplet. Holds the 280 GB `thinclient_index`.
- Stays allocated across droplet destroy/create → **wake = attach (seconds), not copy**.
- Covered by credits (block storage is a Service, not an excluded Add-On).

---

## 3. Cost model (3 weeks, scale-to-zero)

| Item | Rate | 3-week cost | Covered by credit? |
|---|---|---|---|
| Persistent volume 300 GB | $0.10/GB·mo | ~**$21** | yes |
| Boot snapshot (~40 GB) | $0.06/GB·mo | ~**$1.7** | yes |
| Compute (128 GB) | $1.00/h **only while up** | rest (~**$175**) | yes |
| Gateway / reaper | NAS | **$0** | — |

→ **~$175 left for live compute = ~175 droplet-hours** over 3 weeks (~8 h/day every
day, or far more if demos are occasional). **Budget holds comfortably** as long as the
droplet is never pinned 24/7. Bandwidth: 128 GB tier includes ~6 TB transfer; demo
JSON is negligible.

**Killswitch math:** hard cap e.g. `MAX_RUNTIME_HOURS_PER_DAY=8` guarantees
≤ 8 × 21 × $1 = $168 + $23 standing = **$191 worst case**, inside $200.

---

## 4. Known risks / open decisions (resolve in Phase 0)

1. **One-time 280 GB upload — must be RESUMABLE (source host is not always-on).** The
   index lives on the build/dev box, which runs intermittently, so the transfer cannot
   assume a single ~31 h-at-20 Mbps continuous run. Use a resumable/chunked method:
   `rsync --partial --append-verify --inplace` (re-runnable, picks up where it left
   off), or chunked upload to DO Spaces (`s3cmd`/`rclone` with retries) then pull
   internally to the volume. Drive it from a re-entrant script (cron/systemd) that
   survives source reboots. **This is the main setup-time cost and the long pole.**
2. **Abstracts incomplete.** Only `social_sciences__recent` has local abstracts
   (`build_status.json`); all other sections fetch abstracts **live from OpenAlex** at
   rerank → latency + API budget under concurrency. `OPENALEX_API_KEY` is in `.env`.
   Decide: finish sidecars vs accept live-fetch vs scope demo to social_sciences.
3. **Dense leg not built** (`dense_done:false`) → natural-language queries weaker
   (loses the measured +0.17 NDCG@10 NL win). Decide: build full-corpus dense vs accept.
4. **CPU rerank throughput.** Cross-encoder is torch/CPU (~5 s warm/query, no int8/ONNX
   export yet). At 5–15 concurrent this is the bottleneck. Highest-leverage pre-demo
   task: **export an int8 ONNX cross-encoder** + run multiple uvicorn workers.
5. **Reaper correctness = budget safety.** The idle reaper MUST reliably destroy; the
   hard daily killswitch is the backstop. Test both before going live.
6. **Auth.** Server has no native auth. Add a **bearer token** (link token + MCP header)
   and keep `demo.<domain>` secret (Anthropic's connector range is shared).
7. **Build currently stopped** mid-`phase:build` (not packed, dense not done); confirm
   the index state to snapshot is the one you want.

---

## 5. Build phases

- **Phase −1 — WAIT (current):** index build must fully finish (dense + pack) before
  anything below starts. See the GATE note at the top. Quality decisions (#2/#3/#4) are
  deferred until the build completes, then revisited against the finished artifacts.
- **Phase 0 — confirm + decide:** credits (done: $204.98, exp 2026-07-31), region,
  decisions #2/#3/#4 above, index upload path (#1).
- **Phase 1 — seed the volume:** create 300 GB volume + seed droplet; get the 280 GB
  index onto it; detach.
- **Phase 2 — bake the image:** snapshot OS + code + models + cloud-init + pre-warm.
- **Phase 3 — NAS gateway:** FastAPI app (token landing + wake + proxy + waiting room),
  cloudflared hostname `demo.<domain>`, single-flight lock.
- **Phase 4 — safety:** idle reaper + hard daily killswitch + DO firewall + bearer auth.
- **Phase 5 — dry run:** measure wake time, warm-query latency, behavior at 5–15
  concurrent; tune tier (128↔192) and worker count.
- **Phase 6 (quality, optional pre-demo):** int8 ONNX cross-encoder, abstract sidecars
  for remaining sections, full-corpus dense leg.

---

## 6. Efficacy (to be recorded — none measured yet)

Per repo doc rules, the following are UNMEASURED and must be filled from the Phase-5 dry
run before the demo is called ready: wake latency (click→serving), warm single-query
latency at 150M, sustained latency at 5/10/15 concurrent, $/demo-hour actuals. Mark any
pre-measurement numbers `*est.*`.
