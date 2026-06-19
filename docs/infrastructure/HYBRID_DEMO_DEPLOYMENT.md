# Hybrid demo deployment — NAS broker + DO scale-to-zero compute

**Date:** 2026-06-16 · **Status:** PLAN (nothing built yet) · **Goal:** host a public
tech demo of the **full 150M thin-client stack** for a **5–15 concurrent, self-serving**
audience, CPU-only, on **$200 DigitalOcean credit that must last ≥3 weeks**
(Student Pack credit expires 2026-07-31).

This is the companion build-plan to the analysis in `THIN_CLIENT_SWAP.md` (engines)
and `STORAGE_BUDGET_150M.md` (footprint). It supersedes the DO-only sketch discussed
in chat.

---

## 0. UPDATE 2026-06-19 — re-sized for Qdrant `on_disk` sparse (supersedes the BMP sizing below)

Two facts moved since the 2026-06-16 plan was written, and they change the
**droplet tier, the cost model, and the wake architecture**. The §1–§5 BMP-era
numbers below are kept for history but are **superseded by this section**:

1. **The 150M build finished** (2026-06-16T03:00:21, `phase:done`) — the Phase-−1
   WAIT gate is satisfied. The relevant in-flight work is now the **off-BMP →
   Qdrant `on_disk` migration** (`THIN_CLIENT_SWAP.md` § "Off-BMP migration"),
   whose 30M gate passed 2026-06-19 on RAM + latency + throughput.
2. **Qdrant `on_disk` deletes the BMP RAM wall that forced the 128 GB droplet.**
   The whole expensive shape below (128 GB @ $1/h, scale-to-zero juggling sized
   to fit ~175 droplet-hours) existed *only* because BMP held the sparse leg
   fully resident (~211 GB all-sections). Qdrant serves sparse from page cache.

**Measured 30M → extrapolated 150M (Qdrant `on_disk`, `splade_par30`):**

| Axis | 30M (measured) | 150M (*est.*, ×5) | Old BMP plan |
|---|---|---|---|
| Sparse RAM at rest | 1.4 GB | ~7 GB | ~211 GB all-sections |
| Sparse warm working set | 5.3 GB | ~26 GB *est.* (page cache, reclaimable) | (would not fit) |
| Sparse on-disk | 41 GB | ~205 GB *est.* | 280 GB |

**The serving bottleneck flips from RAM → CPU.** With sparse off the RAM budget,
the binding constraint is the **CPU cross-encoder rerank** (~5 s warm/query,
torch-CPU). So size by **vCPU for rerank parallelism**, not by 128 GB RAM.

**Re-sized droplet candidates (CPU-for-rerank + room for the ~26 GB Qdrant/tantivy
working set):**

| Tier | vCPU / RAM | Rate *est.* | Fit |
|---|---|---|---|
| `g-8vcpu-32gb` | 8 / 32 GB | ~$0.25/h | working set resident with some page-cache churn; fewer rerank workers |
| **`g-16vcpu-64gb`** (recommended) | 16 / 64 GB | ~$0.50/h | working set comfortably resident + 16 vCPU for concurrent rerank |

*(Exact DO SKU + price confirmed in Phase 0 — these are list-price estimates.)*

### Hot/cold business-hours schedule (owner request 2026-06-19)

Keep the droplet **hot 09:00–17:00 on weekdays** (landing page fully loaded,
instant queries — no 2–4 min wake) and **scale-to-zero after hours / weekends**
(boot-on-request via the existing wake path). The broker owns this policy
(`scripts/demo_broker/`, §2A) via a `BUSINESS_HOURS` window + timezone; outside
the window it falls back to the on-demand wake + idle-reaper behaviour.

**Cost model — 3 weeks, `g-16vcpu-64gb` @ ~$0.50/h *est.*:**

| Item | Basis | 3-week cost *est.* |
|---|---|---|
| Hot 9–5 weekdays | 8 h × 5 d × 3 wk = 120 h × $0.50 | ~$60 |
| After-hours boot-on-request | ~20 h occasional × $0.50 | ~$10 |
| Persistent volume (250 GB) | $0.10/GB·mo × ~0.7 mo | ~$17 |
| Boot snapshot (~40 GB) | $0.06/GB·mo | ~$2 |
| **Total** | | **~$89** (vs old $175–191) |

On the `g-8vcpu-32gb` tier the same schedule lands ≈ **$54**. Both leave large
headroom under the $204.98 credit — enough that **always-on the 32 GB tier**
(504 h × $0.25 + ~$19 standing ≈ **$145**) would also fit, if instant 24/7 is
ever wanted. Recommendation: **hot/cold 9–5 on `g-16vcpu-64gb`** — instant during
business hours, CPU headroom for 5–15 concurrent rerank, and ~$89 keeps a wide
budget margin.

**Killswitch math (revised):** hard cap `MAX_RUNTIME_HOURS_PER_DAY=10` →
≤ 10 × 21 × $0.50 = $105 compute + ~$19 standing = **$124 worst case**, inside
$200 even if the schedule logic misfires.

### What this section changes vs §1–§5 below
- Droplet tier: **128 GB @ $1/h → 64 GB @ ~$0.50/h** (or 32 GB @ ~$0.25/h).
- Engine on the droplet: **BMP mmap stack → Qdrant `on_disk` sparse** + tantivy
  BM25F + usearch dense + CPU cross-encoder rerank. Seed payload is the **Qdrant
  storage dir** (~205 GB *est.*), not the BMP `thinclient_index` (280 GB).
- Wake model: **pure scale-to-zero → hot 9–5 / cold after-hours** (above).
- Open quality gate inherited from the migration: **NDCG parity at 150M is
  UNMEASURED** (`THIN_CLIENT_SWAP.md`) — do not call the demo quality-ready until
  the full-scale ingest + parity eval lands.

> Numbers marked *est.* are extrapolations from the 30M gate; replace with
> measured values from the full 150M ingest and the Phase-5 dry run (§6).

> **GATE (updated 2026-06-19): the original "wait for build" gate is SATISFIED.**
> The 150M build reached `phase:done` 2026-06-16T03:00:21 (dense + pack complete).
> The remaining blocker for the *final* seed is the **off-BMP → Qdrant `on_disk`
> migration**: its 30M gate passed (RAM/latency/throughput), the full 150M ingest
> (~9.8 h) + NDCG parity are pending. **Interim work that is engine-independent —
> the NAS broker, int8 ONNX cross-encoder, Cloudflare/auth, and the resumable seed
> mechanism (tested on the 41 GB 30M Qdrant storage) — can and should start now**
> (see §0). Only the final volume seed + Phase-5 dry run wait on the 150M ingest.
> **Note:** the build/source host is **not always-on** — the later ~205 GB Qdrant
> storage upload (§4.1) must be **resumable/chunked**, never a single
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
4. **CPU rerank throughput.** ✅ **int8 ONNX export DONE + MEASURED 2026-06-19**
   (`scripts/export_ce_onnx_int8.py`, results
   `data/eval_results/ce_onnx_int8_bench_20260619_0302.json`). On this 16-core AVX2
   host, `models/sfu-cross-encoder-v1` (6-layer BERT, 384-hid) dynamic-int8:
   **p50 latency 114.0 ms → 5.96 ms (19.1×), throughput 5.5 → 144 pairs/s, model
   90.9 → 23.2 MB (3.9×)**; ranking-safe: Pearson r = 0.994 on logits, **top-half
   ordering Jaccard = 1.0** (int8 preserves the order). This removes the rerank
   bottleneck — a 50-candidate rerank drops from ~5.7 s to ~0.3 s. Remaining: run
   multiple uvicorn workers on the droplet; **NDCG quality parity is a separate
   LLM-judge eval** (`eval_cross_encoder.py`), not yet run — do not claim quality
   parity from the latency bench alone.
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

**MEASURED so far (interim work, 2026-06-19):**
- **int8 ONNX cross-encoder** (rerank bottleneck, risk #4): p50 114.0→5.96 ms (19.1×),
  5.5→144 pairs/s, 90.9→23.2 MB, logit r=0.994, top-half ranking Jaccard 1.0.
  Script `scripts/export_ce_onnx_int8.py`, results
  `data/eval_results/ce_onnx_int8_bench_20260619_0302.json`. Quality (NDCG) parity
  still UNMEASURED (separate LLM-judge eval).
- **Broker schedule decision core** (`scripts/demo_broker/scheduler.py`): 13/13 unit
  tests (hot-window keep-alive, prewarm boot, after-hours scale-to-zero, idle reaper,
  daily killswitch). Control-plane only; no live DO timing yet (dry-run).
- **Multi-tenant broker (per-person, account-free)** (`scripts/demo_broker/`,
  added 2026-06-19): per-invitee unguessable-token links (`mint_invite.py`) →
  isolated sessions with their own `/mcp` bearer; **push notification to owner on
  session start + errors** (ntfy/Slack/Discord webhook); **durable usage+debug**
  (sqlite `invites`/`sessions`/`events` + rotating JSONL) and owner-gated
  `/admin/report`; **safety caps** = per-session rate/daily limits + Zotero-write
  blocked for visitors (JSON-RPC `tools/call` inspected at the proxy, 403).
  **No OpenAlex key per user** — OpenAlex is keyless; the droplet uses one shared
  polite-pool identity, rate-capped. Tested: 13 tenancy unit tests + a full
  `/start`→`/mcp`→`/admin/report` integration test (dry-run, no DO spend). Live
  wake-time / per-tenant latency still UNMEASURED (needs a real droplet).
- **Resumable seed** (`scripts/seed_demo_volume.sh`): interrupt→resume→checksum-verify
  validated; dry-run enumerated the real 41 GB / 30M Qdrant storage (2,509 files).
