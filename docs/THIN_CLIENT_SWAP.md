# Thin-Client Swap — this container's architecture (2026-06-11)

This dev container is the **thin-client testbed**: OpenSearch is removed from
it entirely and the serving path runs the validated no-JVM stack. The original
`sfu-library-mcp` container (150M-doc OpenSearch 3.7 cluster at host port 9200)
is untouched and is used only as a **read-only export source and eval baseline**.

## What runs here now

| Leg | Engine | Config |
|---|---|---|
| Lexical BM25F | tantivy 0.26 | title^3 + abstract, `index_option='freq'` (positions-off lossless lever), fast-field year/type/is_oa filters, no stored text |
| Sparse SPLADE | bmp 0.2.6 | 8-bit block-max impacts, **bsize=256 + chunked top-term clustered insertion** (measured −54% size, −61% latency at equal recall), 2M-doc shards, weights ×70 quantization (saturation-free for SPLADE log1p ≤ 3.64), sidecar post-filtering |
| Dense | usearch 2.x | b1 Hamming mmap + int8 rescore memmap (binary+rescore 32×, validated R@10 0.996); coverage = dense-POC 600k until full corpus is encoded, **plus the query-driven warm cache below** |
| Dense warm cache | `lib/thinclient/dense_cache.py` | query-driven delta over the static seed (RAM Hamming + int8 rescore, SQLite durability, 500k-doc cap); `SFU_DENSE_WARMCACHE=0` disables (default on); design in `DENSE_WARMCACHE_RESEARCH.md` |
| Fusion | Python RRF k=60 | unchanged (`federated_search.py`) |
| Rerank | unchanged | abstracts via hot sidecar / OpenAlex fetch (below) |

Serving swap: `SFU_SEARCH_BACKEND=thinclient` (default) makes `tools.py`
construct `lib.thinclient.retriever.ThinClientRetriever` — a drop-in for
`OpenSearchRetriever` (same `search/dense_search/is_available` surface and
result shape). `opensearch` value restores the legacy backend (evals use it
against the original cluster).

Serving hardening (2026-06-11 evening): SPLADE query-encoder fallback (lexical
+ dense still serve if the encoder dies), per-leg latency metrics + JSONL query
log (`SFU_METRICS_LOG_PATH` / `SFU_QUERY_LOG_PATH`), and four index-management
MCP tools: `get_index_status`, `list_personas`, `request_section_unpack`,
`reload_index`.

## Hot/cold profile structure (fully implemented)

- Subject sections: `social_sciences`, `med_bio`, `phys_eng`, `cs_math`,
  `other` — priority-ordered disjoint title+abstract vocabulary classifier
  (`lib/thinclient/sections.py`, ported from the measured eval).
- **Era sub-sections (primary fast axis):** each subject section is BUILT as
  two date-range sub-sections, `<base>__recent` / `<base>__archive` (boundary
  publication_year 2010, routed at build time in the builder; the export spool
  stays base-keyed). Year-filtered queries prune the contradicting era at
  query time (`retriever._era_skip`).
- Personas (`PERSONAS`) are era-qualified: subject personas default to the
  recent era (`political_science` → `social_sciences__recent`) with
  `*_historical` variants for the archive era, plus era-wide `contemporary` /
  `historical` and the testbed `all_hot`. Hot sections stay live **with a
  zstd-dictionary abstract sidecar**; cold sections are packed.
- **Packing operates on built artifacts** (`lib/thinclient/packer.py`,
  tar+zstd-19 LDM): unpack = pure decompression at disk speed with SHA-256
  artifact parity — replacing the old rebuild-from-JSONL flow that ran at
  2,016 docs/s (~2.4 h per section at 150M). Archives are kept after unpack,
  so re-packing an unwritten section is just deleting the live dir.
- Abstract policy (measured: extern_display is ranking-lossless): hot = local
  sidecar; cold = OpenAlex API mget before rerank. Never title-only rerank.
  Sidecar format is **v3**: script-bucketed (latin/cyrillic/han/…) clustered
  32 KB blocks, zstd-19 with per-script trained dicts (measured 64.7→52.9 GB
  at 150M); readers transparently handle v1 (per-doc) and v2 (+ru dict).
- `meta.sqlite` is **v2**: W-ids as INTEGER PRIMARY KEY with a TEXT overflow
  table (measured ~10 GB at 150M vs 24 GB v1); the retriever auto-detects
  v1/v2 schemas.
- CLI: `python -m lib.thinclient.packer {pack|unpack|repack|status} <root> [section]`.

## Index layout

```
data/thinclient_index/
  manifest.json        provenance, persona, section states, checksums
  build_status.json    resumable build checkpoint (phase/section granularity)
  build_progress_<base>.json   per-spool-slice BUILD checkpoint (transient;
                       deleted when the section completes)
  meta.sqlite          v2: int-PK id -> title/doi/year/type/is_oa/section
  sections/<base>__<era>/   tantivy/ + splade_NNN.bmp (+vocab.zst sidecars)
                            + abstracts.sqlite (hot)
  packed/<base>__<era>.tar.zst
  dense/               b1.usearch + rescore_int8.npy + ids.json
  dense_cache/         query-driven warm-cache delta (SQLite + int8)
```

## Migration (150M docs, kept)

`scripts/build_thinclient_index.py` streams the corpus out of the original
cluster's `openalex_works` (sliced scroll; `_source` still carries the
production `sparse_field` weights, so **no re-encoding**), classifies sections,
and builds all artifacts checkpointed (export/build/pack/dense phases).
Monthly rebuilds later: snapshot JSONL → `lib/thinclient/doc_encoder.py`
(the GPU TRT batch SPLADE encoder, extracted from the retired indexer) →
same builder.

Operational helpers around the running 150M migration:
- `scripts/wait_and_resume_migration.sh` — polls the source cluster's `_count`
  and auto-launches the build once ≥150M docs are stable (used when the
  source cluster goes down mid-export).
- `scripts/restart_at_build_phase.sh` — armed to restart the run at the
  export→build boundary so the BUILD phase picks up post-launch builder code
  (era routing, abstracts v3, meta v2, b256 clustering).

Build RAM (measured 2026-06-12): the era split runs TWO builders per section
worker, and `--build-workers 3` OOM-killed the 150M BUILD on the 31 GB host
(`BrokenProcessPool` ~1 min in, swap 5 GB deep) **at the then-default
`--bmp-shard-docs 2_000_000`** — observed worker RSS ~10.5-11.3 GB each at 2
workers. Code defaults stay `--build-workers 2` / 2M shard docs;
`BMP_CLUSTER_CHUNK_DOCS 250k`, `TANTIVY_WRITER_HEAP 512 MB`. The 150M run's
supervisor conf overrides to `--build-workers 3 --bmp-shard-docs 1_000_000`
(2026-06-12): halving the shard buffer is what makes the third worker fit —
*estimated* from the RSS breakdown, throughput/RSS to be confirmed from the
run's own logs. Trade-off: ~2x BMP shard count (plus one boundary rotation
per slice, see below) → more query-time fan-out, mitigated by vocab-sidecar
skip + era pruning; retrieval-latency impact UNMEASURED at 150M.
Measured under full 3-way load (2026-06-13): ~6.2 GB/worker early, drifting
+~0.35 GB/h each (tantivy segment churn across per-slice commits) — pool now
uses `max_tasks_per_child=1` so each section task starts in a fresh process;
per-worker throughput ~1,050 docs/s non-hot / ~375 docs/s hot (abstracts
zstd-19 is the hot-section bottleneck) vs ~1,400/s solo — the third worker
nets ~+12% aggregate, not +50%.

WAL high-water-mark swap-thrash + fix (2026-06-13): the RSS budget above was
not the only memory sink. A plain sqlite `commit()` in WAL mode only does a
PASSIVE checkpoint, which never shrinks the `-wal` file below its high-water
mark, so the per-section `meta_*.sqlite-wal` grew unbounded across slices
(measured mid-run: other 5.8G, social_sciences 3.3G, cs_math 1.9G ≈ 11G
combined). That page-cache load drove the 31 GB host into swap (free RAM
1.5 GB, swap 6.2 GB deep, one worker crawling at ~54 KB/s in `D` state) even
though per-worker RSS looked fine. Fix: `SectionBuilder.slice_checkpoint()`
and the hot-section `AbstractStoreWriter.checkpoint()` now issue
`PRAGMA wal_checkpoint(TRUNCATE)` after each per-slice commit, resetting the
`-wal` to ~0 every slice. Crash-safe (data is durable from the commit before
the truncate) — re-validated with `test_build_resume.py` (clean==chaos, 11
kills). Procedure that day: dropped to `--build-workers 2` to clear the
thrash (RAM used 29G→9G, free →21G), shipped the truncate, then restored
`--build-workers 3`.

Per-slice truncate was INSUFFICIENT — intra-slice fix (2026-06-13): the truncate
above only fires at a slice boundary, but the dominant sections have ~2-2.5h
slices and the meta rows are not committed until that boundary, so a single
slice's uncommitted `-wal` still grew to GB before any truncate could run. Over
a 9.5h 3-worker run only ONE slice committed (cs_math); `other`/`social_sciences`
never reached a boundary and their `-wal` reballooned to 5.8G/5.3G, dragging
free RAM back to 5.0G. Fix: `SectionBuilder.add()` now commits + TRUNCATEs the
meta `-wal` every `META_WAL_TRUNCATE_DOCS` docs (default 250k ≈ ~200MB cap,
`SFU_META_WAL_TRUNCATE_DOCS` env override). Safe because meta rows are
`INSERT OR REPLACE` (idempotent on the whole-slice re-feed resume performs) and
only the meta db is touched — tantivy/BMP still commit/rotate at the slice
boundary, so the atomic-slice resume model is unchanged. Validated with
`test_build_resume.py` at `SFU_META_WAL_TRUNCATE_DOCS=3000` (clean==chaos, 12
mid-build kills, mid-slice truncates firing under the chaos). Bounded-WAL
efficacy at 150M scale: to be confirmed from the post-fix run's WAL sizes.

Per-slice BUILD checkpointing (2026-06-12): a silent whole-session kill cost
~8 h because BUILD resume granularity was the whole section (10-15 h each at
150M). The section worker now checkpoints after every spool slice
(`build_progress_<base>.json`): meta + abstracts commit (both sqlite sidecars
moved `journal_mode OFF -> WAL` so a SIGKILL can't corrupt them), tantivy
commit, and the open BMP shard rotates at the slice boundary (shards never
span slices, so resume just deletes shards >= the recorded `next_shard`).
A crash now costs at most one slice (~10-25 min) instead of the section.
Edge cases handled: tantivy double-commit window (first re-fed slice probes
for its first doc id and skips re-adds), abstracts dict-training buffers
(persisted as `pending_*` meta rows at checkpoints — training still waits
for the full 20k sample), orphaned abstract blocks from re-fed slices
(dropped at finish). Slice-boundary BMP rotation is safe: measured per-slice
era minimum is 18.6k docs (med_bio archive), above the 5k tail bar.
Kill/resume parity test: `scripts/tests/test_build_resume.py` (SIGKILL every
8-25 s until completion; asserts meta/tantivy/BMP-search/abstracts parity
against an uninterrupted build of the same real-data spool subset).

Process supervision (2026-06-12): a second BUILD run died SILENTLY ~17 min in
(whole session gone, no traceback, no OOM event, no low-RAM sample — killer
unidentified; nohup/setsid detachment from tool-spawned shells is fragile
here). The migration now runs under **supervisord**:
`scripts/migration-150m.supervisor.conf` → `/etc/supervisor/conf.d/`,
`autorestart=unexpected` + checkpointed phases make restarts free;
`sudo supervisorctl status migration-150m` to check, exit 0 = DONE (no
restart). Remove the conf after the build completes.

### BUILD COMPLETE — measured results (2026-06-16)

The 150M migration reached `phase: done` at **2026-06-16T03:00:21**
(`data/thinclient_index/{build_status.json,manifest.json}`,
`logs/migration_150m.log`). **150,413,098 docs**, 5 personas × 2 eras = 10
sub-sections. Engines as built (from `manifest.json`): tantivy 0.26 BM25F
(title^3/abstract, freq-only), bmp 0.2.6 (b256 clustered, `splade_quant_scale`
70), usearch b1 + int8 rescore (32×).

As-built on-disk footprint — persona `political_science`, hot
`social_sciences__recent` live + 9 cold sub-sections packed (measured `du`,
`manifest.json`):

| Component | Size | Notes |
|---|---|---|
| packed cold (9 × `tar.zst`) | 45.0 GB | live 84.3 GB → 45.0 GB, aggregate **1.875×** |
| hot section live (`social_sciences__recent`) | 34.0 GB | incl. 26,322,924 abstracts (zstd-19 script dicts) |
| `meta.sqlite` | **24.9 GB** | meta v2 INTEGER-PK; **exceeds the ~10 GB v2 estimate** in `STORAGE_BUDGET_150M.md` §1 — corrected there, cause not yet diagnosed |
| dense leg | 0.36 GB | 600k vectors, dim 384: b1 118 MB + int8 rescore 230 MB |
| **Serving total (on disk)** | **≈ 104 GB** (97 GiB) | vs ≈ 95 GB estimate; the +9 GB is entirely `meta.sqlite` |

> **⚠ On-disk ≠ resident RAM (measured 2026-06-17).** The 68.6 GB of BMP SPLADE
> shards are NOT mmap'd — `bmp.Searcher` (0.2.6, no mmap mode) deserializes each
> `*.bmp` into **anonymous RAM at a measured 3.07×** its on-disk size (probe:
> 569 MB shard → 1747 MB steady-state RSS; ratio holds 12 MB–569 MB). `_load()`
> builds all 192 shards eagerly, so the **full 150M SPLADE leg needs ~211 GB
> resident** — it cannot be served, even one-engine-at-a-time, on the 24 GB host.
> tantivy (29 GB), `meta.sqlite` (24 GB) and dense (0.34 GB) are genuinely
> mmap/paged and stay ~0 resident; the RAM wall is BMP alone. `retriever._load()`
> now samples `MemAvailable` before each shard and aborts with a clear
> `RuntimeError` (gate `SFU_LOAD_MEM_FLOOR_GB`, default 1.5) instead of being
> SIGKILLed mid-construction.

#### Why SPLADE serves fine at 1M/15M but OOMs at 150M — and how this relates to hot/cold

This OOM appeared for the first time during the 2026-06-18 parity benchmark. It is
**not a behaviour change** — it is scale meeting a design property that was known
and documented, plus an eval access pattern that bypasses the hot/cold mitigation.

- **BMP has always loaded its shards fully into RAM.** It is a load-into-memory
  block-max engine (`bmp.Searcher` → `load_into_memory()`); there is no mmap mode.
  At 1M docs the SPLADE set is ~1 GB → ~3 GB resident, so it fits trivially — every
  prior parity run used `data/thinclient_1m`. 150M was the **first query-time load
  of the full 68.6 GB BMP set** (→ ~211 GB). Scale exposed a latent cost; nothing
  regressed.

- **Was the "Rust engine = more RAM-efficient" design intent wrong? No — but it was
  scoped to the laptop/subset tier and to the *other* legs.** Per
  `THIN_CLIENT_STACK_RESEARCH.md`, the thin-client was recommendation #2 — the
  **laptop tier (~15M docs)** replacement for OpenSearch's JVM (measured tantivy
  BM25F **45 MB RSS** vs a 2 GB JVM heap; tens-of-MB idle; mmap dense via usearch
  `view()`). The sparse leg was **always expected to be RAM-resident** (Seismic
  "index is fully RAM-resident, ~8 GB / 9M docs"; the same doc flags Qdrant's
  "sparse must be on_disk at 150M, **~300 GB in-RAM otherwise**"). At the 15M tier
  the sparse leg is ~9–13 GB — laptop-feasible. So "more RAM-efficient" was true for
  BM25F + dense + cold-start at the *designed tier*; it never promised a disk-backed
  sparse leg at 150M. The 150M thin-client (this testbed) pushes the laptop stack
  past its tier. In the deployment plan, **Full/150M was always the *server*
  (OpenSearch) tier**; thin-client was the ~10–15M / 20–40 GB laptop/subset tier.

- **Why OpenSearch never hit this.** OpenSearch served SPLADE as Lucene
  `rank_features` (impact-quantized postings) stored on disk and **mmap'd
  (`MMapDirectory`, page-cache-resident)**, with a bounded JVM heap. It never makes
  the whole sparse index resident — cold postings stay on disk and the OS reclaims
  them under pressure. BMP traded exactly that away for a fully-resident structure
  (faster per query, instant in-RAM block-max), so **at 150M the BMP sparse leg is a
  RAM *regression* vs the OpenSearch leg it replaced** — on the one axis OpenSearch
  handled well. BM25F (tantivy) and dense (usearch) kept the mmap property and have
  no such problem.

- **Is it just the benchmark, or real use? Both — at different thresholds.** The
  ~211 GB figure *is* a benchmark/access-pattern artifact: full-corpus parity forces
  **all 10 sections live at once**, which is precisely what the hot/cold persona
  system exists to avoid. Designed serving keeps **one persona's hot section live**
  (manifest `hot_sections = ['social_sciences__recent']`) and the other 9 **packed**
  (`packer.py`); `_load()` only loads section dirs physically present under
  `sections/`, so production normally loads one section, not ten. **But the
  underlying ceiling is real, not an edge case:** even a single hot `__recent`
  section at 150M is ~13 GB on disk → **~40 GB resident**, still over a 24 GB host.
  So:
  - **Local/laptop:** fine at the intended ~15M tier (hot section is a few GB);
    **not viable at 150M** with BMP as-is (one hot section already busts a laptop).
  - **Server-to-client:** the 150M tier. One-persona hot/cold fits a ~64–128 GB
    server; serving many personas hot at once (or all sections, like the eval)
    trends toward ~211 GB and is unreasonable — that needs the mmap fix.
  - **Gotcha:** the index currently ships **all 10 sections unpacked** (a build/eval
    state), so pointing the live server at it as-is would OOM identically until the
    cold sections are packed away. And a query routed to a *cold* section triggers an
    on-demand unpack→load of that section's BMP (~tens of GB) — so even correct
    hot/cold can spike on an off-hot query on a small host.

  **Fair summary:** it is an edge-case interaction between an all-sections access
  pattern and the hot/cold persona system *layered on top of* a structural
  per-section RAM cost. The eval triggers the extreme; the structural cost (BMP
  resident, no mmap) is what makes even normal single-persona serving strain small
  hosts at 150M. The durable fix is a mmap-backed / disk-resident sparse engine
  (BMP fork with mmap, Seismic/PISA, or SPLADE impacts as tantivy payloads — same
  page-cache property OpenSearch had); hot/cold is a partial mitigation, not a cure.

#### Off-BMP migration → Qdrant on_disk sparse (30M GATE PASSED on RAM+latency; NDCG parity still unmeasured — 2026-06-19)

The chosen durable fix for the BMP RAM wall is **Qdrant `on_disk=true` sparse**
(no JVM, page-cache resident like the old OpenSearch leg). Migration is **GPU-free**:
`data/thinclient_index/spool_backup/` (123 GB, 5 sections) already holds the
precomputed SPLADE `sparse_field`, so we re-ingest those vectors rather than
re-encoding. Ingest path: `scripts/spike_qdrant_sparse.py build-spool-par`
(8 workers, gRPC, `wait=True` bounded in-flight) into a collection whose sparse
index is `on_disk`. Token space: every encoder token → its `bert-base-uncased`
vocab id (shared doc/query u32 space).

- **Resumability (commit 19bb44e):** per-worker checkpoints under
  `data/qdrant_spike/ckpt/<collection>/` (slice/line/id-offset/done), fsync'd per
  batch; deterministic point IDs (`id_base+local`) make replay idempotent, so a
  container restart resumes mid-slice. Auto-resume on checkpoint presence; a fresh
  start drops the collection + stale checkpoints. `_wait_qdrant()` decouples
  start ordering from the server.
- **Operational:** runs under supervisord (`scripts/qdrant-offbmp.supervisor.conf`,
  programs `qdrant-spike` + `qdrant-ingest-30m`); Qdrant storage
  `data/qdrant_spike/storage`, REST 6333 / gRPC 6334.
- **Spike signal (validation):** `on_disk` sparse RAM is flat/sub-linear (544 MB
  at-rest @ 5M vs BMP ~7 GB); bounded parallel ingest holds <10 GB RSS. See `a41cd6d`.
- **30M gate — MEASURED 2026-06-19** (collection `splade_par30`, 30,000,000 docs,
  `status=green`, 16 segments; eval `scripts/spike_qdrant_sparse.py measure`,
  results `data/qdrant_spike/measure_par30.log` + `logs/qdrant_ingest_30m.log`):
  - **RAM (the thesis): 1,441 MB resident at rest** for 30M on_disk sparse — vs BMP,
    whose load-into-RAM 3.07× would put 30M at tens of GB resident. Flat, page-cache
    backed. Querying warms page cache to ~5.3 GB RSS (reclaimable, not allocation).
  - **Search latency, 40 eval queries (Qdrant search time only, encode excluded):**
    warm steady-state **p50 ~66 ms / p95 ~85 ms / max ~95 ms** (two passes:
    67.4/83.4/97.7 and 65.6/88.6/93.4). Cold first-touch pass was p50 208 / p95 1333 /
    max 1502 ms — i.e. the tail is page-cache warm-up, not steady cost.
  - **Ingest throughput: 4,234 docs/s end-to-end** (`wait=True`, 8 workers), 30M in
    7,086 s; peak ingest RSS 11.4 GB. Extrapolates to full ~150M ≈ **~9.8 h**.
  - **NOT measured: NDCG quality parity.** `measure` reports latency+RSS only. The
    30M subset is not directly comparable to the 150M **NDCG@10 0.561** baseline
    (§ BUILD COMPLETE). Low *a-priori* risk: Qdrant stores **f32** sparse values
    while BMP 8-bit-quantizes (weights ×70), so on_disk ranking should be ≥ BMP on
    quality — but this remains **unmeasured / design only** until a full-scale (or
    matched-qrel) retrieval eval is run. Do not claim NDCG parity yet.
- **Verdict:** gate **PASSES on RAM + throughput + latency**; GO/NO-GO for the full
  ~9.8 h 150M ingest is pending user sign-off, with NDCG parity to be measured at
  full scale. Phases 5–7 (wire retriever sparse leg to Qdrant, decommission BMP /
  reclaim 68.6 GB, doc sync) are post-GO.

Per-section pack ratio (`manifest.json`, live/packed) ranged 1.86–2.01×,
aggregate **1.875×** — **below the 2.10× bsize-32 assumption**, confirming the
`STORAGE_BUDGET_150M.md` §1 "*denser b256 shards will pack slightly less
(est.)*" note. Pack phase 01:33→02:59 (~86 min, 9 cold sub-sections); dense leg
36 s.

Resolved "to-be-confirmed" items: the 3-worker / `--bmp-shard-docs 1_000_000`
build reached DONE without the OOM/swap-thrash recurring (the failure mode the
per-slice + intra-slice `wal_checkpoint(TRUNCATE)` fixes targeted), so the
bounded-WAL fix held at 150M scale. NOTE: the post-build meta-merge of `other`
(54.9M docs) is a single bulk `INSERT OR REPLACE` (NOT the bounded per-slice
path) and produced a transient ~24 GB `meta.sqlite-wal` that checkpoint-drained
into `meta.sqlite` cleanly; final `-wal`/`-shm` truncated to 0.

Post-build housekeeping: remove the supervisord conf (build done); the
`data/thinclient_index/spool_backup/` build intermediate (**131 GB**) is
reclaimable once the index is validated. **Retrieval efficacy (NDCG@10 /
latency / per-leg parity) at full 150M scale is UNMEASURED** until
`scripts/eval_thinclient_parity.py` runs against this index (the only prior
parity result, `data/eval_results/thinclient_parity_20260611_0646.json`,
predates this full build) — see Validation below.

Validation: `scripts/eval_thinclient_parity.py` (per-leg overlap vs baseline,
LLM-judged NDCG@10, latency). **At 150M, run it OOM-safe** via
`scripts/run_parity_safe.sh` — the legacy single-process path loaded BOTH
retrievers and interleaved `tc.search`/`baseline.search` per query, keeping the
thin-client and the OpenSearch 150M cluster hot at once and
OOM-killing on the 24 GB host. The eval is now split into one-engine-per-process
subcommands joined offline: `record-tc` (thin-client only, `SFU_DENSE_WARMCACHE=0`)
→ `record-os` (OpenSearch only, thin-client process already exited) → `compare`
(pure offline join, identical summary schema). `ThinClientRetriever` has no
`close()`, so only a fresh process frees the mmap set; the orchestrator sequences
the phases as separate processes with page-cache drops + `MemAvailable` preflight
between them, an in-loop mem-floor guard that flushes a partial record and exits
cleanly before the OOM-killer fires (the killer left "no traceback" above), and
optional `PAUSE_OS=1` to `docker pause` OpenSearch during phase A. The split caps
peak RAM at one engine rather than the sum — **but that is still not enough at
150M: one engine (the thin-client SPLADE leg) is ~211 GB resident (see the BMP
note above), so `record-tc` OOMs on the 24 GB host before it serves a single
query** (verified 2026-06-17: `record-tc` SIGKILLed mid-`_load()` at ~32 GB
committed; the host `.wslconfig` 24 GB cap contained the kill — only the process
died, Docker + OpenSearch survived). Holding the whole SPLADE engine resident in
one process would need ≈232 GB — but that is **not a serving recommendation**: a
search DB should not keep the index in RAM, and this stack's other legs (tantivy
BM25F, `meta.sqlite`, dense) already serve from mmap at ~0 resident. The 211 GB is
the artifact of BMP 0.2.6 being load-into-memory only, taken across all sections
at once. The real fixes are an mmap-backed SPLADE engine (drops serving RAM to the
hot working set) or hot/cold residency (serve the live section only); see the
serving-RAM box in `STORAGE_BUDGET_150M.md`. The wave eval below sidesteps the
issue entirely for benchmarking.

**150M parity numbers on the 24 GB host: use `scripts/eval_parity_section_waves.py
run`** (added 2026-06-17). It exploits the fact that the legs already merge across
sections/shards by plain score-concatenation: BM25F runs on tantivy (mmap, ~0
resident) in one pass via `SFU_SKIP_BMP=1`, and the SPLADE/BMP leg is recorded
**shard by shard, each wave in a fresh process** (the only way to free BMP RAM),
then merged offline. Because BMP scores are corpus-independent dot products, the
shard-partition merge reproduces the full-corpus top-K **exactly** — validated
2026-06-17 on `data/thinclient_1m`: wave output is byte-for-byte identical to a
direct full `record-tc` on both legs, all queries. Emits a record-tc-schema JSON
consumed unchanged by `eval_thinclient_parity.py compare`. Time-not-RAM bound:
it reads the 68.6 GB BMP set once across ~30 waves (`--resident-budget-gb`,
default 8). The legacy single-process `record-tc` stays **BLOCKED on host RAM**.

**MEASURED 150M parity (2026-06-18, first full-corpus run).** 40 diverse queries,
LLM-judged NDCG@10, thin-client (this index) vs the 150M OpenSearch baseline.
Thin-client recorded via section-shard-waves (27 waves, 8 GB budget, ~43 min,
min RAM 15.4 GB, peak swap 2.1 GB, no OOM); OpenSearch via `record-os`; joined by
`compare`. **On the common judged set (34 queries with a judged doc in both),
thin-client RRF NDCG@10 = 0.561 vs OpenSearch 0.449 (Δ +0.112; thin-client wins
29/34 queries, loses 5).** Overlap@50 vs OpenSearch baseline: bm25f 0.532, splade
0.399, rrf 0.476 (the thin-client uses tantivy BM25F + BMP SPLADE, not OpenSearch's
analyzers, so per-leg sets differ while final RRF quality is higher). Latency is
NOT comparable across these runs (different hardware; tc splade latency is
reconstructed encode+search, hydration excluded). Records:
`data/eval_results/thinclient_parity_waves_150m.json` (summary),
`parity_record_tc_waves_150m.json` (tc), `parity_record_os_20260618_0028.json`
(os). Eval: `scripts/eval_parity_section_waves.py` + `eval_thinclient_parity.py
compare`. Caveat: NDCG coverage is judge-cache-limited (34/40 queries scored).
The base harness is **IMPLEMENTED 2026-06-17, smoke-tested on `data/thinclient_1m`**
(compare output schema-identical to `thinclient_parity_20260616_0536.json`).
Permanent pytest
suite
`scripts/tests/test_thinclient_stack.py` (13 tests: meta v1/v2, era routing +
pruning, abstracts v1/v2/v3, dense cache, metrics/reload, MCP index tools).
Storage research: `docs/LEXICAL_STORAGE_RESEARCH.md`,
`docs/STORAGE_BUDGET_150M.md`, `scripts/eval_text_compression.py`.

## Removed from this container

Training (SPLADE finetune, embedding/CE/LambdaMART trainers, miners,
generators), OpenSearch indexing pipeline (splade_indexer, opensearch_sync,
reindex shells, watchdogs), cloud GPU offload, docker/opensearch templates,
the devcontainer OpenSearch service+volume, training data and superseded
checkpoints (embed v2/v3, sfu-splade-v1). Kept: all eval/benchmark harnesses,
dense ANN code + artifacts, snapshot downloader + 304-part snapshot, serving
path, query-side encoders.

Host-side cleanup after container recreate:
`docker rm -f claudebox-sfu-library-thinclient-opensearch &&
docker volume rm claudebox-sfu-library-thinclient-opensearch`
