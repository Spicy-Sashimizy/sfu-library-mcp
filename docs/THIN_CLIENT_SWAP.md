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
| **Serving total** | **≈ 104 GB** (97 GiB) | vs ≈ 95 GB estimate; the +9 GB is entirely `meta.sqlite` |

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
thin-client (~104 GB mmap) and the OpenSearch 150M cluster hot at once and
OOM-killing on the 31 GB host. The eval is now split into one-engine-per-process
subcommands joined offline: `record-tc` (thin-client only, `SFU_DENSE_WARMCACHE=0`)
→ `record-os` (OpenSearch only, thin-client process already exited) → `compare`
(pure offline join, identical summary schema). `ThinClientRetriever` has no
`close()`, so only a fresh process frees the mmap set; the orchestrator sequences
the phases as separate processes with page-cache drops + `MemAvailable` preflight
between them, an in-loop mem-floor guard that flushes a partial record and exits
cleanly before the OOM-killer fires (the killer left "no traceback" above), and
optional `PAUSE_OS=1` to `docker pause` OpenSearch during phase A. Peak RAM ≈ one
engine, never the sum. **IMPLEMENTED 2026-06-17, smoke-tested on `data/thinclient_1m`
(compare output schema-identical to `thinclient_parity_20260616_0536.json`); 150M
parity numbers still UNMEASURED until the full run completes.** Permanent pytest
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
