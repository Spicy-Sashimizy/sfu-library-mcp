# Whole-DB Storage Budget @150.4M docs — measured levers + hypothetical totals

**Date:** 2026-06-11 · **Updated:** 2026-06-16 (§1 now carries the AS-BUILT 150M
footprint; the completed build supersedes the earlier `meta.sqlite` estimate) ·
**Status:** all numbers below are MEASURED unless marked *est.* · Sources: `scripts/eval_storage_levers.py` →
`data/eval_results/storage_levers_eval.json`, `scripts/eval_text_compression.py`
→ `data/eval_results/text_compression_eval.json`, component sizes from the 1M
validation build (`data/thinclient_1m`), full method detail in
`LEXICAL_STORAGE_RESEARCH.md` §8–9.

## 1. Thin-client DB — as built vs all levers applied (all sections live)

| Component | As built | Lever | After | Notes |
|---|---|---|---|---|
| BMP SPLADE shards | 167.6 GB | bsize 256 + clustered (section, top-SPLADE-term) insertion | **76.7 GB** | −54.2% at recall@50 0.967 vs 0.966; also −61% query latency |
| tantivy BM25F | 44.5 GB | clustered ordering tested: **+0.7% — null result, not adopted** | 44.8 GB | tantivy already delta/bit-packs postings; ordering didn't help |
| Abstract sidecar | 64.7 GB | clustered 32 KB blocks + per-(section×language) zstd-19 dicts (3.42×) | **52.9 GB** | bit-exact; 100-doc rerank fetch ≪ 300 ms budget |
| meta.sqlite | 24.1 GB | numeric PK (W-id digits) + dict-zstd titles/DOIs + enum-int type/section (1.679×) | **14.4 GB** | round-trip verified 200/200 |
| **Total (all live, all hot)** | **300.9 GB** | | **188.8 GB** | **−112.1 GB (−37.3%), all lossless** |

**Implementation status (2026-06-11 evening):** the three winning levers are now
IN the builder, not just measured — BMP b256+clustered (`af09836`), abstracts v3
script-bucketed 32 KB blocks + per-script dicts (`d978ba3`), meta v2 INTEGER-PK
schema (`9a6d195`; implemented v2 measured **~10 GB** at 150M, better than the
14.4 GB lever estimate above — **SUPERSEDED 2026-06-16: the completed 150M
build's `meta.sqlite` measured 24.9 GB, not ~10 GB; the early figure did not
hold at full build, cause not yet diagnosed — see "AS BUILT" below and
`THIN_CLIENT_SWAP.md` §BUILD COMPLETE**). Sections have since also split into
era sub-sections (`7fe689a`), which changes layout but not totals.

**Persona steady state — AS BUILT (measured 2026-06-16, supersedes the estimate
below):** political_science (hot `social_sciences__recent` live + 9 cold
sub-sections packed): **≈ 104 GB** on disk — hot live 34.0 GB (incl. 26.3M
abstracts), packed cold 45.0 GB (live 84.3 GB → **1.875×** aggregate, range
1.86–2.01×), `meta.sqlite` **24.9 GB**, dense 0.36 GB. Excludes the reclaimable
131 GB `spool_backup/` build intermediate. The 1.875× ratio confirms the est.
below that b256 packs under the 2.10× bsize-32 figure; the +9 GB vs the ≈ 95 GB
estimate is entirely `meta.sqlite` (24.9 GB built vs 14 GB assumed). Source:
`data/thinclient_index/manifest.json`, `logs/migration_150m.log` (DONE
2026-06-16T03:00:21).

Persona steady state (ESTIMATE 2026-06-11, superseded above): hot live ≈ 34 GB
+ packed cold ≈ 47 GB (2.10× est.) + meta 14 GB = **≈ 95 GB**. *Pack ratio was
measured on bsize-32 artifacts; denser b256 shards will pack slightly less
(est.).*

Old-era levers already baked into this design (do NOT double-count): SPLADE
stored once (old `nosrc_splade`, −36.9%), positions off (old `freqs`, −8.6%),
no dup id field (−1.0%).

## 2. BMP matrix (100k real docs, production SPLADE vectors, alpha=0.8)

| variant | B/doc | vs b32/corpus | ms/q | recall@50 vs exact |
|---|---|---|---|---|
| b32 corpus (paper config, first build) | 1404.8 | — | 27.2 | 0.966 |
| b32 clustered | 1160.1 | −17.4% | 13.3 | 0.960 |
| b64 clustered | 961.9 | −31.5% | 14.1 | 0.962 |
| b128 clustered | 789.5 | −43.8% | 13.6 | 0.966 |
| **b256 clustered (ADOPTED)** | **642.9** | **−54.2%** | **10.7** | **0.967** |
| b32 without compress_range | 1681.1 | +19.7% | 27.1 | 0.966 |

### BMP 0.2.6 quirks (measured; hardened in `lib/thinclient/builder.py`)
- **u8 impact saturation at 255**: quantization scale 1000 collapsed recall to
  0.281. Adopted scale 70 (saturation-free for SPLADE log1p weights ≤ 3.64),
  explicit clamp; recall 0.968–0.970, size unchanged. Score scale is a global
  multiplier — index/query scale mismatch does not change ranking.
- **Heavy score-tie queries panic** (Rust unwrap, `search.rs:131`): zero
  occurrences across all real-data runs; retriever guards per shard.
- **Indexes < ~500 docs return empty** (between 150 and 500 with
  compress_range): builder absorbs tail shards < 5,000 docs into the previous
  shard and warns if a whole section is sub-minimum.
- **All-query-terms-absent-from-shard panics**: per-shard `*.vocab.zst`
  sidecars + skip (since first build).

## 3. Hypothetical: same tech applied to the MAINLINE Lucene DB

Measured head-to-head on the same 20k abstracts: mainline-style storage
(16 KB zstd-6 blocks, no trained dict ≈ Lucene stored-fields format) =
**546.0 B/doc**; the recommended sidecar config = **351.8 B/doc** → the new
tech beats the mainline's existing optimization by **35.6%**.

| Mainline state | @150M | Degradation |
|---|---|---|
| Original live index | 274 GB | — |
| Existing lossless config (`combined2`) | 141.3 GB | none (80/80 score parity, measured 2026-06-10) |
| + abstracts externalized into the recommended sidecar | **≈ 112 GB** | **zero ranking change** (extern_display variant measured score-identical; sidecar bit-exact). Operational: sidecar joins the update pipeline; `_reindex` can no longer rebuild abstracts from the index alone |
| + BP doc-ID reordering (only untested lever left) | ≈ 106–111 GB *est.* | lossless (equal-score tie order only) |

Mainline waterfall: **274 → 141.3 → ~112 GB (−59% total), fully lossless.**
The thin-client testbed pays ~77 GB more disk than optimized mainline at 150M
(BMP's block-max speed structures) in exchange for the no-JVM/laptop
properties; at the 15M deployment tier the same build scales to ~19 GB
all-live / ~10 GB persona steady.

## 4. Degradation accounting (why "lossless" is justified)

- Stored text: SHA-256 bit-exact round trips (abstracts, titles, DOIs).
- IDs: exact integer encodings (delta-varint 2.66 B/id, 4.5× vs raw).
- BMP bsize/ordering: changes storage layout, not stored scores; recall@50
  0.967 vs 0.966 baseline at the production alpha=0.8.
- Doc reordering: scores order-invariant; only equal-score tie order shuffles
  (same accepted caveat as force_merge in the original evals).
- Pre-existing (independent of these levers, previously validated): BMP 8-bit
  impacts; dense binary+rescore ΔNDCG@10 +0.0008.
- REJECTED lossy options, for the record: truncate-128tok (4.34×, cosine
  0.955) and stopword-drop (2.25×, cosine 0.968) both fail the 0.98
  embedding-cosine gate; SLM rank-decode (gpt2) fails the 300 ms rerank
  budget by 1,411× AND same-machine bit-exactness.

## 5. Reproduce

```bash
.venv/bin/python3 scripts/eval_storage_levers.py        # BMP matrix + meta + pack + mainline-sim
.venv/bin/python3 scripts/eval_text_compression.py      # sidecar variants + languages [--slm]
```
