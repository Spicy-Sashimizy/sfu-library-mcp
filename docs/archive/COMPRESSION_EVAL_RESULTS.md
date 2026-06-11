# Index Compression Evaluation Results

**Date:** 2026-06-10
**Context:** Thin-client / localized deployment requires shrinking the local search stack
(`docs/LOCALIZED_DEPLOYMENT_PLAN.md`). Two measured evaluations + literature research:

- `scripts/eval_lexical_lossless.py` → `data/eval_results/lexical_lossless_eval.json`
- `scripts/eval_dense_compression.py` → `data/eval_results/dense_compression_eval.json`

---

## 1. Lexical leg (BM25F postings + SPLADE rank_features + stored _source) — LOSSLESS levers

Measured on a 1M-doc subset of the live `openalex_works` (150,413,098 docs / 274 GB,
zstd level 3). Sizes are live-segment bytes after `force_merge(1)`. **Parity** = 40 mixed
queries × {BM25F, SPLADE}, top-50 vs baseline: ALL variants returned identical ID sets and
identical score multisets (max Δscore 0.0000); only equal-score tie order ever differed.

| Variant | What it does | Saved | ~150M est. | Search lossless? | Incremental adds? |
|---|---|---|---|---|---|
| `force_merge` only | merge to 1 segment | ~4% | 263 GB | yes (tie order may shuffle) | yes — but new segments accumulate; re-merge periodically. Best for read-only bundles |
| `zstd6` | codec level 3→6 | 4.4% | 262 GB | yes | yes, fully transparent |
| `zstd_no_dict` | zstd without dict | **−1.1% (bigger)** | 277 GB | yes | yes — **not worth it** |
| `nosrc_splade` | `_source.excludes: sparse_field` (weights were stored TWICE: postings + JSON) | **36.9%** | **173 GB** | yes — weights stay in rank_features postings | yes for adds. ⚠️ future `_reindex` from this index can no longer rebuild sparse_field (re-encode or keep snapshot parts). Check `_recovery_source` purged after merge |
| `extern_display` | also exclude `abstract` from `_source`; serve display text from external store | **59.3%** | **112 GB** | yes for ranking; abstract no longer retrievable from the index (needs sidecar store) | yes; sidecar must be updated in the same pipeline |
| `freqs` | `index_options: freqs` on title/abstract/concepts (drop positions) | 8.6% | 250 GB | yes for ALL current query shapes (BM25F multi_match + rank_feature use no phrase/span) | yes. Forecloses future phrase queries unless reindexed |
| `noid` | stop indexing the dynamically-mapped duplicate `id` text+keyword field | 1.0% | 271 GB | yes (`openalex_id` is canonical) | yes |
| **`combined`** (zstd6 + nosrc_splade + freqs + noid) | | **48.0%** | **142.5 GB** | **yes — verified** | yes (with the nosrc reindex caveat) |
| combined + extern_display | | ~62% (computed) | **~105 GB** | yes + sidecar | yes + sidecar |

All mapping/_source changes require ONE full reindex (~13 h at measured 3,137 docs/s
pipeline throughput; pure server-side `_reindex` is faster since sparse_field is in
`_source` today — that is exactly the double-storage being removed).

### Round 2 — researched levers, now MEASURED (12-variant run, 2026-06-10)

| Method | Measured | Parity | Verdict |
|---|---|---|---|
| Index sorting (`index.sort: publication_year` desc) | **0.4%** | ✅ 80/80 lossless | not worth the reindex on this corpus |
| doc_values off (year/type/is_oa) | **−0.0%** | ✅ 80/80 lossless | negligible; skip |
| 2 shards vs 1 (live index has 2) | 1→2 costs **0.7%**; single-shard saves ~0.7% | ❌ **40/80 score-multiset** — per-shard IDF shifts BM25 scores, confirmed empirically | small win but NOT strictly lossless; re-validate evals if done |
| `combined2` (combined + sortyear + dvoff) | **48.4%** → **141.3 GB** | ✅ 79/80 exact, 80/80 scores | new lossless ceiling, +0.4pp over `combined` |

### Top-3 lossless levers by measured savings (single methods)

1. **`_source.excludes: sparse_field` — 35.0%** (274 → 178 GB). The SPLADE weights
   were stored twice; postings keep serving search identically.
2. **`index_options: freqs` (drop text positions) — 8.6%** (→ 250 GB). Lossless for
   every query shape the stack issues (no phrase/span queries anywhere).
3. **zstd level 6 — 4.4%** (→ 262 GB). One settings change. (force_merge to 1 segment
   is worth a similar ~4% and stacks with everything.)

(`extern_display` at **58.7%** beats them all but is "lossless + sidecar": ranking
identical, abstracts served from an external store — rank it #1 if the sidecar is
acceptable.) All stacked = `combined2` **48.4%**, or ~62% with externalized display.

### Still-untested lossless (research-only)

| Method | Expected | Notes |
|---|---|---|
| **BP doc-ID reordering** (`BPIndexReorderer`, Lucene 9.8+ misc) | 1.5–10% of postings | Offline Lucene tool on the read-only artifact; adds land unordered until re-run. Expert effort |
| Shipping: `_flush` + `tar \| zstd --long` | 5–15% transport-only | Snapshot `compress:true` is metadata-only — not a lever |
| Ruled out | — | norms removal (changes BM25), `match_only_text` (constant TF), alternative postings formats (Bloom/Direct/FST: bigger or RAM-bound, no back-compat), `_field_names`, compound format, FM-index/succinct self-indexes (substring search ≠ ranked retrieval; postings can't be FM-indexed; zstd already beats it on stored text) |

---

## 2. Dense ANN HNSW leg — quantization (minimal-loss levers)

Same 600k POC corpus (`openalex_works_dense`, 384-dim fp32 sfu-academic-embed-v5), same
360 paraphrase queries + LLM-judge ground truth as `eval_dense_poc.py`. faiss HNSW
(M=16, efC=256) mirrors the lucene config; `+rescore` = 4× oversampled candidates
re-ranked with fp32 from disk (≈ OpenSearch 2.17 disk-based mode / BBQ).

| Variant | B/vec | × | 150M vectors | R@10 vs exact | 3-leg ΔNDCG@10 | Verdict |
|---|---|---|---|---|---|---|
| fp32 (today) | 1536 | 1× | 231 GB | 0.994 | baseline | |
| fp16 | 768 | 2× | 116 GB | 0.994 | +0.0000 | free |
| **int8** | 384 | 4× | **58 GB** | 0.991 | **+0.0000** | **free — adopt now** (lucene `sq`) |
| int4+rescore | 192 | 8× | 29 GB | 0.997 | −0.0001 | free with rescore |
| pq48+rescore | 48 | 32× | 7.2 GB | 0.985 | −0.0018 | OK; dominated by binary |
| **binary+rescore** | 48 | 32× | **7.2 GB** | 0.996 | **+0.0008** | **holds at 32× — the 150M-scale answer** (OS 2.17+ `mode: on_disk`) |
| binary (no rescore) | 48 | 32× | 7.2 GB | 0.732 | −0.0064 | rescore is mandatory |
| trunc 384→192 | 768 | 2× | 116 GB | 0.846 | −0.0032 | **bad** — v5 is not MRL-trained; quantize, don't truncate |

HNSW graph overhead (M=16 ≈ 128 B/vec ≈ 19 GB at 150M) is NOT reduced by vector
quantization. Rescore variants keep fp32 (or fp16) vectors on disk, not in RAM.

Dense-leg-only NDCG@10 (overall): fp32 0.7117 / int8 0.7122 / binary+rescore 0.7121.

### Headline for the localized deployment plan

- Lexical: 274 GB → **142.5 GB lossless** (→ ~105 GB with externalized display text).
- Dense at 150M: 250 GB naive fp32+graph → **~26 GB RAM-resident** (binary codes + graph)
  with fp32/fp16 rescore vectors on disk — or 58 GB flat with plain int8.
- These multiply with doc-count subsetting (§4 of LOCALIZED_DEPLOYMENT_PLAN.md): a 15M-doc
  laptop tier lands ≈ **14 GB lexical + 0.7 GB dense codes**.
