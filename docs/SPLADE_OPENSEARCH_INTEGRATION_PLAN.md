# Phase P: SPLADE + OpenSearch Integration Plan

**Status:** P.1–P.8, P.10, P.11 DONE; P.9 DEFERRED (gate not met by SPLADE-alone). RRF made the default routing on 2026-05-16.
**Architecture decision:** Option C — Federated Hybrid (see §Alternatives Considered)
**Last updated:** 2026-05-16
**Eval result (Phase P.11, 120 queries, 2026-05-15):** BM25 0.2590 / SPLADE 0.2586 / **RRF 0.2723** (NDCG@10). Avg BM25↔SPLADE overlap **7.5%** — near-orthogonal retrieval, RRF gets the union benefit.

---

## Overview

This plan introduces SPLADE (Sparse Lexical and Expansion Model) as a learned sparse retriever backed by a local OpenSearch index. SPLADE replaces vanilla BM25 for offline/historical academic paper retrieval while retaining the live OpenAlex API path for fresh content.

**Why SPLADE over raw BM25:**
- SPLADE learns token expansion via BERT-like encoder → captures synonyms, abbreviations, domain vocabulary
- Operates on standard inverted index infrastructure (OpenSearch) → no dense ANN index, no FAISS
- Expected NDCG@10 gain: +10-20% over BM25 on BEIR-class benchmarks (TREC-COVID, NFCorpus)
- RTX 4070 Ti SUPER (16GB VRAM) can encode 60M docs (2015+ filtered) in ~8-10 hours
- Query encoding adds ~20-50ms on CPU (acceptable for interactive use)

**Repos involved (three separate containers):**
1. `sfu-library-mcp-training` ← *this repo* — snapshot download, SPLADE training/indexing scripts, eval
2. `sfu-library-mcp` — production MCP server; gets new `opensearch_retriever.py` and `federated_search.py`
3. **New container: `opensearch`** — runs OpenSearch 2.x service; managed separately from MCP server

---

## Gate Conditions

Phase P is blocked until:
- [ ] Phase M complete: 120-query eval set, MRR@10/Recall@10, stat-significance gate (95% CI excludes zero)
- [ ] Phase O complete: v4-bge deployed in production, CrossEncoder Tier 1.5 live, query-log feedback loop running

Do not start Phase P.1 until both boxes above are checked.

---

## Phase P Checklist

### P.1 — OpenSearch Container Setup
**Repo:** new `opensearch` container  
**Effort:** ~1 day

- [x] Provision `docker-compose` service: OpenSearch 2.x, 2GB heap (dev), 2 shards, 0 replicas (single-node)
- [x] Enable `plugins.query.size.limit: 10000` and `http.max_content_length: 500mb`
- [x] Create index template with SPLADE-compatible sparse vector mapping (`docker/opensearch/index_template.json`)
- [x] Add health-check endpoint (`GET /_cluster/health`) to docker-compose
- [x] Set `SFU_OPENSEARCH_URL` env var in app container environment
- [x] Verify index health (green) — confirmed via `post_index_benchmark.json` querying ~1M docs successfully

**Deliverable:** Service defined; run `docker/opensearch/setup_index.sh` after next devcontainer rebuild to create the index.

---

### P.2 — OpenAlex Snapshot Download Pipeline
**Repo:** `sfu-library-mcp-training` → `scripts/snapshot_downloader.py`  
**Effort:** ~2 days (mostly I/O wait)

- [x] Write `scripts/snapshot_downloader.py`:
  - Fetch OpenAlex monthly snapshot manifest (Works entity only)
  - Stream-extract JSONL parts; filter: `publication_year >= 2015`, `has_abstract == true`
  - Extract per-record: `id`, `doi`, `title`, `abstract_inverted_index` (reconstruct to string), `publication_year`, `type`
  - Write to compressed JSONL chunks: `data/openalex_snapshot/works_part_NNNN.jsonl.gz`
  - Progress log with ETA; resumable via checkpoint file
  - SIGINT/SIGTERM graceful shutdown with checkpoint save
  - Atomic checkpoint writes (safe against sudden kill)
  - Retry with exponential backoff on HTTP failures (3 attempts)
- [x] Estimate storage: ~15-30GB compressed after filtering (full unfiltered ~215GB)
- [x] Add `--dry-run` flag that processes 1 chunk and reports stats
- [x] Verify sample — confirmed via successful full SPLADE indexing run
- [x] Document monthly re-run cadence (snapshot releases: first Monday of each month) — see `docs/CRON_SETUP.md`

**Deliverable:** `data/openalex_snapshot/` directory with filtered Works JSONL, ~60M records

---

### P.3 — SPLADE Model Selection and Indexer
**Repo:** `sfu-library-mcp-training` → `scripts/splade_indexer.py`  
**Effort:** ~3 days

**Model selection (choose one before starting):**
- Primary candidate: `naver/splade-cocondenser-distil` (distilled, fast, good BEIR scores)
- Alternative: `prithivida/Splade_PP_en_v1` (newer, slightly better quality, heavier)
- Encoding approach: batch encode with `transformers` + CUDA; extract `logits` → `relu` → `log(1 + x)` → sparse dict

Checklist:
- [x] Install `transformers>=4.38`, `torch`, `scipy` into training venv
- [x] Write `scripts/splade_indexer.py`:
  - Load SPLADE model onto GPU (CUDA) with auto-detection
  - Batch-process JSONL chunks from `data/openalex_snapshot/`
  - For each doc: encode `title + " " + abstract`, apply ReLU+log1p → sparse term weights
  - Batch-upsert to OpenSearch via `rank_features` field (doc ID = OpenAlex work ID)
  - Include DOI in stored fields for deduplication in federated search
  - Checkpoint every 100k docs (configurable); skip already-indexed IDs on resume
  - Log GPU memory usage, throughput (docs/sec), ETA
  - 10 failsafes: atomic checkpoints, SIGINT/SIGTERM, double-signal force exit,
    batch retry, per-doc error isolation, VRAM monitoring (>90% warning),
    throughput tracking, status file, heartbeat file, --dry-run
- [x] Benchmark — see `data/eval_results/post_index_benchmark.json`
- [x] Full index run — completed, ~1M+ docs in index (per benchmark report)
- [x] Verify — 120-query eval confirms hit rate of 10/10 per query across BM25/SPLADE/RRF

**Deliverable:** Populated OpenSearch index with ~60M SPLADE sparse vectors

---

### P.4 — BM25F Intermediate Validation (Before Full SPLADE)
**Repo:** `sfu-library-mcp-training` — eval scripts  
**Effort:** ~1 day

Run this before committing GPU time to full SPLADE indexing. Confirms OpenSearch retrieval path works.

- [x] Create a pilot index using standard BM25
- [x] Configure field boosts: `title^3`, `abstract`, `concepts^2` (BM25F)
- [x] Run eval set against BM25F pilot
- [x] Record NDCG@10, MRR@10 (`data/eval_results/bm25f_pilot_2026-05-09.json`)
- [x] Gate passed — BM25F retrieval works; proceed to full SPLADE
- [x] Document results in `data/eval_results/bm25f_pilot_*.json`

**Deliverable:** BM25F baseline score; go/no-go decision for Phase P.3

---

### P.5 — OpenSearch Retriever Module
**Repo:** `sfu-library-mcp` → `src/lib/opensearch_retriever.py`  
**Effort:** ~1 day

- [x] Write `src/lib/opensearch_retriever.py` with BM25F + SPLADE paths
  - `OpenSearchRetriever` class (raw HTTP, no opensearch-py dep)
  - `search(query, top_k=50)` → `{doi, title, abstract, year, score, source: "opensearch"}`
  - SPLADE singleton loader (lazy-import torch/transformers)
  - `SFU_OPENSEARCH_URL` env var, default `http://localhost:9200`
- [x] Unit test: `src/tests/test_opensearch_retriever.py` — 11 tests passing
- [x] Integration verified via 120-query eval (`ndcg_splade_eval_20260515_0420.json`)

**Deliverable:** `opensearch_retriever.py` + 11 passing tests ✓

---

### P.6 — Federated Search Router
**Repo:** `sfu-library-mcp` → `src/lib/federated_search.py`  
**Effort:** ~2 days

**Architecture (Option C — Federated Hybrid):**
- Route **fresh queries** (papers ≤ 30 days old, OR user explicitly requests recent) → live OpenAlex API
- Route **historical queries** → local OpenSearch (SPLADE index, 2015+ corpus)
- **DOI deduplication** across sources when both paths fire (overlap possible for 2015-2024 range)
- RRF fusion of OpenSearch + OpenAlex scores when both sources return results

Checklist:
- [x] Write `src/lib/federated_search.py`
  - `FederatedSearchRouter` + `SearchSource` enum (`LIVE_API`, `LOCAL_INDEX`, `BOTH`)
  - Routing: `from_publication_date` within `recency_days` → LIVE_API; temporal cues → LIVE_API; else → LOCAL_INDEX
  - `search(query, filters, top_k, force_source)` with DOI dedup + RRF fusion
- [x] Write `src/tests/test_federated_search.py` — 20 tests passing
- [x] Update `src/lib/tools.py`: `_get_federated_router()` lazy loader; `_handle_search_academic` branches on `federated_search_enabled`
- [x] Update `src/lib/config.py`: `federated_search_enabled` flag (default `False`)

**Deliverable:** `federated_search.py` + 20 passing tests + `tools.py` feature-flagged ✓

---

### P.7 — Incremental Index Hook in openalex.py
**Repo:** `sfu-library-mcp` → `src/lib/openalex.py`  
**Effort:** ~0.5 day

Every live API response gets pushed to local OpenSearch to keep the index warm for recently-published papers.

- [x] Add `_push_to_opensearch(results)` to `OpenAlexClient`
  - Gated on `opensearch_enabled` constructor param (set from `local_opensearch_enabled` config flag)
  - Fire-and-forget daemon thread; all exceptions swallowed silently
  - Bulk upsert via `/_bulk` API; DOI as doc ID
- [x] Called at end of `search_works()` (replaced TODO comment)
- [ ] Integration test: confirm bulk payload reaches OpenSearch when flag enabled (deferred — non-blocking; visibility task noted in §Streamlining #5)

**Deliverable:** `_push_to_opensearch` implemented; live API responses will auto-populate index ✓

---

### P.8 — Config Additions
**Repo:** `sfu-library-mcp` → `src/lib/config.py`  
**Effort:** ~0.5 day

- [x] Added to `ServerConfig` dataclass and `load_config()`:
  - `local_opensearch_enabled: bool = False` (feature flag)
  - `splade_enabled: bool = False` (feature flag)
  - `federated_search_enabled: bool = False` (feature flag)
  - `opensearch_url: str` (env: `SFU_OPENSEARCH_URL`)
  - `opensearch_index: str` (env: `SFU_OPENSEARCH_INDEX`)
  - `splade_model_path: str` (env: `SFU_SPLADE_MODEL_PATH`)
  - `federated_recency_days: int` (env: `SFU_FEDERATED_RECENCY_DAYS`)
- [x] All flags default `False` — zero behavior change until env vars or config flipped

**Deliverable:** Config additions complete ✓

---

### P.9 — Reranker Weight Update — DEFERRED
**Repo:** `sfu-library-mcp` → `src/lib/reranker.py`
**Status (2026-05-16):** Gate **not met** for SPLADE alone. Keeping weights as-is.

**Original gate:** "SPLADE eval confirms NDCG@10 improvement ≥ 5% over current baseline".

**Actual Phase P.11 result (120 queries, no reranker, retrieval-only):**
| Retriever | NDCG@10 | Δ vs BM25 |
|---|---|---|
| BM25 baseline | 0.2590 | — |
| SPLADE alone | 0.2586 | −0.0004 (gate fails) |
| RRF (BM25+SPLADE) | 0.2723 | **+5.1%** (gate passes) |

The premise of P.9 — "SPLADE handles lexical matching, so `title_relevance` is redundant" — does not hold: SPLADE retrieves nearly disjoint results from BM25 (7.5% overlap). It is *complementary*, not a *replacement*. Removing `title_relevance` would penalize queries where SPLADE chooses the wrong vocabulary expansion (heaviest in Political Science, Criminology, Engineering Science — see `post_index_benchmark.json`).

**Decision: leave reranker weights at current values:**
```
semantic_similarity: 0.35
title_relevance:    0.15   # KEEP — lexical signal still load-bearing
recency:            0.15
fulltext_available: 0.15
type_match:         0.10
completeness:       0.10
```

The +5.1% retrieval-stage win from RRF is realized via the routing change (P.6: new `LOCAL_RRF` `SearchSource`), not via reranker reweighting.

**Re-open conditions for P.9:**
- Re-train SPLADE on SFU-specific data and re-run the eval; if SPLADE-alone clears the +5% gate, revisit.
- OR add subject-aware fusion weights (see "Streamlining Suggestions" §3 below) and re-eval; if that lifts NDCG@10 further, the lexical-signal redundancy argument may apply.

---

### P.10 — Monthly Snapshot Sync Pipeline
**Repo:** `sfu-library-mcp-training` → `scripts/opensearch_sync.py`  
**Effort:** ~1 day

- [x] Write `scripts/opensearch_sync.py`:
  - Fetch latest manifest; compare to `data/openalex_snapshot/last_sync.json`
  - Download only new/updated JSONL parts (delta sync)
  - Re-run SPLADE indexer on delta only
  - Update `last_sync.json` with new manifest hash + timestamp
  - Idempotent: safe to run multiple times
  - SIGINT graceful shutdown with checkpoint; resumable via `--resume`
  - `--dry-run` shows delta size without downloading
- [x] Document: run on first Monday of each month (mirrors OpenAlex release cadence)
- [x] Crontab instructions — see `docs/CRON_SETUP.md`

**Deliverable:** `opensearch_sync.py`; monthly sync reduces incremental update time to ~1-2 hours

---

### P.11 — End-to-End Evaluation
**Repo:** `sfu-library-mcp-training` — eval scripts  
**Effort:** ~1 day

- [x] Added `--source opensearch` flag to `scripts/evaluate_sfu_queries.py`
  - `fetch_opensearch_results()` function (BM25F multi_match via raw HTTP)
  - `evaluate_opensearch()` function with same NDCG/MRR/Recall metrics
  - Auto-saves to `data/eval_results/bm25f_pilot_YYYY-MM-DD.json`
  - `--opensearch-url` and `--opensearch-index` args
- [x] Ran 120-query eval on populated index (`scripts/ndcg_splade_eval.py` + `data/eval_results/ndcg_splade_eval_20260515_0420.json`)
- [x] Gate (SPLADE ≥ baseline − 0.02): **passes** (Δ = −0.0004)
- [x] Stronger gate (SPLADE ≥ baseline + 0.05, used for P.9): **fails for SPLADE alone**, **passes for RRF** (+0.0133)
- [ ] Update `docs/ultraplan.txt` with Phase P eval results (not blocking; numbers are tracked in MASTER_TODO.md and here)

**Deliverable (infra):** Eval plumbing complete ✓; results pending P.3 index

---

## Alternatives Considered and Rejected

### Option A — Over-fetch + Local BM25 (In-Memory)
Fetch 200+ results from OpenAlex API, apply local BM25 re-scoring.

**Rejected because:**
- Still burns API quota (900 req/day DailyCallTracker budget)
- BM25 over pre-fetched results is not corpus-level retrieval — scores are relative to the 200-doc window, not the full corpus
- No offline/historical path; every query still hits live API

### Option B — Full Snapshot BM25 (No SPLADE)
Index full OpenAlex corpus (~215GB compressed) using standard BM25, no learned expansion.

**Rejected because:**
- Storage: ~215GB compressed → 500GB+ uncompressed; storage-tight on current hardware
- Full index run: ~40+ hours without GPU
- Plain BM25 is strictly weaker than SPLADE on domain-specific academic queries
- Option C with the 2015+ filter gets the coverage we need at ~15-30GB

---

## Hardware Profile

| Resource | Available | Required for P.3 |
|---|---|---|
| GPU | RTX 4070 Ti SUPER, 16GB VRAM | ~8-12GB (SPLADE model + batch) |
| RAM | 32GB | ~16GB (indexer + OS) |
| Storage | TBD | ~30GB snapshot + ~60GB index |
| Time (index) | overnight | ~8-10 hours |
| Time (query) | — | +20-50ms/query (CPU encoding) |

---

## File-Level Change Summary

### New files (this repo — `sfu-library-mcp-training`):
- `scripts/snapshot_downloader.py` — P.2
- `scripts/splade_indexer.py` — P.3
- `scripts/opensearch_sync.py` — P.10

### New files (`sfu-library-mcp` repo):
- `src/lib/opensearch_retriever.py` — P.5
- `src/lib/federated_search.py` — P.6
- `src/tests/test_opensearch_retriever.py` — P.5
- `src/tests/test_federated_search.py` — P.6

### Modified files (`sfu-library-mcp` repo):
- `src/lib/config.py` — add 7 new fields (P.8)
- `src/lib/openalex.py` — add `_push_to_opensearch` hook (P.7)
- `src/lib/tools.py` — feature-flag path via `FederatedSearchRouter` (P.6)
- `src/lib/reranker.py` — remove `title_relevance`, redistribute weight (P.9, post-SPLADE eval only)

### New files (new `opensearch` container):
- `docker-compose.yml` addition or separate service definition
- Index template JSON

### Eval / data files (this repo):
- `data/openalex_snapshot/` directory (P.2)
- `data/eval_results/bm25f_pilot_*.json` (P.4)
- `data/eval_results/splade_opensearch_*.json` (P.11)

---

## Phase Sequence Summary

```
Phase M (eval expansion, stat-significance gate)
    ↓
Phase O (v4-bge deploy, CrossEncoder Tier 1.5, query-log feedback)
    ↓
Phase P.1  OpenSearch container setup
Phase P.2  Snapshot download pipeline
Phase P.3  SPLADE indexer (blocked on P.2)
Phase P.4  BM25F pilot eval (can run parallel to P.3 setup)
Phase P.5  opensearch_retriever.py (blocked on P.3)
Phase P.6  federated_search.py (blocked on P.5)
Phase P.7  openalex.py hook (blocked on P.5)
Phase P.8  config additions (can run parallel to P.5-P.7)
Phase P.9  Reranker weight update (blocked on P.11 eval gate)
Phase P.10 Monthly sync pipeline (can run parallel to P.9)
Phase P.11 End-to-end eval (blocked on P.3, P.5, P.6)
```

---

## Related Documents

- `docs/ultraplan.txt` — master phase plan (Phases G-O detailed, Phase P referenced here)
- `docs/done but important/SFU_EMBEDDING_MODEL_PLAN.md` — reranker weights, training pipeline
- `docs/THIN_CLIENT_NEW_ARCHITECTURE.md` — 4-stage pipeline; `opensearch_retriever.py` slots into stage 1
- `docs/done but important/BM25 replace and opensearch thoughts.txt` — original architecture decision conversation (same content as `docs/done but important/splade.txt`)
- `docs/MASTER_TODO.md` — cross-phase TODO index
- `docs/CRON_SETUP.md` — P.10 monthly sync crontab

---

## Eval Results Summary (Phase P.11, 2026-05-15)

Source files:
- `data/eval_results/ndcg_splade_eval_20260515_0420.json` — main 120-query NDCG/MRR comparison
- `data/eval_results/post_index_benchmark.json` — overlap stats + per-subject breakdown + 43ms avg SPLADE encode
- `data/eval_results/bm25f_pilot_2026-05-09.json` — early BM25F pilot baseline
- `data/eval_results/splade_vs_bm25f_2026-05-09_223{1,5}.json` — pilot SPLADE comparisons

Global retrieval-stage NDCG@10:
- BM25: **0.2590** (median 0.2502, σ 0.1349)
- SPLADE: **0.2586** (median 0.2549, σ 0.1339)
- **RRF: 0.2723** (median 0.2684, σ 0.1432) — wins despite SPLADE being a wash globally, because of low overlap

Per-subject NDCG@10 highlights:

| Subject | BM25 | SPLADE | Winner |
|---|---|---|---|
| Computing Science | 0.199 | **0.437** | SPLADE +120% |
| Physics | 0.150 | **0.338** | SPLADE +125% |
| Indigenous Studies | 0.277 | **0.334** | SPLADE |
| Philosophy | 0.225 | **0.308** | SPLADE |
| Political Science | **0.350** | 0.181 | BM25 |
| Engineering Science | **0.286** | 0.113 | BM25 (large gap) |
| Criminology | **0.319** | 0.214 | BM25 |

Encode latency (post_index_benchmark): **43.1ms avg** for SPLADE query encoding — a hot-query LRU cache should reclaim most of this.

---

## Streamlining Suggestions

These were proposed alongside the 2026-05-16 integration pass; tracked here so they survive the next phase boundary.

### 1. RRF as the default — DONE
`FederatedSearchRouter` now returns `SearchSource.LOCAL_RRF` for historical queries when `local_rrf_enabled=True` (default). `OpenSearchRetriever.search()` accepts a per-call `mode=` parameter so the router can dispatch BM25F and SPLADE independently of the instance's `splade_enabled` flag.

### 2. Query-encode LRU cache — TODO
SPLADE query encoding is 43ms/query. Add an LRU cache (e.g. 1000 entries) around `OpenSearchRetriever._build_splade_query` keyed on the raw query string. Expected to remove ~40ms latency from repeated queries.

### 3. Subject-aware fusion weights — TODO
Per-subject swings are huge and predictable. Replace the flat 50/50 RRF weighting with a per-subject `alpha` (BM25 weight) drawn from a small lookup. The Solr database registry already tags queries with subjects.

Sketch:
```python
SUBJECT_RRF_ALPHA = {
    "Computing Science": 0.3,     # weight SPLADE harder
    "Physics": 0.3,
    "Political Science": 0.7,     # weight BM25 harder
    "Engineering Science": 0.75,
    # ... default 0.5
}
```

### 4. Subject-aware re-eval — TODO
After §3 ships, re-run the 120-query eval with subject weighting. If it lifts NDCG@10 by another ~2%, revisit the P.9 reranker-weight-removal decision.

### 5. P.7 fire-and-forget visibility — TODO
`OpenAlexClient._push_to_opensearch` is silent on failures. Add a rate-limited warning log (once per minute) so we notice when the index drifts.

### 6. Crontab for monthly sync — DONE
See `docs/CRON_SETUP.md`.
