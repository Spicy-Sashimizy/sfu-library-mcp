# Phase P: SPLADE + OpenSearch Integration Plan

**Status:** PENDING — gated on Phase O completion  
**Phase O gated on:** Phase M completion  
**Architecture decision:** Option C — Federated Hybrid (see §Alternatives Considered)  
**Last updated:** 2026-05-08

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

- [ ] Provision `docker-compose` service: OpenSearch 2.x, 8GB heap, 2 shards, 1 replica
- [ ] Enable `plugins.query.size.limit: 10000` and `http.max_content_length: 500mb`
- [ ] Create index template with SPLADE-compatible sparse vector mapping:
  ```json
  "sparse_field": { "type": "rank_features" }
  ```
- [ ] Verify index health (green) and REST API accessible on port 9200
- [ ] Set `OPENSEARCH_URL` env var and document in container README
- [ ] Add health-check endpoint (`GET /_cluster/health`) to docker-compose

**Deliverable:** Running OpenSearch 2.x instance, empty index, accessible from MCP server container

---

### P.2 — OpenAlex Snapshot Download Pipeline
**Repo:** `sfu-library-mcp-training` → `scripts/snapshot_downloader.py`  
**Effort:** ~2 days (mostly I/O wait)

- [ ] Write `scripts/snapshot_downloader.py`:
  - Fetch OpenAlex monthly snapshot manifest (Works entity only)
  - Stream-extract JSONL parts; filter: `publication_year >= 2015`, `has_abstract == true`
  - Extract per-record: `id`, `doi`, `title`, `abstract_inverted_index` (reconstruct to string), `publication_year`, `type`
  - Write to compressed JSONL chunks: `data/openalex_snapshot/works_YYYY-MM_part_NNNN.jsonl.gz`
  - Progress log with ETA; resumable via checkpoint file
- [ ] Estimate storage: ~15-30GB compressed after filtering (full unfiltered ~215GB)
- [ ] Add `--dry-run` flag that processes 1 chunk and reports stats
- [ ] Verify sample: spot-check 50 records for abstract reconstruction correctness
- [ ] Document monthly re-run cadence (snapshot releases: first Monday of each month)

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
- [ ] Install `transformers>=4.38`, `torch`, `scipy` into training venv
- [ ] Write `scripts/splade_indexer.py`:
  - Load SPLADE model onto GPU (CUDA)
  - Batch-process JSONL chunks from `data/openalex_snapshot/`
  - For each doc: encode `title + " " + abstract`, apply ReLU+log1p → sparse term weights
  - Batch-upsert to OpenSearch via `rank_features` field (doc ID = OpenAlex work ID)
  - Include DOI in stored fields for deduplication in federated search
  - Checkpoint every 100k docs; skip already-indexed IDs on resume
  - Log GPU memory usage, throughput (docs/sec), ETA
- [ ] Benchmark: target ≥ 2,000 docs/sec on RTX 4070 Ti SUPER
- [ ] Full index run: ~60M docs ÷ 2,000/sec ≈ 8-10 hours (schedule overnight)
- [ ] Verify: random 100-doc sample — query each doc's title, confirm it ranks in top 3

**Deliverable:** Populated OpenSearch index with ~60M SPLADE sparse vectors

---

### P.4 — BM25F Intermediate Validation (Before Full SPLADE)
**Repo:** `sfu-library-mcp-training` — eval scripts  
**Effort:** ~1 day

Run this before committing GPU time to full SPLADE indexing. Confirms OpenSearch retrieval path works.

- [ ] Create a 500k-doc pilot index (2022-2024, has_abstract, random sample) using standard BM25
- [ ] Configure field boosts: `title^3`, `abstract^1`, `keywords/concepts^2`  
  (mirrors current OpenAlex text search; this is BM25F)
- [ ] Run Phase M eval set (120 queries) against BM25F pilot
- [ ] Record NDCG@10, MRR@10 — compare to OpenAlex live API baseline from Phase M
- [ ] Gate: if BM25F pilot doesn't beat or match OpenAlex API NDCG@10, diagnose before continuing
- [ ] Document results in `data/eval_results/bm25f_pilot_YYYY-MM-DD.json`

**Deliverable:** BM25F baseline score; go/no-go decision for Phase P.3

---

### P.5 — OpenSearch Retriever Module
**Repo:** `sfu-library-mcp` → `src/lib/opensearch_retriever.py`  
**Effort:** ~1 day

- [ ] Write `src/lib/opensearch_retriever.py` with:
  - `OpenSearchRetriever` class wrapping `opensearch-py` client
  - `search(query: str, top_k: int = 50) -> list[dict]` method
  - SPLADE query encoding: load model once (singleton), encode query → sparse term dict
  - Build `rank_features` query body for OpenSearch
  - Return normalized result dicts: `{doi, title, abstract, year, score, source: "opensearch"}`
  - Connection retry + circuit breaker (same pattern as `SemanticScholarClient`)
  - `SFU_OPENSEARCH_URL` env var, default `http://localhost:9200`
- [ ] Unit test: `src/tests/test_opensearch_retriever.py`
  - Mock OpenSearch responses
  - Verify query encoding produces non-empty sparse dict
  - Verify result normalization
- [ ] Integration test (manual): point at local OpenSearch, run 5 known queries, check top results

**Deliverable:** `opensearch_retriever.py` + tests; importable from `federated_search.py`

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
- [ ] Write `src/lib/federated_search.py`:
  - `FederatedSearchRouter` class
  - `route(query: str, filters: dict) -> SearchSource` enum: `LIVE_API`, `LOCAL_INDEX`, `BOTH`
  - Routing logic:
    - If `filters.get("from_publication_date")` within last 30 days → `LIVE_API`
    - If query has temporal cues ("recent", "2025", "latest") → `LIVE_API`
    - Otherwise → `LOCAL_INDEX`
    - Always `BOTH` if `federated_both_enabled` config flag is set
  - `search(query, filters, top_k)` method: dispatches, deduplicates by DOI, RRF-fuses
  - DOI normalization: strip `https://doi.org/` prefix for comparison
- [ ] Write `src/tests/test_federated_search.py`:
  - Test routing logic for fresh/historical/temporal-cue cases
  - Test DOI dedup: overlapping result sets collapse correctly
  - Test RRF fusion with mock scores
- [ ] Update `src/lib/tools.py`: replace direct `openalex.py` calls with `FederatedSearchRouter` when `federated_search_enabled` flag is set (feature-flagged)
- [ ] Update `src/lib/config.py`: add `federated_search_enabled` flag (default `False`)

**Deliverable:** `federated_search.py` + tests; `tools.py` updated with feature flag path

---

### P.7 — Incremental Index Hook in openalex.py
**Repo:** `sfu-library-mcp` → `src/lib/openalex.py`  
**Effort:** ~0.5 day

Every live API response gets pushed to local OpenSearch to keep the index warm for recently-published papers.

- [ ] Add `_push_to_opensearch(results: list[dict])` private method in `OpenAlexClient`
  - Only active when `local_opensearch_enabled` config flag is True
  - Fire-and-forget: async, non-blocking, catch all exceptions silently (never degrade live search)
  - Batch upsert: DOI as doc ID, avoid re-encoding if doc already indexed
  - Skip SPLADE encoding in this path — store raw text; encode lazily or queue for background worker
- [ ] Integration point: call `_push_to_opensearch` at end of `search()` method
- [ ] Test: mock OpenSearch client; verify push is called with correct payload; verify exception swallowed

**Deliverable:** Live API responses begin populating local index; zero latency impact on hot path

---

### P.8 — Config Additions
**Repo:** `sfu-library-mcp` → `src/lib/config.py`  
**Effort:** ~0.5 day

- [ ] Add to `Config` class (maintain existing `rrf_enabled`, `rerank_enabled`, `crossencoder_enabled` pattern):
  ```python
  local_opensearch_enabled: bool = False      # Master switch for OpenSearch path
  splade_enabled: bool = False                # Use SPLADE encoding vs BM25F on OpenSearch
  federated_search_enabled: bool = False      # Route queries through FederatedSearchRouter
  opensearch_url: str = "http://localhost:9200"
  opensearch_index: str = "openalex_works"
  splade_model_path: str = "naver/splade-cocondenser-distil"
  federated_recency_days: int = 30            # Queries within N days → live API
  ```
- [ ] All flags default to `False` — zero behavior change until explicitly enabled
- [ ] Document each flag in config docstring

**Deliverable:** Config additions; no behavior change until flags flipped

---

### P.9 — Reranker Weight Update
**Repo:** `sfu-library-mcp` → `src/lib/reranker.py`  
**Effort:** ~0.5 day

**IMPORTANT:** Do not run this step until Phase P.3 (full SPLADE index) is live and evaluated.

Current weights (from `SFU_EMBEDDING_MODEL_PLAN.md`):
```
semantic_similarity: 0.35
title_relevance: 0.15   ← PATCH: compensates for no query ownership; remove once SPLADE ships
recency: 0.15
fulltext_available: 0.15
type_match: 0.10
completeness: 0.10
```

Post-SPLADE target weights:
```
semantic_similarity: 0.50   # +0.15 from title_relevance removal
title_relevance: 0.00       # REMOVE — SPLADE handles lexical matching
recency: 0.15
fulltext_available: 0.15
type_match: 0.10
completeness: 0.10
```

- [ ] After SPLADE eval confirms NDCG@10 improvement ≥ 5% over current baseline:
  - Remove `title_relevance` signal from `_compute_feature_scores` in `reranker.py`
  - Redistribute weight: `semantic_similarity → 0.50`
  - Run full 120-query Phase M eval set before and after weight change
  - Require: weight change does not reduce NDCG@10 vs pre-change baseline (within 95% CI)
- [ ] Update `SFU_EMBEDDING_MODEL_PLAN.md` with new weights and rationale

**Deliverable:** Updated reranker weights; eval confirms no regression

---

### P.10 — Monthly Snapshot Sync Pipeline
**Repo:** `sfu-library-mcp-training` → `scripts/opensearch_sync.py`  
**Effort:** ~1 day

- [ ] Write `scripts/opensearch_sync.py`:
  - Fetch latest manifest; compare to `data/openalex_snapshot/last_sync.json`
  - Download only new/updated JSONL parts (delta sync)
  - Re-run SPLADE indexer on delta only
  - Update `last_sync.json` with new manifest hash + timestamp
  - Idempotent: safe to run multiple times
- [ ] Document: run on first Monday of each month (mirrors OpenAlex release cadence)
- [ ] Add to crontab instructions in container README

**Deliverable:** `opensearch_sync.py`; monthly sync reduces incremental update time to ~1-2 hours

---

### P.11 — End-to-End Evaluation
**Repo:** `sfu-library-mcp-training` — eval scripts  
**Effort:** ~1 day

- [ ] Extend `scripts/evaluate_sfu_queries.py` to support `--source opensearch` flag
- [ ] Run 120-query Phase M eval set against SPLADE/OpenSearch endpoint
- [ ] Compare to Phase M baselines (v4-bge + RRF with live OpenAlex API)
- [ ] Required gate: SPLADE NDCG@10 ≥ Phase M OpenAlex baseline − 0.02 (allow 2% slack for corpus staleness)
- [ ] Record results in `data/eval_results/splade_opensearch_YYYY-MM-DD.json`
- [ ] Update `docs/ultraplan.txt` with Phase P eval results

**Deliverable:** SPLADE eval results; go/no-go for production Phase O cutover

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
- `docs/BM25 replace and opensearch thoughts.txt` — original architecture decision conversation (same content as `docs/splade.txt`)
- `docs/MASTER_TODO.md` — cross-phase TODO index
