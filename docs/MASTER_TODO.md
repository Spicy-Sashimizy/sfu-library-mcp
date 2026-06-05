# Master TODO Index

Cross-phase status tracker. See individual plan docs for full checklists.  
Last updated: 2026-06-05 (reconciled against commit history through `48f5a36`, 2026-05-31)

> **Reconciliation note (2026-06-05):** This index had drifted behind the code.
> Corrected below: **Q2.1/Q2.2 are DONE and wired** (commit `fae67d7`, call-site
> wiring in `src/lib/tools.py:947-961`); **Q3.1/Q3.2 are DONE and wired** (CE v1 +
> embedder v5, commits `7b247d2`→`8639b14`); **Q3.3 is prep-only, never run**.
> **Q2.3 (SPLADE model swap) is still open** — the 2026-05-29 full reindex
> re-encoded with the *same* `prithivida/Splade_PP_en_v1`, not the Naver model.
> The index is **~150M docs (150,413,098), 406 GB**, NOT "~1M" as several rows
> below originally claimed (reindex completed 2026-05-29, `indexer_status.json`).

---

## Phase Status

| Phase | Description | Status | Blocked By |
|---|---|---|---|
| M | Expand eval to 120 queries; MRR@10/Recall@10; stat-sig gate | **DONE** — LLM-judged benchmark complete 2026-05-16 | — |
| N | LambdaMART (Tier 2) + GUI tracking layer | **FRAMEWORK SHIPPED** (2026-06-05) — wired behind default-off `lambdamart_enabled`; training deferred (needs lightgbm + judged dataset run) | — |
| O | Deploy v4-bge; CrossEncoder Tier 1.5; query-log feedback | Largely superseded — CE v1 + embedder v5 wired (Q3.2); query-log loop still open | — |
| P | SPLADE + OpenSearch (Federated Hybrid) | **DONE** — RRF wired, benchmark confirms +0.097 vs OpenAlex | — |
| Q | SPLADE optimization (k-tuning, model swap, training) | **IN PROGRESS** — Q1 ✅, Q2.1/Q2.2 ✅, Q3.1/Q3.2 ✅; **open: Q2.3, Q2.4, Q3.3** | — |

---

## Phase M — Eval Expansion — DONE (2026-05-16)

**Completed:** LLM-judged TREC-style benchmark replaces citation-count proxy.  
**Results:** `docs/LLM_BENCHMARK_RESULTS_2026-05-16.md` | raw: `data/eval_results/benchmark_llm_judge_final.json`  
**Why old method was wrong:** `docs/BENCHMARK_METHODOLOGY.md`

Key finding: citation-count proxy was understating local index quality by 2.5×. SPLADE NDCG was actually 0.6534, not 0.2586.

- [x] 120-query eval set complete (`data/sfu_eval_queries.json`)
- [x] NDCG@10, MRR@10, P@10(≥2) computed for all methods
- [x] LLM-judged topical relevance (Claude Haiku, TREC 0-3 scale)
- [x] OpenAlex live included as fourth comparison method
- [x] Stat-sig confirmed: RRF beats OpenAlex by +0.097 NDCG (71/120 wins, 34% ties, 7% losses)

---

## Phase N — LambdaMART Tier 2 + GUI Tracking — FRAMEWORK SHIPPED (2026-06-05)

The learned-reranker (Tier 2 stacking) framework is now wired end-to-end behind a
default-off flag, plus the analytics tracking layer the GUI dashboard consumes.
**Training is deferred** — it needs `lightgbm` installed and a judged dataset built
(OpenSearch was down / no real query log when this landed).

**Reranker (default off; no-op without lightgbm + `models/lambdamart_v1.txt`):**
- [x] `src/lib/lambdamart_features.py` — shared 6-feature vector (train == infer single source of truth). `bm25_rank_score` deliberately dropped (positional signal with no judge-set equivalent).
- [x] `src/lib/reranker.py` — `rerank_with_lambdamart` + lazy-load sentinel (mirrors the cross-encoder pattern).
- [x] `src/lib/tools.py` — Stage 3 in `_maybe_rerank` (after the cross-encoder); retrieval pool widened.
- [x] `src/lib/config.py` — `SFU_FEATURE_LAMBDAMART_ENABLED` (default off) + `SFU_LAMBDAMART_MODEL_PATH`.

**Training pipeline (run when lightgbm is available):**
- [x] `scripts/build_lambdamart_dataset.py` — joins LLM-judge grades → OpenAlex metadata → shared features → JSONL. Offline dry-run verified (120 q / 1256 rows via eval cache).
- [x] `scripts/train_lambdamart.py` reworked — judged-grade primary path, citation-proxy query-log fallback, `--eval` GroupKFold NDCG@10 vs semantic baseline, feature-importance sidecar.
- [x] `src/lib/openalex.py` — `get_works_batch` (paged batch fetch for the builder).
- [ ] **TODO (deferred):** `sudo .venv/bin/pip install lightgbm`; run `build_lambdamart_dataset.py` (full fetch) + `train_lambdamart.py --eval`; flip the flag only on a positive delta.

**GUI tracking layer (from the Claude Design analytics dashboard handoff):**
- [x] `src/lib/engagement.py` — validated click-through event log; impressions = propensity denominators.
- [x] `src/lib/analytics.py` — `build_analytics_bundle()`: ndcg-by-subject, reranker-signal, position-bias/propensity, model-versions, session-replay, KPIs (non-LambdaMART panels are explicit stubs).
- [x] `src/lib/model_registry.{json,py}` — model-versions table (Admin activate/rollback).
- [x] `record_engagement` MCP tool + `GET /analytics` + `POST /engagement` (`src/sfu_library_mcp_http.py`).

---

## Phase O — Production Deploy (largely superseded by Q1/Q3)

**Plan:** `docs/done but important/ultraplan.txt` (search: "Phase O")  
**Files to touch (`sfu-library-mcp` repo):**
- [x] `src/lib/config.py` — flip `crossencoder_enabled = True` — **DONE** (Q1.4, `719c093`)
- [x] `src/lib/reranker.py` — CrossEncoder Tier 1.5 wiring — **DONE & UPGRADED**: now serves SFU-fine-tuned **CE v1** with exists()-fallback to base `cross-encoder/ms-marco-MiniLM-L-6-v2` (`8639b14`)
- [ ] `src/lib/tools.py` — query-log feedback loop plumbing — **STILL OPEN** (the one genuinely-incomplete Phase O item)
- [x] Model deployment: `SFU_EMBEDDING_MODEL_PATH` → **embedder v5** (supersedes the v4-bge target; `eb67a2d`)

**Gate:** Phase M stat-sig gate passed.

---

## Phase P — SPLADE + OpenSearch

**Plan:** `docs/SPLADE_OPENSEARCH_INTEGRATION_PLAN.md`  
**Architecture:** Option C — Federated Hybrid  
**Gate:** Phase O complete

### Sub-phases:
| Sub-phase | Description | Repo | Status |
|---|---|---|---|
| P.1 | OpenSearch container | `.devcontainer/docker-compose.yml` + `docker/opensearch/` | **DONE** — service, index template, setup script |
| P.2 | Snapshot downloader | `sfu-library-mcp-training` | **DONE** — ran 2026-05-13, ~60M filtered Works in `data/openalex_snapshot/` |
| P.3 | SPLADE indexer | `sfu-library-mcp-training` | **DONE** — index populated. (Throughput first confirmed on a ~1M pilot; **full corpus reindexed 2026-05-29 to 150,413,098 docs / 406 GB**, 0 errors — `indexer_status.json`. Evals dated 05-15/16 ran against the earlier ~1M pilot index.) |
| P.4 | BM25F pilot eval infra | `sfu-library-mcp-training` | **DONE** — pilot run 2026-05-09 (`data/eval_results/bm25f_pilot_2026-05-09.json`) |
| P.5 | `opensearch_retriever.py` | `src/lib/opensearch_retriever.py` | **DONE** — BM25F + SPLADE paths; `mode=` param added 2026-05-16 for RRF dispatch |
| P.6 | `federated_search.py` | `src/lib/federated_search.py` | **DONE** — `LOCAL_RRF` source added 2026-05-16; default for historical queries |
| P.7 | `openalex.py` incremental hook | `src/lib/openalex.py` | **DONE** — `_push_to_opensearch` fire-and-forget |
| P.8 | Config additions | `src/lib/config.py` | **DONE** — `local_rrf_enabled` added 2026-05-16; opensearch + federated flags now default-on |
| P.9 | Reranker weight update | `src/lib/reranker.py` | **DEFERRED** — original +5% gate fails for SPLADE-alone (Δ −0.0004 vs BM25); passes for RRF (+5.1%). See SPLADE plan §P.9 |
| P.10 | Monthly sync pipeline | `sfu-library-mcp-training` | **DONE** (script + cron doc in `docs/CRON_SETUP.md`) |
| P.11 | End-to-end eval | `scripts/ndcg_splade_eval.py` | **DONE** — 120-query eval 2026-05-15 (`data/eval_results/ndcg_splade_eval_20260515_0420.json`): BM25 0.2590 / SPLADE 0.2586 / **RRF 0.2723** |

---

## Code-level TODO Markers

| File | Line | Note | Status |
|---|---|---|---|
| `src/lib/config.py` | near feature flags block | 7 new config fields needed (P.8) | **DONE** |
| `src/lib/reranker.py` | `title_relevance` weight | Remove after SPLADE ships (P.9) | **DEFERRED** — SPLADE-alone failed the +5% gate; RRF passes but doesn't motivate removing `title_relevance` |
| `src/lib/openalex.py` | end of `search_works()` method | `_push_to_opensearch` hook (P.7) | **DONE** |
| `src/lib/tools.py` | `_handle_search_academic` | Route via FederatedSearchRouter (P.6) | **DONE** |

---

## Key Eval Data

| Dataset | Location | Size | Notes |
|---|---|---|---|
| Eval queries | `data/sfu_eval_queries.json` | 120 queries | Gold standard — 49 subjects |
| LLM judge cache | `data/eval_results/llm_judge_cache.json` | 321KB | All 120 queries cached — reruns free |
| **LLM benchmark (final)** | `data/eval_results/benchmark_llm_judge_final.json` | 68KB | **Current ground truth** — 2026-05-16 |
| LLM benchmark report | `data/eval_results/benchmark_report_final.md` | 17KB | Human-readable wiring decisions |
| Old citation-proxy eval | `data/eval_results/ndcg_splade_eval_20260515_0420.json` | 65KB | **SUPERSEDED** — 2.5× undercount, do not use |
| Training triplets | `data/sfu_training_triplets.jsonl` | 9,234 triplets | anchor/pos/neg for Q3 training |
| BM25F pilot results | `data/eval_results/bm25f_pilot_2026-05-09.json` | — | Phase P.4, historical |
| Post-index benchmark | `data/eval_results/post_index_benchmark.json` | — | Overlap + encode-latency stats |

---

## Current Best Model

RRF (BM25F + SPLADE, local OpenSearch index) with cross-encoder reranker:  
**NDCG@10 = 0.8245 (SPLADE alone) / 0.7229 (RRF)** — 120-query LLM-judged, 2026-05-17 (post-Q1)  
vs pre-Q1 baseline: SPLADE +0.171, RRF +0.087, BM25F +0.038  
vs OpenAlex relevance sort: **0.5391** (RRF delta +0.184)

Previous pre-Q1 baseline: SPLADE 0.6534 / RRF 0.6362 / BM25F 0.6249 (2026-05-16)  
Previous model baseline: `sfu-academic-embed-v4-bge` NDCG@10 = 0.6689 (35-query Phase L eval, 2026-05-04) — different eval set, not directly comparable.

**Q1 milestone cleared.** Subject routing (Q2.1/Q2.2) since landed and wired.

**Post-training (Q3.2) component evals — 2026-05-24** (best current signals; no single full-pipeline end-to-end number exists yet):
- **Cross-encoder v1** (SFU hard-neg triplets): NDCG@10 **0.7405** / MRR 0.7942 — +9.5% vs base 0.6763. Wired in `reranker.py`.
- **Embedder v5** (re-fine-tuned from v4-bge): NDCG@10 **0.7717** / MRR 0.8375 — +23.8% vs v4 0.6233. Wired via `SFU_EMBEDDING_MODEL_PATH`.

Remaining open lever toward ≥0.7500 end-to-end: Q2.3 model swap and/or Q3.3 SPLADE fine-tune.

---

## Phase Q — SPLADE Optimization (Next Priority)

**Plan:** `docs/SPLADE_OPTIMIZATION_ROADMAP.md`  
**DigitalOcean credits available:** 205 (~$42–51 needed for full training path)

### Q1 — Quick wins (no infra cost):
- [x] Q1.1 RRF k-param sweep — SWEPT 2026-05-16; k has **no effect** when BM25F/SPLADE overlap is ≤10%. k=60 kept. Root cause of RRF < SPLADE is BM25F quality dilution, not k — fixed by cross-encoder (Q1.4).
- [x] Q1.2 BM25F most_fields + tie_breaker=0.5 — `src/lib/opensearch_retriever.py` + `scripts/benchmark_llm_judge.py` (2026-05-16)
- [x] Q1.3 SPLADE top_k=64 + scaling_factor=4 — `src/lib/opensearch_retriever.py` + benchmark aligned to same DSL (2026-05-16)
- [x] Q1.4 Flip cross-encoder ON — `src/lib/config.py` → `crossencoder_enabled = True` (2026-05-16; full benchmark confirmed 2026-05-17: SPLADE 0.8245, RRF 0.7229)

### Q2 — Medium effort (no cloud cost):
- [x] Q2.1 Zero-coverage subject fallback routing (15 subjects → always LIVE_API) — **DONE** `fae67d7` (`ALWAYS_LIVE_SUBJECTS` in `src/lib/federated_search.py:32`); call-site wiring via `detect_subject()` in `src/lib/tools.py:947-961`
- [x] Q2.2 Subject-aware routing for Anthropology edge case — **DONE** `fae67d7` (`SOFT_LIVE_SUBJECTS`, overridable by `prefer_local`)
- [ ] Q2.3 SPLADE model swap → `naver/splade-cocondenser-ensembledistil` + re-index — **OPEN**. Indexer still defaults to `prithivida/Splade_PP_en_v1` (`scripts/splade_indexer.py:95`); the 2026-05-29 reindex used that same model. A swap now costs a full 150M re-encode (~2.4 h local), not "11 min."
- [ ] Q2.4 Index expansion: arXiv stat/urban/policy, PubMed Central for bio — **OPEN**

### Q3 — Training (DigitalOcean):
- [x] Q3.1 Mine hard negatives from local OpenSearch (CPU, free) — **DONE** `7b247d2`/`fe3051a` (atomic writes); 9,932-record pool
- [x] Q3.2 Cross-encoder fine-tune on SFU triplets — **DONE locally (free, not DO)** `c8be236`→`8639b14`; CE v1 wired in `reranker.py` (+9.5% NDCG). Embedder v5 also trained + wired (`eb67a2d`, +23.8% vs v4).
- [ ] Q3.3 SPLADE fine-tune on SFU corpus — **PREP-ONLY, NEVER RUN** (`0d9e8f1` cloud orchestrator + `59d01c0` local hardened pipeline). ~$5-8 DO, L40S 48GB, 3-5 GPU hrs — bf16/cost-safe; ≤$15.70 worst case. Orchestrator `scripts/cloud/run_splade_finetune.sh` (auto-teardown + self-destruct + checkpoint round-trip). **Run-time prereqs unmet: install doctl + Write-scope DO token + `--ssh-key-id`.** Local alternative `scripts/run_splade_pipeline_local.sh` (~4-5 h, $0) also not yet run.

**Target:** RRF NDCG@10 ≥ 0.70 after Q1+Q2; ≥ 0.74 after Q3.

---

## Phase P — Retrieval Layer Eval (2026-05-15/16, 120 queries)

OpenSearch-only NDCG@10 (no reranker, no semantic similarity, raw retrieval):

| Retriever | NDCG@10 | MRR@10 | Note |
|---|---|---|---|
| BM25 | 0.2590 | 0.6567 | baseline |
| SPLADE | 0.2586 | 0.6048 | Δ vs BM25: −0.0004 |
| **RRF (BM25+SPLADE)** | **0.2723** | 0.6154 | Δ vs BM25: **+5.1%** |

**Avg BM25↔SPLADE overlap: 7.5%** — near-orthogonal retrieval. This is why
RRF wins despite SPLADE matching BM25 globally. RRF was made default in
`FederatedSearchRouter` on 2026-05-16.

Subject-level extremes (full table in `post_index_benchmark.json`):
- SPLADE >> BM25: Computing Science (0.44 vs 0.20), Physics (0.34 vs 0.15)
- BM25 >> SPLADE: Political Science (0.35 vs 0.18), Criminology (0.32 vs 0.21), Engineering Science (0.29 vs 0.11)
- Suggests a future subject-aware fusion-weight pass.
