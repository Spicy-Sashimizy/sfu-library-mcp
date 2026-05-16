# Master TODO Index

Cross-phase status tracker. See individual plan docs for full checklists.  
Last updated: 2026-05-16

---

## Phase Status

| Phase | Description | Status | Blocked By |
|---|---|---|---|
| M | Expand eval to 120 queries; MRR@10/Recall@10; stat-sig gate | **NEXT PRIORITY** | — |
| N | LambdaMART, latency budget, query logs | Deferred | Phase O |
| O | Deploy v4-bge; CrossEncoder Tier 1.5; query-log feedback | Pending | Phase M |
| P | SPLADE + OpenSearch (Federated Hybrid) | Pending | Phase O |

---

## Phase M — Eval Expansion

**Plan:** `docs/ultraplan.txt` (search: "Phase M")  
**Files to touch:**
- [ ] `scripts/evaluate_sfu_queries.py` — add MRR@10, Recall@10 metrics
- [ ] `scripts/eval_compare.py` — add stat-significance output (95% CI)
- [ ] `data/sfu_eval_queries.json` — expand from 35 → 120 queries
- [ ] `scripts/generate_sfu_training_data.py` — may need query dedup pass

**Gate:** 95% CI of NDCG@10 delta must exclude zero before Phase O starts.

---

## Phase O — Production Deploy

**Plan:** `docs/ultraplan.txt` (search: "Phase O")  
**Files to touch (`sfu-library-mcp` repo):**
- [ ] `src/lib/config.py` — flip `crossencoder_enabled = True`
- [ ] `src/lib/reranker.py` — CrossEncoder Tier 1.5 wiring (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [ ] `src/lib/tools.py` — query-log feedback loop plumbing
- [ ] Model deployment: `SFU_EMBEDDING_MODEL_PATH` → `sfu-academic-embed-v4-bge`

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
| P.3 | SPLADE indexer | `sfu-library-mcp-training` | **DONE** — index populated; `data/eval_results/post_index_benchmark.json` confirms ~1M docs throughput target met |
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
| Eval queries | `data/sfu_eval_queries.json` | 120 queries (expanded for Phase M) | Gold standard |
| Eval cache | `data/openalex_eval_cache.json` | 227 result sets, 12MB | Pre-fetched results |
| Training triplets | `data/sfu_training_triplets.jsonl` | 9,234 triplets | anchor/pos/neg |
| BM25F pilot results | `data/eval_results/bm25f_pilot_2026-05-09.json` | 120 queries | Phase P.4 |
| SPLADE eval (final) | `data/eval_results/ndcg_splade_eval_20260515_0420.json` | 120 queries | Phase P.11 — RRF wins |
| Post-index benchmark | `data/eval_results/post_index_benchmark.json` | overlap + encode-latency stats | Drives RRF-default design |

---

## Current Best Model

`sfu-academic-embed-v4-bge` with RRF enabled:  
**NDCG@10 = 0.6689** (35-query Phase L eval, 2026-05-04)

Phase M will re-establish this baseline on 120 queries with stat-significance.

---

## Phase P — Retrieval Layer Eval (2026-05-15, 120 queries)

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
