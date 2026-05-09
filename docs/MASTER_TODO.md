# Master TODO Index

Cross-phase status tracker. See individual plan docs for full checklists.  
Last updated: 2026-05-09

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
| P.1 | OpenSearch container | `.devcontainer/docker-compose.yml` + `docker/opensearch/` | **DONE** — service, index template, setup script added |
| P.2 | Snapshot downloader | `sfu-library-mcp-training` | **DONE** (script) — `snapshot_downloader.py` with checkpoint/resume, SIGINT handling, --dry-run; needs actual run |
| P.3 | SPLADE indexer | `sfu-library-mcp-training` | **DONE** (script) — `splade_indexer.py` with 10 failsafes, checkpoint/resume, VRAM monitoring; needs P.2 data + OpenSearch |
| P.4 | BM25F pilot eval infra | `sfu-library-mcp-training` | **DONE** — `evaluate_sfu_queries.py --source opensearch` path added; needs P.1 index populated |
| P.5 | `opensearch_retriever.py` | `src/lib/opensearch_retriever.py` | **DONE** — BM25F + SPLADE paths, 11 unit tests passing |
| P.6 | `federated_search.py` | `src/lib/federated_search.py` | **DONE** — routing, DOI dedup, RRF fusion, 20 unit tests; `tools.py` feature-flagged |
| P.7 | `openalex.py` incremental hook | `src/lib/openalex.py` | **DONE** — `_push_to_opensearch` fire-and-forget, gated on `local_opensearch_enabled` |
| P.8 | Config additions | `src/lib/config.py` | **DONE** — 7 new fields (3 bool flags + 4 scalars), all default-off |
| P.9 | Reranker weight update | `src/lib/reranker.py` | Pending — gated on P.11 eval gate (SPLADE NDCG ≥ baseline − 0.02) |
| P.10 | Monthly sync pipeline | `sfu-library-mcp-training` | **DONE** (script) — `opensearch_sync.py` with delta detection, resume, idempotency; needs P.2+P.3 first run |
| P.11 | End-to-end eval | `scripts/evaluate_sfu_queries.py` | **DONE** (infra) — `--source opensearch` + `evaluate_opensearch()` + auto-save to `data/eval_results/`; needs P.3 index to measure NDCG |

---

## Code-level TODO Markers

| File | Line | Note | Status |
|---|---|---|---|
| `src/lib/config.py` | near feature flags block | 7 new config fields needed (P.8) | **DONE** |
| `src/lib/reranker.py` | `title_relevance` weight | Remove after SPLADE ships (P.9) | Pending eval gate |
| `src/lib/openalex.py` | end of `search_works()` method | `_push_to_opensearch` hook (P.7) | **DONE** |
| `src/lib/tools.py` | `_handle_search_academic` | Route via FederatedSearchRouter (P.6) | **DONE** |

---

## Key Eval Data

| Dataset | Location | Size | Notes |
|---|---|---|---|
| Eval queries | `data/sfu_eval_queries.json` | 35 queries (Phase M: expand to 120) | Gold standard |
| Eval cache | `data/openalex_eval_cache.json` | 227 result sets, 12MB | Pre-fetched results |
| Training triplets | `data/sfu_training_triplets.jsonl` | 9,234 triplets | anchor/pos/neg |
| BM25F pilot results | `data/eval_results/bm25f_pilot_*.json` | TBD (Phase P.4) | — |
| SPLADE eval results | `data/eval_results/splade_opensearch_*.json` | TBD (Phase P.11) | — |

---

## Current Best Model

`sfu-academic-embed-v4-bge` with RRF enabled:  
**NDCG@10 = 0.6689** (35-query Phase L eval, 2026-05-04)

Phase M will re-establish this baseline on 120 queries with stat-significance.
