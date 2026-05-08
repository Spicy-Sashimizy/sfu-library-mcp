# Master TODO Index

Cross-phase status tracker. See individual plan docs for full checklists.  
Last updated: 2026-05-08

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
| P.1 | OpenSearch container | new `opensearch` container | Pending |
| P.2 | Snapshot downloader | `sfu-library-mcp-training` | Pending |
| P.3 | SPLADE indexer | `sfu-library-mcp-training` | Pending (needs P.2) |
| P.4 | BM25F pilot eval | `sfu-library-mcp-training` | Pending |
| P.5 | `opensearch_retriever.py` | `sfu-library-mcp` | Pending (needs P.3) |
| P.6 | `federated_search.py` | `sfu-library-mcp` | Pending (needs P.5) |
| P.7 | `openalex.py` incremental hook | `sfu-library-mcp` | Pending (needs P.5) |
| P.8 | Config additions | `sfu-library-mcp` | Pending |
| P.9 | Reranker weight update | `sfu-library-mcp` | Pending (needs P.11 eval gate) |
| P.10 | Monthly sync pipeline | `sfu-library-mcp-training` | Pending |
| P.11 | End-to-end eval | `sfu-library-mcp-training` | Pending (needs P.3, P.5, P.6) |

---

## Code-level TODO Markers

The following files contain `# TODO(Phase P)` markers pointing here:

| File | Line | Note |
|---|---|---|
| `src/lib/config.py` | near feature flags block | 7 new config fields needed (P.8) |
| `src/lib/reranker.py` | `title_relevance` weight | Remove after SPLADE ships (P.9) |
| `src/lib/openalex.py` | end of `search()` method | Add `_push_to_opensearch` hook (P.7) |
| `src/lib/tools.py` | `_handle_search_academic` | Route via FederatedSearchRouter (P.6) |

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
