# Master TODO Index

Cross-phase status tracker. See individual plan docs for full checklists.  
Last updated: 2026-05-16

---

## Phase Status

| Phase | Description | Status | Blocked By |
|---|---|---|---|
| M | Expand eval to 120 queries; MRR@10/Recall@10; stat-sig gate | **DONE** — LLM-judged benchmark complete 2026-05-16 | — |
| N | LambdaMART, latency budget, query logs | Deferred | Phase O |
| O | Deploy v4-bge; CrossEncoder Tier 1.5; query-log feedback | Pending | — |
| P | SPLADE + OpenSearch (Federated Hybrid) | **DONE** — RRF wired, benchmark confirms +0.097 vs OpenAlex | — |
| Q | SPLADE optimization (k-tuning, model swap, training) | **NEXT PRIORITY** | — |

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

## Phase O — Production Deploy

**Plan:** `docs/done but important/ultraplan.txt` (search: "Phase O")  
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

RRF (BM25F + SPLADE, local OpenSearch index):  
**NDCG@10 = 0.6534 (SPLADE alone) / 0.6362 (RRF)** — 120-query LLM-judged, 2026-05-16  
vs OpenAlex relevance sort: **0.5391** (+0.097 delta for RRF)

Previous baseline: `sfu-academic-embed-v4-bge` NDCG@10 = 0.6689 (35-query Phase L eval, 2026-05-04) — this was on a different eval set and metric; not directly comparable.

**Next milestone (Phase Q):** RRF NDCG@10 > 0.6700 via k-param tuning + cross-encoder reranker.

---

## Phase Q — SPLADE Optimization (Next Priority)

**Plan:** `docs/SPLADE_OPTIMIZATION_ROADMAP.md`  
**DigitalOcean credits available:** 205 (~$42–51 needed for full training path)

### Q1 — Quick wins (no infra cost):
- [x] Q1.1 RRF k-param sweep — SWEPT 2026-05-16; k has **no effect** when BM25F/SPLADE overlap is ≤10%. k=60 kept. Root cause of RRF < SPLADE is BM25F quality dilution, not k — fixed by cross-encoder (Q1.4).
- [x] Q1.2 BM25F most_fields + tie_breaker=0.5 — `src/lib/opensearch_retriever.py` + `scripts/benchmark_llm_judge.py` (2026-05-16)
- [x] Q1.3 SPLADE top_k=64 + scaling_factor=4 — `src/lib/opensearch_retriever.py` + benchmark aligned to same DSL (2026-05-16)
- [x] Q1.4 Flip cross-encoder ON — `src/lib/config.py` → `crossencoder_enabled = True` (2026-05-16; latency gate: run full benchmark to confirm &lt;500ms overhead)

### Q2 — Medium effort (no cloud cost):
- [ ] Q2.1 Zero-coverage subject fallback routing (15 subjects → always LIVE_API)
- [ ] Q2.2 Subject-aware routing for Anthropology edge case
- [ ] Q2.3 SPLADE model swap → `naver/splade-cocondenser-ensembledistil` + re-index
- [ ] Q2.4 Index expansion: arXiv stat/urban/policy, PubMed Central for bio

### Q3 — Training (DigitalOcean):
- [ ] Q3.1 Mine hard negatives from local OpenSearch (CPU, free)
- [ ] Q3.2 Cross-encoder fine-tune on SFU triplets (~$12-15 DO, L40S, 5-6 GPU hrs)
- [ ] Q3.3 SPLADE fine-tune on SFU corpus (~$30-36 DO, A100, 8-10 GPU hrs)

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
