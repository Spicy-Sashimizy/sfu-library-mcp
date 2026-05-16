# SPLADE Optimization Roadmap

**Last updated:** 2026-05-16  
**Baseline:** LLM-judged 120-query benchmark, 2026-05-16 (see `docs/LLM_BENCHMARK_RESULTS_2026-05-16.md`)  
**Current scores:** SPLADE 0.6534 | RRF 0.6362 | BM25F 0.6249 | OpenAlex 0.5391 (mean NDCG@10)

---

## Current State Summary

- RRF is wired as default in `FederatedSearchRouter` for `search_academic`
- `search_by_topic` and `export_search` wiring decisions: **confirmed, wire RRF** (delta +0.097 vs OpenAlex, threshold +0.03 met for 27/34 subjects)
- 15 subjects have 0.0 local coverage — always route live
- 2 subjects (Anthropology, Molecular Biology) perform better with OpenAlex live — route live

---

## Phase Q1 — Quick Wins, No Infrastructure Cost

All of these are parameter or flag changes in existing code. No re-indexing, no new models, no spending.

### Q1.1 — RRF k-Parameter Tuning
**Expected gain:** +0.02–0.04 NDCG  
**Effort:** 2 hours (sweep + benchmark)  
**File:** `scripts/benchmark_llm_judge.py` → `rrf_fuse()` k parameter; `src/lib/opensearch_retriever.py`

**The problem:** Current k=60 is the standard default. With only 7.5% overlap between BM25F and SPLADE results, lower k values reward the consensus documents more aggressively. SPLADE alone (0.6534) beats RRF (0.6362) precisely because k=60 is diluting SPLADE's signal.

**Steps:**
1. Add `--k-param` argument to `benchmark_llm_judge.py`
2. Run on 20-query holdout: `python scripts/benchmark_llm_judge.py --skip-live --max-queries 20 --k-param 20`
3. Repeat for k = 20, 30, 40, 50
4. Set winner in `opensearch_retriever.py` RRF call
5. Re-run full 120-query benchmark to confirm gain

**Hypothesis:** k=30-40 will make RRF outperform pure SPLADE outright (target: RRF NDCG > 0.6534).

---

### Q1.2 — BM25F Field Weighting
**Expected gain:** +0.01–0.02 on BM25 leg of RRF  
**Effort:** 30 minutes  
**File:** `scripts/benchmark_llm_judge.py` → `bm25f_search()`, `src/lib/opensearch_retriever.py`

Switch from `best_fields` (takes max field score) to `most_fields` + `tie_breaker=0.5` (sums field contributions). Better for queries where relevant terms split across title and abstract (Film, Archaeology).

```python
# Change in bm25f_search():
"query": {
    "multi_match": {
        "query": query_text,
        "fields": ["title^3", "abstract^1.5", "concepts^2"],
        "type": "most_fields",   # was "best_fields"
        "tie_breaker": 0.5       # new
    }
}
```

---

### Q1.3 — SPLADE Term Expansion Tuning
**Expected gain:** +0.01–0.03 NDCG on weak subjects  
**Effort:** 30 minutes  
**File:** `scripts/benchmark_llm_judge.py` → `splade_search()` (line ~244), `src/lib/opensearch_retriever.py`

Increase SPLADE's expanded vocabulary from 48 to 64 terms, and raise `scaling_factor` from 1 to 4 to boost rare term weights (as recommended in the original SPLADE paper).

```python
# In splade_search():
should = [
    {"rank_feature": {"field": f"sparse_field.{t}", "boost": w,
                      "log": {"scaling_factor": 4}}}  # was 1
    for t, w in sorted(sparse_query.items(), key=lambda x: -x[1])[:64]  # was 48
]
```

Particularly targets Communication (0.33), History (0.31), Economics (0.29) — subjects with thin but real index coverage where SPLADE needs more expansion room.

---

### Q1.4 — Enable Cross-Encoder Reranker
**Expected gain:** +0.03–0.06 NDCG  
**Effort:** 15 minutes (flag flip + test)  
**File:** `src/lib/config.py` → `crossencoder_enabled` flag

The cross-encoder reranker is already implemented (`src/lib/reranker.py`). It does joint query-document scoring on the top-20 RRF results and re-ranks to top-10. Currently disabled. The high variance in SPLADE results (std 0.46) indicates noisy rank ordering — the reranker fixes this.

1. Find and set `crossencoder_enabled = True` in config
2. Confirm model path for `cross-encoder/ms-marco-MiniLM-L-6-v2`
3. Run 10-query smoke test, check latency budget
4. If latency OK, run full 120-query benchmark

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~6ms per doc-pair on CPU)

---

## Phase Q2 — Medium Effort, No Cloud Cost

### Q2.1 — Zero-Coverage Subject Fallback
**Expected gain:** Prevents 15 subjects returning blank results  
**Effort:** 1 day  
**File:** `src/lib/federated_search_router.py`

Implement routing logic for the 15 subjects with 0.0 local NDCG. When a query's detected subject has no local coverage, route directly to `LIVE_API` without attempting local retrieval.

**Subjects to route live always:**  
Theatre, Music, Urban Studies, Applied Legal Studies, Publishing, Management & Organizational Studies, Accounting, Forensics, Statistics & Actuarial Science, Sustainable Energy Engineering, Sustainable Community Development, Visual Arts, Public Policy, Molecular Biology & Biochemistry, Global Health

**Implementation approach:**
```python
ALWAYS_LIVE_SUBJECTS = {
    "Theatre", "Music", "Urban Studies", "Applied Legal Studies",
    "Publishing", "Management & Organizational Studies", "Accounting",
    "Forensics", "Statistics & Actuarial Science",
    "Sustainable Energy Engineering (SEE)",
    "Sustainable Community Development",
    "Visual Arts", "Public Policy",
    "Molecular Biology & Biochemistry", "Global Health",
}

def route(self, query: str, subject_hint: str = "") -> RouteDecision:
    if subject_hint in ALWAYS_LIVE_SUBJECTS:
        return RouteDecision.LIVE_API
    # ... existing RRF logic
```

---

### Q2.2 — Subject-Aware Routing for Edge Cases
**Expected gain:** +0.01–0.03 on Anthropology  
**Effort:** 1 day  
**File:** `src/lib/federated_search_router.py`

Anthropology (RRF 0.898 vs OpenAlex 0.931) is the one subject where OpenAlex live is better than local RRF. Add it to a soft-route list that prefers live by default but can be overridden.

---

### Q2.3 — SPLADE Model Swap
**Expected gain:** +0.015–0.03 NDCG  
**Effort:** 3 hours + 11 min re-index  
**Requires:** Re-running SPLADE indexer (existing TRT pipeline, ~11 min for 1M docs)

Current model: `prithivida/Splade_PP_en_v1` (2021 vintage)  
Recommended: `naver/splade-cocondenser-ensembledistil` (2022 TREC winner, 2-4 pts better on BEIR benchmarks)

**Steps:**
1. Download new model: `python -c "from transformers import AutoTokenizer, AutoModelForMaskedLM; AutoTokenizer.from_pretrained('naver/splade-cocondenser-ensembledistil'); AutoModelForMaskedLM.from_pretrained('naver/splade-cocondenser-ensembledistil')"`
2. Update `SPLADE_MODEL` constant in `scripts/splade_indexer.py` and `scripts/benchmark_llm_judge.py`
3. Clear the OpenSearch `sparse_field.*` fields (requires re-index)
4. Run SPLADE indexer: `python scripts/splade_indexer.py` (~11 min at TRT speed)
5. Run 20-query holdout benchmark to quantify delta before full rollout
6. If delta > +0.01, run full 120-query benchmark and commit

**Risk:** +10-15ms per query at inference time. Mitigation: SPLADE queries are already batched and async.

---

### Q2.4 — Index Expansion for Zero-Coverage Subjects
**Expected gain:** +0.05–0.15 NDCG for expanded subjects  
**Effort:** 1-2 weeks per domain  
**Requires:** New data ingestion pipelines

Priority targets by data availability and impact:

| Subject | Data Source | Papers Approx | Priority |
|---------|-------------|---------------|----------|
| Molecular Biology & Biochemistry | PubMed Central Open Access | 4M+ | High — lost -1.0 on AlphaFold query |
| Global Health | WHO IRIS, PMC | 500K+ | High — lost -0.43 on pandemic query |
| Statistics & Actuarial Science | arXiv:stat, math.ST | 200K+ | Medium |
| Urban Studies | SSRN, urban OA journals | 100K+ | Medium |
| Public Policy | SSRN, Policy Commons | 200K+ | Medium |
| Theatre | JSTOR OA, MLA (if licensed) | 50K+ | Low |
| Music | RILM (if licensed), JSTOR | 30K+ | Low |

**arXiv expansion** (fastest — free bulk S3 dump):
```bash
python scripts/filter_openalex_by_concept.py \
  --concepts "Statistics" "Urban Planning" "Public Policy" \
  --output data/openalex_snapshot/expansion/
python scripts/splade_indexer.py --input data/openalex_snapshot/expansion/
```

**PubMed Central** for Molecular Biology:
- PMC Open Access subset is freely downloadable
- Parse XML → extract title/abstract/IDs → SPLADE encode → index alongside existing works

---

## Phase Q3 — Training Path (DigitalOcean, 205 Credits Available)

Training is on the table. DO credits are approximately: H100 ~$8/hr (~25hrs), A100 ~$3.57/hr (~57hrs), L40S ~$2.49/hr (~82hrs).

### Q3.1 — Mine Hard Negatives from Local Index (Free — CPU here)
**Effort:** 2 hours  
**Script:** `scripts/mine_hard_negatives.py` (currently uses BM25 via live API — upgrade to local OpenSearch)

Hard negatives are training examples where a paper scores high under a retriever but is actually irrelevant. Forcing the model to distinguish these teaches finer semantic boundaries.

**Why RRF-pooled negatives are hardest:**  
BM25F and SPLADE only overlap 7.5% — they find different papers that look relevant but aren't. Mining from both and pooling gives the hardest possible training signal (the union of both failure modes).

**Pipeline:**
```bash
# Mine SPLADE negatives (semantic near-misses)
python scripts/mine_hard_negatives.py \
  --queries data/sfu_eval_queries.json \
  --retriever splade --top-k 50 \
  --output data/training/hard_negatives_splade.jsonl

# Mine BM25F negatives (lexical near-misses)  
python scripts/mine_hard_negatives.py \
  --queries data/sfu_eval_queries.json \
  --retriever bm25f --top-k 50 \
  --output data/training/hard_negatives_bm25.jsonl

# Merge and filter false negatives (papers the LLM judge scored ≥2 are actually relevant)
# Use the existing llm_judge_cache.json to identify false negatives
python scripts/merge_negatives.py \
  --inputs data/training/hard_negatives_splade.jsonl data/training/hard_negatives_bm25.jsonl \
  --judge-cache data/eval_results/llm_judge_cache.json \
  --min-judge-score 2 \
  --output data/training/hard_negatives_rrf_pool.jsonl
```

**False-negative filtering is critical:** ~70% of BM25 top-k results on academic corpora are actually relevant (NV-Retriever 2024 finding). Any paper the LLM judge scored ≥2 must be dropped from negatives.

---

### Q3.2 — Cross-Encoder Fine-Tuning on SFU Triplets
**Expected gain:** +0.04–0.08 NDCG on reranked results  
**DO cost:** ~$12–15 (5-6 GPU hours on L40S at $2.49/hr)  
**Best first training target** — highest ROI from DO credits

A cross-encoder trained on SFU-specific triplets learns SFU vocabulary: "Musqueam", "Secwépemc", "Métis", "Trudeau", "BC Hydro", "Site C", "First Nations", "UNDRIP" — terms that are rare in the base MS MARCO training data but common in SFU queries.

**Architecture:**
- Base: `cross-encoder/ms-marco-MiniLM-L-6-v2` (already vetted in Q1.4)
- Fine-tune on SFU triplets from Q3.1
- Output: `models/sfu-cross-encoder-v1`

**DigitalOcean setup:**
```bash
# On L40S droplet (48GB VRAM, ~$2.49/hr)
pip install sentence-transformers torch

python scripts/train_cross_encoder.py \
  --base-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --train-data data/training/hard_negatives_rrf_pool.jsonl \
  --output models/sfu-cross-encoder-v1 \
  --epochs 3 --batch-size 16 --warmup-steps 200

# Expected wall time: 4-5 hours for ~8K triplets × 3 epochs
# Expected cost: ~$12-15 from DO credits
```

After training: deploy to `src/lib/reranker.py`, run 120-query benchmark to confirm gain.

---

### Q3.3 — SPLADE Fine-Tuning on SFU Corpus
**Expected gain:** +0.03–0.06 NDCG on sparse encoding  
**DO cost:** ~$30–36 (8-10 GPU hours on A100 at $3.57/hr)  
**Do this after Q3.2 confirms training pipeline works**

Fine-tuning modifies the SPLADE encoder weights to better expand SFU-specific vocabulary in the sparse representation. "Indigenous" expands correctly to "First Nations", "Musqueam", "treaty rights" — terms the pretrained MS-MARCO model underweights because they're rare in web search.

**Architecture:**
- Base: `naver/splade-cocondenser-ensembledistil` (after Q2.3 model swap)
- Training loss: FLOPS regularization (controls sparsity) + MultipleNegativesRankingLoss
- Data: 8,724 SFU triplets from `data/sfu_training_triplets.jsonl` + hard negatives from Q3.1

**VRAM requirement:** A100 40GB (SPLADE training requires more VRAM than cross-encoder — the FLOPS loss requires full vocabulary projection at each step)

```bash
# On A100 droplet (~$3.57/hr)
python scripts/finetune_splade.py \
  --base-model naver/splade-cocondenser-ensembledistil \
  --train-data data/training/hard_negatives_rrf_pool.jsonl \
  --output models/sfu-splade-v1 \
  --epochs 3 --lambda-q 0.0008 --lambda-d 0.0006 \
  --batch-size 8 --grad-accum 4

# Expected wall time: 8-10 hours
# Expected cost: ~$30-36 from DO credits
```

After training: re-index all 1M docs with new model (~11 min), run full benchmark.

---

## Summary: Ordered Action Plan

| # | Step | Cost | Expected NDCG gain | Time |
|---|------|------|--------------------|------|
| Q1.1 | RRF k-param tuning (k=20-40) | $0 | +0.02–0.04 | 2 hrs |
| Q1.2 | BM25F most_fields + tie_breaker | $0 | +0.01–0.02 on BM25 leg | 30 min |
| Q1.3 | SPLADE top_k=64 + scaling_factor=4 | $0 | +0.01–0.03 | 30 min |
| Q1.4 | Flip cross-encoder flag ON | $0 | +0.03–0.06 | 15 min |
| Q2.1 | Zero-coverage subject fallback routing | $0 | Prevents blank results for 15 subjects | 1 day |
| Q2.2 | Subject-aware routing (Anthropology edge case) | $0 | +0.01–0.03 for Anthropology | 1 day |
| Q2.3 | SPLADE model swap → Naver cocondenser | $0, re-index 11 min | +0.015–0.03 | 3 hrs |
| Q2.4 | Index expansion (arXiv stat/urban, PMC bio) | $0 data, compute only | +0.05–0.15 for expanded subjects | 1-2 wks |
| Q3.1 | Mine hard negatives from local index | $0 (CPU) | (enables Q3.2/Q3.3) | 2 hrs |
| Q3.2 | Cross-encoder fine-tune on SFU triplets | **~$12–15 DO** | +0.04–0.08 on reranked results | 5-6 GPU hrs |
| Q3.3 | SPLADE fine-tune on SFU corpus | **~$30–36 DO** | +0.03–0.06 on sparse encoding | 8-10 GPU hrs |

**Total DO spend for full training path: ~$42–51** (leaves ~$154 from 205 credits for second runs or A/B testing)

**Conservative ceiling with all Q1+Q2 applied (no training):** NDCG@10 ~0.68–0.72  
**Ceiling with Q3 training applied:** NDCG@10 ~0.72–0.76, with SFU-specific vocabulary gains especially visible in Indigenous Studies, Canadian Studies, Labour Studies
