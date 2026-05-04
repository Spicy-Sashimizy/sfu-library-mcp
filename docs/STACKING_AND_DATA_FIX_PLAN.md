# Stacking + Data Pipeline Fix Plan

Implementation plan covering: Solr coverage verification, root-cause fix for the cleaning step, rank-fusion stacking, retraining, and a measurement protocol that decides each step before the next one runs.

---

## Phase 0: Verification — Was the entire SFU Solr corpus actually used?

Verification ran against `data/solr_cache.json` (766 records) and `data/sfu_training_triplets.jsonl` (raw, 4,014 triplets).

### Subject coverage (97% reached, 3% missed)

| Metric | Count | Coverage |
|---|---|---|
| Subjects in Solr | 108 | — |
| Subjects in raw training data | 105 | 97% |
| Subjects in *clean* training data | 77 | 71% |
| Subjects missing entirely | 6 | — |

Missing subjects (each backed by ≥9 SFU databases — not edge cases):

- Biomedical Physiology and Kinesiology (BPK) - Sports (20 dbs)
- Database Trials (1 db) — can ignore, administrative bucket
- Environmental Education (9 dbs)
- French Literature (17 dbs)
- Global Health (21 dbs)
- Health Care & Epidemiology (17 dbs)

26 additional subjects appeared in <10 triplets — too thin to teach the model anything.

### Provider coverage (14% reached, 86% missed)

| Metric | Count | Coverage |
|---|---|---|
| Providers in Solr | 108 | — |
| Providers tagged in training metadata | 15 | 14% |

This is the headline failure: `provider_aware` strategy was supposed to teach the model SFU's subscription landscape, and 86% of providers never appeared.

### Strategy hit rates (raw, before cleaning)

| Strategy | Plan target | Got | Hit |
|---|---|---|---|
| `synthetic_query` | 3,000 | 2,869 | **96%** ✓ |
| `cross_subject_hard_negative` | 3,000 | 471 | 16% |
| `citation_pair` | 5,000 | 409 | 8% |
| `provider_aware` | 4,000 | 265 | 7% |

### Cleaning step destroys 83% of triplets

| File | Triplets | Dropped from raw |
|---|---|---|
| `sfu_training_triplets.jsonl` (raw) | 4,014 | — |
| `sfu_training_triplets.clean.jsonl` | 700 | **83%** |

Per-strategy loss in cleaning:

| Strategy | Raw → Clean | Loss |
|---|---|---|
| `synthetic_query` | 2,869 → 279 | 90% |
| `cross_subject_hard_negative` | 471 → 375 | 20% |
| `citation_pair` | 409 → 42 | 90% |
| `provider_aware` | 265 → 4 | 99% |

### Root cause

`scripts/generate_sfu_training_data.py:786` deduplicates by `anchor[:200]` — first 200 characters of the anchor string. Most strategies emit *multiple positive papers per anchor* (a single synthetic query is intentionally paired with many topically-relevant papers), so the dedup keeps only the first triplet per anchor and discards the rest.

This is not by design — it actively destroys the contrastive learning signal, which benefits from many (anchor, positive) pairs sharing an anchor.

### Verdict on "was the entirety of Solr properly trained"

**No, not properly.** The retrieval was largely complete (97% subject coverage in raw data) but the cleaning step then threw away 83% of those triplets, leaving 71% subject coverage and only 14% provider coverage in what actually trained the model. The +2.2% gain in v1 came from ~5% of the planned dataset.

---

## Phase 1: Fix the dedup bug

**File:** `scripts/generate_sfu_training_data.py`, function `run_quality_checks` (line 742).

**Change:**
- Remove the `anchor[:200]` global dedup.
- Keep the `(anchor, positive)` pair dedup — it correctly removes exact duplicate training pairs.
- Add length checks for the negative as well (currently only anchor and positive are checked).

**Expected outcome:** clean dataset grows from 700 → ~3,800 triplets without re-running any API calls. Subject coverage rises from 71% → 97%. Provider coverage rises from 4 records → 250+ provider-aware triplets.

---

## Phase 2: Re-clean existing raw triplets (no regeneration)

Run the fixed quality-check pass over the raw file. No OpenAlex calls needed. Takes seconds.

```bash
.venv/bin/python3 -m scripts.generate_sfu_training_data --reclean-only
```

(A `--reclean-only` flag will be added in Phase 1.)

---

## Phase 3: Rank-fusion stacking (RRF) in evaluation

This is the .bat-stacking-idea, scoped to the smallest version that produces signal.

**Approach:** The current eval script reranks OpenAlex results by *embedding cosine alone*, discarding the BM25 ranking that OpenAlex returned. RRF combines the two:

```
score(d) = 1 / (60 + rank_bm25(d)) + 1 / (60 + rank_embed(d))
```

No score normalization, no learned weights, no training. Standard k=60 from the literature.

**Why this is high-leverage:** the original benchmark showed BM25 (0.31) and embedding (0.12) fail on different queries — the canonical case where rank fusion outperforms either ranker alone. The Indigenous Studies query that scored 0.998 on BM25 and 0.0002 on MiniLM is recovered immediately by RRF.

**Implementation:** Extend `scripts/evaluate_sfu_queries.py` with a `--fusion rrf` flag.

---

## Phase 4: Benchmark — 4-way comparison on the 35 SFU eval queries

| Configuration | What's measured |
|---|---|
| BM25-only (OpenAlex order) | Lower bound for the lexical signal |
| MiniLM-only (off-the-shelf) | Lower bound for embedding rerank |
| sfu-academic-embed-v1-only | Current production candidate |
| RRF(BM25, sfu-academic-embed-v1) | Stacking gain over the fine-tune |

Decision rule:
- If RRF > v1 by ≥3% NDCG: ship RRF immediately, do Phase 5.
- If RRF ≤ v1: do not ship RRF; fall through to Phase 6 only.

---

## Phase 5: Port RRF to the production reranker (gated on Phase 4 win)

**File:** `src/lib/reranker.py`. Production docs come from Primo, not OpenAlex, but the doc *order* returned by Primo is itself a BM25-derived signal we can fuse against the embedding rank.

**Change:**
- New scoring path: `_compute_rrf_scores(docs, semantic_scores)` returns RRF of (Primo order, embedding cosine order).
- New env flag `SFU_FEATURE_RRF_ENABLED` (default off, on once benchmark confirms the win).
- When enabled, the RRF score replaces the `semantic_similarity` term in the weighted sum (RRF *is* the fusion of lexical + semantic, so token-overlap `title_relevance` weight redistributes accordingly).
- Tests in `src/tests/test_reranker.py` covering: RRF math correctness, fallback when no embedding scores, fallback when RRF flag off.

---

## Phase 6: Retrain on the cleaned dataset (gated on Phase 2 success)

Only worth doing if Phase 2 actually recovers ~3,800 triplets.

**Script:** `scripts/train_embedding_model.py` (already exists, no changes).

**Hyperparameters:** identical to v1 for a clean A/B comparison.

**Output:** `models/sfu-academic-embed-v2/`.

**Hardware note:** This is the long step. The container has no GPU passthrough so CPU training would take 15–20 hours. The plan calls for running this on the user's gaming PC (RTX). I will NOT kick off training in the container; I will leave a `scripts/run_training_v2.sh` ready and the user runs it on the host.

---

## Phase 7: Final benchmark + decision

Re-run Phase 4 with v2 in place of v1. Compare:

| Configuration | NDCG@10 expected (rough) |
|---|---|
| v1 only (today's baseline) | 0.5876 |
| v2 only (after dedup fix + retrain) | 0.62–0.66 |
| RRF(BM25, v1) | 0.61–0.65 |
| RRF(BM25, v2) | **target: ≥0.66** |

Decision:
- If RRF(BM25, v2) ≥ 0.66 → ship as `sfu-academic-embed-v2 + RRF`.
- If RRF(BM25, v2) < 0.66 → revisit base model (BGE-base instead of MiniLM) before further data work.

---

## What is explicitly out of scope

- **Tier 2/Tier 3 stacking (LambdaMART, learned blends)** — only worth building if RRF (Tier 1) shows the fusion approach has legs. Defer until Phase 4 results.
- **Scraping SFU full-text content** — license forbids bulk scraping; titles+abstracts via OpenAlex are sufficient.
- **Switching base model to BGE-base** — only revisit if Phase 7 shows MiniLM has plateaued.
- **Real query-log training** — the highest-quality signal but requires the model to be deployed first. Future work, not blocking.

---

## Success criteria

| Metric | Threshold | How measured |
|---|---|---|
| Cleaned triplets recovered | ≥ 3,500 (vs. 700 today) | wc -l on cleaned file |
| Subject coverage in clean data | ≥ 95% (vs. 71% today) | scripts/inspect_coverage.py |
| Provider coverage in training | ≥ 50% (vs. 14% today) | same |
| RRF NDCG@10 vs v1-only | ≥ +3% | scripts/evaluate_sfu_queries.py |
| Final NDCG@10 (v2 + RRF) | ≥ 0.66 (vs. 0.5876 today) | same |
| Production reranker tests | All passing | pytest src/tests/ |
