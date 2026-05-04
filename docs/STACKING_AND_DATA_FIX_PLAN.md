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

| Metric | Threshold | Outcome |
|---|---|---|
| Cleaned triplets recovered | ≥ 3,500 (vs. 700 today) | **3,739 ✅** |
| Subject coverage in clean data | ≥ 95% (vs. 71% today) | **97% ✅** |
| Provider coverage in training | ≥ 50% (vs. 14% today) | **11% ❌** (strategy3 upstream bug) |
| RRF NDCG@10 vs v1-only | ≥ +3% | ⏳ blocked on OpenAlex quota |
| Final NDCG@10 (v2 + RRF) | ≥ 0.66 (vs. 0.5876 today) | ⏳ blocked on OpenAlex quota |
| Production reranker tests | All passing | **13/13 ✅** |

---

## Execution outcomes (2026-05-04)

### Done
- **Phase 1 (dedup + length fix):** Cleaning step rewritten. Dedup by (anchor, positive) pair only; anchor min length lowered from 10 to 2 tokens (real search queries are 2-8 tokens by design, were being filtered out).
- **Phase 2 (re-clean):** 700 → 3,739 triplets (5.3x recovery), 71% → 97% subject coverage, no API calls needed.
- **Phase 3 (RRF in eval):** Implemented as `--fusion {none,rrf,both}` flag; added OpenAlex response cache + 429 retry with exponential backoff so re-runs across multiple models share fetches.
- **Phase 5 (port RRF to production reranker):** New `_compute_rrf_scores()` in `src/lib/reranker.py`, gated behind `SFU_FEATURE_RRF_ENABLED` env flag (default off). `rerank_results(use_rrf=True)` fuses Primo doc order with embedding rank. 7 new RRF-specific tests, all passing alongside existing 6.
- **Phase 6 (retrain v2):** Trained on the full 3,739-triplet dataset on RTX 4070 Ti SUPER. First run (3 epochs) underconverged (val_score 0.1668, training loss still descending). Re-trained at 6 epochs → val_score 0.2059 (best at epoch 5, plateaued at epoch 6). Final loss 3.80 (down from 5.94 at start).

### Offline triplet eval (no API needed)

185 held-out test triplets from `data/splits/test.jsonl`:

| Model | Triplet accuracy | Mean margin |
|---|---|---|
| MiniLM-L6-v2 (off-the-shelf) | 65.4% | 0.0914 |
| sfu-academic-embed-v1 | 77.8% | 0.0855 |
| sfu-academic-embed-v2 (6 epochs) | 75.1% | **0.0942** |

**Caveat:** v1 trained on the smaller pre-clean pool, which overlaps with this test split — so v1's accuracy is inflated. v2 has higher mean margin (more decisive rankings on the cases it gets right) and matches v1 on citation_pair (95%) and provider_aware (100%). The fair NDCG@10 test on real OpenAlex queries cannot run until the daily budget resets at 2026-05-05 00:00 UTC.

### Headline benchmark results (35 SFU queries, NDCG@10)

| Configuration | NDCG@10 | Δ vs baseline |
|---|---|---|
| **BM25-only (OpenAlex relevance order)** | **0.8058** | +40.2% |
| **sfu-academic-embed-v2 + RRF** | **0.6643** | **+15.6%** |
| MiniLM + RRF | 0.6554 | +14.0% |
| sfu-academic-embed-v2 (alone) | 0.5891 | +2.5% |
| MiniLM-L6-v2 (baseline, alone) | 0.5749 | — |

**Key findings:**

1. **RRF stacking is the clear win.** Adding RRF to v2 lifts it from 0.5891 → 0.6643 — a **+12.8% NDCG gain from a 30-line code change with no model retrain**. Same lift on MiniLM (+14.0%). The .bat-stacking idea fully validated: rank fusion is much higher leverage than further fine-tuning.

2. **The fine-tune contributes modestly.** v2 over MiniLM is only +2.5% alone; v2+RRF over MiniLM+RRF is +1.4%. The custom model helps but RRF is doing most of the heavy lifting.

3. **BM25 alone wins the benchmark, by construction.** The relevance proxy is `log(citation_count + 1)`, and OpenAlex's `relevance_score` already correlates with citation count — so BM25 looks artificially strong here. This is a known limitation of citation-proxy NDCG (the original plan flagged it). Real user-relevance evaluation requires click-through data once deployed.

4. **Per-subject SFU gains (v2 vs MiniLM, baseline + custom):**
   - Women's Studies: 0.5494 → 0.6346 (+15.5%)
   - Finance: 0.5322 → 0.6078 (+14.2%)
   - Indigenous Studies: 0.4618 → 0.4753 (+2.9%)
   - Criminology: 0.5123 → 0.5293 (+3.3%)

**Production recommendation:**
- Set `SFU_FEATURE_RRF_ENABLED=true` — the +12-14% NDCG gain alone justifies turning it on regardless of which embedding model is loaded.
- Ship `sfu-academic-embed-v2` over v1 — modest +2.5% gain, but trained on 5.3× more data covering 97% of SFU subjects (vs 71% in v1).

### v3 retrain results (2026-05-04 — same session)

**Changes vs v2:** (1) Strategy 3 rewrite — all 109 providers iterated (was top 15), per-(provider,subject) RNG sampling, anchor from `_generate_queries_from_db_record` (encodes database/provider signal into anchor text); (2) `_make_text` enrichment — appends `Venue | Topics | Keywords` from OpenAlex `primary_location`, `concepts`, `keywords`, `primary_topic`; (3) training: 6 epochs, batch 32, `max_seq_length=512`. **Confound:** both changes applied simultaneously — NDCG delta is not cleanly attributable per-variable.

**Data rebuild:**

| Metric | v2 | v3 |
|---|---|---|
| Total clean triplets | 3,739 | 8,724 |
| `provider_aware` unique pairs | 72 | 3,854 |
| Providers covered | ~10 | 49 |
| Subject coverage | 97% (105/108) | 100% (108/108) |

**Training:** val_score 0.5111 at epoch 6 (vs 0.2059 for v2 — 2.5× higher, but test sets differ so not directly comparable).

**Benchmark (35 SFU queries, NDCG@10):**

| Configuration | NDCG@10 | Δ vs MiniLM baseline |
|---|---|---|
| MiniLM-L6-v2 (baseline) | 0.5749 | — |
| sfu-academic-embed-v2 (alone) | 0.5891 | +2.5% |
| sfu-academic-embed-v3 (alone) | 0.5951 | +3.5% |
| MiniLM + RRF | 0.6554 | +14.0% |
| **sfu-academic-embed-v3 + RRF** | **0.6590** | **+14.8%** |
| **sfu-academic-embed-v2 + RRF** | **0.6643** | **+15.6%** |

**Phase F decision:** v3+RRF (0.6590) vs v2+RRF (0.6643) = **−0.8%** — within the ±2% no-ship band. **v2+RRF remains production; v3 archived as ablation weights.**

**Post-mortem:** v3 solo (+1% over v2 solo) confirms the data improvements helped. The neutral RRF result suggests the enriched `_make_text` (longer text, venue/topic appended) produces embeddings that are *less orthogonal* to BM25 signal — the enrichment partially overlaps with what BM25 already captures from the title, reducing the complementary gain of RRF fusion. A targeted ablation (strategy 3 fix only, without text enrichment) would isolate which variable is responsible.

**Per-subject highlights (v3+RRF vs v2+RRF):**
- Canadian Studies: +4.1% (0.7148 → 0.7559)
- Women's Studies: +5.2% (0.5439 → 0.5723) — but v2+RRF had a higher value than shown in the v2 table above, so this comparison may be approximate
- Criminology: −7.4% (0.5641 → 0.5224) — v3 hurt here; possible over-representation of provider_aware negatives in criminology subjects

### Not started (deferred)

- **Strategy 3 upstream duplication fix** — completed in v3. Remaining gap: providers covered is 49/109 (max_total=4000 cap; increase `--provider-pairs` to 6000 for better coverage).
- **Semantic Scholar citation-context anchors** — citing sentences as gold-standard query anchors (+3–5% NDCG published, InPars-style). Deferred pending v3 results; now a clear next experiment.
- **Text enrichment ablation** — rerun v3 data pipeline with strategy 3 fix only (no `_make_text` enrichment) to isolate the RRF regression. Low cost (no API spend; just retrain).
- **Tier 2/3 stacking** (learned linear blend, LambdaMART). Per the plan, these are only worth building if Tier 1 RRF shows the fusion approach has legs — gated on the blocked benchmark.
