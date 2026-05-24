# Session Notes — 2026-05-24

A running record of work, findings, and decisions from this session, for picking up later. Companion to `LOCALIZED_DEPLOYMENT_PLAN.md`, `SPLADE_OPTIMIZATION_ROADMAP.md`, `MASTER_TODO.md`.

---

## 1. Headline outcomes

The Q3.1 hard-negative mining paid off in **two measured production wins**, both wired in:

| Model | Trained on | NDCG@10 | vs baseline | Status |
|---|---|---|---|---|
| **Cross-encoder v1** | query-doc hard-neg triplets | **0.7405** (MRR 0.7942) | **+9.5%** vs base 0.6763 | ✅ wired in `reranker.py` (with exists()-fallback to base) |
| **Embedder v5** | query-doc hard-neg triplets | **0.7717** (MRR 0.8375) | **+23.8%** vs v4 0.6233 | ✅ wired via `SFU_EMBEDDING_MODEL_PATH` |

Control that proves it's the *mined data*: training the cross-encoder on the **old citation-pair** triplets *regressed* (−0.016); the query-doc hard-neg data won both times. (Attribution caveat: the mined data changed *both* format (query-doc) and negatives (hard) vs the old data — no ablation isolating hard-negativity alone.)

---

## 2. Completed work

- **Bug backlog — all 12 items fixed + ~214 tests** (2 parallel agents, pushed). Highlights: filters dropped on local path, subject-routing wiring (heuristic detector), SPLADE serving rewired to local ONNX (offline), year-key mismatch, cross-encoder candidate-width, circuit-breaker over-count, `_push_to_opensearch` sparse_field/year, sync delta full-reindex, downloader crash-durability, + new data-pipeline test suite (was zero coverage).
- **Q3.1 hard-neg mining** — 9,932-record pool; re-run locally with **atomic writes** (temp→fsync→`os.replace`) added to `mine_hard_negatives.py` + `merge_negatives.py`. Reproduced exactly (commit `fe3051a`).
- **Query-doc triplets** — `scripts/build_ce_triplets.py` → 9,135 triplets / 120 queries (positives fetched from OpenSearch by judge-cache score≥2; negatives from the mined pool).
- **Q3.2 cross-encoder** — `train_cross_encoder.py`, trained + eval'd (`eval_cross_encoder.py`) + wired.
- **Embedder v5** — `train_embedding_model.py` re-fine-tuned from v4-bge on the triplets; eval'd (`eval_embedder.py`) + wired.
- **Q3.3 cloud SPLADE** — `finetune_splade.py` hardened: cost caps ($20/12h ceiling), auto-destroy droplet, checkpoint round-trips. Launch-ready behind doctl + Write-scope.
- **Hardened LOCAL SPLADE pipeline** — `scripts/run_splade_pipeline_local.sh` + `pipeline_state.py` (commit `59d01c0`): mine→triplets→fine-tune→re-encode→benchmark, **<10h** (~4-5h), stage-resume, **OOM auto-backoff** (smoke-verified 2.36 GB peak on 16 GB), 10h watchdog, verification gates. **NOT run** — user starts it manually. Caught 2 real bugs (ONNX/TRT cache keyed by sentinel→old-model reuse; stale `indexer_checkpoint.json` completed→resume would skip everything).
- **Dense-ANN POC — DONE** (`scripts/eval_pipeline.py`; results `data/eval_results/pipeline_dense_comparison.json`): 600K-doc `openalex_works_dense` index + 360 diverse paraphrased queries (120 keyword / 240 natural); A=RRF(bm25f+splade) vs B=+dense, **both through the full production rerank path** (embed v5 → cross-encoder v1), scored on the **LLM-judge cache**. Headline (FINAL reranked NDCG@10): the large retrieval-only recall gain (**+0.292 overall, +0.338 natural**) **survives reranking** at **+0.119 overall / +0.170 natural / +0.017 keyword** (A overall 0.429 → B 0.548; A natural 0.302 → B 0.471; A keyword 0.684 → B 0.701). Dense's win is **almost entirely on natural-language queries**; keyword is ~flat. ⚠️ Deltas are an **OPTIMISTIC upper bound** — dense is a 600K judged-coverage subset vs the 150M lexical index, and the judge cache is keyword-seeded (understates dense's NL edge). The *direction* (survives reranking; NL >> keyword) is robust; *magnitudes* need a full-scale dense index + NL-seeded judgments (see `LOCALIZED_DEPLOYMENT_PLAN.md` §9).
- **3 investigations + 1 research pass** — optimization audit, model-upgrade analysis, bug/test audit, alternative-technique research (top idea: the dense-ANN leg).

---

## 3. Key findings / corrections to the docs

- **Index is ~150M docs / 406 GB, NOT ~1M** as the roadmap assumed → reindex/re-encode is **~2.4 h local** (4070 Ti @ 17.4K docs/sec), not "11 min." Corpus is already recency-filtered to **~2015+** and recent-year-dense.
- **SPLADE serving model bug:** prod config pointed `splade_model_path` at `naver/splade-cocondenser-distil` (an un-cached HF id) — **not loadable offline**, and a mismatch with the indexed model. Fixed to use the local `models/splade_onnx` ONNX encoder.
- **`.env` had stale `-clone` paths**: `SFU_EMBEDDING_MODEL_PATH` pointed at a nonexistent `/workspaces/sfu-library-mcp-clone/...` dir (fixed → `-training`, now → v5). `PROJECT_NAME`/`PYTHONPATH` still `-clone` (vestigial; compose overrides PYTHONPATH).
- **7 pre-existing test failures** in `test_tools_unit.py` are a test-isolation bug (federated router bypasses the `_get_openalex` mock and hits the live local index), not a regression.
- **Hardware:** local RTX 4070 Ti SUPER (16 GB). SPLADE full fine-tune wants ~40–48 GB → on 16 GB needs the cut-down config (batch 2/grad-accum 16/seq 128/grad-checkpoint) or LoRA; verified it fits at 2.36 GB peak. Cross-encoder + embedder fine-tunes fit 16 GB easily.

---

## 4. Benchmarking system (current state + the fix needed)

- **Ground truth:** LLM judge cache (`data/eval_results/llm_judge_cache.json`), TREC 0–3, generated by **Claude Haiku** (`claude-haiku-4-5-20251001`), ~4,034 entries / 120 queries. Cached → reruns of seen pairs are free. Optional local Ollama (`qwen3:1.7b`) judge backend exists in `benchmark_llm_judge.py`.
- **Canonical harness family (consistent, offline-safe, local-ONNX SPLADE):** `eval_cross_encoder.py`, `eval_embedder.py`, `eval_dense_poc.py` — same cache, NDCG@10/MRR@10 (+Recall@50).
- **Outliers being retired:** `benchmark_llm_judge.py` (uses the stale HF SPLADE encoder → **fails offline**; also doesn't run the cross-encoder) and `ndcg_splade_eval.py` (**citation-count proxy** — superseded, ~2.5× undercount).
- **Task #21 — DONE (the swap):** `run_splade_pipeline_local.sh` stage 5 now calls `eval_pipeline.py` (LLM-judge harness, full rerank path) instead of `ndcg_splade_eval.py` (citation proxy). It query-encodes SPLADE with `models/splade_onnx_fp16` so the query encoder matches the model the index was just re-encoded with (stage 4 also now rebuilds `models/splade_onnx`, which it had been deleting without recreating — a latent query-side bug). `eval_pipeline.py` already toggles legs (A/B) + rerank path, covering most of the "one configurable comparator" goal; full standard+diverse parity across *every* workflow remains a nice-to-have.
- **No current end-to-end full-pipeline (retrieval→embed→CE) number exists yet** — the component evals above are the best current signals (~0.74–0.77 reranked).

---

## 5. Cost / time analysis

| Step | Time | Cost |
|---|---|---|
| Hard-neg mining (Q3.1) | ~5–10 min | $0 (local; reuses cached judge) |
| Build triplets | ~2–3 min | $0 |
| Cross-encoder / embedder fine-tune | ~2–4 min each | $0 (local 4070 Ti) |
| SPLADE fine-tune (cloud L40S $1.57/hr) | ~10–40 min + ~15–30 min setup | **~$5–8** (capped $20) |
| SPLADE fine-tune (local, OOM-safe/LoRA) | ~1–2 h | $0 (16 GB, fragile) |
| Re-encode 150M to deploy | ~2.4 h | $0 (local; mandatory regardless of where you train) |
| **Full SPLADE pipeline LOCAL** | **~4–5 h (<10 h)** | **$0** |

The only dollar cost in the whole chain is the cloud SPLADE fine-tune (~$5–8); everything else is free local. Re-encode can't be meaningfully offloaded to cloud (index is local → transfer overhead negates the gain).

---

## 6. Deployment architecture (see LOCALIZED_DEPLOYMENT_PLAN.md)

Four tiers from one codebase: **Full** (server, 406 GB) · **Subset** (laptop, ~20–40 GB year×citation bundle) · **Dynamic warm-cache** (self-building, recommended laptop default) · **Live-only** (OpenAlex + SFU rerank, no index/GPU). Retrieval (SPLADE/dense) needs an index; reranking (CE v1, embedder v5) does not. Warm-cache + Zotero/local-PDF seeding is the recommended low-footprint path — full design + prior art in the deployment plan doc.

---

## 7. Infra setup status
- **DigitalOcean:** token in `.env` (validated); L40S 48 GB @ $1.57/hr available; **Write-scope unconfirmed**; `doctl` not installed. GPU droplets region-restricted (use nyc2/tor1/atl1).
- **Azure:** SFU institutional tenant `1sfu` (tenant id `04e8677e-...`). Storage account `sfulibraryml4036007092` + key + container set in `.env`. Subscription `d6209381-...`. Service-principal **blocked** (tenant disallows app registration) → use interactive `az login` or ask SFU IT; `az` not installed. (Note: GPU blocked on student subs — Azure is CPU-only here.)
- Both `.env` blocks are untracked/gitignored (secrets stay local).

---

## 8. Open items / next steps
- **Tasks #13** (re-encode SPLADE on new snapshots — coordinate model + snapshot; **blocked** on OpenAlex snapshot access), **#19** (dense POC — **DONE**, see §2), **#21** (benchmark consistency — **DONE**, stage-5 swap), **#22** (this doc + warm-cache scope — done).
- **Tomorrow:** user runs `bash scripts/run_splade_pipeline_local.sh` — stage 5 is already swapped to the LLM-judge comparator (#21). Output `data/eval_results/pipeline_splade_eval_*.json`; compare its **config-A** numbers to the pre-reindex baseline (**overall 0.429 / keyword 0.684 / natural 0.302** final-reranked NDCG@10) to read off the SPLADE fine-tune's effect.
- **Decide:** dense-ANN go/no-go — the POC says **go for natural-language** (+0.170 NDCG@10 surviving rerank), but confirm at **full scale** + with **NL-seeded judgments** before productizing (LOCALIZED_DEPLOYMENT_PLAN §9). Also: laptop index engine, year×citation selector + `cited_by_count` indexing, warm-cache productization.
