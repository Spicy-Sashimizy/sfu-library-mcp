# Benchmark Methodology: Old vs New

**Last updated:** 2026-06-18  
**Context:** Explains why the citation-count benchmark was replaced and how the LLM-judged method works; plus the 150M thin-client-vs-OpenSearch parity method and measured results (see last section).

---

## The Problem: Citation Count Was Marking Its Own Homework

The original benchmark (`scripts/ndcg_splade_eval.py`, result at `data/eval_results/ndcg_splade_eval_20260515_0420.json`) used **OpenAlex `cited_by_count`** as the relevance signal.

### How it worked (old method)

1. Run query against local OpenSearch → get 10 results from BM25F, SPLADE, RRF
2. For each returned paper, call the OpenAlex API to fetch its `cited_by_count`
3. Treat citation count as a relevance score (higher citations = more relevant)
4. Compute NDCG@10 from those citation scores

### Why this was wrong

**Bug 1 — Fame ≠ fit.**  
Citation count measures how famous a paper is globally, not how relevant it is to the query. A 1988 paper with 5,000 citations is not more relevant to "Northern Gateway pipeline environmental assessment BC" than a 2022 paper with 40 citations that directly addresses it. SFU's niche subjects (Indigenous Studies, Labour Studies, Canadian Studies, Latin American Studies) structurally produce papers with low citation counts — they are small disciplines. SPLADE was correctly finding relevant papers and being penalized for returning them.

**Bug 2 — search_by_topic was tautological.**  
`search_by_topic` sorts OpenAlex results by `cited_by_count:desc`. If you then grade performance by `cited_by_count`, you are measuring whether the sort worked — not whether results are topically relevant. OpenAlex's own sort scores NDCG ≈ 1.0 by definition. Any local retriever that returns a different ordering (finding more relevant but less famous papers) scores lower. This is textbook circular evaluation.

### The numerical evidence

Same 120 queries, same retrieval methods. Only the grading signal changed:

| Method | Old NDCG (citation proxy) | New NDCG (LLM judge) | Ratio |
|--------|--------------------------|---------------------|-------|
| SPLADE mean | 0.2586 | 0.6534 | **2.5× understated** |
| RRF mean | 0.2724 | 0.6362 | **2.3× understated** |
| Old max score | 0.6569 | 1.0000 | ceiling was capped |

**6 queries scored below 0.05 NDCG under the old method.**  
When the LLM judged those same queries for topical relevance, average score was **0.817**.  
The most extreme case: `Canadian documentary film Indigenous representation` scored **0.000** (papers not famous enough) but **0.986** (papers perfectly on-topic).

---

## The New Method: TREC-Style LLM Judging

**Script:** `scripts/benchmark_llm_judge.py`  
**Judge:** `claude-haiku-4-5-20251001` via `claude -p` subprocess (no API key needed — uses Claude Code session)  
**Results:** `data/eval_results/benchmark_llm_judge_final.json`  
**Report:** `data/eval_results/benchmark_report_final.md`

### How it works

1. **Retrieve** top-10 results from all four methods per query: BM25F, SPLADE, RRF, OpenAlex relevance sort
2. **Pool** — merge unique papers across all methods into one pool of ~25-30 papers per query (TREC-style pooling so no method is disadvantaged)
3. **Judge** — Claude Haiku reads each paper's title + abstract and scores it 0-3:
   - 3 = perfectly relevant (directly answers the query)
   - 2 = highly relevant (strongly related)
   - 1 = marginally relevant (tangential)
   - 0 = not relevant
4. **Cache** — all judgments written to `data/eval_results/llm_judge_cache.json` — reruns are free
5. **Score** — NDCG@10, MRR@10, P@10(≥2) computed per method using the Haiku scores as ground truth

### Why this beats citation count

- **Independent judge:** Haiku cannot favor any retriever because it sees papers without knowing which method returned them
- **Topical relevance, not fame:** Haiku reads the actual content and decides if it answers the query
- **No tautology:** The judge is completely decoupled from the sort logic of any retriever
- **Works for niche subjects:** Haiku can assess relevance of a 2023 paper on Musqueam land rights just as well as a 2001 paper on protein folding

### Performance

- 4 parallel subprocess workers → ~5-8 min for all 120 queries (cold)
- Cached reruns → ~1-2 min (cache hit rate ~100%)
- Cost: $0 additional — uses Claude Code's existing session

---

## Key Takeaway

The old benchmark made local search *look* worse than it was by 2.3-2.5×. The LLM-judged benchmark reveals that SPLADE and RRF are finding highly topically relevant papers on 32 of 34 subjects with local index coverage. The low mean NDCG (0.63-0.65) is caused by 15 zero-coverage subjects scoring 0.0, not by retrieval failing on covered subjects.

---

## Running the parity eval at 150M (the RAM wall + how to get numbers anyway)

`scripts/eval_thinclient_parity.py` compares the thin-client stack against the
OpenSearch baseline. The hard constraint at 150M is RAM, and the cause is
specific: **BMP (the SPLADE engine, 0.2.6) has no mmap mode** — `bmp.Searcher`
deserializes each `*.bmp` shard into **anonymous RAM at a measured 3.07×** its
on-disk size (probe 2026-06-17: 569 MB shard → 1747 MB resident; ratio holds
12 MB–569 MB). The full 150M SPLADE leg is **68.6 GB on disk → ~211 GB resident**
(NOT the "~104 GB mmap" earlier docs claimed — that 104 GB is on-disk total;
tantivy/`meta.sqlite`/dense are genuinely mmap/paged and stay ~0 resident). So
even one engine cannot be held in a single process on the 24 GB host (capped via
`.wslconfig` 2026-06-17), let alone both.

`retriever._load()` now samples `MemAvailable` before each shard and aborts with
a clean `RuntimeError` (gate `SFU_LOAD_MEM_FLOOR_GB`, default 1.5) instead of
being SIGKILLed mid-construction. The legacy `combined` single-process mode is
refused by default for the same reason.

### Section-shard-wave eval — gets 150M numbers on the 24 GB host (added 2026-06-17)

`scripts/eval_parity_section_waves.py run` produces a record-tc-schema file
without ever holding the SPLADE set in one process. It exploits the fact that the
legs **already merge across sections/shards by plain score-concatenation**:

- **BM25F** runs on tantivy (mmap, ~0 resident) in one pass via the retriever's
  `SFU_SKIP_BMP=1` flag (load tantivy/meta/dense, skip BMP). Output is identical
  to `record-tc`'s bm25f field (same code path).
- **SPLADE/BMP** scores are corpus-independent dot products, so each shard's
  top-K is recorded in a **fresh process per wave** (the only way to free BMP
  RAM — the retriever has no `close()`), and merged offline. First-fit-decreasing
  bin-packs shards into waves under `--resident-budget-gb` (default 8).

This is **exact, not approximate** — validated 2026-06-17 on `data/thinclient_1m`:
wave-merged output is byte-for-byte identical to a direct full `record-tc` on both
legs, all queries. The OpenSearch side is recorded separately (`record-os`, its
own container, low RAM) and joined by `compare` (pure offline set-math + judge
cache, no engine live). Time-, not RAM-, bound: it reads the 68.6 GB BMP set once.

```bash
# thin-client record on a small host (≈43 min at 150M, 27 waves, 8 GB budget):
scripts/eval_parity_section_waves.py run --index-root data/thinclient_index \
    --queries 40 --resident-budget-gb 8 --output <tc_record.json>
# OpenSearch baseline (separate process):
scripts/eval_thinclient_parity.py record-os --baseline-url <url> --queries 40 --output <os_record.json>
# join → summary:
scripts/eval_thinclient_parity.py compare --tc-record <tc_record.json> --os-record <os_record.json> --output <summary.json>
```

The `run_parity_safe.sh` record-then-replay split (one engine per process, page-
cache drops, `MemAvailable` preflight, in-loop mem-floor guard) remains the path
**once the host can hold one engine** (~232 GB) — at 24 GB its `record-tc` still
OOMs in `_load()`, so use the wave eval instead.

### MEASURED 150M parity (2026-06-18, first full-corpus run)

40 diverse queries, LLM-judged NDCG@10 (same judge cache + method as above),
thin-client (full `data/thinclient_index`) vs the 150M OpenSearch baseline.
Thin-client recorded via section-shard-waves (27 waves, 8 GB budget, ~43 min,
min RAM 15.4 GB available, peak swap 2.1 GB — no OOM).

**NDCG@10 on the common judged set** (34 queries with a judged doc in *both*
engines — apples-to-apples, equal denominator):

| Metric | Thin-client (150M) | OpenSearch (150M) | Δ |
|---|---|---|---|
| RRF NDCG@10 | **0.561** | 0.449 | **+0.112** (tc better) |
| Per-query wins | **29 / 34** | 5 / 34 | tc wins 85% |

Overlap@50 vs OpenSearch baseline (agreement, not quality — different engines):
bm25f **0.532**, splade **0.399**, rrf **0.476**.

As-reported means (unequal denominators, for the record): tc_rrf 0.537 (39 judged
queries) / os_rrf 0.449 (34); recomputed on the common 34 for fairness above.
NDCG coverage is judge-cache-limited (34/40 queries gradable). **Latency is NOT
comparable** across these runs (different hardware; tc splade latency is
reconstructed encode+search, hydration excluded) — reported in the summary for
completeness only, not as a head-to-head.

Records: `data/eval_results/thinclient_parity_waves_150m.json` (summary),
`parity_record_tc_waves_150m.json` (tc), `parity_record_os_20260618_0028.json`
(os). Full architecture context: `THIN_CLIENT_SWAP.md`; serving-RAM math:
`STORAGE_BUDGET_150M.md`.
