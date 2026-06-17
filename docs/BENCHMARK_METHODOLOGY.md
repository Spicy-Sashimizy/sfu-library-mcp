# Benchmark Methodology: Old vs New

**Last updated:** 2026-05-16  
**Context:** Explains why the citation-count benchmark was replaced and how the LLM-judged method works.

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

## Running the parity eval OOM-safe at 150M (added 2026-06-17)

`scripts/eval_thinclient_parity.py` compares the thin-client stack against the
OpenSearch baseline. At 150M docs the thin-client serving set (~104 GB mmap) and
the OpenSearch 150M cluster cannot both stay hot on the 31 GB host, so the legacy
single-process mode (now `combined`, kept only for small indices) OOM-kills. Use
the record-then-replay split instead:

```bash
scripts/run_parity_safe.sh [QUERIES] [INDEX_ROOT] [BASELINE_URL]
# PAUSE_OS=1 also `docker pause`s OpenSearch during the thin-client phase
```

It runs each engine in its **own process** (the retriever has no `close()`, so a
fresh process is the only way to release the mmap working set), drops page cache
between phases, preflights `MemAvailable`, and aborts cleanly via an in-loop
mem-floor guard before the OOM-killer can fire. Phases:

1. `record-tc` — thin-client only (`SFU_DENSE_WARMCACHE=0`) → top-50 ids + latency
2. `record-os` — OpenSearch only (thin-client process already exited)
3. `compare` — pure offline join → the standard `thinclient_parity_*.json` summary

Because overlap is set math on id lists and NDCG uses the offline judge cache, no
comparison step needs both engines live. Peak RAM ≈ one engine, never the sum.
**150M parity numbers are UNMEASURED until the full run lands** — smoke-tested on
`data/thinclient_1m` only (schema-identical to the prior parity file).
