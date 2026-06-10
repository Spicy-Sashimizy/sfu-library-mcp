# SFU Library Thin Client — Development Workspace

Self-contained handoff for building the **no-JVM laptop search stack**:
`tantivy-py` (BM25F) + `pyseismic-lsr` (SPLADE) + `usearch`/`faiss` (dense,
binary+rescore), fused with app-side RRF — replacing OpenSearch on consumer
devices while the 150M-doc server stays on OpenSearch (upgrading to 3.x).

## Why this stack (evidence, all measured in the parent repo)

| Finding | Where |
|---|---|
| OpenSearch lexical index is 274 GB; **48% lossless reduction verified** (zstd6 + `_source` minus sparse_field + positions-off + dup-field-off), 80/80 query parity | `docs/COMPRESSION_EVAL_RESULTS.md` |
| Dense leg: **binary+rescore holds quality at 32×** (3-leg ΔNDCG +0.0008); int8 free at 4× | `docs/COMPRESSION_EVAL_RESULTS.md` |
| Engine research: keep OpenSearch server-side; **leave it on laptops** (JVM floor). Best laptop stack = tantivy + seismic + usearch | `docs/SEARCH_ENGINE_ALTERNATIVES.md` |
| Hot/cold sectioning: home section live + others packed ≈ **29–31% extra saving**, lossless unpack parity, ~2× pack ratio | `docs/THIN_CLIENT_STACK_RESEARCH.md` (results §) and parent `data/eval_results/sectioned_index_eval.json` |
| Hands-on POC of this exact stack on 100k real docs | `prototype/thin_client_poc.py`, results in `docs/THIN_CLIENT_STACK_RESEARCH.md` |

## Deployment tiers this workspace targets

From `docs/LOCALIZED_DEPLOYMENT_PLAN.md`: the **subset tier** (~10–15M docs) and
**dynamic warm-cache tier** run locally. At 15M docs the lexical artifact is
~14 GB (with the lossless config) and dense binary codes ~0.7 GB; per-persona
hot/cold sectioning brings steady-state to ~16 GB lexical-equivalent.

## What to build here

1. `retriever.py` — port of `src/lib/opensearch_retriever.py`'s interface
   (same return shape) backed by tantivy + seismic + usearch. RRF fusion is
   already app-side in the parent repo (`federated_search.py`) — keep that.
2. Index builder — monthly batch artifact: corpus JSONL → tantivy index +
   seismic index + usearch binary index (+ fp16/int8 rescore matrix on disk).
3. Section pack/unpack (hot/cold) — see `prototype/thin_client_poc.py` and the
   sectioned eval in the parent repo (`scripts/eval_sectioned_index.py`).
4. Eval parity — wire to the parent repo's NDCG@10 harness
   (`data/eval_results/llm_judge_cache.json` + `diverse_queries.json`) before
   trusting any engine swap.

## Getting started

```bash
pip install -r requirements.txt
# copy data artifacts listed in DATA_MANIFEST.md from the parent repo, then:
python prototype/thin_client_poc.py --docs 100000
```

## Key facts for the retriever port

- BM25F = `multi_match most_fields` over title^3/abstract — in tantivy:
  boolean SHOULD of per-field parsed queries with `Query.boost_query(q, 3.0)`.
- SPLADE query encoding: production ONNX encoder (`models/splade_onnx`,
  `encode_splade()` in the parent retriever) — identical on laptop.
- Seismic has **no native filters** — post-filter or partition indexes by
  year-bucket/section; tantivy has fast-field range filters (year) natively.
- Dense: 384-dim L2-normalized (sfu-academic-embed-v5); binarize sign(x),
  search hamming, rescore top-4k candidates with fp32/fp16 from an mmapped
  matrix (validated recipe: R@10 ≈ 0.996 vs exact at 32× compression).
- Seismic indexes are **static** — rebuilt monthly, which matches the OpenAlex
  snapshot cadence. The warm-cache tier needs an updatable sparse fallback
  (brute-force dot product over the small cache is fine).
