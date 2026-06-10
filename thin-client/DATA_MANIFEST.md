# Data artifacts to copy from the parent repo (`sfu-library-mcp-training`)

| Artifact | Parent path | Size | Needed for |
|---|---|---|---|
| Dense embedder v5 | `models/sfu-academic-embed-v5/` | ~135 MB | dense query encoding (384-dim) |
| SPLADE ONNX encoder | `models/splade_onnx/` | ~450 MB | SPLADE query/doc encoding |
| LLM-judge ground truth | `data/eval_results/llm_judge_cache.json` | ~10 MB | NDCG@10 eval parity |
| Diverse eval queries | `data/eval_results/diverse_queries.json` | ~1 MB | eval parity (360 queries) |
| Dense POC vectors (600k) | `data/dense_compression/vectors_600k.npy` + `ids_600k.json` | ~950 MB | dense-leg experiments without re-encoding |
| 100k POC corpus | `data/thin_client_poc/docs_100k.jsonl` | ~700 MB | engine experiments (incl. sparse_field weights) |
| Section archives | `data/sectioned/*.jsonl.zst` | ~930 MB total | hot/cold unpack experiments (1M docs, 5 sections) |
| Retriever reference | `src/lib/opensearch_retriever.py` | — | interface to mirror + `encode_splade()`/`encode_dense()` |
| RRF reference | `src/lib/federated_search.py` | — | fusion logic to reuse |

Corpus at scale: the full filtered OpenAlex snapshot lives in
`data/openalex_snapshot/works_part_*.jsonl.gz` (~15–30 GB compressed, 150M docs
with title/abstract/year/type; **no concepts/topics fields** — sectioning must
classify on title+abstract, or the snapshot downloader must be extended to keep
`concepts` before subject sectioning can use real metadata).

Eval methodology to reproduce (scripts in parent `scripts/`):
`eval_dense_compression.py`, `eval_lexical_lossless.py`, `eval_sectioned_index.py`,
`thin_client_poc.py` — all write JSON to `data/eval_results/`.
