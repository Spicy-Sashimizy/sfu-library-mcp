# Thin-Client Swap — this container's architecture (2026-06-11)

This dev container is the **thin-client testbed**: OpenSearch is removed from
it entirely and the serving path runs the validated no-JVM stack. The original
`sfu-library-mcp` container (150M-doc OpenSearch 3.7 cluster at host port 9200)
is untouched and is used only as a **read-only export source and eval baseline**.

## What runs here now

| Leg | Engine | Config |
|---|---|---|
| Lexical BM25F | tantivy 0.26 | title^3 + abstract, `index_option='freq'` (positions-off lossless lever), fast-field year/type/is_oa filters, no stored text |
| Sparse SPLADE | bmp 0.2.6 | 8-bit block-max impacts, **bsize=256 + chunked top-term clustered insertion** (measured −54% size, −61% latency at equal recall), 2M-doc shards, weights ×70 quantization (saturation-free for SPLADE log1p ≤ 3.64), sidecar post-filtering |
| Dense | usearch 2.x | b1 Hamming mmap + int8 rescore memmap (binary+rescore 32×, validated R@10 0.996); coverage = dense-POC 600k until full corpus is encoded, **plus the query-driven warm cache below** |
| Dense warm cache | `lib/thinclient/dense_cache.py` | query-driven delta over the static seed (RAM Hamming + int8 rescore, SQLite durability, 500k-doc cap); `SFU_DENSE_WARMCACHE=0` disables (default on); design in `DENSE_WARMCACHE_RESEARCH.md` |
| Fusion | Python RRF k=60 | unchanged (`federated_search.py`) |
| Rerank | unchanged | abstracts via hot sidecar / OpenAlex fetch (below) |

Serving swap: `SFU_SEARCH_BACKEND=thinclient` (default) makes `tools.py`
construct `lib.thinclient.retriever.ThinClientRetriever` — a drop-in for
`OpenSearchRetriever` (same `search/dense_search/is_available` surface and
result shape). `opensearch` value restores the legacy backend (evals use it
against the original cluster).

Serving hardening (2026-06-11 evening): SPLADE query-encoder fallback (lexical
+ dense still serve if the encoder dies), per-leg latency metrics + JSONL query
log (`SFU_METRICS_LOG_PATH` / `SFU_QUERY_LOG_PATH`), and four index-management
MCP tools: `get_index_status`, `list_personas`, `request_section_unpack`,
`reload_index`.

## Hot/cold profile structure (fully implemented)

- Subject sections: `social_sciences`, `med_bio`, `phys_eng`, `cs_math`,
  `other` — priority-ordered disjoint title+abstract vocabulary classifier
  (`lib/thinclient/sections.py`, ported from the measured eval).
- **Era sub-sections (primary fast axis):** each subject section is BUILT as
  two date-range sub-sections, `<base>__recent` / `<base>__archive` (boundary
  publication_year 2010, routed at build time in the builder; the export spool
  stays base-keyed). Year-filtered queries prune the contradicting era at
  query time (`retriever._era_skip`).
- Personas (`PERSONAS`) are era-qualified: subject personas default to the
  recent era (`political_science` → `social_sciences__recent`) with
  `*_historical` variants for the archive era, plus era-wide `contemporary` /
  `historical` and the testbed `all_hot`. Hot sections stay live **with a
  zstd-dictionary abstract sidecar**; cold sections are packed.
- **Packing operates on built artifacts** (`lib/thinclient/packer.py`,
  tar+zstd-19 LDM): unpack = pure decompression at disk speed with SHA-256
  artifact parity — replacing the old rebuild-from-JSONL flow that ran at
  2,016 docs/s (~2.4 h per section at 150M). Archives are kept after unpack,
  so re-packing an unwritten section is just deleting the live dir.
- Abstract policy (measured: extern_display is ranking-lossless): hot = local
  sidecar; cold = OpenAlex API mget before rerank. Never title-only rerank.
  Sidecar format is **v3**: script-bucketed (latin/cyrillic/han/…) clustered
  32 KB blocks, zstd-19 with per-script trained dicts (measured 64.7→52.9 GB
  at 150M); readers transparently handle v1 (per-doc) and v2 (+ru dict).
- `meta.sqlite` is **v2**: W-ids as INTEGER PRIMARY KEY with a TEXT overflow
  table (measured ~10 GB at 150M vs 24 GB v1); the retriever auto-detects
  v1/v2 schemas.
- CLI: `python -m lib.thinclient.packer {pack|unpack|repack|status} <root> [section]`.

## Index layout

```
data/thinclient_index/
  manifest.json        provenance, persona, section states, checksums
  build_status.json    resumable build checkpoint
  meta.sqlite          v2: int-PK id -> title/doi/year/type/is_oa/section
  sections/<base>__<era>/   tantivy/ + splade_NNN.bmp (+vocab.zst sidecars)
                            + abstracts.sqlite (hot)
  packed/<base>__<era>.tar.zst
  dense/               b1.usearch + rescore_int8.npy + ids.json
  dense_cache/         query-driven warm-cache delta (SQLite + int8)
```

## Migration (150M docs, kept)

`scripts/build_thinclient_index.py` streams the corpus out of the original
cluster's `openalex_works` (sliced scroll; `_source` still carries the
production `sparse_field` weights, so **no re-encoding**), classifies sections,
and builds all artifacts checkpointed (export/build/pack/dense phases).
Monthly rebuilds later: snapshot JSONL → `lib/thinclient/doc_encoder.py`
(the GPU TRT batch SPLADE encoder, extracted from the retired indexer) →
same builder.

Operational helpers around the running 150M migration:
- `scripts/wait_and_resume_migration.sh` — polls the source cluster's `_count`
  and auto-launches the build once ≥150M docs are stable (used when the
  source cluster goes down mid-export).
- `scripts/restart_at_build_phase.sh` — armed to restart the run at the
  export→build boundary so the BUILD phase picks up post-launch builder code
  (era routing, abstracts v3, meta v2, b256 clustering).

Build RAM (measured 2026-06-12): the era split runs TWO builders per section
worker, and `--build-workers 3` OOM-killed the 150M BUILD on the 31 GB host
(`BrokenProcessPool` ~1 min in, swap 5 GB deep). Defaults are now
`--build-workers 2`, `BMP_CLUSTER_CHUNK_DOCS 250k`, `TANTIVY_WRITER_HEAP
512 MB` (~6-8 GB/worker). Chunk halving keeps the measured 256-doc-block
locality win; smaller tantivy heap only means more segment flushes.

Validation: `scripts/eval_thinclient_parity.py` (per-leg overlap vs baseline,
LLM-judged NDCG@10, latency) and the permanent pytest suite
`scripts/tests/test_thinclient_stack.py` (13 tests: meta v1/v2, era routing +
pruning, abstracts v1/v2/v3, dense cache, metrics/reload, MCP index tools).
Storage research: `docs/LEXICAL_STORAGE_RESEARCH.md`,
`docs/STORAGE_BUDGET_150M.md`, `scripts/eval_text_compression.py`.

## Removed from this container

Training (SPLADE finetune, embedding/CE/LambdaMART trainers, miners,
generators), OpenSearch indexing pipeline (splade_indexer, opensearch_sync,
reindex shells, watchdogs), cloud GPU offload, docker/opensearch templates,
the devcontainer OpenSearch service+volume, training data and superseded
checkpoints (embed v2/v3, sfu-splade-v1). Kept: all eval/benchmark harnesses,
dense ANN code + artifacts, snapshot downloader + 304-part snapshot, serving
path, query-side encoders.

Host-side cleanup after container recreate:
`docker rm -f claudebox-sfu-library-thinclient-opensearch &&
docker volume rm claudebox-sfu-library-thinclient-opensearch`
