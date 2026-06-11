# Thin-Client Swap — this container's architecture (2026-06-11)

This dev container is the **thin-client testbed**: OpenSearch is removed from
it entirely and the serving path runs the validated no-JVM stack. The original
`sfu-library-mcp` container (150M-doc OpenSearch 3.7 cluster at host port 9200)
is untouched and is used only as a **read-only export source and eval baseline**.

## What runs here now

| Leg | Engine | Config |
|---|---|---|
| Lexical BM25F | tantivy 0.26 | title^3 + abstract, `index_option='freq'` (positions-off lossless lever), fast-field year/type/is_oa filters, no stored text |
| Sparse SPLADE | bmp 0.2.6 | 8-bit block-max impacts (bsize=32), 2M-doc shards, weights ×100 quantization, sidecar post-filtering |
| Dense | usearch 2.x | b1 Hamming mmap + int8 rescore memmap (binary+rescore 32×, validated R@10 0.996); coverage = dense-POC 600k until full corpus is encoded |
| Fusion | Python RRF k=60 | unchanged (`federated_search.py`) |
| Rerank | unchanged | abstracts via hot sidecar / OpenAlex fetch (below) |

Serving swap: `SFU_SEARCH_BACKEND=thinclient` (default) makes `tools.py`
construct `lib.thinclient.retriever.ThinClientRetriever` — a drop-in for
`OpenSearchRetriever` (same `search/dense_search/is_available` surface and
result shape). `opensearch` value restores the legacy backend (evals use it
against the original cluster).

## Hot/cold profile structure (fully implemented)

- Sections: `social_sciences`, `med_bio`, `phys_eng`, `cs_math`, `other` —
  priority-ordered disjoint title+abstract vocabulary classifier
  (`lib/thinclient/sections.py`, ported from the measured eval).
- Personas (`PERSONAS`): political_science / computer_science / health_science /
  interdisciplinary_cogsci / all_hot. Hot sections stay live **with a
  zstd-dictionary abstract sidecar**; cold sections are packed.
- **Packing operates on built artifacts** (`lib/thinclient/packer.py`,
  tar+zstd-19 LDM): unpack = pure decompression at disk speed with SHA-256
  artifact parity — replacing the old rebuild-from-JSONL flow that ran at
  2,016 docs/s (~2.4 h per section at 150M). Archives are kept after unpack,
  so re-packing an unwritten section is just deleting the live dir.
- Abstract policy (measured: extern_display is ranking-lossless): hot = local
  sidecar (~0.5 KB/doc); cold = OpenAlex API mget before rerank. Never
  title-only rerank.
- CLI: `python -m lib.thinclient.packer {pack|unpack|repack|status} <root> [section]`.

## Index layout

```
data/thinclient_index/
  manifest.json        provenance, persona, section states, checksums
  build_status.json    resumable build checkpoint
  meta.sqlite          id -> title/doi/year/type/is_oa/section
  sections/<name>/     tantivy/ + splade_NNN.bmp + abstracts.sqlite (hot)
  packed/<name>.tar.zst
  dense/               b1.usearch + rescore_int8.npy + ids.json
```

## Migration (150M docs, kept)

`scripts/build_thinclient_index.py` streams the corpus out of the original
cluster's `openalex_works` (sliced scroll; `_source` still carries the
production `sparse_field` weights, so **no re-encoding**), classifies sections,
and builds all artifacts checkpointed (export/build/pack/dense phases).
Monthly rebuilds later: snapshot JSONL → `lib/thinclient/doc_encoder.py`
(the GPU TRT batch SPLADE encoder, extracted from the retired indexer) →
same builder.

Validation: `scripts/eval_thinclient_parity.py` (per-leg overlap vs baseline,
LLM-judged NDCG@10, latency). Storage research + tests:
`docs/LEXICAL_STORAGE_RESEARCH.md`, `scripts/eval_text_compression.py`.

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
