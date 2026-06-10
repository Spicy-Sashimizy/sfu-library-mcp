# Should the stack leave OpenSearch? (researched 2026-06-10)

**Question:** would moving off OpenSearch yield quality, efficiency, or storage gains?
**Answer:** No for the 150M server tier (upgrade to OpenSearch 3.x instead). Yes for the
laptop thin-client tier (no JVM on consumer machines; use static rebuilt artifacts).

Stack requirements used as criteria: BM25F (`multi_match most_fields`), SPLADE as
`rank_features` (top-256 terms/doc, per-term `rank_feature` bool/should), planned 384-dim
HNSW dense leg (binary+rescore 32× validated in `docs/COMPRESSION_EVAL_RESULTS.md`),
year/type/is_oa filters, app-side RRF (Python), monthly upserts by `_id`, scroll/export.

## Per-alternative verdicts

| Option | BM25F | SPLADE | Dense | Single-node RAM | Migration | Laptop fit |
|---|---|---|---|---|---|---|
| **OpenSearch 3.x (stay+upgrade)** | identical | identical + two-phase + Seismic RFC | derived source (−3× vector `_source`), disk-based BQ | JVM floor stays | near-zero | poor |
| Elasticsearch 9.x | identical | identical | BBQ GA (fast) | same JVM | very low | poor; synthetic `_source` is Enterprise-tier |
| **Qdrant** | **lost** (BM25-as-sparse only) | exact dot product, native sparse | native + quantization | sparse must be on_disk at 150M (~300 GB in-RAM otherwise) | moderate | good |
| Milvus 2.5 | partial | native (WAND/MAXSCORE) | native | heavy (etcd+MinIO) | mod-high | poor |
| Tantivy / tantivy-py | near-identical | **not implementable from Python** (no FeatureField equivalent) | none | tens of MB | moderate | excellent (lexical only) |
| Embedded Lucene | identical | identical (FeatureField IS rank_features) | identical | few hundred MB heap | **high** (own a Java service / PyLucene) | moderate |
| SEISMIC (pyseismic-lsr) | n/a | exact, ~90% recall, 1–2 orders faster (SIGIR'24) | n/a | ~8 GB / 9M docs | low (sidecar) | excellent for subset tiers; static index only |
| Vespa | equivalent | exact (tensors) | native | ≥4–8 GB floor | very high (full rewrite) | no |
| SQLite FTS5 / DuckDB | weak | none | none | tiny | low | degraded fallback only |
| Quickwit | n/a | n/a | n/a | — | — | wrong shape: append-only logs, expensive deletes; team acquired by Datadog |

## Key findings

- **Quality:** no engine improves ranking math — BM25/SPLADE/cross-encoder are the quality,
  and they're engine-portable. Risks run the other way (Tantivy has no rank_features
  equivalent; Qdrant has no true BM25F; weight-into-term-frequency hacks change BM25 math).
- **Storage:** Lucene FOR/PFOR postings are near-SOTA among production engines; no
  production alternative beats the measured lossless config (~142 GB) materially.
- **Efficiency:** the real OpenSearch tax is the JVM + cluster machinery on small machines.
  The "Tantivy 2× faster than Lucene" headline is heavily caveated (single-field, 5M docs,
  page-cache-resident; see jpountz's analysis).
- **The SPLADE speed/efficiency wins are landing inside OpenSearch:** two-phase neural
  sparse exists in 2.15+ (≈up to 9.8× speedup, <0.04% relevance loss — usable against the
  existing rank_features field TODAY), and SEISMIC is RFC'd into k-NN with a PoC showing
  P50 17.5× / P99 35.9× at recall 99.7%→92.4% (k-NN#2715).

## Recommendations

1. **Server tier: stay; plan an OpenSearch 2.19 → 3.x upgrade** (derived source for
   vectors, disk-based binary quantization matching the validated 32× recipe, GPU index
   builds, Lucene 10, Seismic path).
2. **Laptop tier: leave OpenSearch.** Best stack: `tantivy-py` (BM25F) +
   `pyseismic-lsr` (SPLADE, static monthly-rebuilt index — matches the update cadence) +
   faiss/usearch (dense, binary+rescore), fused by the existing Python RRF. No JVM,
   tens-of-MB idle RAM. Runner-up: Qdrant local mode (one engine: sparse+dense+filters+
   upserts) if dropping the BM25F leg survives an NDCG ablation.
3. **Cheapest high-information experiment:** enable the two-phase neural sparse processor
   on the current 2.19 index and re-run the NDCG@10 harness — tests most of the SPLADE
   speedup with zero migration.

Full citations in the research transcript; load-bearing sources:
jpountz.github.io/2025/05/12/analysis-of-Search-Benchmark-the-Game.html ·
github.com/quickwit-oss/tantivy-py · lucene.apache.org FeatureField javadoc ·
qdrant.tech/articles/sparse-vectors/ · github.com/tuskanny/seismic (arXiv:2404.18812) ·
opensearch.org/blog/Introducing-a-neural-sparse-two-phase-algorithm/ ·
github.com/opensearch-project/k-NN/issues/2715 · docs.vespa.ai/en/vespa-quick-start.html ·
elastic.co/search-labs/blog/elasticsearch-bbq-vs-opensearch-faiss
