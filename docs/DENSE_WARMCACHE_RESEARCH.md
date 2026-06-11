# Query-Driven Dense Warm Cache — research + design (2026-06-11)

The "index that builds itself from what users request" line of research, re-traced
and turned into the v1 implementation in `src/lib/thinclient/dense_cache.py`.

## Where the idea comes from

**In-repo:** `docs/LOCALIZED_DEPLOYMENT_PLAN.md` §5 "Dynamic warm-cache" (design,
previously unimplemented): a read-through, usage-driven local index that caches
exactly what the user actually searches; §7 cites Frieder et al., *Caching
Historical Embeddings in Conversational Search* (arXiv 2211.14155, ACM TWEB 2023
— up to 75% hit rate from temporal locality). §8 decision #6 deferred wiring
dense into the warm-cache write path.

**Literature:** the classic lineage is **database cracking / adaptive indexing**
(Idreos et al., CIDR 2007; EDBT 2012; VLDB 2012) — indexes built incrementally as
a byproduct of query processing. Ported to vector search:

- **CrackIVF** (PVLDB 18, 2025, arXiv 2503.01823): answer with near-brute-force
  immediately, build IVF partitions progressively from the query distribution;
  10–1000× faster index initialization.
- **Quake** (arXiv 2506.03437): workload-adaptive partition maintenance.
- Maintenance-side: **FreshDiskANN** (two-tier fresh/static + background merge),
  **SPFresh** (SOSP 2023, in-place LIRE rebalancing), LSM-VEC.
- Risk literature: HNSW recall depends on insertion order — up to ~12.8 pp swing,
  query-correlated insert order is the bad case (arXiv 2405.17813).

## Verified facts (this container, usearch 2.25.3)

- usearch b1 incremental `add()` works on a mutable index; concurrent add+search
  from multiple threads: zero errors. `view=True` (how the serving seed is
  loaded) is **immutable**; duplicate keys **raise** — dedup is mandatory.
- CPU dense encode with `models/sfu-academic-embed-v5`: ~87 docs/s (batch 32).
- Bulk GPU encode of 150M ≈ 13.5 h on the RTX 4070 + ~84 GB artifacts; the warm
  cache costs ~0.6–1.7 s background CPU per query and ~0.5 GB per 1M cached docs,
  and converges to the *user's active domain* rather than the full corpus —
  it is the automatic answer to "which subset to ship".

## v1 design (implemented)

Two-tier delta, FreshDiskANN-style, with CrackIVF's answer-first philosophy:

- **Seed** = the existing static `dense/` artifacts (immutable mmap view +
  int8 memmap), untouched.
- **Delta** = `dense/delta.sqlite` (sole durability point: id, key, int8 vector,
  doc metadata) + in-RAM packed binary codes searched by **brute-force Hamming**
  (popcount LUT) + exact int8 rescore. Brute force sidesteps HNSW insert-order
  degradation entirely at delta scale (sub-ms below ~200k docs); scores are
  corpus-independent dot products, so seed and delta merge exactly by score.
- **Ingest**: every hydrated local result (BM25F/SPLADE/dense legs) is enqueued
  fire-and-forget; a single background worker dedups (seed ids + delta ids),
  CPU-encodes title+abstract in batches of 32, commits to SQLite, then appends
  to the RAM arrays. Bounded queue (drop-newest on overflow), hard doc cap.
- **Hydration**: delta rows carry their own title/doi/year/type/is_oa, so warm
  docs hydrate and post-filter even when absent from `meta.sqlite`.
- Gated by `SFU_DENSE_WARMCACHE` (default on; `0` disables).

### Deferred to v2

- usearch mutable delta index above the brute-force threshold.
- `merge_dense_delta(index_root)`: fold delta into the seed artifacts (bulk
  rebuild erases incremental graph debt), truncate delta.
- OpenAlex live-API ingest hook (`openalex.py:_push_to_opensearch` repurpose) —
  v1 captures the dominant path (local legs) only.
- ONNX-int8 export of the v5 encoder (2–3× CPU encode).

### Success metrics

1. Cross-query recall growth on a replayed query log (expect in-domain payoff
   after ~10–50 queries, per LOCALIZED §5 model).
2. Delta-vs-exact recall ≥ 0.97 (trivially exact in v1: brute force).
3. p95 `search_rrf` unchanged with cache on.
4. Cache-hit fraction of the dense leg vs query count.
