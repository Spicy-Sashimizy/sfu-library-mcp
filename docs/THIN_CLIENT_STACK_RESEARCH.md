# Thin-Client Search Stack — Research + Measured POC

**Date:** 2026-06-10 · **Question:** can `tantivy-py + pyseismic-lsr + usearch/faiss`
replace OpenSearch on consumer devices (no JVM, minimal RAM)?
**Answer: yes — architecture validated hands-on; one substitution recommended
(BMP instead of Seismic as the default SPLADE leg for end-user machines).**

## Measured POC (this repo, real data — `scripts/thin_client_poc.py`)

100,000 real docs from the production corpus (title/abstract + production SPLADE
`sparse_field` weights), 20 eval paraphrases, queries encoded with the production
ONNX SPLADE encoder. Results (`data/eval_results/thin_client_poc.json`):

| Engine | Build | Index size | Latency | Notes |
|---|---|---|---|---|
| tantivy 0.26 (BM25F: title^3 + abstract, bool-should) | 19.4 s | 65 MB | 8.0 ms/q | year range filter works natively |
| seismic 0.5 (SPLADE, default build params) | 49.5 s | 315 MB | 4.4 ms/q | no filters; index is RAM-resident |
| usearch 2.25 b1 binary + fp32 rescore (dense) | 1.6 s | 20 MB | 0.34 ms/q | **recall@10 = 0.975 vs exact** |
| RRF fusion (production k=60) | — | — | — | works; lexical∩sparse overlap@50 = 6.8 (legs are complementary) |

### Head-to-head vs OpenSearch (same 100k corpus, same 20 BM25F queries, matched role)

A lexical-only OpenSearch index was built with identical content to the tantivy index
(BM25 title/abstract + filter fields, no SPLADE postings, no stored text):

| | OpenSearch 2.19 (1 shard, force-merged, zstd) | tantivy 0.26 (mmap) |
|---|---|---|
| Index size | 209 MB¹ | **65 MB** |
| BM25F latency (top-50) | 19.7 ms/query² | **4.4 ms/query** |
| Process RAM | 2,048 MB heap (configured) + off-heap | **45 MB peak RSS** |

¹ OpenSearch copy still carries `doi`/`id` dynamic fields in `_source` (~tens of MB).
² Includes HTTP+JSON transport (inherent to its architecture; tantivy is in-process);
measured while the cluster had background reindex load, so treat as indicative —
the RAM column is the structural difference that matters for the thin client.

Extrapolation to a 15M-doc laptop tier: tantivy ~10 GB (shrinkable: positions off,
no stored text → metadata in SQLite), sparse leg ~9–13 GB RAM if Seismic
(disqualifying) vs single-GB-scale with BMP, dense ~3 GB on disk (0.7 GB hot).
The whole stack starts in seconds and idles at tens of MB plus mmapped pages —
versus OpenSearch's multi-GB JVM floor.

## Key research findings (full sources in the research transcript)

### tantivy-py 0.26 — production-ready lexical leg
- `parse_query(..., field_boosts={"title": 3.0})` gives multi_match-style BM25F directly.
- Native filters: integer fast-field range queries (year), term queries (type/is_oa).
- `index_option='freq'` drops positions (same lossless trick we validated on
  OpenSearch: 8.6% saved, parity-identical) — set it for abstract/title.
- mmap-served, <10 ms open, near-zero resident RAM. Wheels: win/mac(arm64)/linux.
- Confirmed limitation: NO custom similarity from Python — SPLADE cannot ride on
  tantivy (term-repetition hack distorts scoring; unfixable from Python).

### SPLADE leg — BMP primary, Seismic optional power-user backend
- **pyseismic-lsr 0.5**: fastest published algorithm (SIGIR'24; 185 µs/q on 8.8M docs)
  and it worked first-try on our data, BUT: PyPI ships ONE wheel per release,
  Linux-x86_64 single-ABI only (never Windows, never mac arm64) — you'd build and
  ship 6 wheels yourself; index is fully RAM-resident (~7.9 GB @ 8.8M MS MARCO →
  ~9–13 GB @ 15M); no filters; load-at-start = 5–15 s cold start.
- **BMP 0.2.6 (`pip install bmp`, pisa-engine, SIGIR'24)**: block-max pruning over
  8-bit quantized impacts; 2.9–7.5× faster than nearest competitor on SPLADE;
  **real Windows wheels**, 0.6 MB; much smaller RAM. Input is CIFF (one-time
  SPLADE-vectors→CIFF converter needed — see thin-client workspace TODO).
  `Searcher("idx.bmp").search({"token": w, ...}, k=10)` — same dict shape our
  encoder already produces. Gap: build mac-arm64/linux-aarch64 wheels in CI.
- Fallbacks ranked: BMP → self-built Seismic wheels (Linux power users) →
  brute-force sparse over the warm cache only. pyterrier_pisa ruled out
  (linux-only, JVM-adjacent). tantivy term-repetition ruled out (wrong math).

### Dense leg — usearch over faiss
- usearch 2.25: 63 wheels incl. **Windows arm64** + musl; `b1` Hamming with
  `Index.view()` mmap serving (instant cold start, ~0 resident); exact rescore
  built in. faiss-cpu = community wheels, win-arm64 dropped in 1.14.2; keep as
  alternative (IVF-PQ, IDSelector) not default.
- Recipe (validated in our dense compression eval at 600k: R@10 0.996 at 32×):
  b1 index via view() + int8/fp32 `np.memmap` rescore of 4–20× over-fetched
  candidates. 15M docs: 720 MB codes + ~5.8 GB i8 rescore matrix, both on disk.

### Filters across the stack
Native only in tantivy. Pattern: sidecar numpy arrays keyed by internal doc-id
(year uint16, type uint8, is_oa bitmask ≈ 60 MB at 15M docs, memmapped) →
over-fetch 10–20× from sparse/dense legs → mask → RRF. Assign internal ids
sorted by publication_year so year filters become contiguous range tests.
Tantivy backstops recall under very selective filters.

### Packaging (Tauri plan from `archive/THIN_CLIENT_PLAN.md`, superseded — kept for the numbers)
Tauri + PyInstaller-onedir Python sidecar is the established pattern.
Sidecar ≈ 60–90 MB (tantivy 3.8 MB + usearch 0.3 MB + bmp 0.6 MB + numpy +
CPython) (+16 MB if faiss). Cold start ~2–4 s with BMP. Ship per-arch builds;
never compile native deps with `target-cpu=native`.

## Recommended architecture

```
Tauri shell ──▶ Python sidecar (PyInstaller onedir)
  query ──▶ ONNX encoders (SPLADE int8 + dense 384-d)
     ├─ tantivy: BM25F title^3/abstract, native year/type/oa filters
     ├─ bmp:     SPLADE block-max (seismic optional Linux backend)
     └─ usearch: b1 mmap + int8 memmap rescore
  → sidecar numpy filter masks → RRF k=60 (existing code) → SQLite metadata
Monthly build box: corpus JSONL → all three indexes + sidecars, shipped as a
versioned bundle (the hot/cold section pack/unpack from eval_sectioned_index.py
layers on top: home sections live, others as zstd-19 archives at ~2× ratio).
```

Target steady-state RAM: **<2–3 GB** including encoders — fits the archived
THIN_CLIENT_PLAN's ~2.1–2.5 GB budget alongside Qwen3-1.7B only on 16 GB
machines; on 8 GB machines run the search stack + embedding-only reranking.

## Abstract storage strategy — hot/cold hybrid (decided 2026-06-11)

**Measured fact:** moving abstracts out of the index causes ZERO ranking change —
the `extern_display` lossless variant keeps abstracts *indexed* (BM25F scores
identical: 80/80 score-multiset parity, max Δ 0.0000) and only drops the stored
copy. **But abstracts are NOT display-only:** both rerank stages consume
`title + abstract` for every candidate (`src/lib/reranker.py:184,302`), and the
rerank is worth ~+0.12 NDCG@10 — so the top ~50–100 candidates' abstracts must be
fast-fetchable per query, not just the displayed top-10.

Policy per section temperature:
- **HOT sections → local abstract sidecar** (zstd-dictionary SQLite/LMDB).
  Cost ≈ 0.5 KB/doc compressed (measured from stored-field deltas): ~1.8 GB for a
  typical persona's home section at the 15M tier (~17.6 GB for the largest
  section at 150M). Keeps everyday queries fully offline at full rerank quality.
- **COLD sections → no local abstracts; remote fetch before rerank** (one mget of
  ~100 docs ≈ 150 KB, +100–300 ms). Off-domain queries already imply
  connectivity/unpack; warm-cache ingest gets abstracts free from OpenAlex live
  responses. Prefer the OpenAlex API (inverted-abstract parser already in
  `openalex.py`) over exposing the project's OpenSearch.
- **Never fetch-at-display-only:** reranking on titles alone for the other ~90
  candidates is the one variant that measurably loses quality.

## Hot/cold persona profiling — measured metrics (1M-doc eval, scaled)

Sections (combined lossless config; classified on title+abstract — the live
index has no concepts metadata):

| Section | share | live @150M | packed @150M (ratio) |
|---|---|---|---|
| social_sciences | 23.4% | 52.3 GB | 30.6 GB (1.7×) |
| med_bio | 11.4% | 32.0 GB | 13.6 GB (2.4×) |
| phys_eng | 5.5% | 16.0 GB | 6.8 GB (2.3×) |
| cs_math | 10.3% | 29.6 GB | 12.5 GB (2.4×) |
| other (catch-all) | 49.4% | 98.0 GB | 76.0 GB (1.3×) |
| **all live total** | | **227.9 GB** | **139.5 GB all-packed** |

Per persona (home section live + everything else packed), lexical leg only:

| Persona | steady @150M | peak* @150M | steady @15M | peak* @15M | saved |
|---|---|---|---|---|---|
| political_science | 161.2 GB | 259.2 GB | 16.1 GB | 25.9 GB | 29.2% |
| computer_science | 156.6 GB | 254.6 GB | 15.6 GB | 25.4 GB | 31.3% |
| health_science | 157.9 GB | 255.9 GB | 15.7 GB | 25.5 GB | 30.7% |
| interdisciplinary (3 hot) | 196.8 GB | 294.8 GB | 19.6 GB | 29.4 GB | 13.6% |

\* peak = steady + largest cold section (`other`) temporarily unpacked live.
Add the dense leg: +7.2 GB binary codes (+~19 GB HNSW graph) at 150M, ~0.7+1.9 GB
at 15M; add hot-section abstract sidecar per the policy above.

Operational caveats at scale: unpack-by-rebuild ran at **2,016 docs/s**
(114k-doc section in 57 s) → a med_bio-sized section is ~14 min at the 15M tier
and **~2.4 h at 150M** — fine as a one-time library expansion, not per-query;
sub-section archives (year-slices) or segment-level restore cut this. The
`other` catch-all (49% of docs, 1.3× pack ratio) caps savings — richer section
vocabularies or indexing real `concepts` metadata (requires snapshot re-download;
see DATA_MANIFEST.md) is the highest-leverage improvement to this scheme.

## Risk table (top items)

| Risk | Mitigation |
|---|---|
| BMP lacks mac-arm64/linux-aarch64 wheels | build 2 wheels in CI (maturin); pin versions |
| Seismic/BMP have no filters | sidecar masks + over-fetch (pattern above) |
| usearch b1 recall on 384-d | 10–20× over-fetch + exact rescore (measured 0.975–0.996 R@10) |
| Small-team deps (Seismic 129★) | static monthly rebuild; MIT licenses; vendorable |
| SPLADE→CIFF converter doesn't exist yet | small one-time tool; spec is simple protobuf (ciff-hub) |

Next steps live in `thin-client/README.md` (handoff workspace).

---

# Appendix: per-engine survey — should the stack leave OpenSearch? (researched 2026-06-10)

*(Merged from the former `SEARCH_ENGINE_ALTERNATIVES.md`, 2026-06-11. This is the
survey whose verdicts the POC above acted on.)*

**Question:** would moving off OpenSearch yield quality, efficiency, or storage gains?
**Answer:** No for the 150M server tier (upgrade to OpenSearch 3.x instead). Yes for the
laptop thin-client tier (no JVM on consumer machines; use static rebuilt artifacts).

Stack requirements used as criteria: BM25F (`multi_match most_fields`), SPLADE as
`rank_features` (top-256 terms/doc, per-term `rank_feature` bool/should), planned 384-dim
HNSW dense leg (binary+rescore 32× validated in `archive/COMPRESSION_EVAL_RESULTS.md`),
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

## Recommendations (as written 2026-06-10; #2 is what this container now runs)

1. **Server tier: stay; plan an OpenSearch 2.19 → 3.x upgrade** (derived source for
   vectors, disk-based binary quantization matching the validated 32× recipe, GPU index
   builds, Lucene 10, Seismic path). *(Applies to the ORIGINAL sfu-library-mcp container,
   not this testbed — see `deprecated/OPENSEARCH3_UPGRADE.md`.)*
2. **Laptop tier: leave OpenSearch.** Best stack: `tantivy-py` (BM25F) +
   `pyseismic-lsr` (SPLADE, static monthly-rebuilt index — matches the update cadence) +
   faiss/usearch (dense, binary+rescore), fused by the existing Python RRF. No JVM,
   tens-of-MB idle RAM. *(Adopted, with BMP substituted for Seismic per the POC above.)*
   Runner-up: Qdrant local mode (one engine: sparse+dense+filters+
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
