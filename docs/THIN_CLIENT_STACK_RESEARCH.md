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

### Packaging (Tauri plan from THIN_CLIENT_PLAN.md)
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

Target steady-state RAM: **<2–3 GB** including encoders — fits the
THIN_CLIENT_PLAN's ~2.1–2.5 GB budget alongside Qwen3-1.7B only on 16 GB
machines; on 8 GB machines run the search stack + embedding-only reranking.

## Risk table (top items)

| Risk | Mitigation |
|---|---|
| BMP lacks mac-arm64/linux-aarch64 wheels | build 2 wheels in CI (maturin); pin versions |
| Seismic/BMP have no filters | sidecar masks + over-fetch (pattern above) |
| usearch b1 recall on 384-d | 10–20× over-fetch + exact rescore (measured 0.975–0.996 R@10) |
| Small-team deps (Seismic 129★) | static monthly rebuild; MIT licenses; vendorable |
| SPLADE→CIFF converter doesn't exist yet | small one-time tool; spec is simple protobuf (ciff-hub) |

Next steps live in `thin-client/README.md` (handoff workspace).
