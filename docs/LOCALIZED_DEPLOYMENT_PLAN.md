# Localized Deployment & Dynamic Warm-Cache Plan

**Status:** Design / scope (not yet implemented)
**Last updated:** 2026-05-24
**Context:** How to deploy the SFU library MCP to client machines (incl. laptops) without shipping the full 150M-doc / 406 GB OpenSearch index. Companion to `SESSION_NOTES_2026-05-24.md`.

---

## 1. Problem

The full system runs against a local OpenSearch index of **150,413,098 docs / 406 GB** (≈ 2.7 KB/doc; the snapshot is already recency-filtered to ~2015+). That cannot run on a laptop. We need deployment tiers that trade footprint for retrieval quality, plus a way to make the local index *small but useful*.

Key architectural fact: **retrieval vs reranking** have different needs.
- **Retrieval** (BM25F, SPLADE, dense) finds candidates from a corpus → **needs an index**.
- **Reranking** (cross-encoder v1, dense embedder v5) re-scores a *given* candidate set → **no index needed** (encode on the fly).

So the SFU-tuned **cross-encoder and dense embedder work even with no local index** (reranking OpenAlex's live returns). **SPLADE's value requires a local index** (it's a retriever; as a reranker it's dominated by the cross-encoder).

---

## 2. Deployment tiers

| Tier | Local components | Footprint | Retrieval quality | Needs |
|---|---|---|---|---|
| **Full** | 150M index + BM25F+SPLADE RRF + rerank | server, 406 GB, OpenSearch, GPU to build | best (RRF beats OpenAlex relevance by +0.10–0.18 NDCG@10) | server |
| **Subset** | ~10–15M index (year×citation) + RRF + rerank | laptop, ~20–40 GB, OpenSearch | good on recent/mainstream; tail → live | curated bundle |
| **Dynamic warm-cache** *(recommended laptop default)* | rerankers + self-building local SPLADE/dense cache | small, capped (~GBs); **no GPU for the user** | OpenAlex + growing cross-query recall + rerank | OpenAlex key |
| **Live-only** | rerankers only; OpenAlex live | few hundred MB; no OpenSearch/GPU | OpenAlex relevance + SFU rerank (recall-bounded by OpenAlex) | OpenAlex key |

All four share one codebase — they differ only by router config (`LIVE_API` vs `LOCAL_RRF`) and whether/what index is present.

---

## 3. Index size analysis

~2.7 KB/doc. Estimated composition: stored `_source` (title+abstract) ~35–45%, SPLADE `sparse_field` (256 terms/doc) ~25–35%, BM25 inverted index ~20–30%, metadata ~5%.

Year distribution (from the live index — corpus is already ~2015+ and **recent-year-dense**):

| Range | Docs | ~Size |
|---|---|---|
| ≥ 2015 (≈ all) | ~145.9M | ~394 GB |
| ≥ 2020 | ~103.8M | ~280 GB |
| ≥ 2022 | ~83.4M | ~225 GB |

**Critical:** a year selector *alone* won't reach laptop scale — even "2022+" is 83M docs / 225 GB. Reaching ~10–15M (~20–40 GB) needs **year × a citation/quality floor** (drop the long tail of never-cited papers). A pure time-window cut is too coarse at this density.

### Year × citation "time-frame selector" feature
Offer a 2-axis selector (time window + citation floor) with **live size/doc-count estimates per combo**, build standard **pre-built bundles** server-side (`_reindex` filter → snapshot), host on Blob/Spaces, user restores locally. Bundles include the **pre-encoded `sparse_field`** so the user needs no GPU. Live-API fallback for out-of-range queries.
- ⚠️ Depth axis needs a **`cited_by_count` field indexed** — verify it exists; if not, adding it requires a re-index.

---

## 4. Compression strategies

Ranked by impact (cost = local GB/RAM; performance = recall + latency):

1. **Subset doc count** (dominant, linear): year + citation + subject scoping. 150M→15M = ~40 GB; →5M = ~13 GB. *Improves latency* (fewer postings). Cost: long-tail recall, mitigated by live fallback.
2. **`index.codec: best_compression`** (zstd): ~15–25% off stored fields, ~free perf cost. Trivial.
3. **SPLADE forward-index / sparse compression** (from prior art — see §7): quantized pseudo-tf (DeepCT-style scale+round), StreamVByte / SIMDBP-256 integer compression, flat compact structures (up to ~6.2× on doc-level indices), FLOPS/L1 regularization to minimize non-zeros (already used in our fine-tune). Reduce doc-side top-k 256→128/96 for further size at modest recall cost.
4. **Trim/truncate stored `_source`** (abstracts): big savings; loses full-text re-encode/display.
5. **Dense vectors (if dense leg added): int8 scalar quantization** (~4× vs fp32) and/or **Matryoshka truncation** (384→128, needs MRL-trained embedder). Note: dense needs ~50× more storage than BM25, so quantization is mandatory at scale.

Combined (subset + codec + SPLADE top-k + truncated abstracts) ≈ ~30–40% reduction *on top of* subsetting.

---

## 5. Dynamic warm-cache — SCOPE

A **read-through, usage-driven local index** that builds itself from real searches. Solves §3's "which subset to ship" automatically — it caches exactly what the user actually searches.

### Architecture
```
query ──> [local warm cache: SPLADE/dense retrieval] ─┐
     └──> [OpenAlex live (user key) + Semantic Scholar]┼─> RRF fuse ─> rerank (CE v1 + embed v5) ─> results
                                                        │
              (async, post-response) encode returned docs ──> upsert into local cache
```

### Write path (cache warming) — async, zero user-facing latency
- After returning results, **SPLADE- and dense-encode the docs OpenAlex/S2 returned** and **upsert into the local cache index** (dedup by OpenAlex id).
- Done in the background so it never adds to query latency.
- **Codebase status:** `src/lib/openalex.py` already has a `_push_to_opensearch` hook (fire-and-forget push of live results to local). It currently *skips* writing `sparse_field` (a deliberate guard, see SESSION_NOTES item #6). For the warm cache, flip it to **encode-on-ingest** (SPLADE via the local ONNX encoder in `models/splade_onnx`; dense via `embedding.encode_texts`).

### Read path (local + live fusion)
- Query the warm cache (SPLADE/dense over accumulated docs) **and** live API; fuse with RRF (already implemented in `federated_search.py`).
- Cold start = empty cache → pure live. Warms over usage.

### Eviction (keep it "minimal")
- **LRU + TTL + hard size cap** (e.g., N docs / X GB). Evicted docs just get re-fetched live if needed. Keeps the cache to exactly the user's active domain.

### Seeding (cold-start mitigation) — see §6
- Pre-warm from the user's **Zotero library / local PDF downloads** so the cache is useful from day one.

### Honest limits
- **It's a cache, not a fresh corpus.** Recall is bounded by *what's been seen* — it can't surface a doc OpenAlex never returned for any past query.
- **Genuine gain: cross-query recall.** A doc OpenAlex returned for query A becomes locally retrievable for query B *even if OpenAlex's ranking for B would miss it* — and this grows with usage. Plus speed + partial offline for repeat/related queries.
- Cold start until warmed; encode-on-ingest CPU cost (async-mitigated).

### Implementation steps
1. Flip `_push_to_opensearch` → encode-on-ingest (SPLADE + dense), upsert by id, atomic (reuse the temp→fsync→`os.replace` pattern already in `mine_hard_negatives.py`).
2. Stand up a small local `warm_cache` index (knn_vector + sparse_field + text), `index.knn:true`.
3. Read path: enable local+live RRF fusion for the warm-cache profile.
4. Eviction: LRU/TTL/size-cap maintenance job.
5. Seeding: Zotero/local-PDF ingest (see §6).
6. Config profile `deployment=warm_cache`.

### Cold-start → warm performance model (to measure)
| Searches | Cache state | Behavior |
|---|---|---|
| 0 | empty | pure live (= live-only tier) |
| ~10–50 | warming | cross-query recall starts paying off in-domain |
| ~100+ | warm for domain | local SPLADE/dense leg adds recall + speed; offline-capable for repeat domains |

---

## 6. Local downloads / Zotero seeding (the academic-user insight)

**Insight:** academics *download* the papers they use — so their most-relevant, most-recent documents already live on disk (PDF folders, **Zotero library**). Use that as a high-value seed for the warm cache.

- **This project already integrates Zotero** (`src/lib/zotero.py`, 872 LOC) — the user's Zotero library *is* their curated local corpus. Ingest it: parse the library's items/PDFs → extract title+abstract (+ full text) → SPLADE/dense-encode → seed the warm cache.
- Covers the "very recent / not-yet-in-snapshot" gap: papers a researcher just downloaded may be newer than the OpenAlex snapshot, and they're exactly what that researcher cares about.
- **Strong prior art (validates the approach):** a whole ecosystem of local Zotero semantic-search tools exists — *RAG-Assistant-for-Zotero* (BGE/SPECTER/MiniLM, local), *ZotSeek* (**fuses semantic + keyword with RRF, 100% local** — same fusion philosophy as us), *ARCHILLES* (Zotero/Obsidian/Calibre/folders), *zotero-rag*, *Zotero Chunk RAG*. We can mirror their PDF-ingest + local-embedding patterns and reuse our existing Zotero client + fine-tuned SFU models.
- Generalize beyond Zotero: a "watched folder" of downloaded PDFs → same ingest pipeline.

---

## 7. Prior art (researched 2026-05-24)

- **Caching Historical Embeddings in Conversational Search** (Frieder et al., arXiv 2211.14155 / ACM TWEB 2023): client-side **document embedding cache** exploiting temporal locality of retrieved docs; **up to 75% hit rate without degrading quality**, reducing back-end load. Closest academic precedent for the warm cache.
- **Incremental indexing for RAG**: update only changed docs/embeddings; **document-level embedding caches persist across sessions**; **warm-start** preloads frequent vectors. (Morphik, Medium incremental-indexing writeups.)
- **Query-result caching** in search engines (caching historical query results; Algolia local result cache; OpenSearch index-request/shard cache). Mature for *query→result* reuse (complements our *doc-level* cache).
- **Local-first Zotero RAG tools** (see §6) — proven on-device semantic search over researchers' PDF libraries; ZotSeek uses RRF.
- **SPLADE / sparse index compression**: quantized pseudo-tf (DeepCT-style), FLOPS/L1 sparsification (we already use FLOPS in fine-tuning), StreamVByte/SIMDBP-256 4-bit integer compression, "Forward Index Compression for Learned Sparse Retrieval," hybrid thresholding for sparsification, flat compact structures (~6.2× reduction). Note dense ≈ 50× BM25 storage → quantization mandatory.

Sources:
- https://arxiv.org/pdf/2211.14155
- https://dl.acm.org/doi/full/10.1145/3578519
- https://medium.com/@vasanthancomrads/incremental-indexing-strategies-for-large-rag-systems-e3e5a9e2ced7
- https://www.morphik.ai/blog/retrieval-augmented-generation-strategies
- https://docs.opensearch.org/latest/search-plugins/caching/index/
- https://github.com/introfini/ZotSeek
- https://github.com/aahepburn/RAG-Assistant-for-Zotero
- https://archilles.org/
- https://arxiv.org/html/2602.05445v1 (Forward Index Compression for Learned Sparse Retrieval)
- https://dl.acm.org/doi/full/10.1145/3634912 (Effective and Efficient Sparse Neural IR)

---

## 8. Open decisions
1. Warm-cache eviction policy + size cap (per-user vs shared/team).
2. Is `cited_by_count` indexed? (gates the year×citation selector depth axis).
3. Zotero/folder ingest: full-text chunks vs title+abstract only.
4. Laptop engine: keep OpenSearch (heavy JVM) vs a lighter embedded engine (Lucene/Tantivy + FAISS/sparse) for true laptop-class deploys.
5. Which tier(s) to productize first (recommended: warm-cache as the laptop default, live-only as the zero-infra fallback).
