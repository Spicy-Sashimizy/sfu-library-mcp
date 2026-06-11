# Lexical Storage Research: Shrinking the 150M-Doc Abstract Sidecar and Postings Beyond Codec Compression

**Date:** 2026-06-11
**Scope:** Web research on storage/compression alternatives for the OpenAlex 150M-work lexical tier (tantivy BM25F + BMP SPLADE + usearch dense), with hard constraint: **fetch + decode ~100 abstracts (~1 KB each) in < 300 ms** for the cross-encoder reranker (worth +0.12 NDCG@10 — must not be degraded).
**Baseline:** Lucene index 274 GB → 142 GB (zstd-6 codec, no SPLADE weights in _source, no positions, no dup field) → ~105 GB with externalized abstract sidecar.

---

## 0. The decode budget, made explicit

- 100 abstracts × ~1 KB ≈ **100 KB of text** to materialize per query, budget 300 ms ⇒ minimum required decode throughput ≈ **0.33 MB/s sustained, random-access**, plus seek/page-in overhead on laptop NVMe (~100 random 4K reads ≈ 5–20 ms, negligible).
- Byte-level codecs (zstd, Brotli, FSST) decode at **hundreds of MB/s to GB/s** — they consume ~0.1–1 ms of the budget. The budget is therefore *not* a constraint for any conventional codec; it is **only** a constraint for neural methods.
- Neural arithmetic-coding decode is *sequential per stream*: one LM forward pass per token position. 100 docs × ~250 tokens = 25,000 token steps; batching across the 100 docs reduces this to ~250 *sequential batched* forward passes. At a (optimistic) 15–40 ms/batched-step for a 1.7B Q4 model on a laptop GPU, that is **4–10 s ⇒ 13–30× over budget**, before counting VRAM contention with the cross-encoder, SPLADE, and dense models already resident. This single calculation rules out LLM-as-decompressor for the hot rerank path (details and sources in §1).

---

## 1. Executive summary table

Ranking metric: (realistic compression ratio on ~1 KB English abstracts) × (decode speed vs budget) × (random-access fit). "Ratio" = original/compressed, *expected on this corpus* — flagged (M) measured in cited source on comparable data, (E) extrapolated/estimated, (C) vendor/author claim not independently reproduced.

| Rank | Method | Ratio on ~1 KB abstracts | Decode speed | Random access | Fits 100-abstracts/300 ms? | Python/pip, license | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **zstd + trained dictionary** (per-doc frames) | ~2–3× (E; 2.5–10× (M) on 1 KB JSON records, less for prose) | ~500–2000 MB/s; 25–55% slower with dict on small blocks (M) — still ≫ budget | Per-doc, O(1) | **Yes** (≪1 ms) | `zstandard` pip, BSD | **Test first** |
| 2 | **Similarity-clustered block compression** (sort docs by topic/MinHash, zstd-19 blocks of 16–64 docs, optional dict) | ~3–4.5× (E; approaches large-window zstd) | One block decode per miss ≈ 30–100 µs | Per-block (16–64 KB) | **Yes** | pip zstandard + offline clustering | **Test** — best ratio/complexity trade |
| 3 | **FSST** (+ entropy stage) | ~2× alone (M, text columns); ~2.5–3× with secondary zstd (E) | 1–3 GB/s (M) | Per-string, finest-grained; equality-preserving | **Yes** | C/Rust libs (`fsst` crates, cwida/fsst, MIT); thin Python via duckdb/vortex | Test if per-doc O(1) without frames is required |
| 4 | **Brotli-11 + custom dictionary** | ~2.2–3× (E; built-in 120 KB text dict helps short prose) | ~300–500 MB/s | Per-doc | **Yes** | `brotli` pip, MIT | Alternative to #1; benchmark head-to-head |
| 5 | **Token-ID recoding** (BPE IDs + freq-ordered varint + zstd) | ~2.5–3× (E; +0.8–7 pp over byte zstd, (M) on enwik8) | tokenizer decode ~10s MB/s — fine | Per-doc | **Yes** | `tokenizers` pip | Marginal gain over #1/#2; low priority |
| 6 | **Static index pruning / impact quantization** (postings, not stored text) | postings −20–60% (M, with measurable quality cost beyond ~50%) | n/a (faster queries) | n/a | n/a | PISA/BMP tooling, Apache/MIT | Apply 8-bit impacts (done by BMP); pruning only with eval harness |
| 7 | **BP doc-ID reordering** | postings −1.5% (M, Lucene nightly) to −10–15% (M, literature on web corpora) | n/a (also speeds queries) | n/a | n/a | Lucene `BPIndexReorderer`; not built into tantivy | Cheap win if order is free to choose; modest |
| 8 | **ts_zip / ts_sms-class small-LM neural lossless** (RWKV 169M / Nacrith 135M) | ~5–7× (M on enwik-class text) | ≤1 MB/s on RTX 4090 (C/M); laptop ~0.2–0.4 MB/s | Per-doc possible (independent streams) | **Borderline-NO** (~100 ms on 4090 *best case*; 300 ms–1 s+ on laptop; GPU contention) | binaries only (ts_zip, no source); Nacrith code N/A | Cold-tier only |
| 9 | **LLMZip / FineZip / LMCompress (7–8B LM lossless)** | ~8–11× (M) | ~9 MB/**hour** compress (M, A6000); decode same order | Per-doc possible | **NO** (3–4 orders of magnitude over) | research code, GPL/research | Ruled out for serving |
| 10 | **Lossy semantic/abstractive compression** (LLMLingua-2, RECOMP, SemanticZip-style) | 2–20× "token reduction" (M/C) | LLMLingua-2 fast to *compress*; decode = LM generation if reconstruction needed | n/a (store compressed form, feed directly) | Only if reranker consumes compressed text directly | `llmlingua` pip, MIT | Risky: threatens the +0.12 NDCG@10; test only with full harness |
| — | **FM-index / succinct text self-index** | ~0.4–1× of text | poor locality on disk | substring access, not doc-fetch oriented | technically yes for extraction, but no ranking benefit | sdsl-lite | Already ruled out; confirmed correct call (§4.4) |
| — | **Store OpenAlex `abstract_inverted_index` natively** | <1× (it is *larger* than plain text) | n/a | n/a | n/a | n/a | Reconstruct once, store plain text compressed (§4.3) |

**Bottom line:** Nothing neural fits the hot path. The realistic envelope for the abstract sidecar is **~2–4.5×** over raw UTF-8 (vs ~1.6–1.8× for naive per-doc zstd-6 without a dictionary), i.e. an estimated 150 GB raw-text-equivalent sidecar portion compressing to roughly **25–40 GB instead of ~60 GB** — combined with the already-measured 105 GB config, a plausible **~80–90 GB total** target. These are estimates; §6 gives the test protocol.

---

## 2. Neural / LLM-based text compression

### 2.1 Lossless: LM + arithmetic coding (bit-exact)

The core mechanism in all of these: the LM provides next-token probabilities; an arithmetic coder turns the true token sequence into a near-entropy bitstream. Decoding replays the LM **one token at a time** — this sequentiality is fundamental and is what kills serving latency.

**LLMZip** (Valmeekam et al., arXiv [2306.04050](https://arxiv.org/abs/2306.04050), June 2023; [code](https://github.com/vcskaushik/LLMzip))
- LLaMA-7B + arithmetic coding: **0.71 bits/char** on text8-class data (M) ≈ ~11× over raw UTF-8; beats BSC, ZPAQ, paq8h.
- Throughput: catastrophic. Per the FineZip authors, an LLMZip-style system with Llama3-8B needs **~9.5 days to compress 10 MB** (~0.044 MB/h) on an A6000 ([arXiv 2409.17141](https://arxiv.org/html/2409.17141v1)). Decompression is the same compute (must replay the LM). **Verdict: NO** — 5–6 orders of magnitude off the serving budget.

**Language Modeling Is Compression** (Delétang et al., DeepMind, arXiv [2309.10668](https://arxiv.org/abs/2309.10668), ICLR 2024; [code](https://github.com/google-deepmind/language_modeling_is_compression))
- Chinchilla-70B compresses enwik9 slices to **8.3%** of raw size (M) vs gzip 32.3%, LZMA2 23%. Important caveat the paper itself makes: counting the *model size* in the compressed size destroys the win for any corpus smaller than ~the model itself. For a fixed 150M-doc corpus with the model already shipped (Qwen3-1.7B), the amortization argument actually holds — the blocker is purely decode speed.

**ts_zip** (Bellard, [bellard.org/ts_zip](https://bellard.org/ts_zip/), 2023–2024)
- RWKV 169M v4, 8-bit quantized, deterministic BF16 eval (reproducible across hardware — solves the bit-exactness-across-devices trap that GPU float nondeterminism creates for arithmetic coding).
- Ratios (M): enwik8 **1.106 bpb**, enwik9 1.084 bpb, alice29.txt 1.142 bpb (vs xz 1.99/1.71/2.55) ≈ ~7× over raw.
- Speed (C, author): "**up to 1 MB/s on a RTX 4090**" for compression *and* decompression; 4 GB RAM. On a thin-client laptop GPU expect 0.2–0.4 MB/s (E). 100 KB of abstracts ⇒ **~100 ms on a 4090 best case, 300 ms–1 s on laptop hardware**, while competing for the same GPU as the cross-encoder. Binaries only, no source, no license for redistribution. **Verdict: NO for hot path; plausible for a cold tier** (e.g., abstracts of docs never surfaced in top-1000 in query logs).

**ts_sms** (Bellard, [bellard.org/ts_sms](https://bellard.org/ts_sms/), Dec 2024)
- Same idea specialized for *short* texts: arithmetic-coding-compatible padding avoids encoding message length; example on page: 155-char message → 28 base64 chars vs Brotli's 88 (M, single example). Directly relevant evidence that LM coding keeps its advantage at ~1 KB scale (where zstd/Brotli lose theirs). Same speed/licensing objections as ts_zip.

**FineZip** (Mittu et al., arXiv [2409.17141](https://arxiv.org/abs/2409.17141), Sept 2024)
- "Online memorization" (LoRA-finetune on the corpus being compressed) + dynamic (fixed-window) context ⇒ parallelizable batches. Llama-3-8B on A6000: 10 MB in **~4 h**; 4-bit quantized: **67 min (~9 MB/h)**, ratio 0.128→0.145 (vs gzip 0.324, bzip2 0.237, LLMZip 0.116) (M).
- 54× faster than LLMZip is still ~5 orders of magnitude slower than zstd. The corpus-finetune idea (a LoRA adapter as a "dictionary") is the one transferable insight. **Verdict: NO for serving.**

**LMCompress** (Li et al., *Nature Machine Intelligence* 7:794–799, May 2025; [paper](https://www.nature.com/articles/s42256-025-01033-7), arXiv [2407.07723](https://arxiv.org/abs/2407.07723))
- Domain-finetuned LLaMA3-8B reaches ~**⅓ the compressed size of zpaq** on text, ~4× better than bz2 (M). State of the art on ratio; no serving-speed story at all (same per-token decode). Confirms ceiling: a finetuned 8B could plausibly hit **~10–12×** on abstracts — for archival only.

**Nacrith** (arXiv [2602.19626](https://arxiv.org/abs/2602.19626), Feb 2026)
- The most practical 2026 datapoint: SmolLM2-**135M** + online predictor ensemble + 32-bit arithmetic coder, llama.cpp backend ("~7× faster single-token decode than PyTorch"), 500 MB GGUF, **1.2 GB VRAM/worker**. enwik8 **0.939 bpb** (M) — beats ts_zip by ~15%, CMIX by 44%, FineZip by 8%, with a 60× smaller model. Code availability not stated.
- Even so: 0.939 bpb ≈ 8.5× ratio, but decode remains ~1 MB/s-class. A 135M model at ~2–5 ms/batched step ⇒ ~0.5–1.3 s for 100 abstracts on a discrete GPU (E). **Closest-to-viable neural option; still over budget and VRAM-expensive on a thin client.**

**Can the shipped Qwen3-1.7B double as the decompressor?** Mechanically yes (its logits + arithmetic coding = a compressor, per DeepMind), and it would amortize model weights to zero. But: (a) per-token decode for 1.7B Q4 on laptop hardware is ~20–60 tok/s single-stream, ~1–3k tok/s well-batched (E from llama.cpp-class benchmarks, e.g. ~24 tok/s for Qwen3-class Q4 on M4 Pro, [llama.cpp #19366](https://github.com/ggml-org/llama.cpp/issues/19366), [Qwen speed docs](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)) ⇒ 25k tokens ≈ **8–25 s hot path**; (b) arithmetic coding requires **bit-exact reproducible logits** — any llama.cpp version bump, quantization change, GPU/CPU kernel difference silently corrupts the entire sidecar (ts_zip solves this with a custom deterministic runtime; llama.cpp does not guarantee it); (c) KV-cache + weights contend with reranker VRAM. **Verdict: NO for hot path; defensible only for cold-tier archival with a frozen, deterministic runtime.**

### 2.2 Lossy: semantic / abstractive reconstruction

Distinct proposition: don't reconstruct bytes; store a short form whose *meaning* suffices for the consumer (here, the cross-encoder).

- **LLMLingua / LLMLingua-2** (Microsoft, arXiv [2310.05736](https://arxiv.org/abs/2310.05736), EMNLP 2023; [project](https://llmlingua.com/llmlingua.html); `pip install llmlingua`, MIT). Token-pruning by small-LM perplexity; up to 20× claimed with "little performance loss" *on QA/reasoning tasks* (M for their tasks, C for any other consumer). Compression is a one-time offline cost; the **compressed text is consumed directly — no decode step at all**, so it trivially fits the latency budget and *shrinks reranker input length* (a latency win). The open question is purely quality: cross-encoders are trained on natural prose; perplexity-pruned text is distribution-shifted input.
- **RECOMP** (Xu et al., arXiv [2310.04408](https://arxiv.org/abs/2310.04408), ICLR 2024): extractive/abstractive compressors for retrieved docs in RAG — evidence that 6–10× shorter contexts can preserve downstream task accuracy, again task-dependent.
- **SemanticZip** (arXiv [2605.24541](https://arxiv.org/html/2605.24541), May 2026): LLM-as-semantic-decompressor pilot; 19–46% token reduction; introduces **Critical Atom Recall / Weighted Atom Recall** as fidelity metrics. Authors are candid it's a 5-example pilot, no baselines — cite for the *measurement methodology*, not the numbers.
- Degradation measurement (consensus across these): string metrics (ROUGE-L), semantic metrics (BERTScore), and — the only one that matters here — **end-task delta** (rerank-score shift, NDCG@10 on the existing harness). §6.2 operationalizes this.

**Verdict:** the only neural family compatible with the budget, because decode is free. But it gambles the +0.12 NDCG@10 the reranker earns. Worth a *contained* experiment (§5, candidate 5) — e.g., LLMLingua-2 at conservative 2× on abstracts > 1.5 KB only — with an automatic abort threshold of, say, > 0.005 NDCG@10 loss.

---

## 3. Dictionary / statistical methods for short academic texts

### 3.1 zstd with trained dictionaries — primary candidate

- Mechanism: `zstd --train` (COVER/fastCOVER) builds a shared dictionary (typical 16–112 KB) used as implicit history for every tiny frame. Exactly designed for "many ~1 KB records" ([zstd homepage](https://facebook.github.io/zstd/), [manual](https://github.com/facebook/zstd/blob/dev/programs/zstd.1.md)).
- Measured reference point: `github-users` set, ~10K records of ~1 KB — dictionary compression improves ratio dramatically (~2.6× → ~10×) *and* speeds up both directions on that JSON-heavy data (M, vendor benchmark). English prose has less boilerplate than JSON; expect **~2–3×** for abstracts with a 110 KB dict trained on 10–100 MB of abstracts (E — must measure). Train **per-domain dictionaries** (e.g., by OpenAlex topic/field: medicine vs physics abstracts share much more internal phrasing) and store a 1-byte dict ID per doc; this is the cheap version of "domain-specific static dictionaries" and typically buys another 5–15% (E).
- Caveat (M): with a dict on small independent blocks, *decompression slows 25–55%* at higher levels ([zstd issue #4175](https://github.com/facebook/zstd/issues/4175), Nov 2024) — from ~2 GB/s to ~550–700 MB/s. Irrelevant at our 0.33 MB/s requirement.
- Random access: compress each abstract as an independent frame; offsets in a fixed-width table (or use the [seekable format](https://github.com/facebook/zstd/tree/dev/contrib/seekable_format) — note its own guidance: frames < 1 KB "hurt compression ratio considerably", which is precisely why the dictionary is mandatory here).
- Availability: `zstandard` / stdlib `compression.zstd` (Python 3.14+), BSD. Degradation: none (lossless, stable format, dict must be versioned and immutable — losing the dict loses the data).
- **Fits budget: YES (≈0.2 ms for 100 docs).**

### 3.2 Brotli with custom/shared dictionaries

- Brotli ships a built-in ~120 KiB static dictionary of English/HTML substrings — it already beats zstd on short English text at high levels; v1.1.0 adds *custom* shared dictionaries on top ([RFC draft](https://datatracker.ietf.org/doc/html/draft-vandevenne-shared-brotli-format-04), [Chrome write-up](https://developer.chrome.com/blog/shared-dictionary-compression), [DebugBear guide](https://www.debugbear.com/blog/shared-compression-dictionaries)). Web-delta numbers (84–90% smaller than plain Brotli) are for *versioned-resource* dictionaries — not transferable to abstracts; ignore as marketing for this use case.
- Expected on abstracts: **~2.2–3×** at quality 9–11 with a trained dict (E); decode ~300–500 MB/s. `brotli` pip, MIT. Same random-access framing as zstd. **Fits budget: YES.** Worth a head-to-head vs zstd-dict; historically Brotli wins by ~5–10% on small English text, zstd wins on decode speed (both irrelevant margins here, so pick the better ratio).

### 3.3 FSST — random-access string compression

- Boncz/Neumann/Leis, VLDB 2020 ([paper PDF](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf), [code cwida/fsst](https://github.com/cwida/fsst), MIT; pure-Rust [spiraldb/fsst](https://github.com/spiraldb/fsst); used in DuckDB ([PR #4366](https://github.com/duckdb/duckdb/pull/4366)) and Vortex/Lance ecosystems).
- 255 symbols of up to 8 bytes; ~**2× on text** (M), **1–3 GB/s** encode/decode (M), and the killer feature: **per-string random access with zero framing overhead** — you can decode one abstract, or even compare compressed bytes for equality, without touching neighbors. A second-stage zstd over FSST output recovers some entropy coding (DuckDB-style), at the cost of reintroducing blocks.
- For tantivy specifically: FSST is the natural fit if the sidecar is a column in a Lance/Vortex/Parquet-class file rather than a custom blob store.
- **Fits budget: YES.** Expected ratio alone (~2×) is below zstd-dict; choose it only if O(1) *sub-block* access or compressed-domain equality matters.
- See also **OnPair** (arXiv [2508.02280](https://arxiv.org/abs/2508.02280), Aug 2025): BPE-class ratios with FSST-class random access, strings compressed independently; young, code availability unclear — watch, don't depend.

### 3.4 smaz / shoco / unishox-class short-string codecs

- [smaz](https://github.com/antirez/smaz) (static English dictionary, ~28% saving (M) on dictionary words), [shoco](https://ed-von-schleck.github.io/shoco/) (trainable entropy coder, ~33% (M), ASCII-only, never expands ASCII), [Unishox](https://www.theoj.org/joss-papers/joss.03919/10.21105.joss.03919.pdf) (Unicode-capable). All target *tens of bytes* (SMS, keys). At 1 KB they are strictly dominated by zstd-dict on both ratio and ecosystem maturity. **Verdict: not applicable; skip.**

### 3.5 Token-level recoding (store BPE IDs, not UTF-8)

- Idea: abstracts → BPE token IDs (≈4.3 chars/token for o200k-class vocab) → varint → entropy code. Raw effect ≈ 2.1–2.2× before entropy coding.
- 2026 measurements: tokenizer→LZ beats raw-LZ for weak compressors, but **tokenized varint streams can be slightly *worse* inputs for zstd/LZMA** than raw UTF-8 ([An Information-Theoretic Perspective on LLM Tokenizers, arXiv 2601.09039](https://arxiv.org/html/2601.09039v1)); **frequency-ordered token IDs + varint** recover this: +7.1 pp for zlib, +1.7 pp LZMA, **+0.76 pp zstd** on enwik8 ([Frequency-Ordered Tokenization for Better Text Compression, arXiv 2602.22958](https://arxiv.org/html/2602.22958v1), Feb 2026) (M).
- Net: ~0–10% over a tuned zstd-dict pipeline, plus a tokenizer-version pinning liability, plus lossy-by-default on whitespace/unicode edge cases unless byte-fallback is kept (then it's bit-exact). One genuine synergy: if abstracts are stored as Qwen3 token IDs, the reranker/SLM could skip re-tokenization. **Fits budget: YES; priority: low.**

---

## 4. Structural alternatives for the stored-text sidecar

### 4.1 Similarity-clustered block layout (recommended)

Lucene's stored fields already do block compression (16–128 KB chunks) in *docID order*. The sidecar can beat it by choosing the order: cluster similar abstracts (same OpenAlex topic/subfield, or MinHash buckets), write clusters contiguously, compress 16–64-doc blocks with zstd-19 + per-domain dict. Cross-doc redundancy inside a block (shared methodology phrasing, venue boilerplate, near-dup preprint/published pairs) becomes LZ matches.
- Expected: **~3–4.5×** (E) — between per-doc dict (~2–3×) and whole-corpus zstd long-window; the same intuition that makes BP reordering shrink postings (§5.1) applies to stored text.
- Random access: read+decode one ~32 KB block per cache-missing abstract: ~30–100 µs decode each, ≤10 ms worst-case for 100 misses. Block cache makes repeat queries cheaper. **Fits budget: YES.**
- This composes with everything in §3 and is the highest-leverage structural change.

### 4.2 Dedup / MinHash clustering + delta chains

- Storage-systems literature (post-dedup delta compression, e.g. [Ddelta](https://www.sciencedirect.com/science/article/abs/pii/S0166531614000790); MinHash + Union-Find near-dup clustering with ~400× clustering speedups reported) shows high gains **when corpora contain near-duplicates**. OpenAlex does: preprint vs published versions, retraction notices, multi-source merges of the same work.
- But explicit delta-chains (bsdiff-style, chain depth > 1) break O(1) access and add fragility. The pragmatic version is §4.1: put near-dups in the same compression block and let zstd find the delta. A separate win available regardless: **exact/near-dup detection to store one abstract per cluster** with a pointer — pure win, no decode cost. Expected total: **5–15%** extra (E, corpus-dependent — measure dup rate first with a MinHash pass).

### 4.3 OpenAlex native `abstract_inverted_index` vs plain text

- OpenAlex ships abstracts as `{"word": [positions...]}` **for legal reasons, not size** ([OpenAlex docs](https://docs.openalex.org/api-entities/works/work-object), [community discussion](https://bmkramer.github.io/SesameOpenScience_site/thought/202411_open_abstracts/)). As JSON it is *larger* than the plain text it encodes (every word carries quoting, brackets, and integer positions); it also compresses worse (positions are high-entropy).
- **Verdict: reconstruct once at ingest, store compressed plain text.** The only reason to retain the inverted form is the legal posture of "not storing plaintext abstracts" — if that matters for redistribution of the thin-client bundle, note that a zstd-dict-compressed plaintext store is equally "not human-readable at rest" but reconstructs trivially; consult licensing, not engineering.

### 4.4 Succinct self-indexes (FM-index) — confirming the prior ruling

- FM-index ([Wikipedia](https://en.wikipedia.org/wiki/FM-index); Ferragina–Manzini) gives substring search over text stored in ~entropy-compressed space. Two disqualifiers for this stack, both confirmed in literature: (1) it answers *locate/extract*, not **ranked** retrieval — BM25/SPLADE need per-term postings with impacts, which the FM-index does not materialize cheaply; (2) BWT access patterns are cache/disk-hostile — "the non-sequential access pattern … makes the FM-index a poor choice for disk-based search" ([Large-Scale Pattern Search Using Reduced-Space On-Disk Suffix Arrays, arXiv 1303.6481](https://arxiv.org/pdf/1303.6481)). Modern use is genomics and n-gram lookup ([Infini-gram mini, arXiv 2506.12229](https://arxiv.org/pdf/2506.12229)), not BM25 serving. Document *extraction* from an FM-index is also far slower than block-zstd. **Prior ruling stands.**

---

## 5. Postings-side, beyond-codec

### 5.1 BP / recursive graph bisection doc-ID reordering

- Dhulipala et al., KDD 2016 ([arXiv 1602.08820](https://arxiv.org/abs/1602.08820)): reorder docIDs to minimize log-gaps; reference implementations: Lucene [`BPIndexReorderer`](https://lucene.apache.org/core/9_12_0/misc/org/apache/lucene/misc/index/BPIndexReorderer.html) ([PR #12489](https://github.com/apache/lucene/pull/12489), [issue #12665](https://github.com/apache/lucene/issues/12665)), PISA [`reorder-docids --bp`](https://pisa.readthedocs.io/en/latest/document_reordering.html), [enhanced-graph-bisection](https://github.com/jmmackenzie/enhanced-graph-bisection) (Mackenzie et al.).
- Measured spread is wide and honest reporting matters: Lucene's own benchmark observed **only ~1.5% postings (doc) compression improvement** on wikimedium-class data, while the original paper and PISA-era results show **up to ~10–15%+** index-size reduction on web corpora, plus consistent *query speedups* from clustering (the speedup, not the size, is why Lucene pursued it; see [jpountz's benchmark analysis](https://jpountz.github.io/2025/05/12/analysis-of-Search-Benchmark-the-Game.html), May 2025, which benchmarks `lucene-bp` and tantivy variants head-to-head).
- tantivy has **no built-in BP**; but since the sidecar redesign already imposes a custom doc order (§4.1), you get most of the benefit by **feeding tantivy docs in cluster order** — one ordering can serve both postings compression and stored-text block locality. (BMP additionally benefits: BP-style ordering makes block maxima tighter ⇒ better pruning — see the BMP paper below.)
- Expected here: **2–8%** of postings size (E), plus query-latency upside. Cheap because the ordering pass is shared.

### 5.2 Impact quantization (8-bit)

- BMP (Mallia, Suel, Tonellotto, SIGIR 2024, [arXiv 2405.01117](https://arxiv.org/abs/2405.01117), [pisa-engine/BMP](https://github.com/pisa-engine/BMP)) already **requires 8-bit quantized impacts**; measured quality cost of 8-bit quantization is negligible (≤1% RR@10 vs exhaustive in the BMP paper; Anserini-style quantization literature concurs; 4-bit costs ~0.05 recall@1000 for SPLADE per [superblock study, arXiv 2602.02883](https://arxiv.org/pdf/2602.02883)). Since the migration already adopts BMP, this is **done by construction** — just don't *also* keep float weights anywhere (the measured lossless config already drops SPLADE weights from _source; keep it that way).

### 5.3 Static index pruning

- Classic: Carmel et al., SIGIR 2001 ([ACM](https://dl.acm.org/doi/10.1145/383952.383958)) — term-based pruning dominates uniform pruning; "prune the index greatly and still get retrieval results almost as good" — concretely, the literature consistently supports **30–50% postings removal with ~negligible P@10 change**, degrading visibly beyond ~60–70% (M, varies by collection). Posting-based variants: [Chen & Lee](https://ceur-ws.org/Vol-480/paper9.pdf).
- For the SPLADE side specifically: **Lassance et al., "A Static Pruning Study on Sparse Neural Retrievers"** (SIGIR 2023, [arXiv 2304.12702](https://arxiv.org/pdf/2304.12702)) — directly applicable: pruning learned-sparse postings (doc-term, term-level) with measured MRR/NDCG curves; SPLADE tolerates moderate pruning well because low-impact expansion terms dominate postings mass.
- Risk profile is different from compression: pruning is **permanently lossy for recall** — it must go through the eval harness, and per-term pruning thresholds should be tuned against NDCG@10/Recall@100 (the rerank candidate pool depends on Recall@100, which is exactly what pruning erodes first). Expected: **20–40% of the BM25+SPLADE postings** at <0.5% NDCG@10 cost (E pending harness run).

---

## 6. Recommended shortlist + test protocol

### 6.1 Shortlist (empirically test on this corpus, in order)

1. **zstd-19 + per-domain trained dictionaries, per-doc frames** — the baseline-beater; 1 day of work; measures the true dictionary win on abstracts.
2. **Similarity-clustered 32 KB blocks (topic/MinHash order) + zstd-19 + dict** — expected best ratio (~3–4.5×); shares its doc ordering with tantivy ingestion (BP-lite for postings + tighter BMP blocks for free).
3. **Brotli-11 + custom dictionary** head-to-head against (1) — pick whichever wins ratio; both trivially fit the budget.
4. **MinHash near-dup pass** — measure the duplicate/near-dup rate first; if >3% of abstracts cluster at >0.8 Jaccard, add cluster-representative dedup + co-location in (2).
5. **(Contained, gated) LLMLingua-2 at 2× on abstracts >1.5 KB** — the only neural option that fits the budget (decode-free); run only behind the degradation protocol below, abort at >0.005 NDCG@10 loss.
6. **(Postings) Static pruning sweep on SPLADE + BM25 postings** (10/20/30/40%) via the existing eval harness, term-based thresholds per Lassance et al.

Explicitly **not** shortlisted: all LM-arithmetic-coding lossless methods (budget: NO by 1.5–6 orders of magnitude; Qwen3-1.7B-as-decompressor additionally founders on bit-exact-reproducibility across runtime upgrades), FM-index (re-confirmed), native inverted-abstract storage (larger than plaintext), smaz/shoco (wrong size class). ts_zip/Nacrith-class small-LM coding is the designated **cold-tier** option if a hot/cold split materializes (ratio ~7–8.5×, decode seconds-per-query acceptable only for cache-miss-tolerant archival).

### 6.2 Information-degradation test protocol

**Lossless methods (1–4, 6-storage):**
- *Bit-exactness:* SHA-256 of every reconstructed abstract vs original over the full 150M corpus (cheap streaming pass); any mismatch = disqualification. Re-run after every dictionary retrain or library version bump; pin dict bytes + codec version in the sidecar header.
- *No quality testing needed* beyond that — but still re-run the rerank E2E latency benchmark (100-candidate fetch p50/p95/p99) on laptop-class hardware, cold page cache.

**Lossy methods (5; also any future semantic store):**
1. **Surface fidelity:** ROUGE-L and BERTScore (deberta-large-mnli or similar) of compressed vs original abstract, distribution over a 50K-doc stratified sample (stratify by field/topic and abstract length).
2. **Reranker-input fidelity (the metric that matters):** for the existing eval query set, run the cross-encoder on (title + original abstract) vs (title + compressed abstract) for the same top-100 candidates; report per-pair **rerank-score delta** distribution and **Kendall-τ of the induced rankings**.
3. **End-to-end:** NDCG@10 / MRR@10 / Recall@100 on the existing harness (the `benchmark_llm_judge` + nDCG eval already in `data/eval_results/`), lossy vs lossless store, ≥3 seeds where sampling applies. Acceptance gate: **ΔNDCG@10 ≥ −0.005** (i.e., spend at most ~4% of the reranker's +0.12 win), else abort.
4. **Adversarial slice:** repeat (2)–(3) on the hardest decile (longest abstracts, math/chem-notation-heavy fields) where token pruning plausibly deletes load-bearing content (numbers, negations, entity names — known LLMLingua failure modes).

**Pruning (6-postings):** Recall@100 is the canary (it feeds the reranker), NDCG@10 the gate; sweep pruning rate and plot both — pick the knee.

---

## 7. Source-quality notes

- Measured-on-comparable-data (highest confidence): zstd dict mechanics + small-block dict slowdown (vendor benchmark + [issue #4175](https://github.com/facebook/zstd/issues/4175)); FSST VLDB numbers; FineZip/LLMZip throughput; BMP SIGIR results; Carmel/Lassance pruning curves; DeepMind/Nature ratio numbers.
- Author-claimed, plausible, unreproduced: ts_zip "1 MB/s on RTX 4090"; Nacrith's "7× faster than PyTorch" and ratio table (single-author preprint, Feb 2026, no code); OnPair claims (Aug 2025, no public benchmarks vs FSST yet).
- Marketing-grade / not transferable: web shared-dictionary "84–90% smaller" figures (versioned-resource deltas, not fresh text); LLMLingua "20× with little loss" (task-specific; our consumer is a cross-encoder, not GPT-4 QA).
- All ratio expectations for *this* corpus marked (E) are exactly that — the shortlist exists to replace them with measurements.

---

## 7. Multilingual corpora + dictionary-structure levers (researched 2026-06-11)

*(Appended extension. The "Source-quality notes" section above was the original report's §7; this section keeps the commissioned title and continues the report. Flags as before: (M) measured in cited source, (E) estimated/extrapolated, (C) author/vendor claim.)*

### 7.1 The corpus is ~25–30% non-English — actual OpenAlex numbers

**Live API measurement (this report, 2026-06-11, `api.openalex.org/works?group_by=language`, 295.45M works) (M):**

| Lang | Share | Lang | Share | Lang | Share |
|---|---|---|---|---|---|
| English | **73.8%** (218.0M) | French | 3.06% | Korean | 1.09% |
| Japanese | 4.26% | Portuguese | 1.84% | Indonesian | 1.08% |
| German | 3.85% | Chinese | 1.70% | Italian | 0.64% |
| Spanish | 3.44% | Russian | 1.12% | Polish/Turkish/Dutch/Ukr. | 0.4–0.5% each |

**But the labels are unreliable.** Céspedes et al., *"Evaluating the linguistic coverage of OpenAlex"* (JASIST 2025, [arXiv 2409.10633](https://arxiv.org/pdf/2409.10633), [Wiley](https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24979); 6,836 hand-verified articles) (M):
- Declared English is 75%; **corrected English is ~68%** — OpenAlex *understates* its multilinguality.
- 14.7% of language labels are false positives. Chinese is worst: only **63.7%** of declared-Chinese is actually Chinese, and ~**5.5M Chinese-language works are mislabeled as English**, plus ~4M Russian and ~3.6M Korean. (OpenAlex runs `langdetect` on title/abstract metadata only.)
- Practical consequence: **do not route on OpenAlex's `language` field.** Run own langid at ingest (fastText lid.176 / lingua-class) on the reconstructed abstract; misrouted docs land in a wrong dictionary and quietly compress worse (see 7.3).
- Caveat (E): abstract *availability* skews more English than the works table (non-English records more often lack abstracts), so the sidecar's non-English share is probably between ~15% and ~26% — measure once at ingest; it sets how much the per-language machinery below is worth.

### 7.2 UTF-8 expansion per script — what a "1 KB abstract" means per language

Per the UTF-8 encoding ([Wikipedia/UTF-8](https://en.wikipedia.org/wiki/UTF-8)): ASCII 1 B/char; Latin-ext/Greek/**Cyrillic/Arabic/Hebrew 2 B/char**; **CJK BMP (Han, kana, Hangul) 3 B/char**; supplementary-plane 4 B/char.

- **Russian/Ukrainian:** ~same character count as an equivalent English abstract but 2 B/char on letters ⇒ **~1.7–1.9× bytes/doc** for the same content (E).
- **Chinese/Japanese:** 3 B/char, but information density is far higher per character — Qwen's own rule of thumb is 1 token ≈ 3–4 chars English vs **1.5–1.8 chars Chinese** ([Qwen tokenization note](https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md), (C/M vendor)). A Chinese abstract conveys equivalent content in ~⅓–¼ the characters, so **bytes/doc roughly comparable to English, sometimes smaller** (E).
- Net effect on the budget: per-language mean-bytes/doc differ enough (±2×) that the sidecar's bytes/doc accounting and any fixed-size block tuning (§4.1) should be computed **per language stratum**, not globally.

### 7.3 zstd trained dictionaries on mixed-language corpora — dilution is real but unmeasured in literature

- **No published controlled study found** comparing one mixed-language trained dictionary vs per-language dictionaries (flagged research gap — our benchmark can be a first datapoint).
- Mechanism argues for dilution (analysis, E): COVER/fastCOVER scores candidate k-byte segments by *corpus-wide* estimated savings ([zstd dictionary system](https://deepwiki.com/facebook/zstd/5-command-line-interface), [python-zstandard docs](https://python-zstandard.readthedocs.io/en/latest/dictionaries.html)). On a 74%-English training mix, dictionary space is allocated ≈ proportional to language mass: a 110 KB dict gives Chinese ~2% ≈ 2 KB of useful history — effectively no dictionary for minority languages, while still paying full dict-management overhead.
- Measured adjacent evidence (M): a mismatched/heterogeneous dictionary can make ratios *worse than no dictionary* ([zstd issue #1703](https://github.com/facebook/zstd/issues/1703)); vendor guidance is explicitly "one dictionary per type of content," small homogeneous samples ([zstd homepage](http://facebook.github.io/zstd/), [training-set sizing #3769](https://github.com/facebook/zstd/issues/3769)).
- **Feasibility here:** every language ≥0.5% share has ≥750K abstracts (≫ the 10–100 MB training-set guidance) — training ~10–15 per-language (or per language×domain) dictionaries is trivial. The §3.1 design already reserves a 1-byte dict-ID per doc; language routing reuses it unchanged.

### 7.4 Brotli's built-in dictionary and non-English text

- The RFC 7932 static dictionary is **13,504 words of English, Spanish, Chinese, Hindi, Russian and Arabic** plus HTML/JS phrases, distilled from a 40-language corpus — i.e., less purely English-biased than folklore says, but Latin/web-weighted ([brotli author on HN, 2021](https://news.ycombinator.com/item?id=27163981)) (C, primary author).
- Same source (C): in brotli-vs-zstd web benchmarks, brotli "wins even when the static dictionary is filled with zeros" — its density edge comes from context modeling, not the dictionary; and Google deliberately benchmarked on a 93-language corpus where only 9.5% of documents were in dictionary-covered languages.
- **No measured per-language penalty table exists publicly** (gap). Expectation (E): the built-in dict's advantage on *short* text is mostly an English/Latin effect and shrinks toward zero on ja/zh/ru; a *custom* trained dictionary (Brotli v1.1 shared-dictionary, §3.2) re-equalizes. Action: the §6.1 item-3 head-to-head must be **stratified by language**, with per-language custom dicts as a third arm.

### 7.5 Token-ID recoding is *broken*, not just worse, with an English WordPiece vocab

- **bert-base-uncased disqualifier (M, documented behavior):** BERT's tokenizer space-splits every CJK-block character before WordPiece ([google-research/bert multilingual.md](https://github.com/google-research/bert/blob/master/multilingual.md)); characters absent from the 30K English vocab become **[UNK]** — WordPiece has *no byte fallback*, so recoding zh/ja abstracts through bert-base-uncased is **lossy and unrecoverable**, not merely inflationary. (Kana/Hangul fare no better: near-total [UNK] in an uncased English vocab.) Any token-recoding variant of §3.5 must therefore exclude this vocab outright.
- Multilingual vocabs fix correctness and ratio: **XLM-R** sentencepiece (250K) achieves bytes/token compression competitive with the 1M-vocab XLM-V across languages (M, comparative studies; see fertility/compression-rate methodology in the tokenizer-evaluation literature, e.g. [language-specific tokenizer design overview](https://www.emergentmind.com/topics/language-specific-tokenizer-design)); **Qwen byte-level BPE** (151,643 vocab, byte fallback ⇒ lossless on arbitrary input) claims higher compression than competitors "in most languages" ([Qwen tech report, arXiv 2309.16609](https://arxiv.org/pdf/2309.16609)) (C/M vendor): ~3–4 chars/token English, ~1.5–1.8 chars/token Chinese.
- Byte-fallback caveat (E): scripts poorly covered by the vocab degrade to ~1 token/byte — the token stream can *exceed* raw UTF-8 for tail languages; entropy coding claws some back. Net position unchanged from §3.5: low priority — but if ever done, **the only admissible vocab is a byte-fallback multilingual one (Qwen/XLM-R class)**, which also matches the shipped Qwen3 reranker stack.

### 7.6 Language-aware compression routing

- Formal literature is thin: closest hits are dictionary-selection-by-word-frequency (DSWF) in [arXiv 2412.15250](https://arxiv.org/pdf/2412.15250) and per-language codec comparisons showing codec *rankings* stay stable across languages while absolute ratios shift ([Compression of text in selected languages, PMC9460191](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9460191/)) (M, 3 languages). Treat routing as engineering practice, not research risk.
- Design for this corpus (composes with §3.1 + §4.1): langid at ingest (not OpenAlex's field, per 7.1) → cluster docs by **(language, topic)** so compression blocks are monolingual by construction → per-(language×domain) zstd dict selected by the existing 1-byte dict-ID. Total dictionary footprint ~15 × 110 KB ≈ 1.6 MB — negligible. Language is plausibly the *strongest single clustering key* for LZ cross-doc matches (E): a mixed-language block wastes window on unmatchable bytes.

### 7.7 Front coding (incremental encoding) — formal treatment and measured ratios

- Folklore form = the user-cited trick ([SO 358240](https://stackoverflow.com/questions/358240), ~50–55% on sorted wordlists (M, anecdotal)): store common-prefix length + new suffix; formalized as incremental/front coding ([Wikipedia](https://en.wikipedia.org/wiki/Incremental_encoding); [Stanford IR Book, blocked storage](https://nlp.stanford.edu/IR-book/html/htmledition/blocked-storage-1.html) — k=4 blocking + front coding saved 1.2 MB + 2.4 MB on the 11.2 MB Reuters term list ≈ **~30% combined** (M)).
- Rigorous benchmark: Brisaboa, Cánovas, Claude, Martínez-Prieto, Navarro, *Compressed String Dictionaries* (SEA 2011, [arXiv 1101.5506](https://arxiv.org/pdf/1101.5506)) (M): on 25.6M ClueWeb words / 18.5M URLs / 30.2M DBpedia URIs, best variants reach **12–30% of original size** with fast ops; **Hu-Tucker-compressed front coding (HTFC) dominates plain FC almost everywhere**; Re-Pair wins space on URLs only; FM-index loses across the board (consistent with §4.4). Production precedent: Apache Druid's segment string columns ([Imply incremental-encoding write-up](https://imply.io/blog/introducing-incremental-encoding-for-apache-druid-dictionary-encoded-columns/)).
- **Random access** = block/bucket front coding: bucket headers stored raw, binary-search headers then sequentially decode ≤ bucket-size entries — microseconds per lookup (M, extract plots in 1101.5506). Utterly negligible vs the 300 ms budget.

### 7.8 Where front coding already exists in this stack

- **tantivy has it already — twice.** The term dictionary has two implementations: the default **FST** (BurntSushi's `fst` crate / the `tantivy-fst` fork) mapping term→ordinal ([tantivy termdict docs](https://docs.rs/tantivy/0.4.2/tantivy/termdict/index.html), [ARCHITECTURE.md](https://github.com/quickwit-oss/tantivy/blob/main/ARCHITECTURE.md)), and **`tantivy-sstable`** — "sorted `&[u8]` keys … encoded using incremental encoding," keys+values zstd-compressed, block-indexed via an FST of block numbers; added because FST lookups need the whole dictionary resident while SSTable gets locality/single-fetch `get` ([tantivy-sstable crate](https://lib.rs/crates/tantivy-sstable), [maintainer on HN](https://news.ycombinator.com/item?id=40942199)). **Conclusion: term-vocabulary compression is already solved upstream; no action, and per-section term vocabularies (application b) are a non-lever.**
- **SQLite has none.** Index b-trees store the **full key in every entry** — no prefix compression, plus ~2-byte cell pointer and header varints per entry ([index key format write-up](https://dev.to/lovestaco/index-key-format-in-sqlite-40pn), [sqlite-users discussion](https://sqlite-users.sqlite.narkive.com/GpXsoNCK/tuning-a-sqlite-database-for-best-compression)). This is the §7.10(c) overhead.

### 7.9 Succinct alternatives: MARISA/LOUDS tries, and what 100M keys cost

- **marisa-trie** (LOUDS + recursive Patricia, [s-yata/marisa-trie](https://github.com/s-yata/marisa-trie); `pip install marisa-trie`): benchmark (M, [project docs](https://github.com/pytries/marisa-trie/blob/master/docs/benchmarks.rst)): **9,805,576 Wikipedia titles → 50.8 MB ≈ 5.2 B/key** (vs darts-clone 376 MB, tx-trie/LOUDS 127 MB); "50–100× less memory than a Python dict," µs lookups. Extrapolated (E): ~100M heterogeneous string keys ≈ 0.5–1 GB.
- **pyahocorasick** is a multi-pattern *matching* automaton, not a compressed dictionary — not applicable to key storage here.
- For **sorted numeric IDs** (our case, see below), succinct tries are the wrong tool: **Elias-Fano** encodes a sorted set of n integers from universe U in n·(⌈log₂(U/n)⌉ + 2) bits with O(1) `select` — the standard in PISA-class index toolkits.

### 7.10 Concrete applications to THIS corpus

**(a) OpenAlex ID key columns ("W" + 8–11 digits).** Strip the constant "W", treat as integers (max ~4.5×10⁹ < 2⁶³). For n = 150M sorted IDs, U ≈ 4.5×10⁹:
- **Elias-Fano:** log₂(U/n)+2 ≈ **7 bits/key ≈ 0.9 B/key → ~130 MB** (E, formula-derived).
- **Delta-varint:** mean gap ~30 ⇒ 1–2 B/key → **~150–300 MB** (E); sampled absolute values every 64–128 keys give near-O(1) access.
- **Front-coded decimal strings:** prefix-len byte + ~2–3 suffix bytes ≈ 3–4 B/key → ~500–600 MB (E).
- **Verdict: front coding loses to numeric encodings here.** The IDs are integers in costume; the sorted-wordlist trick is the right instinct but the wrong tool for this column. (M needed only if measurement is free; the ordering is robust.)

**(b) Per-section term vocabularies:** already front-coded/FST'd inside tantivy (7.8); term dictionaries are MBs against a ~100 GB problem. **No action.**

**(c) SQLite id→rowid key index:** a `TEXT` unique index on 150M 11-char IDs costs ≈ full key + rowid varint + cell overhead ≈ 16–20 B/entry → **~2.5–3 GB** (E from 7.8 format). Two cheaper designs, in order:
1. **Eliminate the index entirely:** `rowid = numeric part of the work ID` (`INTEGER PRIMARY KEY`, fits 2⁶³). Lookup = clustered b-tree walk on the rowid itself, zero extra storage. Works iff sidecar rows can live in ID order.
2. If §4.1's **cluster order** wins the layout (it should), store blobs in cluster order and keep one external **Elias-Fano(IDs) + bit-packed position permutation** (150M × 27 bits ≈ 506 MB, or ×4 B plain = 600 MB): ID → position in O(1), mmap-friendly, no SQLite b-tree at all. Net saving vs naive SQLite index: **~2–2.5 GB**, and lookups drop from b-tree page walks to two array probes.
- **Random-access cost:** every option above is sub-µs–µs per key; 100 lookups consume < 1 ms of the 300 ms budget. The budget is irrelevant to this entire section — these are size/simplicity decisions.

### 7.11 Verdict table

| Method | Expected ratio / size | Random-access fit | Multilingual robustness | Worth testing here? |
|---|---|---|---|---|
| zstd + **per-(language×domain) dicts**, langid routing | 2–3× En; holds for non-En (E) | per-doc O(1) | **Good** (by construction) | **YES — upgrade of shortlist #1** |
| zstd single mixed-corpus dict | 2–3× En, **degraded on ≤4%-share langs** (E; no literature — we'd be first datapoint) | per-doc O(1) | Poor (dilution, 7.3) | Only as control arm |
| Brotli built-in dict on non-English | penalty unmeasured; edge likely →0 on ja/zh/ru (E) | per-doc O(1) | Weak built-in; fixed by custom dict | YES — stratify §6.1 #3 by language |
| **(language, topic)-clustered blocks** + per-block dict | +5–15% over mixed-language blocks (E) | per-block (as §4.1) | **Excellent** | **YES — fold into shortlist #2** |
| Token recoding, bert-base-uncased vocab | n/a — **[UNK]-lossy on CJK** (M) | — | **Broken** | **NO — disqualified** |
| Token recoding, Qwen/XLM-R byte-fallback vocab | ~2× pre-entropy; lossless (M/C) | per-doc O(1) | Good (byte fallback; tail langs inflate) | Low priority (unchanged §3.5) |
| Front coding (block FC/HTFC) on string keys | dict → 12–30% of original (M, 1101.5506) | µs (bucket scan) | n/a | NO for W-IDs (see next row); keep in toolbox for true string keys |
| **Elias-Fano / delta-varint** numeric ID index | **~0.9–2 B/key ≈ 130–300 MB** vs ~2.5–3 GB SQLite TEXT index (E) | O(1)/near-O(1), sub-µs | n/a | **YES — or better, rowid=numeric-ID for zero cost** |
| marisa-trie (LOUDS) key store | ~5 B/key (M, 9.8M titles) | µs | n/a | NO for numeric IDs; revisit only for DOI/alias strings |
| SQLite native TEXT key index | baseline (no prefix compression, full key/entry) | b-tree walk | n/a | NO — replace per 7.10(c) |

**Bottom line for this section:** (1) The multilingual share (~26% declared, more in truth) is large enough that the §6.1 dictionary experiments must be language-stratified and language-routed, at near-zero added complexity (1-byte dict-ID already planned); single mixed dictionaries silently tax 40M+ non-English abstracts. (2) The front-coding instinct is correct and already embodied in tantivy's term dictionary; for the one place it seems newly applicable — the W-ID key column — plain numeric encodings (rowid-as-ID, or Elias-Fano) dominate it by 3–4× and remove ~2.5 GB of SQLite index overhead. Neither lever threatens the 300 ms budget by more than ~1 ms.

### 7.12 Source-quality notes for §7

- Measured, high confidence: OpenAlex API language counts (queried live 2026-06-11); Céspedes et al. JASIST 2025 verification stats; Brisaboa et al. SEA 2011 dictionary benchmarks; marisa-trie benchmark tables; BERT CJK tokenization behavior (documented code behavior); tantivy-sstable format docs.
- Author-claimed, credible, unreproduced: brotli dictionary composition + "wins with zeroed dictionary" (brotli author, HN); Qwen chars/token rules of thumb and "higher compression in most languages."
- Identified literature gaps (we would generate first public numbers): mixed-vs-per-language zstd dictionary dilution; Brotli per-language built-in-dict penalty table.

## 8. MEASURED on this corpus (scripts/eval_text_compression.py, 2026-06-11)

20,000 real abstracts (24 MB, avg 1,203 B/doc), random-access drill = 100
random fetches (the rerank budget is 300 ms). Full JSON:
`data/eval_results/text_compression_eval.json`.

| variant | ratio | B/doc | fetch-100 | budget | lossless |
|---|---|---|---|---|---|
| zstd3 (today's baseline) | 1.98 | 607 | 0.5 ms | OK | yes |
| zstd19 + per-section dicts | 2.82 | 427 | 0.5 ms | OK | yes |
| brotli11 (builtin dict) | 2.59 | 465 | 1.3 ms | OK | yes |
| 32 KB blocks, corpus order | 2.87 | 419 | 10.4 ms | OK | yes |
| **32 KB blocks, clustered (section+top-SPLADE-term)** | **2.99** | **402** | 7.5 ms | OK | yes |
| token-ID varint+zstd-dict | 2.96 | 406 | 44 ms | OK | ids-only (text re-detokenized; CJK lossy) |
| **SLM rank coding (gpt2, LLMZip-style)** | 2.72 | 395 | **423,396 ms** | **FAIL ×1,400** | **NO** — same-machine round trip already breaks (batched-vs-incremental kernel numerics shift logits → rank mismatch), independently confirming §2's reproducibility verdict |
| lossy stopword-drop | 2.25 | 506 | — | — | cos 0.968 → **fails the 0.98 degradation gate** |
| lossy truncate-128tok | 4.34 | 263 | — | — | cos 0.955 → **fails the gate** |

ID key column (sorted W-ids, 12 B/id raw): front coding 7.9 B (1.5×),
**delta-varint of the numeric part 2.66 B (4.5×)** — matches §7's verdict that
numeric encoding beats front coding for this column.

Per-language (py3langid on the sample; scroll-order sample, share not
corpus-representative): per-language dicts beat the mixed dict by **+4.7% (en)
to +37.6% (zh), +47.7% (ru), +57.2% (tr)** — language-aware dictionary routing
is validated and nearly free (1-byte dict id). Brotli's builtin dict loses to
zstd own-dict for every non-English language measured.

**Adopted recommendation:** hot-section abstract sidecar → zstd-19 with
per-(section×language) trained dictionaries; consider clustered 32 KB blocks
(+6% ratio at +7 ms/100 docs) if B/doc matters more than point-lookup
simplicity. SLM decode: cold-tier/archival only, exactly as §2 concluded.
