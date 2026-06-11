#!/usr/bin/env python3
"""Lexical-DB stored-text compression eval — measured ratios, decode speed,
random-access cost, and INFORMATION DEGRADATION for the abstract sidecar.

Implements the test protocol from docs/LEXICAL_STORAGE_RESEARCH.md (§6) on a
real corpus sample. The serving constraint is the rerank fetch: ~100 abstracts
must decode in <300 ms (cross-encoder rerank consumes title+abstract; worth
~+0.12 NDCG@10 — that's the information being protected).

LOSSLESS variants (degradation test = bit-exact round trip, SHA-256):
  zstd3 / zstd19           per-doc frames, no dict (baselines)
  zstd19_dict              per-doc frames + per-SECTION trained dictionaries
  brotli11                 per-doc, built-in 120 KB English dict
  block32k_zstd19          32 KB blocks, corpus order (random access = 1 block)
  block32k_clustered       32 KB blocks, docs ordered by (section, top SPLADE
                           term) — the similarity-clustering lever
  tokenid_varint_zstd      bert-base-uncased token IDs, varint, zstd-dict
                           (NOT bit-exact for whitespace/casing — measured as
                           "token-lossless": ids round-trip, text re-detokenized)
  slm_rank_zstd  (--slm)   "decode with an SLM": LLMZip-style token-RANK coding
                           under a small causal LM (gpt2), ranks zstd-coded.
                           Bit-exact iff the model's logits are reproducible —
                           verified by round trip; decode SPEED is the question.

LOSSY variants (degradation test = dense-embedding cosine original-vs-degraded
with the production v5 bi-encoder + token-retention stats; flagged if cosine
falls below 0.98 — paraphrase-level damage):
  lossy_stopword_drop      drop NLTK-style stopwords (decode-free)
  lossy_trunc128tok        truncate to first 128 wordpiece tokens

Usage
─────
    .venv/bin/python3 scripts/eval_text_compression.py \
        [--docs 20000] [--slm] [--slm-docs 40] [--no-degradation]
Sample is exported once from the source cluster and cached
(data/text_compression_sample.jsonl.gz).
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import zstandard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_text_compression")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SOURCE_URL = os.environ.get("SFU_MIGRATION_SOURCE",
                            "http://host.docker.internal:9200").rstrip("/")
SOURCE_INDEX = os.environ.get("SFU_MIGRATION_INDEX", "openalex_works")
SAMPLE_PATH = REPO_ROOT / "data/text_compression_sample.jsonl.gz"
OUTPUT = REPO_ROOT / "data/eval_results/text_compression_eval.json"
RERANK_FETCH = 100          # abstracts per query the rerank stage needs
BLOCK_BYTES = 32 * 1024
DICT_SIZE = 112 * 1024
RANDOM_FETCH_TRIALS = 30    # how many 100-doc random fetches to time

_STOPWORDS = set("""a about above after again against all am an and any are as at be because been
before being below between both but by could did do does doing down during each few for from
further had has have having he her here hers herself him himself his how i if in into is it its
itself just me more most my myself no nor not now of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while
who whom why will with you your yours yourself yourselves""".split())


# ── sample ────────────────────────────────────────────────────────────────────

def get_sample(n_docs: int) -> list[dict]:
    if SAMPLE_PATH.exists():
        docs = [json.loads(l) for l in gzip.open(SAMPLE_PATH, "rt")]
        if len(docs) >= n_docs:
            return docs[:n_docs]
    import requests
    from lib.thinclient.sections import classify_doc
    logger.info("exporting %d-doc sample from %s/%s", n_docs, SOURCE_URL, SOURCE_INDEX)
    docs: list[dict] = []
    body = {"size": 2000, "query": {"match_all": {}},
            "_source": ["title", "abstract", "sparse_field"]}
    r = requests.post(f"{SOURCE_URL}/{SOURCE_INDEX}/_search?scroll=5m",
                      json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    while len(docs) < n_docs:
        hits = data["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            src = h["_source"]
            ab = src.get("abstract") or ""
            if len(ab) < 100:
                continue
            sparse = src.get("sparse_field") or {}
            top_term = max(sparse, key=sparse.get) if sparse else ""
            docs.append({"id": h["_id"], "abstract": ab,
                         "section": classify_doc(src.get("title"), ab),
                         "top_term": top_term})
            if len(docs) >= n_docs:
                break
        data = requests.post(f"{SOURCE_URL}/_search/scroll",
                             json={"scroll": "5m", "scroll_id": data["_scroll_id"]},
                             timeout=60).json()
    with gzip.open(SAMPLE_PATH, "wt") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    return docs


# ── lossless per-doc variants ────────────────────────────────────────────────

def measure_perdoc(name: str, docs: list[dict], compress, decompress) -> dict:
    payloads = [d["abstract"].encode("utf-8") for d in docs]
    raw = sum(len(p) for p in payloads)
    t0 = time.perf_counter()
    blobs = [compress(i, p) for i, p in enumerate(payloads)]
    enc_secs = time.perf_counter() - t0
    comp = sum(len(b) for b in blobs)

    # bit-exactness
    exact = all(decompress(i, blobs[i]) == payloads[i] for i in range(len(blobs)))

    # rerank-budget drill: RANDOM_FETCH_TRIALS x decode 100 random docs
    rng = random.Random(42)
    t0 = time.perf_counter()
    for _ in range(RANDOM_FETCH_TRIALS):
        for i in rng.sample(range(len(blobs)), min(RERANK_FETCH, len(blobs))):
            decompress(i, blobs[i])
    fetch_ms = (time.perf_counter() - t0) * 1000 / RANDOM_FETCH_TRIALS

    return {"variant": name, "lossless": exact, "ratio": round(raw / comp, 3),
            "bytes_per_doc": round(comp / len(blobs), 1),
            "encode_secs": round(enc_secs, 2),
            "fetch100_ms": round(fetch_ms, 2),
            "fits_300ms_budget": fetch_ms < 300}


def run_perdoc_variants(docs: list[dict]) -> list[dict]:
    out = []

    z3 = zstandard.ZstdCompressor(level=3)
    z3d = zstandard.ZstdDecompressor()
    out.append(measure_perdoc("zstd3", docs,
                              lambda i, p: z3.compress(p),
                              lambda i, b: z3d.decompress(b)))

    z19 = zstandard.ZstdCompressor(level=19)
    out.append(measure_perdoc("zstd19", docs,
                              lambda i, p: z19.compress(p),
                              lambda i, b: z3d.decompress(b)))

    # per-section trained dictionaries
    by_section = defaultdict(list)
    for i, d in enumerate(docs):
        by_section[d["section"]].append(i)
    cdict, ddict = {}, {}
    for sec, idxs in by_section.items():
        samples = [docs[i]["abstract"].encode() for i in idxs[:20000]]
        try:
            zd = zstandard.train_dictionary(DICT_SIZE, samples)
        except zstandard.ZstdError:
            zd = None
        cdict[sec] = (zstandard.ZstdCompressor(level=19, dict_data=zd) if zd else z19)
        ddict[sec] = (zstandard.ZstdDecompressor(dict_data=zd) if zd else z3d)
    out.append(measure_perdoc(
        "zstd19_dict_per_section", docs,
        lambda i, p: cdict[docs[i]["section"]].compress(p),
        lambda i, b: ddict[docs[i]["section"]].decompress(b)))

    try:
        import brotli
        out.append(measure_perdoc(
            "brotli11", docs,
            lambda i, p: brotli.compress(p, quality=11, mode=brotli.MODE_TEXT),
            lambda i, b: brotli.decompress(b)))
    except ImportError:
        logger.warning("brotli not installed — skipping")

    return out


# ── block variants ───────────────────────────────────────────────────────────

def measure_blocks(name: str, docs: list[dict], order: list[int],
                   zdict_per_block_group: bool = True) -> dict:
    """Pack ordered docs into ~32 KB blocks; random access = decode one block."""
    payloads = [docs[i]["abstract"].encode("utf-8") for i in order]
    raw = sum(len(p) for p in payloads)

    samples = [p for p in payloads[:20000]]
    try:
        zd = zstandard.train_dictionary(DICT_SIZE, samples)
        cctx = zstandard.ZstdCompressor(level=19, dict_data=zd)
        dctx = zstandard.ZstdDecompressor(dict_data=zd)
    except zstandard.ZstdError:
        cctx = zstandard.ZstdCompressor(level=19)
        dctx = zstandard.ZstdDecompressor()

    blocks, doc_block = [], {}
    cur, cur_len = [], 0
    t0 = time.perf_counter()
    for pos, p in enumerate(payloads):
        cur.append(p)
        cur_len += len(p)
        doc_block[order[pos]] = (len(blocks), len(cur) - 1)
        if cur_len >= BLOCK_BYTES:
            blocks.append(cctx.compress(b"\x00".join(cur)))
            cur, cur_len = [], 0
    if cur:
        blocks.append(cctx.compress(b"\x00".join(cur)))
    enc_secs = time.perf_counter() - t0
    comp = sum(len(b) for b in blocks)

    def fetch(i: int) -> bytes:
        bi, off = doc_block[i]
        return dctx.decompress(blocks[bi]).split(b"\x00")[off]

    exact = all(fetch(i) == docs[i]["abstract"].encode() for i in order[::97])

    rng = random.Random(42)
    t0 = time.perf_counter()
    for _ in range(RANDOM_FETCH_TRIALS):
        for i in rng.sample(order, min(RERANK_FETCH, len(order))):
            fetch(i)
    fetch_ms = (time.perf_counter() - t0) * 1000 / RANDOM_FETCH_TRIALS

    return {"variant": name, "lossless": exact,
            "ratio": round(raw / comp, 3),
            "bytes_per_doc": round(comp / len(payloads), 1),
            "encode_secs": round(enc_secs, 2), "blocks": len(blocks),
            "fetch100_ms": round(fetch_ms, 2),
            "fits_300ms_budget": fetch_ms < 300}


# ── token-id recoding ────────────────────────────────────────────────────────

def _wordpiece_tokenizer():
    from lib.opensearch_retriever import _resolve_tokenizer_dir
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        str(_resolve_tokenizer_dir(str(REPO_ROOT / "models" / "splade_onnx"))))


def run_tokenid_variant(docs: list[dict]) -> dict:
    tok = _wordpiece_tokenizer()

    def varint(ids: list[int]) -> bytes:
        out = bytearray()
        for v in ids:
            while v >= 0x80:
                out.append((v & 0x7F) | 0x80)
                v >>= 7
            out.append(v)
        return bytes(out)

    def unvarint(b: bytes) -> list[int]:
        out, v, shift = [], 0, 0
        for byte in b:
            v |= (byte & 0x7F) << shift
            if byte & 0x80:
                shift += 7
            else:
                out.append(v)
                v, shift = 0, 0
        return out

    payloads = [d["abstract"] for d in docs]
    raw = sum(len(p.encode()) for p in payloads)
    t0 = time.perf_counter()
    encoded = [varint(tok.encode(p, add_special_tokens=False)) for p in payloads]
    try:
        zd = zstandard.train_dictionary(DICT_SIZE, encoded[:20000])
        cctx = zstandard.ZstdCompressor(level=19, dict_data=zd)
        dctx = zstandard.ZstdDecompressor(dict_data=zd)
    except zstandard.ZstdError:
        cctx = zstandard.ZstdCompressor(level=19)
        dctx = zstandard.ZstdDecompressor()
    blobs = [cctx.compress(e) for e in encoded]
    enc_secs = time.perf_counter() - t0
    comp = sum(len(b) for b in blobs)

    # token-lossless: ids round-trip exactly; TEXT is not byte-identical
    # (wordpiece de-tokenization loses original whitespace/casing).
    ids_ok = all(unvarint(dctx.decompress(blobs[i])) ==
                 tok.encode(payloads[i], add_special_tokens=False)
                 for i in range(0, len(blobs), 97))

    rng = random.Random(42)
    t0 = time.perf_counter()
    for _ in range(RANDOM_FETCH_TRIALS):
        for i in rng.sample(range(len(blobs)), min(RERANK_FETCH, len(blobs))):
            tok.decode(unvarint(dctx.decompress(blobs[i])))
    fetch_ms = (time.perf_counter() - t0) * 1000 / RANDOM_FETCH_TRIALS

    return {"variant": "tokenid_varint_zstd_dict", "lossless": False,
            "token_lossless": ids_ok, "ratio": round(raw / comp, 3),
            "bytes_per_doc": round(comp / len(blobs), 1),
            "encode_secs": round(enc_secs, 2),
            "fetch100_ms": round(fetch_ms, 2),
            "fits_300ms_budget": fetch_ms < 300,
            "note": "ids round-trip; text re-detokenized (casing/whitespace lost)"}


# ── SLM rank coding (LLMZip-style) ───────────────────────────────────────────

def run_slm_variant(docs: list[dict], n_docs: int) -> dict:
    """Token-RANK coding under gpt2: each next-token is stored as its rank in
    the model's sorted next-token distribution; ranks compress extremely well.
    Decode = greedy re-generation following stored ranks (1 forward/token)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

    sub = docs[:n_docs]
    cctx = zstandard.ZstdCompressor(level=19)
    raw = comp = 0
    enc_secs = dec_secs = 0.0
    exact = True

    for d in sub:
        # truncate by TOKENS (gpt2 position limit 1024; CJK byte-fallback BPE
        # can exceed it even for short char counts)
        ids = tok.encode(d["abstract"])[:512]
        text = tok.decode(ids)
        raw += len(text.encode())

        t0 = time.perf_counter()
        ranks = []
        with torch.inference_mode():
            input_ids = torch.tensor([ids], device=device)
            logits = model(input_ids).logits[0]          # (T, V)
            order = torch.argsort(logits, dim=-1, descending=True)
            for t in range(len(ids) - 1):
                ranks.append(int((order[t] == ids[t + 1]).nonzero()[0, 0]))
        blob = cctx.compress(bytes(json.dumps([ids[0]] + ranks), "utf-8"))
        enc_secs += time.perf_counter() - t0
        comp += len(blob)

        # decode: sequential re-generation, 1 forward per token (the honest cost)
        t0 = time.perf_counter()
        stored = json.loads(zstandard.ZstdDecompressor().decompress(blob))
        first, rks = stored[0], stored[1:]
        out_ids = [first]
        with torch.inference_mode():
            for r in rks:
                logits = model(torch.tensor([out_ids], device=device)).logits[0, -1]
                nxt = int(torch.argsort(logits, descending=True)[r])
                out_ids.append(nxt)
        dec_secs += time.perf_counter() - t0
        if out_ids != ids:
            exact = False

    per100_ms = dec_secs / len(sub) * RERANK_FETCH * 1000
    return {"variant": "slm_rank_zstd (gpt2, LLMZip-style)", "lossless": exact,
            "docs_tested": len(sub), "device": device,
            "ratio": round(raw / comp, 3), "bytes_per_doc": round(comp / len(sub), 1),
            "encode_secs": round(enc_secs, 2),
            "fetch100_ms": round(per100_ms, 1),
            "fits_300ms_budget": per100_ms < 300,
            "note": "bit-exact ONLY with identical model/hardware/dtype — the "
                    "reproducibility caveat from LEXICAL_STORAGE_RESEARCH.md; "
                    "decode is 1 LM forward per token"}


# ── lossy variants + degradation metrics ─────────────────────────────────────

def run_lossy_variants(docs: list[dict], measure_degradation: bool) -> list[dict]:
    tok = _wordpiece_tokenizer()
    z19 = zstandard.ZstdCompressor(level=19)

    def stopword_drop(text: str) -> str:
        return " ".join(w for w in text.split() if w.lower() not in _STOPWORDS)

    def trunc128(text: str) -> str:
        ids = tok.encode(text, add_special_tokens=False)[:128]
        return tok.decode(ids)

    variants = [("lossy_stopword_drop", stopword_drop),
                ("lossy_trunc128tok", trunc128)]
    out = []

    emb_model = None
    if measure_degradation:
        from lib.opensearch_retriever import _get_dense_model
        emb_model = _get_dense_model(str(REPO_ROOT / "models" / "sfu-academic-embed-v5"))

    for name, fn in variants:
        raw = comp = 0
        degraded_texts, orig_texts = [], []
        for d in docs[:2000]:
            orig = d["abstract"]
            deg = fn(orig)
            raw += len(orig.encode())
            comp += len(z19.compress(deg.encode()))
            orig_texts.append(orig)
            degraded_texts.append(deg)
        row = {"variant": name, "lossless": False,
               "ratio": round(raw / comp, 3),
               "bytes_per_doc": round(comp / len(orig_texts), 1)}
        if emb_model is not None:
            import numpy as np
            a = emb_model.encode(orig_texts[:500], normalize_embeddings=True,
                                 convert_to_numpy=True, batch_size=64)
            b = emb_model.encode(degraded_texts[:500], normalize_embeddings=True,
                                 convert_to_numpy=True, batch_size=64)
            cos = float(np.mean(np.sum(a * b, axis=1)))
            row["embed_cosine_vs_original"] = round(cos, 4)
            row["degradation_flag"] = cos < 0.98
        out.append(row)
    return out


# ── multilingual analysis ────────────────────────────────────────────────────

def run_language_analysis(docs: list[dict]) -> dict:
    """Per-language compression behaviour: the corpus is NOT all-English, and
    two shortlisted methods are English-biased (Brotli built-in dict; WordPiece
    token recoding). Measures, per detected language with >=200 samples:
    zstd19 ratio with (a) the MIXED-corpus dict, (b) a PER-LANGUAGE dict, and
    (c) brotli11 — quantifying dictionary dilution and English bias."""
    try:
        import py3langid
    except ImportError:
        logger.warning("py3langid not installed — skipping language analysis")
        return {}
    import brotli

    by_lang = defaultdict(list)
    for d in docs:
        lang, _ = py3langid.classify(d["abstract"][:600])
        by_lang[lang].append(d["abstract"].encode("utf-8"))

    mixed_samples = [d["abstract"].encode() for d in docs[:20000]]
    zd_mixed = zstandard.train_dictionary(DICT_SIZE, mixed_samples)
    c_mixed = zstandard.ZstdCompressor(level=19, dict_data=zd_mixed)

    out = {"language_share": {}, "per_language": {}}
    total = len(docs)
    for lang, payloads in sorted(by_lang.items(), key=lambda kv: -len(kv[1])):
        out["language_share"][lang] = round(len(payloads) / total, 4)
        if len(payloads) < 200:
            continue
        raw = sum(len(p) for p in payloads)
        mixed = sum(len(c_mixed.compress(p)) for p in payloads)
        try:
            zd_own = zstandard.train_dictionary(DICT_SIZE, payloads[:20000])
            c_own = zstandard.ZstdCompressor(level=19, dict_data=zd_own)
            own = sum(len(c_own.compress(p)) for p in payloads)
        except zstandard.ZstdError:
            own = None
        br = sum(len(brotli.compress(p, quality=11, mode=brotli.MODE_TEXT))
                 for p in payloads)
        out["per_language"][lang] = {
            "docs": len(payloads),
            "avg_doc_bytes": raw // len(payloads),
            "zstd19_mixed_dict_ratio": round(raw / mixed, 3),
            "zstd19_own_dict_ratio": round(raw / own, 3) if own else None,
            "own_dict_gain_pct": round((mixed - own) / mixed * 100, 1) if own else None,
            "brotli11_ratio": round(raw / br, 3),
        }
    return out


# ── key-column structures (front coding — the sorted-wordlist trick) ────────

def run_id_keycolumn_variants(docs: list[dict]) -> list[dict]:
    """Storage of the SORTED OpenAlex ID key column (sidecar/meta keys).
    Front coding = byte of common-prefix-length with the previous key + new
    suffix (the classic sorted-dictionary incremental encoding); compared with
    delta-varint on the numeric part and plain zstd. Random access uses 1/16
    block restarts (offsets table), like block front coding / Lucene terms."""
    ids = sorted(d["id"] for d in docs)
    raw = sum(len(i) + 1 for i in ids)  # +1 terminator, like the CP/M layout

    # front coding with restarts every 16
    fc = bytearray()
    restarts = []
    prev = ""
    for n, cur in enumerate(ids):
        if n % 16 == 0:
            restarts.append(len(fc))
            common = 0
        else:
            common = 0
            for a, b in zip(prev, cur):
                if a != b:
                    break
                common += 1
            common = min(common, 255)
        suffix = cur[common:].encode()
        fc.append(common)
        fc.append(len(suffix))
        fc.extend(suffix)
        prev = cur
    fc_total = len(fc) + 4 * len(restarts)

    # delta-varint of numeric part (ids are 'W' + digits)
    def varint(v: int) -> bytes:
        out = bytearray()
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
        return bytes(out)

    nums = sorted(int(i[1:]) for i in ids if i[1:].isdigit())
    dv = bytearray()
    last = 0
    for v in nums:
        dv.extend(varint(v - last))
        last = v
    dv_total = len(dv)

    zs = len(zstandard.ZstdCompressor(level=19).compress("\n".join(ids).encode()))

    def row(name, total, lossless=True, note=""):
        return {"variant": name, "lossless": lossless,
                "ratio": round(raw / total, 3),
                "bytes_per_doc": round(total / len(ids), 2), "note": note,
                "scope": "id_key_column"}

    return [
        row("ids_raw_terminated", raw),
        row("ids_front_coded_b16", fc_total,
            note="common-prefix byte + suffix, restart every 16 (random access)"),
        row("ids_delta_varint_numeric", dv_total,
            note="sorted numeric deltas; needs digits-only ids"),
        row("ids_zstd19_blob", zs, note="no random access (whole-list blob)"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=20000)
    parser.add_argument("--slm", action="store_true",
                        help="run the gpt2 rank-coding variant (slow)")
    parser.add_argument("--slm-docs", type=int, default=40)
    parser.add_argument("--no-degradation", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    docs = get_sample(args.docs)
    raw_bytes = sum(len(d["abstract"].encode()) for d in docs)
    logger.info("sample: %d abstracts, %.1f MB raw, avg %d B/doc",
                len(docs), raw_bytes / 1e6, raw_bytes // len(docs))

    results = run_perdoc_variants(docs)

    order_corpus = list(range(len(docs)))
    results.append(measure_blocks("block32k_zstd19_dict_corpus_order", docs, order_corpus))
    order_clustered = sorted(order_corpus,
                             key=lambda i: (docs[i]["section"], docs[i]["top_term"]))
    results.append(measure_blocks("block32k_zstd19_dict_clustered", docs, order_clustered))

    results.append(run_tokenid_variant(docs))
    if args.slm:
        results.append(run_slm_variant(docs, args.slm_docs))
    results.extend(run_lossy_variants(docs, not args.no_degradation))
    results.extend(run_id_keycolumn_variants(docs))
    languages = run_language_analysis(docs)

    out = {"config": {"docs": len(docs), "raw_mb": round(raw_bytes / 1e6, 1),
                      "avg_doc_bytes": raw_bytes // len(docs),
                      "rerank_fetch": RERANK_FETCH, "budget_ms": 300,
                      "source": f"{SOURCE_URL}/{SOURCE_INDEX}"},
           "results": results, "languages": languages}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 100)
    print(f"ABSTRACT-SIDECAR COMPRESSION EVAL — {len(docs):,} real abstracts "
          f"({raw_bytes / 1e6:.0f} MB raw)")
    print("=" * 100)
    print(f"{'variant':<38} {'ratio':>6} {'B/doc':>6} {'fetch100ms':>10} "
          f"{'budget':>7} {'lossless':>9} {'degradation':>12}")
    print("-" * 100)
    for r in results:
        deg = ""
        if "embed_cosine_vs_original" in r:
            deg = f"cos={r['embed_cosine_vs_original']}"
        elif r.get("token_lossless"):
            deg = "ids exact"
        print(f"{r['variant']:<38} {r['ratio']:>6} {r['bytes_per_doc']:>6} "
              f"{r.get('fetch100_ms', '—'):>10} "
              f"{'OK' if r.get('fits_300ms_budget') else ('FAIL' if 'fetch100_ms' in r else '—'):>7} "
              f"{str(r['lossless']):>9} {deg:>12}")
    print("=" * 100)
    if languages:
        print("\nLANGUAGE BREAKDOWN (zstd-dict dilution + Brotli English bias)")
        print("-" * 100)
        print(f"{'lang':<6} {'share':>7} {'docs':>7} {'B/doc':>6} "
              f"{'mixed-dict':>10} {'own-dict':>9} {'own gain':>9} {'brotli11':>9}")
        for lang, st in languages.get("per_language", {}).items():
            print(f"{lang:<6} {languages['language_share'][lang] * 100:>6.1f}% "
                  f"{st['docs']:>7} {st['avg_doc_bytes']:>6} "
                  f"{st['zstd19_mixed_dict_ratio']:>10} "
                  f"{st['zstd19_own_dict_ratio'] or '—':>9} "
                  f"{str(st['own_dict_gain_pct']) + '%' if st['own_dict_gain_pct'] is not None else '—':>9} "
                  f"{st['brotli11_ratio']:>9}")
        print("-" * 100)
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
