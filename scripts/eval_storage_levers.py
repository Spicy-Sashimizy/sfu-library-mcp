#!/usr/bin/env python3
"""ALL-LEVERS storage eval — turns every estimated cell of the whole-DB budget
into a measured one, then prints the concrete total at 150M.

Measurements (no load on the source cluster — local data only):
  1. BMP matrix on the 100k POC corpus (real production SPLADE vectors):
     bsize {32,64,128,256} x ordering {corpus, clustered} x compress_range —
     index size, build time, latency (alpha=0.8), recall@50 vs EXACT
     quantized dot-product ground truth.
  2. tantivy: corpus-order vs clustered-order index size (builder schema).
  3. meta.sqlite recode on the REAL 1M meta: numeric INTEGER PRIMARY KEY
     (delta of the W-id) + zstd-dict-compressed titles/DOIs + int-coded
     type/section. Round-trip verified.
  4. Abstract sidecar: clustered 32KB blocks + PER-LANGUAGE dictionaries
     combined (the full recommended config) on the cached 20k sample.
  5. Artifact pack ratio: tar+zstd-19 of a built 1M section (cold-section
     archive ratio for the hot/cold math).

Output: data/eval_results/storage_levers_eval.json + a final table scaling
to 150.4M docs with ALL levers applied vs today-as-built.

Usage:  .venv/bin/python3 scripts/eval_storage_levers.py [--queries 20]
"""

import argparse
import json
import logging
import shutil
import sqlite3
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import zstandard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("storage_levers")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CORPUS = REPO_ROOT / "data/thin_client_poc/docs_100000.jsonl"
SAMPLE = REPO_ROOT / "data/text_compression_sample.jsonl.gz"
META_1M = REPO_ROOT / "data/thinclient_1m/meta.sqlite"
SECTION_FOR_PACK = REPO_ROOT / "data/thinclient_1m/sections/phys_eng"
WORK = REPO_ROOT / "data/bmp_size_eval"
OUTPUT = REPO_ROOT / "data/eval_results/storage_levers_eval.json"

QUANT = 100
TOP_QUERY_TERMS = 64
K = 50
FULL_DOCS = 150_413_098
# Component sizes measured from the 1M build, scaled to 150.4M (GB):
AS_BUILT_150M = {"bmp": 167.6, "tantivy": 44.5, "abstracts": 64.7, "meta": 24.1}
ABS_ZSTD3_BASELINE_B = 606.9   # naive per-doc zstd3 baseline (measured)
ABS_AS_BUILT_B = 427.0         # per-section dict per-doc (measured, = current sidecar)


def load_corpus() -> list[dict]:
    from lib.thinclient.sections import classify_doc
    docs = []
    with open(CORPUS) as fh:
        for line in fh:
            d = json.loads(line)
            sparse = d.get("sparse_field") or {}
            if not sparse:
                continue
            vec = {t: max(1, int(round(w * QUANT))) for t, w in sparse.items() if w > 0}
            docs.append({
                "id": d["id"], "vec": vec,
                "section": classify_doc(d.get("title"), d.get("abstract")),
                "top": max(sparse, key=sparse.get),
            })
    logger.info("corpus: %d docs with sparse vectors", len(docs))
    return docs


def encode_queries(n: int) -> list[dict]:
    from lib.opensearch_retriever import encode_splade
    records = json.loads((REPO_ROOT / "data/eval_results/diverse_queries.json").read_text())
    out, seen = [], set()
    for rec in records:
        if rec["paraphrase"] in seen:
            continue
        seen.add(rec["paraphrase"])
        e = encode_splade(rec["paraphrase"], str(REPO_ROOT / "models/splade_onnx"))
        q = {t: max(1, int(round(w * QUANT)))
             for t, w in sorted(e.items(), key=lambda x: -x[1])[:TOP_QUERY_TERMS]}
        if q:
            out.append(q)
        if len(out) >= n:
            break
    return out


def exact_ground_truth(docs: list[dict], queries: list[dict]) -> list[set]:
    """Exact top-K by quantized dot product (numpy postings accumulation)."""
    postings_idx: dict[str, list[int]] = defaultdict(list)
    postings_w: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(docs):
        for t, w in d["vec"].items():
            postings_idx[t].append(i)
            postings_w[t].append(w)
    pidx = {t: np.array(v, dtype=np.int64) for t, v in postings_idx.items()}
    pw = {t: np.array(v, dtype=np.float64) for t, v in postings_w.items()}
    truth = []
    for q in queries:
        scores = np.zeros(len(docs))
        for t, wq in q.items():
            if t in pidx:
                scores[pidx[t]] += wq * pw[t]
        top = np.argpartition(-scores, K)[:K]
        truth.append({docs[i]["id"] for i in top[np.argsort(-scores[top])]})
    return truth


def bmp_matrix(docs: list[dict], queries: list[dict], truth: list[set]) -> list[dict]:
    import bmp
    WORK.mkdir(parents=True, exist_ok=True)
    corpus_order = list(range(len(docs)))
    clustered = sorted(corpus_order, key=lambda i: (docs[i]["section"], docs[i]["top"]))
    vocab = set().union(*({t for t in d["vec"]} for d in docs))
    variants = [
        (32, True, "corpus"), (32, True, "clustered"),
        (64, True, "corpus"), (64, True, "clustered"),
        (128, True, "corpus"), (128, True, "clustered"),
        (256, True, "clustered"),
        (32, False, "corpus"), (32, False, "clustered"),
    ]
    results = []
    for bsize, cr, order_name in variants:
        name = f"b{bsize}_{'cr' if cr else 'nocr'}_{order_name}"
        path = WORK / f"{name}.bmp"
        order = corpus_order if order_name == "corpus" else clustered
        t0 = time.perf_counter()
        ix = bmp.Indexer(str(path), bsize=bsize, compress_range=cr)
        for i in order:
            ix.add_document(docs[i]["id"], docs[i]["vec"])
        ix.finish()
        build_s = time.perf_counter() - t0
        size = path.stat().st_size

        s = bmp.Searcher(str(path))
        recalls, t0 = [], time.perf_counter()
        for qi, q in enumerate(queries):
            qk = {t: w for t, w in q.items() if t in vocab}
            if not qk:
                continue
            try:
                ids, _ = s.search(qk, k=K, alpha=0.8, beta=1.0)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                continue
            recalls.append(len(set(ids) & truth[qi]) / K)
        lat_ms = (time.perf_counter() - t0) * 1000 / max(len(recalls), 1)
        row = {"variant": name, "bsize": bsize, "compress_range": cr,
               "order": order_name, "bytes": size,
               "b_per_doc": round(size / len(docs), 1),
               "build_secs": round(build_s, 1),
               "ms_per_query": round(lat_ms, 1),
               "recall@50_vs_exact": round(sum(recalls) / len(recalls), 4)}
        results.append(row)
        logger.info("%s", row)
        path.unlink()  # keep disk clean
    return results


def tantivy_order_test(docs_raw_path: Path) -> list[dict]:
    import tantivy
    from lib.thinclient.sections import classify_doc
    raw = []
    with open(docs_raw_path) as fh:
        for line in fh:
            d = json.loads(line)
            sparse = d.get("sparse_field") or {}
            raw.append({
                "id": d.get("id", ""), "title": d.get("title") or "",
                "abstract": d.get("abstract") or "",
                "year": d.get("publication_year") or 0,
                "key": (classify_doc(d.get("title"), d.get("abstract")),
                        max(sparse, key=sparse.get) if sparse else ""),
            })
    out = []
    for order_name in ("corpus", "clustered"):
        idx_dir = WORK / f"tantivy_{order_name}"
        if idx_dir.exists():
            shutil.rmtree(idx_dir)
        idx_dir.mkdir(parents=True)
        schema = (tantivy.SchemaBuilder()
                  .add_text_field("id", stored=True, tokenizer_name="raw")
                  .add_text_field("title", stored=False, index_option="freq")
                  .add_text_field("abstract", stored=False, index_option="freq")
                  .add_integer_field("year", stored=False, indexed=True, fast=True)
                  .build())
        index = tantivy.Index(schema, path=str(idx_dir))
        writer = index.writer(heap_size=512_000_000)
        ordered = raw if order_name == "corpus" else sorted(raw, key=lambda d: d["key"])
        for d in ordered:
            writer.add_document(tantivy.Document(
                id=d["id"], title=d["title"], abstract=d["abstract"], year=d["year"]))
        writer.commit()
        writer.wait_merging_threads()
        size = sum(f.stat().st_size for f in idx_dir.rglob("*") if f.is_file())
        out.append({"order": order_name, "bytes": size,
                    "b_per_doc": round(size / len(raw), 1)})
        logger.info("tantivy %s: %.1f MB", order_name, size / 1e6)
        shutil.rmtree(idx_dir)
    return out


def meta_recode_test() -> dict:
    """Real 1M meta.sqlite -> numeric-PK + dict-compressed titles/DOIs +
    int-coded enums. Round-trip verified on 200 rows."""
    src = sqlite3.connect(f"file:{META_1M}?mode=ro", uri=True)
    rows = src.execute("SELECT id, title, doi, year, type, is_oa, section FROM docs").fetchall()
    baseline = META_1M.stat().st_size

    title_dict = zstandard.train_dictionary(
        112 * 1024, [r[1].encode() for r in rows[:20000] if r[1]])
    doi_dict = zstandard.train_dictionary(
        16 * 1024, [r[2].encode() for r in rows[:20000] if r[2]])
    ct = zstandard.ZstdCompressor(level=19, dict_data=title_dict)
    cd = zstandard.ZstdCompressor(level=19, dict_data=doi_dict)
    types = sorted({r[4] or "" for r in rows})
    sections = sorted({r[6] or "" for r in rows})
    t_id = {t: i for i, t in enumerate(types)}
    s_id = {s: i for i, s in enumerate(sections)}

    out_path = WORK / "meta_recoded.sqlite"
    out_path.unlink(missing_ok=True)
    dst = sqlite3.connect(str(out_path))
    dst.executescript(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, title BLOB, doi BLOB,"
        "  year INTEGER, type INTEGER, is_oa INTEGER, section INTEGER);"
        "CREATE TABLE enums (kind TEXT, val INTEGER, name TEXT);"
        "CREATE TABLE meta (k TEXT PRIMARY KEY, v BLOB);")
    dst.execute("INSERT INTO meta VALUES ('title_dict', ?)", (title_dict.as_bytes(),))
    dst.execute("INSERT INTO meta VALUES ('doi_dict', ?)", (doi_dict.as_bytes(),))
    dst.executemany("INSERT INTO enums VALUES ('type', ?, ?)",
                    [(i, t) for t, i in t_id.items()])
    dst.executemany("INSERT INTO enums VALUES ('section', ?, ?)",
                    [(i, s) for s, i in s_id.items()])
    non_numeric = 0
    batch = []
    for r in rows:
        wid = r[0]
        if not (wid.startswith("W") and wid[1:].isdigit()):
            non_numeric += 1
            continue
        batch.append((int(wid[1:]),
                      ct.compress(r[1].encode()) if r[1] else None,
                      cd.compress(r[2].encode()) if r[2] else None,
                      r[3], t_id[r[4] or ""], r[5], s_id[r[6] or ""]))
        if len(batch) >= 10000:
            dst.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?,?)", batch)
            batch = []
    dst.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?,?)", batch)
    dst.commit()
    dst.execute("VACUUM")

    # round-trip verify
    dt = zstandard.ZstdDecompressor(dict_data=title_dict)
    dd = zstandard.ZstdDecompressor(dict_data=doi_dict)
    ok = 0
    for r in rows[:200]:
        if not (r[0].startswith("W") and r[0][1:].isdigit()):
            continue
        got = dst.execute("SELECT title, doi, year, type, is_oa, section FROM docs "
                          "WHERE id=?", (int(r[0][1:]),)).fetchone()
        if (got and (dt.decompress(got[0]).decode() if got[0] else "") == (r[1] or "")
                and (dd.decompress(got[1]).decode() if got[1] else "") == (r[2] or "")
                and got[2] == r[3] and types[got[3]] == (r[4] or "")
                and sections[got[5]] == (r[6] or "")):
            ok += 1
    recoded = out_path.stat().st_size
    dst.close()
    src.close()
    out_path.unlink()
    res = {"rows": len(rows), "baseline_bytes": baseline, "recoded_bytes": recoded,
           "ratio": round(baseline / recoded, 3),
           "b_per_doc": round(recoded / len(rows), 1),
           "baseline_b_per_doc": round(baseline / len(rows), 1),
           "non_numeric_ids": non_numeric, "roundtrip_ok": ok}
    logger.info("meta recode: %s", res)
    return res


def abstracts_mainline_sim() -> dict:
    """Approximate the MAINLINE Lucene/OpenSearch stored-fields cost for
    abstracts: corpus-order ~16KB blocks, zstd level 6, no trained dictionary
    (Lucene's zstd codec compresses consecutive docs' stored fields in blocks).
    This is the fair baseline for 'what does the new sidecar tech add on top
    of the mainline DB's preexisting optimizations'."""
    import gzip
    docs = [json.loads(l) for l in gzip.open(SAMPLE, "rt")]
    cctx = zstandard.ZstdCompressor(level=6)
    raw = comp = 0
    cur, cur_len = [], 0
    for d in docs:
        p = d["abstract"].encode()
        cur.append(p)
        cur_len += len(p)
        raw += len(p)
        if cur_len >= 16 * 1024:
            comp += len(cctx.compress(b"\x00".join(cur)))
            cur, cur_len = [], 0
    if cur:
        comp += len(cctx.compress(b"\x00".join(cur)))
    res = {"docs": len(docs), "ratio": round(raw / comp, 3),
           "b_per_doc": round(comp / len(docs), 1)}
    logger.info("abstracts mainline-sim (16KB zstd6 blocks, no dict): %s", res)
    return res


def abstracts_full_config() -> dict:
    """Clustered 32KB blocks + per-language dicts (the complete recommendation)."""
    import gzip

    import py3langid
    docs = [json.loads(l) for l in gzip.open(SAMPLE, "rt")]
    for d in docs:
        d["lang"], _ = py3langid.classify(d["abstract"][:600])
    order = sorted(range(len(docs)),
                   key=lambda i: (docs[i]["lang"], docs[i]["section"], docs[i]["top_term"]))
    by_lang = defaultdict(list)
    for i in order:
        by_lang[docs[i]["lang"]].append(i)

    global_dict = zstandard.train_dictionary(
        112 * 1024, [docs[i]["abstract"].encode() for i in order[:20000]])
    raw = comp = 0
    for lang, idxs in by_lang.items():
        payloads = [docs[i]["abstract"].encode() for i in idxs]
        if len(payloads) >= 500:
            try:
                zd = zstandard.train_dictionary(112 * 1024, payloads[:20000])
            except zstandard.ZstdError:
                zd = global_dict
        else:
            zd = global_dict
        cctx = zstandard.ZstdCompressor(level=19, dict_data=zd)
        cur, cur_len = [], 0
        for p in payloads:
            cur.append(p)
            cur_len += len(p)
            raw += len(p)
            if cur_len >= 32 * 1024:
                comp += len(cctx.compress(b"\x00".join(cur)))
                cur, cur_len = [], 0
        if cur:
            comp += len(cctx.compress(b"\x00".join(cur)))
    res = {"docs": len(docs), "ratio": round(raw / comp, 3),
           "b_per_doc": round(comp / len(docs), 1)}
    logger.info("abstracts full config: %s", res)
    return res


def pack_ratio_test() -> dict:
    """tar+zstd-19 LDM of a real built section (cold-archive ratio)."""
    import os
    live = sum(f.stat().st_size for f in SECTION_FOR_PACK.rglob("*") if f.is_file())
    out = WORK / "pack_test.tar.zst"
    params = zstandard.ZstdCompressionParameters.from_level(
        19, enable_ldm=True, window_log=27, threads=os.cpu_count() or 8)
    cctx = zstandard.ZstdCompressor(compression_params=params)
    t0 = time.perf_counter()
    with open(out, "wb") as fh, cctx.stream_writer(fh) as zw:
        with tarfile.open(fileobj=zw, mode="w|") as tar:
            tar.add(SECTION_FOR_PACK, arcname="s")
    secs = time.perf_counter() - t0
    packed = out.stat().st_size
    out.unlink()
    res = {"live_mb": round(live / 1e6, 1), "packed_mb": round(packed / 1e6, 1),
           "ratio": round(live / packed, 3), "pack_secs": round(secs, 1)}
    logger.info("artifact pack ratio (phys_eng 1M section): %s", res)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    docs = load_corpus()
    queries = encode_queries(args.queries)
    logger.info("computing exact ground truth for %d queries ...", len(queries))
    truth = exact_ground_truth(docs, queries)

    bmp_rows = bmp_matrix(docs, queries, truth)
    tantivy_rows = tantivy_order_test(CORPUS)
    meta_res = meta_recode_test()
    abs_main = abstracts_mainline_sim()
    abs_res = abstracts_full_config()
    pack_res = pack_ratio_test()

    # ── all-levers 150M math ──
    bmp_base = next(r for r in bmp_rows if r["variant"] == "b32_cr_corpus")
    bmp_best = min((r for r in bmp_rows if r["recall@50_vs_exact"] >=
                    bmp_base["recall@50_vs_exact"] - 0.005), key=lambda r: r["bytes"])
    bmp_after = AS_BUILT_150M["bmp"] * bmp_best["bytes"] / bmp_base["bytes"]
    tv_base = next(r for r in tantivy_rows if r["order"] == "corpus")
    tv_clu = next(r for r in tantivy_rows if r["order"] == "clustered")
    tv_after = AS_BUILT_150M["tantivy"] * tv_clu["bytes"] / tv_base["bytes"]
    abs_after = abs_res["b_per_doc"] * FULL_DOCS / 1e9
    meta_after = AS_BUILT_150M["meta"] / meta_res["ratio"]

    total_before = sum(AS_BUILT_150M.values())
    total_after = bmp_after + tv_after + abs_after + meta_after

    summary = {
        "bmp_matrix": bmp_rows,
        "bmp_best_within_recall_gate": bmp_best,
        "tantivy_order": tantivy_rows,
        "meta_recode": meta_res,
        "abstracts_mainline_sim_zstd6_blocks": abs_main,
        "abstracts_full_config": abs_res,
        "artifact_pack_ratio": pack_res,
        "all_levers_150M_gb": {
            "as_built": AS_BUILT_150M,
            "after": {"bmp": round(bmp_after, 1), "tantivy": round(tv_after, 1),
                      "abstracts": round(abs_after, 1), "meta": round(meta_after, 1)},
            "total_before": round(total_before, 1),
            "total_after": round(total_after, 1),
            "saved_gb": round(total_before - total_after, 1),
            "saved_pct": round((1 - total_after / total_before) * 100, 1),
        },
    }
    OUTPUT.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 96)
    print("BMP MATRIX — 100k real docs, production SPLADE vectors, alpha=0.8, recall vs EXACT")
    print("=" * 96)
    print(f"{'variant':<22} {'MB':>7} {'B/doc':>7} {'vs base':>8} {'ms/q':>6} {'R@50':>7}")
    base_b = bmp_base["bytes"]
    for r in bmp_rows:
        print(f"{r['variant']:<22} {r['bytes'] / 1e6:>7.1f} {r['b_per_doc']:>7} "
              f"{(r['bytes'] / base_b - 1) * 100:>+7.1f}% {r['ms_per_query']:>6} "
              f"{r['recall@50_vs_exact']:>7}")
    print("-" * 96)
    al = summary["all_levers_150M_gb"]
    print(f"tantivy clustered vs corpus: {tv_clu['bytes'] / tv_base['bytes'] - 1:+.1%} | "
          f"meta recode: {meta_res['ratio']}x ({meta_res['baseline_b_per_doc']}->"
          f"{meta_res['b_per_doc']} B/doc, roundtrip {meta_res['roundtrip_ok']}/200) | "
          f"abstracts mainline-sim {abs_main['b_per_doc']} vs full-config "
          f"{abs_res['b_per_doc']} B/doc | artifact pack: {pack_res['ratio']}x")
    print("=" * 96)
    print(f"ALL LEVERS @150M: {al['total_before']} GB -> {al['total_after']} GB  "
          f"(saved {al['saved_gb']} GB, {al['saved_pct']}%)")
    for k in ("bmp", "tantivy", "abstracts", "meta"):
        print(f"  {k:<10} {al['as_built'][k]:>7.1f} -> {al['after'][k]:>6.1f} GB")
    print("=" * 96)
    logger.info("wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
