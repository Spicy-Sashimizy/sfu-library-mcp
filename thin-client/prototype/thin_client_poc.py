#!/usr/bin/env python3
"""Thin-client search stack POC: tantivy (BM25F) + seismic (SPLADE) + usearch
(dense binary+rescore) — the no-JVM laptop stack recommended by
docs/SEARCH_ENGINE_ALTERNATIVES.md — exercised on REAL project data.

What it measures (per engine): index build time, on-disk size, mean query
latency, plus dense recall vs exact and a 2-leg RRF fusion smoke test
(tantivy+seismic share the same 100k-doc corpus; usearch uses the dense POC's
cached vectors).

Corpus: 100k docs scrolled from `lexcomp_base` (title/abstract/year/type/is_oa
+ the production SPLADE `sparse_field` weights). Queries: 20 paraphrases from
diverse_queries.json; SPLADE query encoding via the production ONNX encoder.

Usage
─────
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    SFU_OPENSEARCH_URL=http://...:9200 \
    .venv/bin/python3 scripts/thin_client_poc.py [--docs 100000] [--queries 20]
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("thin_client_poc")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
WORK_DIR = REPO_ROOT / "data/thin_client_poc"
DEFAULT_OUTPUT = REPO_ROOT / "data/eval_results/thin_client_poc.json"
SPLADE_ONNX = str(REPO_ROOT / "models/splade_onnx")
DENSE_VECS = REPO_ROOT / "data/dense_compression/vectors_600k.npy"
SOURCE_INDEX = "lexcomp_base"
RRF_K = 60


def opensearch_url() -> str:
    return os.environ.get(
        "SFU_OPENSEARCH_URL",
        "http://claudebox-sfu-library-mcp-training-opensearch:9200",
    ).rstrip("/")


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ── Corpus export ───────────────────────────────────────────────────────────────

def export_corpus(n_docs: int) -> Path:
    path = WORK_DIR / f"docs_{n_docs}.jsonl"
    if path.exists():
        return path
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    url = opensearch_url()
    logger.info("Exporting %d docs from %s …", n_docs, SOURCE_INDEX)
    got = 0
    body = {"size": 2000, "query": {"match_all": {}},
            "_source": ["title", "abstract", "publication_year", "type", "is_oa",
                        "sparse_field"]}
    r = requests.post(f"{url}/{SOURCE_INDEX}/_search?scroll=5m", json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    with open(path, "w") as fh:
        while got < n_docs:
            hits = data["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                doc = h["_source"]
                doc["id"] = h["_id"]
                fh.write(json.dumps(doc) + "\n")
                got += 1
                if got >= n_docs:
                    break
            data = requests.post(f"{url}/_search/scroll",
                                 json={"scroll": "5m", "scroll_id": data["_scroll_id"]},
                                 timeout=60).json()
    logger.info("Exported %d docs -> %s", got, path)
    return path


def load_queries(n: int) -> list[str]:
    records = json.loads((REPO_ROOT / "data/eval_results/diverse_queries.json").read_text())
    out: list[str] = []
    seen: set[str] = set()
    for rec in records:
        q = rec["paraphrase"]
        if q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) >= n:
            break
    return out


# ── Leg 1: tantivy BM25F ────────────────────────────────────────────────────────

def bench_tantivy(corpus: Path, queries: list[str], top_k: int) -> tuple[dict, list[list[str]]]:
    import tantivy
    idx_dir = WORK_DIR / "tantivy_idx"
    fresh = not idx_dir.exists()
    idx_dir.mkdir(parents=True, exist_ok=True)

    schema = (tantivy.SchemaBuilder()
              .add_text_field("id", stored=True, tokenizer_name="raw")
              .add_text_field("title", stored=False)
              .add_text_field("abstract", stored=False)
              .add_integer_field("year", stored=False, indexed=True, fast=True)
              .add_text_field("doctype", stored=False, tokenizer_name="raw")
              .add_boolean_field("is_oa", stored=False, indexed=True)
              .build())
    index = tantivy.Index(schema, path=str(idx_dir))

    build_secs = 0.0
    if fresh:
        t0 = time.perf_counter()
        writer = index.writer(heap_size=256_000_000)
        n = 0
        with open(corpus) as fh:
            for line in fh:
                d = json.loads(line)
                writer.add_document(tantivy.Document(
                    id=d["id"], title=d.get("title") or "",
                    abstract=d.get("abstract") or "",
                    year=d.get("publication_year") or 0,
                    doctype=d.get("type") or "", is_oa=bool(d.get("is_oa")),
                ))
                n += 1
        writer.commit()
        writer.wait_merging_threads()
        build_secs = time.perf_counter() - t0
        logger.info("tantivy: indexed %d docs in %.1fs", n, build_secs)

    index.reload()
    searcher = index.searcher()

    def bm25f_query(text: str):
        # most_fields ≈ SHOULD-sum of per-field queries with boosts (title^3).
        text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
        qt = index.parse_query(text, ["title"])
        qa = index.parse_query(text, ["abstract"])
        return tantivy.Query.boolean_query([
            (tantivy.Occur.Should, tantivy.Query.boost_query(qt, 3.0)),
            (tantivy.Occur.Should, qa),
        ])

    results: list[list[str]] = []
    t0 = time.perf_counter()
    for q in queries:
        hits = searcher.search(bm25f_query(q), top_k).hits
        results.append([searcher.doc(addr)["id"][0] for _, addr in hits])
    search_secs = time.perf_counter() - t0

    # Filtered-query demo: BM25F AND year >= 2020 (uses the fast field).
    fq = tantivy.Query.boolean_query([
        (tantivy.Occur.Must, bm25f_query(queries[0])),
        (tantivy.Occur.Must, tantivy.Query.range_query(
            schema, "year", tantivy.FieldType.Integer, 2020, 3000)),
    ])
    filtered_n = len(searcher.search(fq, top_k).hits)

    return {
        "engine": "tantivy 0.26 (BM25F lexical)",
        "build_secs": round(build_secs, 1),
        "index_mb": round(dir_size(idx_dir) / 1e6, 1),
        "ms_per_query": round(search_secs * 1000 / len(queries), 2),
        "filtered_query_works": filtered_n > 0,
    }, results


# ── Leg 2: seismic SPLADE ───────────────────────────────────────────────────────

def bench_seismic(corpus: Path, queries: list[str], top_k: int) -> tuple[dict, list[list[str]]]:
    from seismic import SeismicIndex
    from lib.opensearch_retriever import encode_splade

    seismic_jsonl = WORK_DIR / "seismic_input.jsonl"
    if not seismic_jsonl.exists():
        with open(corpus) as fin, open(seismic_jsonl, "w") as fout:
            for line in fin:
                d = json.loads(line)
                fout.write(json.dumps({"id": d["id"], "content": "",
                                       "vector": d.get("sparse_field") or {}}) + "\n")

    idx_path = WORK_DIR / "seismic_idx"
    t0 = time.perf_counter()
    if (idx_path.with_suffix(".index.seismic")).exists():
        index = SeismicIndex.load(str(idx_path) + ".index.seismic")
        build_secs = 0.0
    else:
        index = SeismicIndex.build(str(seismic_jsonl))
        build_secs = time.perf_counter() - t0
        index.save(str(idx_path))
        logger.info("seismic: built in %.1fs", build_secs)
    files = list(WORK_DIR.glob("seismic_idx*"))
    size_mb = sum(f.stat().st_size for f in files) / 1e6

    # Encode queries with the production SPLADE ONNX encoder.
    enc = [encode_splade(q, SPLADE_ONNX) for q in queries]
    string_type = f"U{max(1, max((len(t) for e in enc for t in e), default=1))}"
    results: list[list[str]] = []
    t0 = time.perf_counter()
    for e in enc:
        comps = np.array(list(e.keys()), dtype=string_type)
        vals = np.array(list(e.values()), dtype=np.float32)
        hits = index.search(query_id="q", query_components=comps, query_values=vals,
                            k=top_k, query_cut=10, heap_factor=0.7)
        results.append([h[2] for h in hits])  # (query_id, score, doc_id)
    search_secs = time.perf_counter() - t0

    return {
        "engine": "seismic 0.5 (SPLADE sparse)",
        "build_secs": round(build_secs, 1),
        "index_mb": round(size_mb, 1),
        "ms_per_query": round(search_secs * 1000 / len(queries), 2),
        "note": "no native filters; filter post-hoc or partition indexes",
    }, results


# ── Leg 3: usearch dense (binary + fp32 rescore) ───────────────────────────────

def bench_usearch(n_docs: int, n_queries: int, top_k: int) -> dict:
    from usearch.index import Index

    X = np.load(DENSE_VECS)[:n_docs].astype(np.float32)
    Q = X[-n_queries:]  # held-out-ish probes; fine for latency/recall smoke test
    keys = np.arange(len(X), dtype=np.uint64)

    t0 = time.perf_counter()
    bits = np.packbits((X > 0).astype(np.uint8), axis=1)
    index = Index(ndim=X.shape[1], dtype="b1", metric="hamming")
    index.add(keys, bits)
    build_secs = time.perf_counter() - t0
    idx_file = WORK_DIR / "usearch_b1.idx"
    index.save(str(idx_file))

    qbits = np.packbits((Q > 0).astype(np.uint8), axis=1)
    t0 = time.perf_counter()
    cand = index.search(qbits, top_k * 4)
    keys_out = cand.keys.reshape(len(Q), -1)
    # fp32 rescore from "disk" (here: the mmap-able original matrix)
    out = []
    for i in range(len(Q)):
        c = keys_out[i]
        scores = X[c] @ Q[i]
        out.append(c[np.argsort(-scores)][:top_k])
    search_secs = time.perf_counter() - t0

    gt = np.argsort(-(X @ Q.T), axis=0)[:10].T  # exact top-10 per query
    rec10 = float(np.mean([len(set(out[i][:10]) & set(gt[i])) / 10 for i in range(len(Q))]))

    return {
        "engine": "usearch 2.25 b1+fp32-rescore (dense)",
        "build_secs": round(build_secs, 1),
        "index_mb": round(idx_file.stat().st_size / 1e6, 1),
        "ms_per_query": round(search_secs * 1000 / len(Q), 2),
        "recall@10_vs_exact": round(rec10, 4),
        "note": "rescore vectors (fp32/int8) live on disk, mmap-served",
    }


# ── RRF fusion smoke test ───────────────────────────────────────────────────────

def rrf(lists: list[list[str]], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for lst in lists:
        for rank, did in enumerate(lst, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin-client stack POC")
    parser.add_argument("--docs", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    corpus = export_corpus(args.docs)
    queries = load_queries(args.queries)

    t_stats, t_results = bench_tantivy(corpus, queries, args.top_k)
    s_stats, s_results = bench_seismic(corpus, queries, args.top_k)
    u_stats = bench_usearch(args.docs, args.queries, args.top_k)

    fused = [rrf([t_results[i], s_results[i]], args.top_k) for i in range(len(queries))]
    overlap = [len(set(t_results[i]) & set(s_results[i])) for i in range(len(queries))]
    fusion = {
        "fused_lists": len(fused),
        "avg_lexical_sparse_overlap@50": round(sum(overlap) / len(overlap), 1),
        "note": "tantivy+seismic fused with the production RRF (k=60); dense leg "
                "uses the dense-POC corpus so it is benchmarked separately",
    }

    out = {
        "config": {"docs": args.docs, "queries": len(queries), "top_k": args.top_k,
                   "corpus_source": SOURCE_INDEX,
                   "splade_encoder": "production ONNX (models/splade_onnx)"},
        "engines": [t_stats, s_stats, u_stats],
        "fusion": fusion,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 96)
    print(f"THIN-CLIENT STACK POC — {args.docs:,} real docs, {len(queries)} queries, no JVM")
    print("=" * 96)
    print(f"{'engine':<42} {'build s':>8} {'size MB':>8} {'ms/query':>9} {'extra':>20}")
    print("-" * 96)
    for e in (t_stats, s_stats, u_stats):
        extra = ""
        if "recall@10_vs_exact" in e:
            extra = f"R@10={e['recall@10_vs_exact']}"
        if "filtered_query_works" in e:
            extra = f"filters={'OK' if e['filtered_query_works'] else 'FAIL'}"
        print(f"{e['engine']:<42} {e['build_secs']:>8.1f} {e['index_mb']:>8.1f} "
              f"{e['ms_per_query']:>9.2f} {extra:>20}")
    print("-" * 96)
    print(f"RRF fusion: OK across {fusion['fused_lists']} queries; "
          f"avg lexical∩sparse overlap@50 = {fusion['avg_lexical_sparse_overlap@50']}")
    print("=" * 96)
    logger.info("Wrote results -> %s", args.output)


if __name__ == "__main__":
    main()
