#!/usr/bin/env python3
"""SPLADE top-k pruning experiment: index size vs retrieval quality.

For each top-k in K_VALUES, build a throwaway index whose SPLADE vectors are
statically pruned to the top-k highest-weight terms per doc, then measure:
  - on-disk index size (force-merged to 1 segment)
  - nDCG@10 / MRR@10 over the LLM-judged eval queries

This answers "how small can the SPLADE index get before retrieval quality
drops?" so we can pick an operating point for the 425 GB production index.

SAFETY (this script CANNOT harm production):
  * Reads SOURCE_INDEX read-only (scroll). Never writes/deletes it.
  * Only ever creates or deletes indices whose name starts with PRUNE_PREFIX.
    A guard refuses any other target. openalex_works is never touched.
  * Uses the local ONNX model read-only.

Usage:
    .venv/bin/python3 scripts/prune_topk_experiment.py \
        [--k 256,128,64,32,16] [--limit-docs N] [--keep-indices] \
        [--output data/eval_results/prune_topk_experiment.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

OS_URL = "http://claudebox-sfu-library-mcp-training-opensearch:9200"
SOURCE_INDEX = "openalex_works_splade_eval"   # read-only source of real SPLADE docs
SOURCE_SPARSE_FIELD = "sparse_v1"             # full-resolution SPLADE vector to prune
PRUNE_PREFIX = "prune_k"                       # the ONLY namespace we may write/delete
JUDGE_CACHE = REPO_ROOT / "data/eval_results/llm_judge_cache.json"
SPLADE_MODEL = str(REPO_ROOT / "models/splade_onnx")
QUERY_TERMS = 40        # top query terms used to build the rank_feature query
RETRIEVE_K = 50         # hits per query before scoring
K_NDCG = 10

# --- text fields copied across so BM25 stays available + sizes are realistic ---
TEXT_SOURCE_FIELDS = ["openalex_id", "title", "abstract", "doi",
                      "publication_year", "type", "is_oa"]


def _req(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{OS_URL}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _guard(index: str) -> None:
    """Refuse to mutate anything outside the prune_ namespace."""
    if not index.startswith(PRUNE_PREFIX):
        raise SystemExit(f"SAFETY ABORT: refusing to write/delete '{index}' "
                         f"(only {PRUNE_PREFIX}* is allowed)")


# ---------------- nDCG / MRR (mirrors scripts/eval_pipeline.py) ----------------
def _dcg(gains: list[int], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_grades: list[int], ideal_grades: list[int], k: int) -> float:
    idcg = _dcg(sorted(ideal_grades, reverse=True), k)
    return _dcg(ranked_grades, k) / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_grades: list[int], k: int) -> float:
    for i, g in enumerate(ranked_grades[:k]):
        if g > 0:
            return 1.0 / (i + 1)
    return 0.0


# ------------------------------- index build -----------------------------------
def create_prune_index(index: str) -> None:
    _guard(index)
    try:
        _req("DELETE", f"/{index}")
    except Exception:
        pass
    _req("PUT", f"/{index}", {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0,
                     "refresh_interval": "-1",
                     "codec": "zstd", "codec.compression_level": 3},
        "mappings": {"properties": {
            "openalex_id": {"type": "keyword"},
            "doi": {"type": "keyword"},
            "title": {"type": "text"},
            "abstract": {"type": "text"},
            "publication_year": {"type": "integer"},
            "type": {"type": "keyword"},
            "is_oa": {"type": "boolean"},
            "sparse_field": {"type": "rank_features"},
        }},
    })


def prune_vector(sparse: dict, k: int) -> dict:
    if len(sparse) <= k:
        return sparse
    top = sorted(sparse.items(), key=lambda x: -x[1])[:k]
    return dict(top)


def load_pruned(index: str, k: int, limit_docs: int | None) -> tuple[int, float]:
    """Scroll SOURCE_INDEX read-only, prune sparse vec to top-k, bulk index.

    Returns (doc_count, avg_terms_per_doc)."""
    _guard(index)
    scroll = "5m"
    body = {"size": 1000, "_source": TEXT_SOURCE_FIELDS + [SOURCE_SPARSE_FIELD],
            "query": {"match_all": {}}}
    res = _req("POST", f"/{SOURCE_INDEX}/_search?scroll={scroll}", body)
    sid = res["_scroll_id"]
    total_docs = 0
    total_terms = 0
    buf: list[str] = []

    def flush():
        if not buf:
            return
        payload = "\n".join(buf) + "\n"
        req = urllib.request.Request(
            f"{OS_URL}/_bulk", data=payload.encode(), method="POST",
            headers={"Content-Type": "application/x-ndjson"})
        urllib.request.urlopen(req, timeout=300).read()
        buf.clear()

    while True:
        hits = res["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            src = h["_source"]
            sparse = src.get(SOURCE_SPARSE_FIELD) or {}
            if not isinstance(sparse, dict) or not sparse:
                continue
            pruned = prune_vector(sparse, k)
            doc = {f: src[f] for f in TEXT_SOURCE_FIELDS if f in src}
            doc["sparse_field"] = pruned
            total_terms += len(pruned)
            total_docs += 1
            buf.append(json.dumps({"index": {"_index": index, "_id": h["_id"]}}))
            buf.append(json.dumps(doc))
            if len(buf) >= 4000:
                flush()
            if limit_docs and total_docs >= limit_docs:
                break
        if limit_docs and total_docs >= limit_docs:
            break
        res = _req("POST", "/_search/scroll",
                   {"scroll": scroll, "scroll_id": sid})
        sid = res["_scroll_id"]
    flush()
    _req("DELETE", f"/_search/scroll/{sid}") if False else None
    _req("POST", f"/{index}/_forcemerge?max_num_segments=1")
    _req("POST", f"/{index}/_refresh")
    avg = total_terms / total_docs if total_docs else 0.0
    return total_docs, avg


def index_size_mb(index: str) -> float:
    st = _req("GET", f"/{index}/_stats/store")
    return st["indices"][index]["primaries"]["store"]["size_in_bytes"] / 1e6


# ------------------------------- evaluation ------------------------------------
def load_judge() -> dict[str, dict[str, int]]:
    raw = json.loads(JUDGE_CACHE.read_text())
    grades: dict[str, dict[str, int]] = {}
    for key, grade in raw.items():
        if "||" not in key:
            continue
        q, docid = key.split("||", 1)
        grades.setdefault(q, {})[docid] = int(grade)
    return grades


def search_splade(index: str, query_vec: dict) -> list[str]:
    terms = sorted(query_vec.items(), key=lambda x: -x[1])[:QUERY_TERMS]
    should = [{"rank_feature": {"field": f"sparse_field.{t}", "boost": float(w)}}
              for t, w in terms]
    body = {"size": RETRIEVE_K, "_source": ["openalex_id"],
            "query": {"bool": {"should": should}}}
    res = _req("POST", f"/{index}/_search", body)
    out = []
    for h in res["hits"]["hits"]:
        out.append(h["_source"].get("openalex_id") or h["_id"])
    return out


def evaluate(index: str, query_vecs: dict[str, dict],
             gradebook: dict[str, dict[str, int]]) -> dict:
    ndcgs, mrrs = [], []
    for q, gb in gradebook.items():
        if q not in query_vecs:
            continue
        ranked = search_splade(index, query_vecs[q])
        ranked_grades = [gb.get(d, 0) for d in ranked]
        ndcgs.append(ndcg_at_k(ranked_grades, list(gb.values()), K_NDCG))
        mrrs.append(mrr_at_k(ranked_grades, K_NDCG))
    n = len(ndcgs)
    return {"queries": n,
            "ndcg@10": sum(ndcgs) / n if n else 0.0,
            "mrr@10": sum(mrrs) / n if n else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="256,128,64,32,16")
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="cap docs per index (smoke test); default = all")
    ap.add_argument("--keep-indices", action="store_true")
    ap.add_argument("--output",
                    default=str(REPO_ROOT / "data/eval_results/prune_topk_experiment.json"))
    args = ap.parse_args()
    k_values = [int(x) for x in args.k.split(",")]

    print(f"Source (read-only): {SOURCE_INDEX}   field: {SOURCE_SPARSE_FIELD}")
    print(f"K values: {k_values}   limit_docs: {args.limit_docs}\n")

    # 1. encode the judged eval queries ONCE (shared across all k)
    gradebook = load_judge()
    print(f"Judge cache: {len(gradebook)} distinct queries")
    from lib.opensearch_retriever import encode_splade
    t0 = time.time()
    query_vecs = {q: encode_splade(q, SPLADE_MODEL) for q in gradebook}
    print(f"Encoded {len(query_vecs)} queries in {time.time()-t0:.1f}s\n")

    rows = []
    baseline_ndcg = None
    for k in k_values:
        idx = f"{PRUNE_PREFIX}{k}"
        t0 = time.time()
        create_prune_index(idx)
        n_docs, avg_terms = load_pruned(idx, k, args.limit_docs)
        size_mb = index_size_mb(idx)
        build_s = time.time() - t0
        t1 = time.time()
        ev = evaluate(idx, query_vecs, gradebook)
        eval_s = time.time() - t1
        if baseline_ndcg is None:
            baseline_ndcg = ev["ndcg@10"]
        row = {"k": k, "docs": n_docs, "avg_terms": round(avg_terms, 1),
               "size_mb": round(size_mb, 1),
               "ndcg@10": round(ev["ndcg@10"], 4), "mrr@10": round(ev["mrr@10"], 4),
               "ndcg_delta_pct": round(100 * (ev["ndcg@10"] - baseline_ndcg) /
                                       baseline_ndcg, 2) if baseline_ndcg else 0.0,
               "build_s": round(build_s, 1), "eval_s": round(eval_s, 1)}
        rows.append(row)
        print(f"k={k:<4} docs={n_docs:<7} avg_terms={row['avg_terms']:<6} "
              f"size={row['size_mb']:>7.1f}MB  nDCG@10={row['ndcg@10']:.4f} "
              f"({row['ndcg_delta_pct']:+.1f}%)  MRR@10={row['mrr@10']:.4f}  "
              f"[build {build_s:.0f}s eval {eval_s:.0f}s]")
        if not args.keep_indices:
            _guard(idx)
            _req("DELETE", f"/{idx}")

    Path(args.output).write_text(json.dumps(
        {"source": SOURCE_INDEX, "sparse_field": SOURCE_SPARSE_FIELD,
         "limit_docs": args.limit_docs, "rows": rows}, indent=2))
    print(f"\nWrote {args.output}")
    # projection helper
    base = rows[0]
    print("\nProjection to 425 GB prod index (size scales ~linearly with avg_terms,"
          " stored/text fields constant):")
    for r in rows:
        print(f"  k={r['k']:<4} -> ~{425 * r['size_mb'] / base['size_mb']:.0f} GB "
              f"(nDCG {r['ndcg_delta_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
