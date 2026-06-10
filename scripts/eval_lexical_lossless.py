#!/usr/bin/env python3
"""Lossless compression eval for the LEXICAL leg (BM25F postings + SPLADE
rank_features + stored _source) of openalex_works.

Tests every "lossless" size lever on a representative subset of the live index
and verifies that search results are byte-identical (or tie-shuffled only):

    base            faithful copy of the live index config (zstd level 3,
                    dynamic `id` text+keyword field, sparse_field in _source)
    zstd6           index.codec.compression_level 3 → 6
    zstd_no_dict    zstd without dictionary (speed/size tradeoff datapoint)
    nosrc_splade    _source.excludes: ["sparse_field"]  (weights stay in the
                    rank_features postings — they were stored TWICE)
    extern_display  _source.excludes: ["sparse_field", "abstract"] (display
                    text served from an external store; search untouched)
    freqs           index_options: freqs on title/abstract/concepts (drop
                    positions; BM25F multi_match + rank_feature queries never
                    use phrase/span, so scoring is unchanged)
    noid            the dynamically-mapped duplicate `id` field (text+keyword)
                    is no longer indexed (openalex_id keyword is canonical)
    combined        zstd6 + nosrc_splade + freqs + noid

Every variant is measured BEFORE and AFTER force_merge(max_num_segments=1), so
force_merge's own contribution is quantified on each.

Search parity: N queries (from diverse_queries.json) are run in BM25F and
SPLADE mode against base and every variant; top-50 (id, score) lists are
compared. Identical scores with possible equal-score tie swaps == lossless.

Usage
─────
    SFU_OPENSEARCH_URL=http://...:9200 \
    .venv/bin/python3 scripts/eval_lexical_lossless.py \
        [--subset-docs 1000000] [--parity-queries 40] [--keep-indices]
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_lexical_lossless")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DIVERSE = REPO_ROOT / "data/eval_results/diverse_queries.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/eval_results/lexical_lossless_eval.json"
SPLADE_ONNX = str(REPO_ROOT / "models/splade_onnx")

SOURCE_INDEX = "openalex_works"
PREFIX = "lexcomp"
FULL_SCALE_DOCS = 150_413_098
FULL_SCALE_GB = 274.0  # live pri.store.size (2026-06-10, zstd3, multi-segment)
TOP_K = 50


def opensearch_url() -> str:
    return os.environ.get(
        "SFU_OPENSEARCH_URL",
        "http://claudebox-sfu-library-mcp-training-opensearch:9200",
    ).rstrip("/")


OS_URL = opensearch_url()


def req(method: str, path: str, body: dict | None = None, timeout: int = 120) -> dict:
    r = requests.request(method, f"{OS_URL}/{path}", json=body, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.text else {}


# ── Variant definitions ─────────────────────────────────────────────────────────

# Live-index-faithful mapping, incl. the dynamically-created duplicate `id`.
BASE_MAPPING = {
    "doi": {"type": "keyword"},
    "openalex_id": {"type": "keyword"},
    "id": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
    "title": {"type": "text"},
    "abstract": {"type": "text"},
    "concepts": {"type": "text"},
    "publication_year": {"type": "integer"},
    "type": {"type": "keyword"},
    "is_oa": {"type": "boolean"},
    "sparse_field": {"type": "rank_features"},
}


def variant_config(name: str) -> dict:
    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "30s",
        "codec": "zstd",
        "codec.compression_level": 3,
    }
    props = json.loads(json.dumps(BASE_MAPPING))
    mappings: dict = {"properties": props}

    if name in ("zstd6", "combined"):
        settings["codec.compression_level"] = 6
    if name == "zstd_no_dict":
        settings["codec"] = "zstd_no_dict"
    if name in ("nosrc_splade", "combined"):
        mappings["_source"] = {"excludes": ["sparse_field"]}
    if name == "extern_display":
        mappings["_source"] = {"excludes": ["sparse_field", "abstract"]}
    if name in ("freqs", "combined"):
        for f in ("title", "abstract", "concepts"):
            props[f]["index_options"] = "freqs"
    if name in ("noid", "combined"):
        props["id"] = {"type": "keyword", "index": False, "doc_values": False}
    return {"settings": {"index": settings}, "mappings": mappings}


VARIANTS = ["base", "zstd6", "zstd_no_dict", "nosrc_splade", "extern_display",
            "freqs", "noid", "combined"]


# ── Index plumbing ──────────────────────────────────────────────────────────────

def wait_task(task_id: str, label: str) -> None:
    while True:
        t = req("GET", f"_tasks/{task_id}", timeout=60)
        if t.get("completed"):
            status = t.get("task", {}).get("status", {})
            logger.info("  %s done: created=%s total=%s", label,
                        status.get("created"), status.get("total"))
            failures = t.get("response", {}).get("failures") or []
            if failures:
                raise RuntimeError(f"{label} reindex failures: {failures[:3]}")
            return
        time.sleep(10)


def reindex(src: str, dest: str, max_docs: int | None) -> None:
    body: dict = {"source": {"index": src}, "dest": {"index": dest}}
    if max_docs:
        body["max_docs"] = max_docs
    resp = req("POST", "_reindex?wait_for_completion=false&slices=2&refresh=true",
               body, timeout=120)
    wait_task(resp["task"], f"reindex->{dest}")


def store_bytes(index: str) -> tuple[int, int]:
    """Sum LIVE segment sizes via _cat/segments — unlike _stats/store this never
    counts superseded segment files that linger on disk awaiting deletion
    (which double-counted merged indices in earlier runs)."""
    segs = req("GET", f"_cat/segments/{index}?format=json&bytes=b", timeout=60)
    return sum(int(s["size"]) for s in segs), len(segs)


def force_merge(index: str) -> None:
    # force_merge can exceed the HTTP timeout; poll via segment count.
    try:
        req("POST", f"{index}/_forcemerge?max_num_segments=1", timeout=1800)
    except requests.exceptions.ReadTimeout:
        logger.info("  forcemerge HTTP timeout; polling segments…")
        while store_bytes(index)[1] > 1:
            time.sleep(15)
    # Old segments are deleted asynchronously after the merge; measuring too
    # early double-counts old+new. Flush, then wait for the store size to
    # stabilize (two consecutive identical readings) with 1 segment.
    req("POST", f"{index}/_flush", timeout=120)
    req("POST", f"{index}/_refresh", timeout=60)
    prev = -1
    for _ in range(60):
        size, segs = store_bytes(index)
        if segs <= 1 and size == prev:
            return
        prev = size
        time.sleep(5)
    logger.warning("  %s store size did not stabilize; last=%d bytes", index, prev)


def build_variant_index(name: str, subset_docs: int) -> dict:
    index = f"{PREFIX}_{name}"
    if requests.head(f"{OS_URL}/{index}", timeout=30).status_code == 200:
        logger.info("%s already exists — reusing", index)
    else:
        logger.info("Creating %s …", index)
        req("PUT", index, variant_config(name), timeout=60)
        src = SOURCE_INDEX if name == "base" else f"{PREFIX}_base"
        reindex(src, index, subset_docs if name == "base" else None)
    req("POST", f"{index}/_flush", timeout=120)
    req("POST", f"{index}/_refresh", timeout=120)
    prev = -1
    for _ in range(24):  # let post-reindex background merges/deletes settle
        pre_bytes, pre_segs = store_bytes(index)
        if pre_bytes == prev:
            break
        prev = pre_bytes
        time.sleep(5)
    pre_bytes, pre_segs = store_bytes(index)
    if pre_segs > 1:
        logger.info("  force_merge %s (%d segments, %.1f MB)…", index, pre_segs, pre_bytes / 1e6)
        force_merge(index)
    post_bytes, _ = store_bytes(index)
    count = req("GET", f"{index}/_count", timeout=60)["count"]
    return {"index": index, "docs": count,
            "pre_merge_mb": round(pre_bytes / 1e6, 1),
            "post_merge_mb": round(post_bytes / 1e6, 1)}


# ── Search parity ───────────────────────────────────────────────────────────────

def parity_check(variants: dict[str, dict], queries: list[str]) -> dict[str, dict]:
    """Run BM25F + SPLADE on every variant index, compare to base top-50."""
    from lib.opensearch_retriever import OpenSearchRetriever

    def hits_for(index: str, query: str, mode: str) -> list[tuple[str, float]]:
        r = retrievers[index]
        docs = r.search(query, top_k=TOP_K, mode=mode)
        return [(d.get("openalex_id") or d.get("doi") or "", round(d.get("score", 0.0), 4))
                for d in docs]

    retrievers = {v["index"]: OpenSearchRetriever(url=OS_URL, index=v["index"],
                                                  splade_model_path=SPLADE_ONNX, timeout=30)
                  for v in variants.values()}
    base_idx = variants["base"]["index"]

    baseline: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for q in queries:
        for mode in ("bm25f", "splade"):
            baseline[(q, mode)] = hits_for(base_idx, q, mode)

    report: dict[str, dict] = {}
    for name, v in variants.items():
        if name == "base":
            continue
        stats = {"queries": 0, "exact_order": 0, "same_id_set": 0,
                 "same_score_multiset": 0, "max_score_diff": 0.0}
        for q in queries:
            for mode in ("bm25f", "splade"):
                b = baseline[(q, mode)]
                h = hits_for(v["index"], q, mode)
                stats["queries"] += 1
                if h == b:
                    stats["exact_order"] += 1
                if {i for i, _ in h} == {i for i, _ in b}:
                    stats["same_id_set"] += 1
                if sorted(s for _, s in h) == sorted(s for _, s in b):
                    stats["same_score_multiset"] += 1
                for (_, sb), (_, sh) in zip(b, h):
                    stats["max_score_diff"] = max(stats["max_score_diff"], abs(sb - sh))
        report[name] = stats
        logger.info("parity %s: exact %d/%d, id-set %d/%d, score-multiset %d/%d, maxΔscore %.4f",
                    name, stats["exact_order"], stats["queries"],
                    stats["same_id_set"], stats["queries"],
                    stats["same_score_multiset"], stats["queries"],
                    stats["max_score_diff"])
    return report


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Lossless lexical-index compression eval")
    parser.add_argument("--subset-docs", type=int, default=1_000_000)
    parser.add_argument("--parity-queries", type=int, default=40)
    parser.add_argument("--diverse-queries", default=str(DEFAULT_DIVERSE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--keep-indices", action="store_true",
                        help="don't delete lexcomp_* test indices at the end")
    args = parser.parse_args()

    logger.info("OpenSearch: %s   subset: %d docs", OS_URL, args.subset_docs)

    variants: dict[str, dict] = {}
    for name in VARIANTS:
        variants[name] = build_variant_index(name, args.subset_docs)
        logger.info("%-16s %8.1f MB pre-merge  %8.1f MB merged",
                    name, variants[name]["pre_merge_mb"], variants[name]["post_merge_mb"])

    records = json.loads(Path(args.diverse_queries).read_text())
    seen: set[str] = set()
    queries: list[str] = []
    for rec in records:  # alternate keyword/natural for a mixed parity set
        q = rec["paraphrase"]
        if q not in seen:
            seen.add(q)
            queries.append(q)
        if len(queries) >= args.parity_queries:
            break
    parity = parity_check(variants, queries)

    base_post = variants["base"]["post_merge_mb"]
    base_pre = variants["base"]["pre_merge_mb"]
    results = []
    for name in VARIANTS:
        v = variants[name]
        saved_pct = (1 - v["post_merge_mb"] / base_post) * 100
        results.append({
            "variant": name,
            "docs": v["docs"],
            "pre_merge_mb": v["pre_merge_mb"],
            "post_merge_mb": v["post_merge_mb"],
            "bytes_per_doc": round(v["post_merge_mb"] * 1e6 / max(1, v["docs"]), 1),
            "saved_vs_base_pct": round(saved_pct, 1),
            "est_full_scale_gb": round(FULL_SCALE_GB * v["post_merge_mb"] / base_post, 1),
            "parity": parity.get(name),
        })

    out = {
        "config": {
            "source_index": SOURCE_INDEX,
            "subset_docs": args.subset_docs,
            "parity_queries": len(queries),
            "parity_modes": ["bm25f", "splade"],
            "top_k": TOP_K,
            "full_scale": {"docs": FULL_SCALE_DOCS, "live_store_gb": FULL_SCALE_GB},
            "note_force_merge": f"base pre-merge {base_pre} MB vs merged {base_post} MB "
                                "isolates force_merge's own contribution",
        },
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 110)
    print(f"LEXICAL LOSSLESS COMPRESSION EVAL — {args.subset_docs:,}-doc subset of {SOURCE_INDEX}")
    print("=" * 110)
    print(f"{'variant':<16} {'pre-merge MB':>12} {'merged MB':>10} {'B/doc':>7} "
          f"{'saved%':>7} {'~150M GB':>9} {'parity (exact/idset/scores of n)':>34}")
    print("-" * 110)
    for r in results:
        p = r["parity"]
        ptxt = "(baseline)" if p is None else (
            f"{p['exact_order']}/{p['same_id_set']}/{p['same_score_multiset']} of {p['queries']}")
        print(f"{r['variant']:<16} {r['pre_merge_mb']:>12.1f} {r['post_merge_mb']:>10.1f} "
              f"{r['bytes_per_doc']:>7.1f} {r['saved_vs_base_pct']:>6.1f}% "
              f"{r['est_full_scale_gb']:>9.1f} {ptxt:>34}")
    print("-" * 110)
    print("parity: exact-order / same-id-set / same-score-multiset out of n query×mode runs @50.")
    print("same scores with tie swaps == lossless ranking; ~150M GB scales the live 274 GB by the")
    print("variant's merged size ratio on this subset.")
    print("=" * 110)

    if not args.keep_indices:
        for name in VARIANTS:
            req("DELETE", f"{PREFIX}_{name}", timeout=120)
        logger.info("Deleted lexcomp_* test indices")
    logger.info("Wrote results -> %s", args.output)


if __name__ == "__main__":
    main()
