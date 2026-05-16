#!/usr/bin/env python3
"""Benchmark search_by_topic and export_search: live OpenAlex baselines vs local index.

Compares five retrieval methods across 120 eval queries:

  Method               | Tool currently uses this        | Would use after wiring
  ---------------------|----------------------------------|------------------------
  openalex_topic       | search_by_topic (baseline)      | stays as fallback
  openalex_relevance   | export_search (baseline)        | stays as fallback
  bm25                 | (local only, not wired)         | -
  splade               | (local only, not wired)         | -
  rrf                  | search_academic (federated)     | search_by_topic + export

The NDCG@10 relevance proxy is log1p(cited_by_count) — same methodology as
ndcg_splade_eval.py so results are directly comparable.

Usage:
    python scripts/benchmark_topic_export.py
    python scripts/benchmark_topic_export.py --skip-live   # local index only (fast)
    python scripts/benchmark_topic_export.py --k 10 --fetch-k 50
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DEFAULT_QUERIES = ROOT / "data" / "sfu_eval_queries.json"
OPENSEARCH_URL = os.environ.get("SFU_OPENSEARCH_URL", "http://claudebox-sfu-library-mcp-training-opensearch:9200")
INDEX = "openalex_works"
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"
CITE_CACHE_FILE = ROOT / "data" / "openalex_cite_cache.json"
OPENALEX_BASE = "https://api.openalex.org"

_env_path = ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip("'").strip('"')

OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "lib-systems@sfu.ca").strip()


# ── Citation cache ────────────────────────────────────────────────────────────

def _load_cite_cache() -> dict:
    if CITE_CACHE_FILE.exists():
        try:
            return json.loads(CITE_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_cite_cache(cache: dict):
    CITE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CITE_CACHE_FILE.write_text(json.dumps(cache))


CITE_CACHE: dict = _load_cite_cache()


def batch_lookup_citations(openalex_ids: list[str]) -> dict[str, int]:
    """Look up cited_by_count for a batch of OpenAlex IDs using the cache."""
    uncached = [oid for oid in openalex_ids if oid not in CITE_CACHE]
    if not uncached:
        return {oid: CITE_CACHE[oid] for oid in openalex_ids}

    for i in range(0, len(uncached), 50):
        batch = uncached[i:i + 50]
        filter_str = "|".join(f"https://openalex.org/{oid}" for oid in batch)
        params: dict = {"filter": f"openalex_id:{filter_str}", "select": "id,cited_by_count", "per_page": 50}
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY
        else:
            params["mailto"] = OPENALEX_MAILTO
        for attempt in range(4):
            try:
                resp = requests.get(
                    f"{OPENALEX_BASE}/works", params=params,
                    headers={"User-Agent": "SFULibraryMCP-Eval/1.0"}, timeout=30,
                )
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                for w in resp.json().get("results", []):
                    oid = w["id"].split("/")[-1]
                    CITE_CACHE[oid] = w.get("cited_by_count", 0)
                break
            except Exception as e:
                if attempt == 3:
                    log.warning("Cite lookup failed: %s", e)
                time.sleep(2 * (attempt + 1))
        if not OPENALEX_API_KEY:
            time.sleep(0.2)

    for oid in uncached:
        if oid not in CITE_CACHE:
            CITE_CACHE[oid] = 0
    _save_cite_cache(CITE_CACHE)
    return {oid: CITE_CACHE.get(oid, 0) for oid in openalex_ids}


# ── Local OpenSearch retrievers ───────────────────────────────────────────────

def bm25f_search(session: requests.Session, query_text: str, k: int = 50) -> list[dict]:
    body = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["title^3", "abstract", "concepts^2"],
                "type": "best_fields",
                "tie_breaker": 0.3,
            }
        },
        "_source": ["openalex_id", "title", "doi"],
    }
    try:
        resp = session.post(f"{OPENSEARCH_URL}/{INDEX}/_search", json=body, timeout=15)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        return [
            {"id": h["_id"],
             "openalex_id": h["_source"].get("openalex_id", h["_id"]),
             "title": h["_source"].get("title", ""),
             "score": h.get("_score", 0)}
            for h in hits if h.get("_source", {}).get("title")
        ]
    except Exception as e:
        log.warning("BM25F search failed: %s", e)
        return []


def splade_search(session: requests.Session, sparse_query: dict, k: int = 50) -> list[dict]:
    if not sparse_query:
        return []
    should = []
    for term, weight in sorted(sparse_query.items(), key=lambda x: -x[1])[:48]:
        should.append({"rank_feature": {"field": f"sparse_field.{term}", "boost": weight, "log": {"scaling_factor": 1}}})
    body = {"size": k, "query": {"bool": {"should": should}}, "_source": ["openalex_id", "title", "doi"]}
    try:
        resp = session.post(f"{OPENSEARCH_URL}/{INDEX}/_search", json=body, timeout=30)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        return [
            {"id": h["_id"],
             "openalex_id": h["_source"].get("openalex_id", h["_id"]),
             "title": h["_source"].get("title", ""),
             "score": h.get("_score", 0)}
            for h in hits if h.get("_source", {}).get("title")
        ]
    except Exception as e:
        log.warning("SPLADE search failed: %s", e)
        return []


def rrf_fuse(lists: list[list[dict]], k_param: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in lists:
        for rank, doc in enumerate(results, start=1):
            did = doc["id"]
            scores[did] = scores.get(did, 0.0) + 1.0 / (k_param + rank)
            if did not in docs:
                docs[did] = doc
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    out = []
    for did, score in ranked:
        d = dict(docs[did])
        d["rrf_score"] = score
        out.append(d)
    return out


# ── OpenAlex live retrievers ──────────────────────────────────────────────────

def _normalize_openalex_hit(w: dict) -> dict:
    """Extract fields used by the benchmark from a raw OpenAlex API work."""
    raw_id = w.get("id", "")
    openalex_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
    return {
        "id": openalex_id,
        "openalex_id": openalex_id,
        "title": w.get("display_name", "") or w.get("title", ""),
        "cited_by_count": w.get("cited_by_count", 0),
        "score": w.get("cited_by_count", 0),
    }


def openalex_search_live(
    session: requests.Session,
    query_text: str,
    sort: str = "relevance_score:desc",
    k: int = 50,
    delay: float = 0.0,
) -> list[dict]:
    """Retrieve from OpenAlex live API with the given sort order."""
    params: dict = {
        "search": query_text,
        "per_page": k,
        "select": "id,display_name,cited_by_count",
    }
    if sort:
        params["sort"] = sort
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    else:
        params["mailto"] = OPENALEX_MAILTO

    for attempt in range(4):
        try:
            resp = session.get(
                f"{OPENALEX_BASE}/works",
                params=params,
                headers={"User-Agent": "SFULibraryMCP-Eval/1.0"},
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            works = resp.json().get("results", [])
            if delay:
                time.sleep(delay)
            return [_normalize_openalex_hit(w) for w in works if w.get("display_name") or w.get("title")]
        except Exception as e:
            if attempt == 3:
                log.warning("OpenAlex live search failed for '%s': %s", query_text[:40], e)
            time.sleep(2 * (attempt + 1))
    return []


# ── Metrics ───────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_rel: list[float], ideal_rel: list[float], k: int) -> float:
    def dcg(rels, k):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    d = dcg(ranked_rel, k)
    ideal = dcg(sorted(ideal_rel, reverse=True), k)
    return d / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked_rel: list[float], k: int) -> float:
    for rank, rel in enumerate(ranked_rel[:k], start=1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark search_by_topic and export_search vs local index")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fetch-k", type=int, default=50)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-live", action="store_true",
                        help="Skip OpenAlex live API calls (run local index only, faster)")
    parser.add_argument("--live-delay", type=float, default=0.15,
                        help="Seconds to sleep between live API calls per query (default 0.15)")
    args = parser.parse_args()

    with open(args.queries) as f:
        eval_queries = json.load(f)
    log.info("Loaded %d eval queries", len(eval_queries))

    log.info("Loading SPLADE encoder...")
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark_splade import SpladeQueryEncoder
    encoder = SpladeQueryEncoder(SPLADE_MODEL, device=args.device)

    session = requests.Session()

    # Decide which methods to run
    live_methods = [] if args.skip_live else ["openalex_topic", "openalex_relevance"]
    local_methods = ["bm25", "splade", "rrf"]
    all_methods = live_methods + local_methods

    method_labels = {
        "openalex_topic":     "OpenAlex (cite-count sort)  ← search_by_topic CURRENT",
        "openalex_relevance": "OpenAlex (relevance sort)   ← export_search CURRENT",
        "bm25":               "BM25F (local OpenSearch)",
        "splade":             "SPLADE (local OpenSearch)",
        "rrf":                "BM25F + SPLADE RRF          ← PROPOSED wiring",
    }

    # Phase 1: retrieve results
    log.info("Phase 1: Retrieving results (%s)...",
             "local only" if args.skip_live else "OpenAlex live + local")
    query_results = []
    all_openalex_ids: set[str] = set()

    for qi, q in enumerate(eval_queries):
        query_text = q["query"]
        subject = q.get("subject", "Unknown")

        sparse = encoder.encode(query_text)
        row: dict = {"query": query_text, "subject": subject}

        if not args.skip_live:
            row["openalex_topic"] = openalex_search_live(
                session, query_text, sort="cited_by_count:desc", k=args.fetch_k,
                delay=args.live_delay,
            )
            row["openalex_relevance"] = openalex_search_live(
                session, query_text, sort="relevance_score:desc", k=args.fetch_k,
                delay=args.live_delay,
            )

        row["bm25"] = bm25f_search(session, query_text, k=args.fetch_k)
        row["splade"] = splade_search(session, sparse, k=args.fetch_k)
        row["rrf"] = rrf_fuse([row["bm25"], row["splade"]])

        for method in all_methods:
            for r in row.get(method, []):
                if r.get("openalex_id"):
                    all_openalex_ids.add(r["openalex_id"])

        query_results.append(row)

        if (qi + 1) % 20 == 0:
            log.info("  Retrieved %d/%d queries", qi + 1, len(eval_queries))

    log.info("  Total unique docs: %d", len(all_openalex_ids))

    # For live methods: cited_by_count already in the response; still cache it
    for qr in query_results:
        for method in live_methods:
            for r in qr.get(method, []):
                oid = r.get("openalex_id", "")
                cbc = r.get("cited_by_count", 0)
                if oid and oid not in CITE_CACHE:
                    CITE_CACHE[oid] = cbc
    _save_cite_cache(CITE_CACHE)

    # Phase 2: batch-lookup missing citation counts (local index results)
    uncached = [oid for oid in all_openalex_ids if oid not in CITE_CACHE]
    log.info("Phase 2: Looking up %d uncached citation counts...", len(uncached))
    cite_map = batch_lookup_citations(list(all_openalex_ids))
    log.info("  Citation counts loaded for %d docs", len(cite_map))

    # Phase 3: compute NDCG
    log.info("Phase 3: Computing NDCG@%d...", args.k)
    ndcg_scores: dict[str, list[float]] = {m: [] for m in all_methods}
    mrr_scores: dict[str, list[float]] = {m: [] for m in all_methods}
    subject_ndcg: dict[str, dict[str, list[float]]] = {m: {} for m in all_methods}
    per_query_detail = []

    for qr in query_results:
        query_text = qr["query"]
        subject = qr["subject"]
        detail_row: dict = {"query": query_text, "subject": subject}

        for method in all_methods:
            results = qr.get(method, [])[:args.fetch_k]
            if not results:
                continue
            # Use cited_by_count from the API response for live methods (already populated),
            # or from the batch lookup for local methods.
            relevance = [math.log1p(cite_map.get(r["openalex_id"], 0)) for r in results]
            ideal = sorted(relevance, reverse=True)
            ndcg = ndcg_at_k(relevance, ideal, args.k)
            mrr = mrr_at_k(relevance, args.k)
            ndcg_scores[method].append(ndcg)
            mrr_scores[method].append(mrr)
            subject_ndcg[method].setdefault(subject, []).append(ndcg)
            detail_row[f"{method}_ndcg"] = round(ndcg, 4)
            detail_row[f"{method}_mrr"] = round(mrr, 4)

        per_query_detail.append(detail_row)

    # ── Print report ──────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("  BENCHMARK: search_by_topic & export_search  —  Baseline vs Local Index")
    print(f"  {len(eval_queries)} queries  |  k={args.k}  |  fetch_k={args.fetch_k}")
    print("=" * 100)

    print(f"\n{'Method':<50} {'NDCG@10':>8} {'Median':>8} {'Std':>7} {'MRR@10':>8} {'N':>5}")
    print("-" * 95)
    for method in all_methods:
        scores = ndcg_scores[method]
        mrrs = mrr_scores[method]
        if not scores:
            print(f"  {method_labels.get(method, method):<48}  (no results)")
            continue
        label = method_labels.get(method, method)
        print(f"  {label:<48}  {np.mean(scores):>8.4f} {np.median(scores):>8.4f} "
              f"{np.std(scores):>7.4f} {np.mean(mrrs):>8.4f} {len(scores):>5}")

    # Delta table for proposed wiring vs current baselines
    if "openalex_topic" in all_methods and "rrf" in all_methods:
        rrf_mean = np.mean(ndcg_scores["rrf"]) if ndcg_scores["rrf"] else 0.0
        topic_mean = np.mean(ndcg_scores["openalex_topic"]) if ndcg_scores["openalex_topic"] else 0.0
        relevance_mean = np.mean(ndcg_scores["openalex_relevance"]) if ndcg_scores.get("openalex_relevance") else 0.0
        bm25_mean = np.mean(ndcg_scores["bm25"]) if ndcg_scores["bm25"] else 0.0
        splade_mean = np.mean(ndcg_scores["splade"]) if ndcg_scores["splade"] else 0.0

        print(f"\n{'=' * 60}")
        print("  WIRING DECISION SUMMARY")
        print(f"{'=' * 60}")
        print(f"  search_by_topic: RRF vs OpenAlex-cite-sort:  {rrf_mean - topic_mean:+.4f} NDCG@10")
        print(f"  export_search:   RRF vs OpenAlex-relevance:  {rrf_mean - relevance_mean:+.4f} NDCG@10")
        print(f"  RRF vs BM25F:                                {rrf_mean - bm25_mean:+.4f} NDCG@10")
        print(f"  RRF vs SPLADE:                               {rrf_mean - splade_mean:+.4f} NDCG@10")

        verdict_topic = "WIRE" if rrf_mean > topic_mean + 0.002 else ("SKIP" if rrf_mean < topic_mean - 0.002 else "NEUTRAL")
        verdict_export = "WIRE" if rrf_mean > relevance_mean + 0.002 else ("SKIP" if rrf_mean < relevance_mean - 0.002 else "NEUTRAL")
        print(f"\n  search_by_topic wiring verdict: {verdict_topic}")
        print(f"  export_search   wiring verdict: {verdict_export}")
        print(f"{'=' * 60}")

    # Subject breakdown
    all_subjects = sorted({s for m in all_methods for s in subject_ndcg[m]})
    col_methods = all_methods[:5]  # truncate for width
    header = f"{'Subject':<45}" + "".join(f" {m[:8]:>8}" for m in col_methods)
    print(f"\n{header}")
    print("-" * (45 + 9 * len(col_methods)))
    for subj in all_subjects:
        row_str = f"{subj[:44]:<45}"
        for method in col_methods:
            scores = subject_ndcg[method].get(subj, [])
            val = np.mean(scores) if scores else 0.0
            row_str += f" {val:>8.4f}"
        print(row_str)

    # Top queries where RRF beats OpenAlex-topic
    if "openalex_topic" in all_methods:
        print(f"\n  Top 10 queries where RRF > OpenAlex-cite-sort:")
        print(f"  {'Query':<58} {'OA-cit':>7} {'RRF':>7} {'Δ':>7}")
        print("  " + "-" * 83)
        deltas_topic = [
            (pq, pq.get("rrf_ndcg", 0) - pq.get("openalex_topic_ndcg", 0))
            for pq in per_query_detail
            if "rrf_ndcg" in pq and "openalex_topic_ndcg" in pq
        ]
        deltas_topic.sort(key=lambda x: -x[1])
        for pq, d in deltas_topic[:10]:
            print(f"  {pq['query'][:57]:<58} {pq.get('openalex_topic_ndcg', 0):>7.4f} "
                  f"{pq.get('rrf_ndcg', 0):>7.4f} {d:>+7.4f}")

        print(f"\n  Top 10 queries where OpenAlex-cite-sort > RRF:")
        print(f"  {'Query':<58} {'OA-cit':>7} {'RRF':>7} {'Δ':>7}")
        print("  " + "-" * 83)
        for pq, d in deltas_topic[-10:]:
            print(f"  {pq['query'][:57]:<58} {pq.get('openalex_topic_ndcg', 0):>7.4f} "
                  f"{pq.get('rrf_ndcg', 0):>7.4f} {d:>+7.4f}")

    # Save results
    output = args.output or (
        ROOT / "data" / "eval_results" / f"benchmark_topic_export_{time.strftime('%Y%m%d_%H%M')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": "topic_export_vs_local_index",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": args.k,
        "fetch_k": args.fetch_k,
        "num_queries": len(eval_queries),
        "skip_live": args.skip_live,
        "methods_run": all_methods,
        "summary": {
            method: {
                "mean_ndcg": round(float(np.mean(ndcg_scores[method])), 4) if ndcg_scores[method] else None,
                "median_ndcg": round(float(np.median(ndcg_scores[method])), 4) if ndcg_scores[method] else None,
                "std_ndcg": round(float(np.std(ndcg_scores[method])), 4) if ndcg_scores[method] else None,
                "mean_mrr": round(float(np.mean(mrr_scores[method])), 4) if mrr_scores[method] else None,
                "label": method_labels.get(method, method),
            }
            for method in all_methods
        },
        "subject_breakdown": {
            method: {
                subj: round(float(np.mean(scores)), 4)
                for subj, scores in subject_ndcg[method].items()
            }
            for method in all_methods
        },
        "per_query": per_query_detail,
    }
    output.write_text(json.dumps(result, indent=2))
    log.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
