#!/usr/bin/env python3
"""NDCG@10 evaluation of OpenSearch retrieval pipelines: BM25F vs SPLADE vs RRF.

Retrieves documents from OpenSearch via three methods, looks up citation counts
from OpenAlex for relevance grading, and computes NDCG@10 — same methodology as
the existing evaluate_sfu_queries.py benchmarks.

Usage:
    python scripts/ndcg_splade_eval.py
    python scripts/ndcg_splade_eval.py --queries data/sfu_eval_queries.json --k 10
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
OPENSEARCH_URL = os.environ.get("SFU_OPENSEARCH_URL", "http://opensearch:9200")
INDEX = "openalex_works"
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"
CITE_CACHE_FILE = ROOT / "data" / "openalex_cite_cache.json"

# Load .env for API key
_env_path = ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip("'").strip('"')

OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()


def load_cite_cache() -> dict:
    if CITE_CACHE_FILE.exists():
        try:
            return json.loads(CITE_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_cite_cache(cache: dict):
    CITE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CITE_CACHE_FILE.write_text(json.dumps(cache))


CITE_CACHE = load_cite_cache()


def batch_lookup_citations(openalex_ids: list[str]) -> dict[str, int]:
    """Look up cited_by_count for a batch of OpenAlex IDs. Returns {id: count}."""
    uncached = [oid for oid in openalex_ids if oid not in CITE_CACHE]
    if not uncached:
        return {oid: CITE_CACHE[oid] for oid in openalex_ids}

    for i in range(0, len(uncached), 50):
        batch = uncached[i:i + 50]
        filter_str = "|".join(f"https://openalex.org/{oid}" for oid in batch)
        params = {
            "filter": f"openalex_id:{filter_str}",
            "select": "id,cited_by_count",
            "per_page": 50,
        }
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY
        else:
            params["mailto"] = "lib-systems@sfu.ca"

        for attempt in range(4):
            try:
                resp = requests.get(
                    "https://api.openalex.org/works",
                    params=params,
                    headers={"User-Agent": "SFULibraryMCP-Eval/1.0"},
                    timeout=30,
                )
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                for w in data.get("results", []):
                    oid = w["id"].split("/")[-1]
                    CITE_CACHE[oid] = w.get("cited_by_count", 0)
                break
            except Exception as e:
                err = str(e)
                if OPENALEX_API_KEY:
                    err = err.replace(OPENALEX_API_KEY, "<redacted>")
                if attempt == 3:
                    log.warning("OpenAlex batch lookup failed: %s", err)
                time.sleep(2 * (attempt + 1))

        if not OPENALEX_API_KEY:
            time.sleep(0.2)

    for oid in uncached:
        if oid not in CITE_CACHE:
            CITE_CACHE[oid] = 0

    save_cite_cache(CITE_CACHE)
    return {oid: CITE_CACHE.get(oid, 0) for oid in openalex_ids}


def bm25f_search(session, query_text: str, k: int = 50) -> list[dict]:
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
            {"id": h["_id"], "openalex_id": h["_source"].get("openalex_id", h["_id"]),
             "title": h["_source"].get("title", ""), "score": h.get("_score", 0)}
            for h in hits if h.get("_source", {}).get("title")
        ]
    except Exception as e:
        log.warning("BM25F search failed: %s", e)
        return []


def splade_search(session, sparse_query: dict, k: int = 50) -> list[dict]:
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
            {"id": h["_id"], "openalex_id": h["_source"].get("openalex_id", h["_id"]),
             "title": h["_source"].get("title", ""), "score": h.get("_score", 0)}
            for h in hits if h.get("_source", {}).get("title")
        ]
    except Exception as e:
        log.warning("SPLADE search failed: %s", e)
        return []


def rrf_fuse(lists: list[list[dict]], k_param: int = 60) -> list[dict]:
    scores = {}
    docs = {}
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


def ndcg_at_k(ranked_relevance: list[float], ideal_relevance: list[float], k: int) -> float:
    def dcg(rels, k):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    d = dcg(ranked_relevance, k)
    ideal = dcg(sorted(ideal_relevance, reverse=True), k)
    return d / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked_relevance: list[float], k: int, threshold: float = 0.0) -> float:
    for rank, rel in enumerate(ranked_relevance[:k], start=1):
        if rel > threshold:
            return 1.0 / rank
    return 0.0


def compute_ndcg_for_results(results: list[dict], cite_map: dict[str, int], k: int) -> tuple[float, list[float]]:
    """Compute NDCG@k for a ranked result list using citation-count relevance."""
    relevance = [math.log1p(cite_map.get(r["openalex_id"], 0)) for r in results]
    ideal = sorted(relevance, reverse=True)
    ndcg = ndcg_at_k(relevance, ideal, k)
    return ndcg, relevance


def main():
    parser = argparse.ArgumentParser(description="NDCG@10 eval: BM25F vs SPLADE vs RRF")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fetch-k", type=int, default=50)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model", default=SPLADE_MODEL,
                        help="Query encoder model — pass the path of the model the INDEX was built with "
                             "(e.g. models/sfu-splade-v1) to avoid encoder mismatch. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open(args.queries) as f:
        eval_queries = json.load(f)
    log.info("Loaded %d eval queries", len(eval_queries))

    log.info("Loading SPLADE encoder: %s", args.model)
    from scripts.benchmark_splade import SpladeQueryEncoder
    encoder = SpladeQueryEncoder(args.model, device=args.device)

    session = requests.Session()

    # Phase 1: Retrieve all results and collect unique openalex_ids
    log.info("Phase 1: Retrieving results from OpenSearch...")
    query_results = []
    all_openalex_ids = set()

    for qi, q in enumerate(eval_queries):
        query_text = q["query"]
        subject = q.get("subject", "Unknown")

        sparse = encoder.encode(query_text)
        bm25_res = bm25f_search(session, query_text, k=args.fetch_k)
        splade_res = splade_search(session, sparse, k=args.fetch_k)
        rrf_res = rrf_fuse([bm25_res, splade_res])

        for r in bm25_res + splade_res + rrf_res:
            all_openalex_ids.add(r["openalex_id"])

        query_results.append({
            "query": query_text,
            "subject": subject,
            "bm25": bm25_res,
            "splade": splade_res,
            "rrf": rrf_res,
        })
        if (qi + 1) % 20 == 0:
            log.info("  Retrieved %d/%d queries", qi + 1, len(eval_queries))

    log.info("  Total unique docs to look up: %d", len(all_openalex_ids))
    uncached = [oid for oid in all_openalex_ids if oid not in CITE_CACHE]
    log.info("  Already cached: %d, need to fetch: %d", len(all_openalex_ids) - len(uncached), len(uncached))

    # Phase 2: Batch lookup citation counts
    log.info("Phase 2: Looking up citation counts from OpenAlex...")
    cite_map = batch_lookup_citations(list(all_openalex_ids))
    log.info("  Citation counts loaded for %d docs", len(cite_map))

    # Phase 3: Compute NDCG for each method
    log.info("Phase 3: Computing NDCG@%d...", args.k)
    methods = ["bm25", "splade", "rrf"]
    ndcg_scores = {m: [] for m in methods}
    mrr_scores = {m: [] for m in methods}
    subject_ndcg = {m: {} for m in methods}
    per_query_detail = []

    for qr in query_results:
        query = qr["query"]
        subject = qr["subject"]
        row = {"query": query, "subject": subject}

        for method in methods:
            results = qr[method][:args.fetch_k]
            if not results:
                continue

            relevance = [math.log1p(cite_map.get(r["openalex_id"], 0)) for r in results]
            ideal = sorted(relevance, reverse=True)
            ndcg = ndcg_at_k(relevance, ideal, args.k)
            mrr = mrr_at_k(relevance, args.k)

            ndcg_scores[method].append(ndcg)
            mrr_scores[method].append(mrr)
            subject_ndcg[method].setdefault(subject, []).append(ndcg)
            row[f"{method}_ndcg"] = round(ndcg, 4)
            row[f"{method}_mrr"] = round(mrr, 4)

            # Top-cited doc rank in this method's results
            if results:
                cites = [(i, cite_map.get(r["openalex_id"], 0)) for i, r in enumerate(results[:args.k])]
                max_cite = max(cites, key=lambda x: x[1])
                row[f"{method}_top_cited_rank"] = max_cite[0] + 1
                row[f"{method}_top_cited_count"] = max_cite[1]

        per_query_detail.append(row)

    # Print results
    print()
    print("=" * 90)
    print(f"  NDCG@{args.k} EVALUATION — OpenSearch Retrieval Pipelines")
    print(f"  {len(eval_queries)} queries | fetch={args.fetch_k} | index={INDEX}")
    print("=" * 90)

    print(f"\n{'Method':<25} {'NDCG@10':>10} {'Median':>10} {'Std':>8} {'MRR@10':>10} {'Queries':>8}")
    print("-" * 75)
    for method in methods:
        scores = ndcg_scores[method]
        mrrs = mrr_scores[method]
        if not scores:
            continue
        label = {"bm25": "BM25F", "splade": "SPLADE", "rrf": "BM25F + SPLADE RRF"}[method]
        print(f"{label:<25} {np.mean(scores):>10.4f} {np.median(scores):>10.4f} "
              f"{np.std(scores):>8.4f} {np.mean(mrrs):>10.4f} {len(scores):>8}")

    # Subject breakdown
    all_subjects = sorted({s for m in methods for s in subject_ndcg[m]})
    print(f"\n{'Subject':<45} {'BM25F':>8} {'SPLADE':>8} {'RRF':>8} {'Δ RRF-BM25':>11}")
    print("-" * 85)
    for subj in all_subjects:
        vals = {}
        for method in methods:
            scores = subject_ndcg[method].get(subj, [])
            vals[method] = np.mean(scores) if scores else 0.0
        delta = vals["rrf"] - vals["bm25"]
        sign = "+" if delta >= 0 else ""
        print(f"{subj[:44]:<45} {vals['bm25']:>8.4f} {vals['splade']:>8.4f} {vals['rrf']:>8.4f} {sign}{delta:>10.4f}")

    # Queries where RRF helps most
    print(f"\n{'Top 10 queries where RRF improves over BM25F:'}")
    print(f"{'Query':<60} {'BM25F':>7} {'RRF':>7} {'Δ':>7}")
    print("-" * 85)
    deltas = [(pq, pq.get("rrf_ndcg", 0) - pq.get("bm25_ndcg", 0)) for pq in per_query_detail]
    deltas.sort(key=lambda x: -x[1])
    for pq, d in deltas[:10]:
        print(f"{pq['query'][:59]:<60} {pq.get('bm25_ndcg', 0):>7.4f} {pq.get('rrf_ndcg', 0):>7.4f} {d:>+7.4f}")

    print(f"\n{'Top 10 queries where RRF hurts vs BM25F:'}")
    print(f"{'Query':<60} {'BM25F':>7} {'RRF':>7} {'Δ':>7}")
    print("-" * 85)
    for pq, d in deltas[-10:]:
        print(f"{pq['query'][:59]:<60} {pq.get('bm25_ndcg', 0):>7.4f} {pq.get('rrf_ndcg', 0):>7.4f} {d:>+7.4f}")

    # Overall delta statistics
    bm25_mean = np.mean(ndcg_scores["bm25"]) if ndcg_scores["bm25"] else 0
    rrf_mean = np.mean(ndcg_scores["rrf"]) if ndcg_scores["rrf"] else 0
    splade_mean = np.mean(ndcg_scores["splade"]) if ndcg_scores["splade"] else 0
    print(f"\n{'='*50}")
    print(f"  RRF vs BM25F:  {rrf_mean - bm25_mean:+.4f} NDCG@{args.k}")
    print(f"  RRF vs SPLADE: {rrf_mean - splade_mean:+.4f} NDCG@{args.k}")
    n_improved = sum(1 for _, d in deltas if d > 0.001)
    n_hurt = sum(1 for _, d in deltas if d < -0.001)
    n_same = len(deltas) - n_improved - n_hurt
    print(f"  Queries improved: {n_improved}  |  same: {n_same}  |  hurt: {n_hurt}")
    print(f"{'='*50}")

    # Save results
    output = args.output or ROOT / "data" / "eval_results" / f"ndcg_splade_eval_{time.strftime('%Y%m%d_%H%M')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": args.k,
        "fetch_k": args.fetch_k,
        "num_queries": len(eval_queries),
        "summary": {
            method: {
                "mean_ndcg": round(float(np.mean(ndcg_scores[method])), 4),
                "median_ndcg": round(float(np.median(ndcg_scores[method])), 4),
                "std_ndcg": round(float(np.std(ndcg_scores[method])), 4),
                "mean_mrr": round(float(np.mean(mrr_scores[method])), 4),
            }
            for method in methods if ndcg_scores[method]
        },
        "subject_breakdown": {
            method: {
                subj: round(float(np.mean(scores)), 4)
                for subj, scores in subject_ndcg[method].items()
            }
            for method in methods
        },
        "per_query": per_query_detail,
    }
    output.write_text(json.dumps(result, indent=2))
    log.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
