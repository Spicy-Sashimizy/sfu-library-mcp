#!/usr/bin/env python3
"""Benchmark SPLADE retrieval vs BM25F on the SFU eval query set.

Encodes each eval query through the SPLADE model, queries OpenSearch using
rank_features, and compares against BM25F results side-by-side.

Usage:
    python scripts/benchmark_splade.py
    python scripts/benchmark_splade.py --top-k 10 --output data/eval_results/splade_vs_bm25f.json
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEFAULT_EVAL_QUERIES = Path(__file__).parent.parent / "data" / "sfu_eval_queries.json"
DEFAULT_OPENSEARCH_URL = "http://opensearch:9200"
DEFAULT_INDEX = "openalex_works"
DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"


class SpladeQueryEncoder:
    def __init__(self, model_name: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.vocab = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        logger.info("SPLADE query encoder loaded on %s", self.device)

    def encode(self, text: str, top_k: int = 64) -> dict[str, float]:
        import torch

        tokens = self.tokenizer(
            [text], max_length=256, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**tokens)

        vec = torch.log1p(torch.relu(output.logits))
        vec = torch.max(vec, dim=1).values.squeeze()

        nonzero = vec.nonzero(as_tuple=True)[0]
        if len(nonzero) == 0:
            return {}

        weights = vec[nonzero]
        if len(nonzero) > top_k:
            topk = torch.topk(weights, top_k)
            nonzero = nonzero[topk.indices]
            weights = topk.values

        sparse = {}
        for idx, w in zip(nonzero.cpu().tolist(), weights.cpu().tolist()):
            token = self.id_to_token.get(idx, "")
            if token and not token.startswith("[") and w > 0.01:
                sparse[token] = round(w, 4)
        return sparse


def splade_search(session, opensearch_url, index, sparse_query, k=50):
    """Query OpenSearch using SPLADE sparse vector via rank_features."""
    if not sparse_query:
        return []

    should_clauses = []
    for term, weight in sorted(sparse_query.items(), key=lambda x: -x[1])[:48]:
        should_clauses.append({
            "rank_feature": {
                "field": f"sparse_field.{term}",
                "boost": weight,
                "log": {"scaling_factor": 1},
            }
        })

    body = {
        "size": k,
        "query": {"bool": {"should": should_clauses}},
        "_source": ["doi", "title", "abstract", "publication_year", "type"],
    }

    try:
        resp = session.post(
            f"{opensearch_url}/{index}/_search",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("SPLADE search failed: %s", e)
        return []

    hits = (data.get("hits") or {}).get("hits") or []
    if not hits:
        return []

    max_score = max(h.get("_score", 0.0) for h in hits) or 1.0
    results = []
    for hit in hits:
        src = hit.get("_source") or {}
        title = src.get("title", "")
        if not title:
            continue
        results.append({
            "id": hit.get("_id", ""),
            "title": title,
            "abstract": src.get("abstract", "")[:200],
            "publication_year": src.get("publication_year"),
            "score": hit.get("_score", 0.0),
            "norm_score": hit.get("_score", 0.0) / max_score,
        })
    return results


def bm25f_search(session, opensearch_url, index, query_text, k=50):
    """Query OpenSearch using BM25F multi_match."""
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
        "_source": ["doi", "title", "abstract", "publication_year", "type"],
    }

    try:
        resp = session.post(
            f"{opensearch_url}/{index}/_search",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("BM25F search failed: %s", e)
        return []

    hits = (data.get("hits") or {}).get("hits") or []
    if not hits:
        return []

    max_score = max(h.get("_score", 0.0) for h in hits) or 1.0
    results = []
    for hit in hits:
        src = hit.get("_source") or {}
        title = src.get("title", "")
        if not title:
            continue
        results.append({
            "id": hit.get("_id", ""),
            "title": title,
            "abstract": src.get("abstract", "")[:200],
            "publication_year": src.get("publication_year"),
            "score": hit.get("_score", 0.0),
            "norm_score": hit.get("_score", 0.0) / max_score,
        })
    return results


def rrf_fuse(result_lists: list[list[dict]], k_param: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across multiple result lists.

    For each doc, score = sum(1 / (k_param + rank)) across all lists where it appears.
    k_param=60 is the standard default from the original RRF paper.
    """
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_param + rank)
            if doc_id not in docs:
                docs[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    fused = []
    for doc_id, score in ranked:
        doc = dict(docs[doc_id])
        doc["rrf_score"] = round(score, 6)
        doc["score"] = round(score, 6)
        fused.append(doc)
    return fused


def overlap_at_k(results_a, results_b, k=10):
    ids_a = {r["id"] for r in results_a[:k]}
    ids_b = {r["id"] for r in results_b[:k]}
    if not ids_a or not ids_b:
        return 0.0
    return len(ids_a & ids_b) / k


def unique_to_splade(splade_results, bm25_results, k=10):
    bm25_ids = {r["id"] for r in bm25_results[:k]}
    return [r for r in splade_results[:k] if r["id"] not in bm25_ids]


def run_benchmark(
    queries_path: Path,
    opensearch_url: str,
    index: str,
    model_name: str,
    k: int,
    device: str,
):
    with open(queries_path) as f:
        eval_queries = json.load(f)

    logger.info("Loading SPLADE encoder...")
    encoder = SpladeQueryEncoder(model_name, device=device)

    session = requests.Session()

    fetch_k = k * 5

    logger.info("Running %d queries (k=%d, fetch=%d) against %s/%s", len(eval_queries), k, fetch_k, opensearch_url, index)
    print()
    print(f"{'#':>3}  {'Subject':<35}  {'BM25F':>5} {'SPLADE':>6} {'RRF':>5}  {'B∩S':>5} {'B∩R':>5} {'S∩R':>5}  {'RRF from both':>13}")
    print("─" * 120)

    per_query = []
    subject_stats = {}
    totals = {"bm25": 0, "splade": 0, "rrf": 0, "bs_olap": 0.0, "br_olap": 0.0, "sr_olap": 0.0, "rrf_from_both": 0}
    encode_times = []

    for qi, q in enumerate(eval_queries):
        query_text = q["query"]
        subject = q.get("subject", "Unknown")

        t0 = time.time()
        sparse = encoder.encode(query_text)
        encode_time = time.time() - t0
        encode_times.append(encode_time)

        splade_results = splade_search(session, opensearch_url, index, sparse, k=fetch_k)
        bm25_results = bm25f_search(session, opensearch_url, index, query_text, k=fetch_k)
        rrf_results = rrf_fuse([bm25_results, splade_results])

        bs_olap = overlap_at_k(bm25_results, splade_results, k)
        br_olap = overlap_at_k(bm25_results, rrf_results, k)
        sr_olap = overlap_at_k(splade_results, rrf_results, k)

        bm25_ids_k = {r["id"] for r in bm25_results[:k]}
        splade_ids_k = {r["id"] for r in splade_results[:k]}
        rrf_top_k = rrf_results[:k]
        rrf_from_both = sum(1 for r in rrf_top_k if r["id"] in bm25_ids_k and r["id"] in splade_ids_k)

        b_hits = len(bm25_results[:k])
        s_hits = len(splade_results[:k])
        r_hits = len(rrf_top_k)

        totals["bm25"] += b_hits
        totals["splade"] += s_hits
        totals["rrf"] += r_hits
        totals["bs_olap"] += bs_olap
        totals["br_olap"] += br_olap
        totals["sr_olap"] += sr_olap
        totals["rrf_from_both"] += rrf_from_both

        if subject not in subject_stats:
            subject_stats[subject] = {
                "bm25": 0, "splade": 0, "rrf": 0,
                "bs_olap": 0.0, "br_olap": 0.0, "sr_olap": 0.0,
                "rrf_from_both": 0, "count": 0,
            }
        ss = subject_stats[subject]
        ss["bm25"] += b_hits; ss["splade"] += s_hits; ss["rrf"] += r_hits
        ss["bs_olap"] += bs_olap; ss["br_olap"] += br_olap; ss["sr_olap"] += sr_olap
        ss["rrf_from_both"] += rrf_from_both; ss["count"] += 1

        per_query.append({
            "query": query_text,
            "subject": subject,
            "bm25_hits": b_hits,
            "splade_hits": s_hits,
            "rrf_hits": r_hits,
            "overlap_bm25_splade": round(bs_olap, 3),
            "overlap_bm25_rrf": round(br_olap, 3),
            "overlap_splade_rrf": round(sr_olap, 3),
            "rrf_from_both_systems": rrf_from_both,
            "rrf_top5": [{"title": r["title"], "rrf_score": r["rrf_score"]} for r in rrf_results[:5]],
            "bm25_top3": [{"title": r["title"], "score": r["score"]} for r in bm25_results[:3]],
            "splade_top3": [{"title": r["title"], "score": r["score"]} for r in splade_results[:3]],
            "encode_time_ms": round(encode_time * 1000, 1),
        })

        print(f"{qi+1:>3}  {subject:<35}  {b_hits:>5} {s_hits:>6} {r_hits:>5}  {bs_olap:>4.0%} {br_olap:>5.0%} {sr_olap:>5.0%}  {rrf_from_both:>13}")

    n = len(eval_queries)
    avg_encode = sum(encode_times) / len(encode_times) * 1000 if encode_times else 0

    print("─" * 120)
    print()
    print("═══ Summary ═══")
    print(f"  Queries evaluated:      {n}")
    print(f"  BM25F avg hits@{k}:      {totals['bm25'] / n:.1f}")
    print(f"  SPLADE avg hits@{k}:     {totals['splade'] / n:.1f}")
    print(f"  RRF avg hits@{k}:        {totals['rrf'] / n:.1f}")
    print(f"  Avg BM25F↔SPLADE overlap: {totals['bs_olap'] / n:.1%}")
    print(f"  Avg BM25F↔RRF overlap:    {totals['br_olap'] / n:.1%}")
    print(f"  Avg SPLADE↔RRF overlap:   {totals['sr_olap'] / n:.1%}")
    print(f"  RRF top-{k} from BOTH:     {totals['rrf_from_both'] / n:.1f} avg ({totals['rrf_from_both']}/{n * k} total)")
    print(f"  Avg encode time:         {avg_encode:.1f} ms")
    print()

    print("  Subject breakdown:")
    print(f"    {'Subject':<40}  {'BM25F':>5} {'SPLADE':>6} {'RRF':>5}  {'B∩S':>5} {'B∩R':>5} {'S∩R':>5}  {'both→RRF':>8}")
    print(f"    {'─'*40}  {'─'*5} {'─'*6} {'─'*5}  {'─'*5} {'─'*5} {'─'*5}  {'─'*8}")
    for subj, s in sorted(subject_stats.items()):
        c = s["count"]
        print(f"    {subj:<40}  {s['bm25']/c:>5.0f} {s['splade']/c:>6.0f} {s['rrf']/c:>5.0f}"
              f"  {s['bs_olap']/c:>4.0%} {s['br_olap']/c:>5.0%} {s['sr_olap']/c:>5.0%}"
              f"  {s['rrf_from_both']/c:>8.1f}")

    print()

    # Show RRF top-1 vs BM25F and SPLADE top-1
    print("═══ RRF vs individual systems: top-1 comparison (sample) ═══")
    shown = 0
    for pq in per_query:
        if not pq["rrf_top5"] or not pq["bm25_top3"] or not pq["splade_top3"]:
            continue
        rrf_t = pq["rrf_top5"][0]["title"]
        bm25_t = pq["bm25_top3"][0]["title"]
        splade_t = pq["splade_top3"][0]["title"]
        if rrf_t == bm25_t and rrf_t == splade_t:
            continue
        print(f"\n  Query: {pq['query'][:80]}")
        print(f"    BM25F  #1: {bm25_t[:95]}")
        print(f"    SPLADE #1: {splade_t[:95]}")
        print(f"    RRF    #1: {rrf_t[:95]}  (score: {pq['rrf_top5'][0]['rrf_score']:.4f})")
        shown += 1
        if shown >= 8:
            break

    # Show queries where RRF pulls from both systems
    high_both = sorted(per_query, key=lambda x: -x["rrf_from_both_systems"])[:5]
    print(f"\n═══ Queries where RRF draws most from BOTH systems (best fusion) ═══")
    for pq in high_both:
        if pq["rrf_from_both_systems"] == 0:
            break
        print(f"\n  Query: {pq['query'][:80]}")
        print(f"    From both: {pq['rrf_from_both_systems']}/{k}  |  BM25F∩SPLADE overlap: {pq['overlap_bm25_splade']:.0%}")
        for r in pq["rrf_top5"][:3]:
            print(f"    → {r['title'][:95]}  ({r['rrf_score']:.4f})")

    result = {
        "benchmark": "splade_vs_bm25f_rrf",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "index": index,
        "docs_in_index": "~1M",
        "k": k,
        "rrf_k_param": 60,
        "num_queries": n,
        "avg_overlap_bm25_splade": round(totals["bs_olap"] / n, 4),
        "avg_overlap_bm25_rrf": round(totals["br_olap"] / n, 4),
        "avg_overlap_splade_rrf": round(totals["sr_olap"] / n, 4),
        "avg_rrf_from_both": round(totals["rrf_from_both"] / n, 2),
        "avg_encode_ms": round(avg_encode, 1),
        "subject_stats": {
            subj: {
                "avg_bm25_hits": round(s["bm25"] / s["count"], 1),
                "avg_splade_hits": round(s["splade"] / s["count"], 1),
                "avg_rrf_hits": round(s["rrf"] / s["count"], 1),
                "avg_overlap_bs": round(s["bs_olap"] / s["count"], 4),
                "avg_overlap_br": round(s["br_olap"] / s["count"], 4),
                "avg_overlap_sr": round(s["sr_olap"] / s["count"], 4),
                "avg_rrf_from_both": round(s["rrf_from_both"] / s["count"], 2),
                "count": s["count"],
            }
            for subj, s in subject_stats.items()
        },
        "per_query": per_query,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark SPLADE vs BM25F")
    parser.add_argument("--queries", type=Path, default=DEFAULT_EVAL_QUERIES)
    parser.add_argument("--opensearch-url", default=DEFAULT_OPENSEARCH_URL)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run_benchmark(
        queries_path=args.queries,
        opensearch_url=args.opensearch_url,
        index=args.index,
        model_name=args.model,
        k=args.top_k,
        device=args.device,
    )

    out = args.output or Path(f"data/eval_results/splade_vs_bm25f_{time.strftime('%Y-%m-%d_%H%M')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
