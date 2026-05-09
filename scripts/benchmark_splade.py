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

    logger.info("Running %d queries (k=%d) against %s/%s", len(eval_queries), k, opensearch_url, index)
    print()
    print(f"{'#':>3}  {'Subject':<40}  {'BM25F':>6}  {'SPLADE':>6}  {'Overlap':>7}  {'SPLADE-unique':>13}")
    print("─" * 120)

    per_query = []
    subject_stats = {}
    splade_hit_count = 0
    bm25_hit_count = 0
    total_overlap = 0.0
    encode_times = []

    for qi, q in enumerate(eval_queries):
        query_text = q["query"]
        subject = q.get("subject", "Unknown")

        t0 = time.time()
        sparse = encoder.encode(query_text)
        encode_time = time.time() - t0
        encode_times.append(encode_time)

        splade_results = splade_search(session, opensearch_url, index, sparse, k=k * 5)
        bm25_results = bm25f_search(session, opensearch_url, index, query_text, k=k * 5)

        olap = overlap_at_k(splade_results, bm25_results, k)
        total_overlap += olap
        splade_unique = unique_to_splade(splade_results, bm25_results, k)

        s_hits = len(splade_results[:k])
        b_hits = len(bm25_results[:k])
        splade_hit_count += s_hits
        bm25_hit_count += b_hits

        if subject not in subject_stats:
            subject_stats[subject] = {"splade_hits": 0, "bm25_hits": 0, "overlap": 0.0, "count": 0, "unique_papers": []}
        subject_stats[subject]["splade_hits"] += s_hits
        subject_stats[subject]["bm25_hits"] += b_hits
        subject_stats[subject]["overlap"] += olap
        subject_stats[subject]["count"] += 1

        per_query.append({
            "query": query_text,
            "subject": subject,
            "splade_hits": s_hits,
            "bm25_hits": b_hits,
            "overlap_at_k": round(olap, 3),
            "splade_unique_count": len(splade_unique),
            "splade_top3": [{"title": r["title"], "score": r["score"]} for r in splade_results[:3]],
            "bm25_top3": [{"title": r["title"], "score": r["score"]} for r in bm25_results[:3]],
            "splade_unique_titles": [r["title"] for r in splade_unique[:5]],
            "encode_time_ms": round(encode_time * 1000, 1),
        })

        print(f"{qi+1:>3}  {subject:<40}  {b_hits:>6}  {s_hits:>6}  {olap:>6.1%}  {len(splade_unique):>13}")

    n = len(eval_queries)
    avg_overlap = total_overlap / n if n else 0
    avg_encode = sum(encode_times) / len(encode_times) * 1000 if encode_times else 0

    print("─" * 120)
    print()
    print("═══ Summary ═══")
    print(f"  Queries evaluated:    {n}")
    print(f"  BM25F avg hits@{k}:    {bm25_hit_count / n:.1f}")
    print(f"  SPLADE avg hits@{k}:   {splade_hit_count / n:.1f}")
    print(f"  Avg overlap@{k}:       {avg_overlap:.1%}")
    print(f"  Avg encode time:      {avg_encode:.1f} ms")
    print()
    print("  Subject breakdown:")
    for subj, s in sorted(subject_stats.items()):
        c = s["count"]
        print(f"    {subj:<40}  BM25F={s['bm25_hits']/c:.0f}  SPLADE={s['splade_hits']/c:.0f}  overlap={s['overlap']/c:.0%}")

    print()

    # Qualitative: show queries where SPLADE found the most unique papers
    unique_rich = sorted(per_query, key=lambda x: -x["splade_unique_count"])[:5]
    if unique_rich and unique_rich[0]["splade_unique_count"] > 0:
        print("═══ Top queries where SPLADE found unique papers (not in BM25F top-k) ═══")
        for pq in unique_rich:
            if pq["splade_unique_count"] == 0:
                break
            print(f"\n  Query: {pq['query'][:80]}")
            print(f"  Subject: {pq['subject']} | SPLADE-unique: {pq['splade_unique_count']}")
            for t in pq["splade_unique_titles"][:3]:
                print(f"    → {t[:100]}")

    # Qualitative: show queries where SPLADE top-1 differs from BM25F top-1
    diff_top1 = [pq for pq in per_query if pq["splade_top3"] and pq["bm25_top3"]
                 and pq["splade_top3"][0]["title"] != pq["bm25_top3"][0]["title"]]
    if diff_top1:
        print(f"\n═══ Different top-1 results: {len(diff_top1)}/{n} queries ═══")
        for pq in diff_top1[:5]:
            print(f"\n  Query: {pq['query'][:80]}")
            print(f"    BM25F #1:  {pq['bm25_top3'][0]['title'][:90]}")
            print(f"    SPLADE #1: {pq['splade_top3'][0]['title'][:90]}")

    result = {
        "benchmark": "splade_vs_bm25f",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "index": index,
        "docs_in_index": "~1M",
        "k": k,
        "num_queries": n,
        "avg_overlap_at_k": round(avg_overlap, 4),
        "avg_splade_hits": round(splade_hit_count / n, 1) if n else 0,
        "avg_bm25_hits": round(bm25_hit_count / n, 1) if n else 0,
        "avg_encode_ms": round(avg_encode, 1),
        "subject_stats": {
            subj: {
                "avg_splade_hits": round(s["splade_hits"] / s["count"], 1),
                "avg_bm25_hits": round(s["bm25_hits"] / s["count"], 1),
                "avg_overlap": round(s["overlap"] / s["count"], 4),
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
