#!/usr/bin/env python3
"""Eval the thin-client stack (tantivy+BMP+usearch) against the OpenSearch
baseline: per-leg overlap, latency, and LLM-judged NDCG@10.

Compares, per query (data/eval_results/diverse_queries.json):
  - thinclient bm25f  vs  OpenSearch bm25f   (top-k id overlap)
  - thinclient splade vs  OpenSearch splade  (top-k id overlap)
  - thinclient RRF    vs  OpenSearch RRF     (overlap + NDCG@10 via the
    LLM-judge cache where judged docs exist)

Overlap vs the full-corpus baseline is only apples-to-apples when the
thin-client index holds the SAME corpus (post-migration). For subset builds
(--limit validation), pass --no-baseline: the run then reports thin-client
functional health only (legs return results, filters honoured, latency,
leg complementarity).

Usage
─────
    .venv/bin/python3 scripts/eval_thinclient_parity.py \
        --index-root data/thinclient_index [--queries 40] [--no-baseline] \
        [--baseline-url http://host.docker.internal:9200]
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_thinclient")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

TOP_K = 50
NDCG_K = 10


def load_queries(n: int) -> list[dict]:
    records = json.loads((REPO_ROOT / "data/eval_results/diverse_queries.json").read_text())
    out, seen = [], set()
    for rec in records:
        if rec["paraphrase"] not in seen:
            seen.add(rec["paraphrase"])
            out.append(rec)
        if len(out) >= n:
            break
    return out


def load_judge() -> dict[tuple[str, str], int]:
    cache = json.loads((REPO_ROOT / "data/eval_results/llm_judge_cache.json").read_text())
    out = {}
    for key, grade in cache.items():
        qk, _, did = key.rpartition("||")
        out[(qk, did)] = grade
    return out


def ndcg_at_k(ranked_ids: list[str], judge: dict, judge_key: str, k: int = NDCG_K) -> float | None:
    grades = [judge.get((judge_key, did)) for did in ranked_ids[:k]]
    known = [g for g in grades if g is not None]
    if not known:
        return None
    dcg = sum((2 ** g - 1) / math.log2(i + 2)
              for i, g in enumerate(grades) if g is not None)
    ideal = sorted((g for (qk, _), g in judge.items() if qk == judge_key),
                   reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None


def rrf(lists: list[list[str]], k: int = 60, top: int = TOP_K) -> list[str]:
    scores: dict[str, float] = {}
    for lst in lists:
        for rank, did in enumerate(lst, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:top]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", default=str(REPO_ROOT / "data/thinclient_index"))
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--baseline-url",
                        default=os.environ.get("SFU_MIGRATION_SOURCE",
                                               "http://host.docker.internal:9200"))
    parser.add_argument("--baseline-index", default="openalex_works")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from lib.thinclient.retriever import ThinClientRetriever
    tc = ThinClientRetriever(index_root=args.index_root, remote_abstracts=False)
    assert tc.is_available(), f"thin-client index at {args.index_root} not available"
    logger.info("thin-client live sections: %s", tc.live_sections())

    baseline = None
    if not args.no_baseline:
        from lib.opensearch_retriever import OpenSearchRetriever
        baseline = OpenSearchRetriever(url=args.baseline_url, index=args.baseline_index,
                                       splade_model_path=str(REPO_ROOT / "models/splade_onnx"),
                                       timeout=30)

    queries = load_queries(args.queries)
    judge = load_judge()

    per_query, lat = [], {"tc_bm25f": [], "tc_splade": [], "os_bm25f": [], "os_splade": []}
    ndcgs = {"tc_rrf": [], "os_rrf": []}
    overlaps = {"bm25f": [], "splade": [], "rrf": []}
    leg_overlap_tc = []

    for rec in queries:
        q, jk = rec["paraphrase"], rec.get("judge_key", rec["paraphrase"])
        t0 = time.perf_counter()
        tc_bm = [d["openalex_id"] for d in tc.search(q, TOP_K, mode="bm25f")]
        lat["tc_bm25f"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        tc_sp = [d["openalex_id"] for d in tc.search(q, TOP_K, mode="splade")]
        lat["tc_splade"].append(time.perf_counter() - t0)
        tc_rrf = rrf([tc_bm, tc_sp])
        leg_overlap_tc.append(len(set(tc_bm) & set(tc_sp)))

        row = {"query": q, "tc_bm25f_n": len(tc_bm), "tc_splade_n": len(tc_sp)}
        n = ndcg_at_k(tc_rrf, judge, jk)
        if n is not None:
            ndcgs["tc_rrf"].append(n)
            row["tc_rrf_ndcg@10"] = round(n, 4)

        if baseline:
            t0 = time.perf_counter()
            os_bm = [d["openalex_id"] for d in baseline.search(q, TOP_K, mode="bm25f")]
            lat["os_bm25f"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            os_sp = [d["openalex_id"] for d in baseline.search(q, TOP_K, mode="splade")]
            lat["os_splade"].append(time.perf_counter() - t0)
            os_rrf = rrf([os_bm, os_sp])
            overlaps["bm25f"].append(len(set(tc_bm) & set(os_bm)) / max(len(os_bm), 1))
            overlaps["splade"].append(len(set(tc_sp) & set(os_sp)) / max(len(os_sp), 1))
            overlaps["rrf"].append(len(set(tc_rrf) & set(os_rrf)) / max(len(os_rrf), 1))
            n = ndcg_at_k(os_rrf, judge, jk)
            if n is not None:
                ndcgs["os_rrf"].append(n)
                row["os_rrf_ndcg@10"] = round(n, 4)
        per_query.append(row)

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {
        "config": {"index_root": args.index_root, "queries": len(queries),
                   "top_k": TOP_K, "baseline": None if args.no_baseline
                   else f"{args.baseline_url}/{args.baseline_index}"},
        "latency_ms": {k: round(avg(v) * 1000, 2) if v else None for k, v in lat.items()},
        "ndcg@10": {k: {"mean": avg(v), "judged_queries": len(v)} for k, v in ndcgs.items()},
        "overlap_vs_baseline@50": {k: avg(v) for k, v in overlaps.items()},
        "tc_leg_complementarity": {"avg_bm25f_splade_overlap@50": avg(leg_overlap_tc)},
        "per_query": per_query,
    }

    out_path = Path(args.output or REPO_ROOT /
                    f"data/eval_results/thinclient_parity_{time.strftime('%Y%m%d_%H%M')}.json")
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 88)
    print(f"THIN-CLIENT PARITY EVAL — {len(queries)} queries vs "
          f"{'(no baseline)' if args.no_baseline else summary['config']['baseline']}")
    print("=" * 88)
    print("latency ms/query:", {k: v for k, v in summary["latency_ms"].items() if v})
    print("NDCG@10:", summary["ndcg@10"])
    if not args.no_baseline:
        print("overlap@50 vs baseline:", summary["overlap_vs_baseline@50"])
    print("tc leg overlap@50 (bm25f∩splade):",
          summary["tc_leg_complementarity"]["avg_bm25f_splade_overlap@50"])
    print("=" * 88)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
