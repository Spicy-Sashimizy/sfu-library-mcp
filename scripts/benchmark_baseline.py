#!/usr/bin/env python3
"""Baseline benchmark: Primo API search → heuristic reranker.

Evaluates the production pipeline end-to-end on the 120-query SFU eval set.
Pipeline: Primo search (limit=30 over-fetch) → rerank_results() → top-10.
Ground truth: seed_doi + top-20 cited-by references from OpenAlex (cached).

Metrics per query: NDCG@10, MRR@10, Recall@10, latency (ms).
Outputs:
  results/baseline_eval.json       — per-query + aggregate + by-subject breakdown
  results/sfu_eval_history.jsonl   — one-line append for eval_compare.py
  data/baseline_eval_cache.json    — OpenAlex reference cache (avoid re-fetching)

Usage:
  cd /workspaces/sfu-library-mcp
  .venv/bin/python3 scripts/benchmark_baseline.py [--dry-run] [--limit N]

Options:
  --dry-run   Skip actual Primo calls; use empty results (for testing metric logic)
  --limit N   Only evaluate first N queries (default: all 120)
"""

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from lib.client import SFULibraryClient
from lib.config import load_config
from lib.reranker import rerank_results

# ── Config ────────────────────────────────────────────────────────────────────
EVAL_QUERIES_PATH = Path("/workspaces/sfu-library-mcp-training/data/sfu_eval_queries.json")
CACHE_PATH = REPO_ROOT / "data" / "baseline_eval_cache.json"
RESULTS_PATH = REPO_ROOT / "results" / "baseline_eval.json"
HISTORY_PATH = REPO_ROOT / "results" / "sfu_eval_history.jsonl"

FETCH_LIMIT = 30       # over-fetch for reranker (mirrors production: limit*3)
RERANK_LIMIT = 10      # return top-10 after rerank
OPENALEX_REFS = 20     # how many cited references to include in ground truth
OPENALEX_DELAY = 0.35  # seconds between OpenAlex calls (rate-limit courtesy)
PIPELINE_ID = "primo_heuristic_reranker_baseline"
PIPELINE_DESC = (
    "Primo API search → heuristic reranker "
    "(title_relevance 0.35, recency 0.20, fulltext 0.20, type_match 0.15, completeness 0.10)"
)

# ── DOI helpers ───────────────────────────────────────────────────────────────

def _normalize_doi(doi: str) -> str:
    """Return lowercase DOI without URL prefix."""
    if not doi:
        return ""
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def extract_doc_doi(doc: dict) -> str:
    """Extract normalized DOI from a Primo PNX document."""
    pnx = doc.get("pnx", {})
    addata = pnx.get("addata", {})
    dois = addata.get("doi", [])
    if dois and dois[0]:
        return _normalize_doi(dois[0])
    # Fallback: scan link fields
    for link_list in pnx.get("links", {}).values():
        if isinstance(link_list, list):
            for link in link_list:
                if isinstance(link, str) and "doi.org/" in link:
                    parts = link.split("doi.org/", 1)
                    if len(parts) > 1:
                        return _normalize_doi(parts[1].split("?")[0])
    return ""


# ── OpenAlex ground truth ─────────────────────────────────────────────────────

def _openalex_headers() -> dict:
    return {
        "Accept": "application/json",
        "User-Agent": "SFULibraryMCP-Benchmark/1.0 (mailto:benchmark@sfu-library-mcp)",
    }


def fetch_openalex_references(seed_doi: str, cache: dict) -> set[str]:
    """Return a set of normalized DOIs relevant to seed_doi.

    Relevant set = {seed_doi} ∪ {top-N references cited by the seed paper}.
    Results are stored in *cache* keyed by seed_doi to avoid re-fetching.
    Returns an empty set (not raising) on any API failure.
    """
    doi_key = seed_doi.lower().strip()
    if doi_key in cache:
        return set(d for d in cache[doi_key] if d)

    relevant: list[str] = []
    try:
        url = f"https://api.openalex.org/works/https://doi.org/{doi_key}"
        resp = requests.get(url, headers=_openalex_headers(), timeout=20)
        if resp.status_code != 200:
            cache[doi_key] = []
            return set()

        work = resp.json()

        # Seed DOI itself (OpenAlex may canonicalize it)
        canonical = _normalize_doi(work.get("doi", ""))
        if canonical:
            relevant.append(canonical)
        elif doi_key:
            relevant.append(doi_key)

        # Fetch DOIs for cited referenced works (referenced_works = what this paper cites)
        ref_ids = work.get("referenced_works", [])[:OPENALEX_REFS]
        if ref_ids:
            # Batch resolve OpenAlex IDs → DOIs via filter API
            id_filter = "|".join(rid.split("/")[-1] for rid in ref_ids)
            batch_url = (
                f"https://api.openalex.org/works"
                f"?filter=openalex:{id_filter}"
                f"&select=doi&per_page={OPENALEX_REFS}"
            )
            time.sleep(OPENALEX_DELAY)
            batch_resp = requests.get(batch_url, headers=_openalex_headers(), timeout=20)
            if batch_resp.status_code == 200:
                for ref_work in batch_resp.json().get("results", []):
                    ref_doi = _normalize_doi(ref_work.get("doi", ""))
                    if ref_doi:
                        relevant.append(ref_doi)

    except Exception:
        pass

    cache[doi_key] = relevant
    return set(d for d in relevant if d)


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_ndcg(docs: list[dict], relevant: set[str], k: int = 10) -> float:
    """NDCG@k with binary relevance.

    Each DOI counts at most once (deduplicates repeated results).
    IDCG uses the full ground-truth size so missing relevant docs are penalized.
    """
    def dcg(rels: list[float]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

    seen: set[str] = set()
    gains: list[float] = []
    for doc in docs[:k]:
        doi = extract_doc_doi(doc)
        if doi and doi in relevant and doi not in seen:
            gains.append(1.0)
            seen.add(doi)
        else:
            gains.append(0.0)

    n_rel = len(relevant)
    ideal = [1.0] * min(n_rel, k) + [0.0] * max(0, k - n_rel)
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def compute_mrr(docs: list[dict], relevant: set[str], k: int = 10) -> float:
    """MRR@k: reciprocal rank of the first relevant result (deduped)."""
    seen: set[str] = set()
    for i, doc in enumerate(docs[:k]):
        doi = extract_doc_doi(doc)
        if doi and doi in relevant and doi not in seen:
            return 1.0 / (i + 1)
        if doi:
            seen.add(doi)
    return 0.0


def compute_recall(docs: list[dict], relevant: set[str], k: int = 10) -> float:
    """Recall@k: unique relevant DOIs found in top k, divided by |relevant|."""
    if not relevant:
        return 0.0
    found = {extract_doc_doi(d) for d in docs[:k]} & relevant
    return len(found) / len(relevant)


# ── Single query eval ─────────────────────────────────────────────────────────

def run_query(
    client: SFULibraryClient,
    query_item: dict,
    gt_cache: dict,
    dry_run: bool = False,
) -> dict:
    """Run one eval query; return metrics dict."""
    query = query_item["query"]
    seed_doi = _normalize_doi(query_item.get("seed_doi", ""))

    # Ground truth: seed + its top-N references
    relevant = fetch_openalex_references(seed_doi, gt_cache)
    relevant.discard("")
    if seed_doi:
        relevant.add(seed_doi)

    if dry_run:
        docs: list[dict] = []
        reranked: list[dict] = []
        latency_ms = 0.0
    else:
        t0 = time.perf_counter()
        raw = client.search(
            query=query,
            limit=FETCH_LIMIT,
            offset=0,
            field="any",
            sort="rank",
        )
        docs = raw.get("docs", []) if raw else []
        reranked = rerank_results(docs, query, RERANK_LIMIT)
        latency_ms = (time.perf_counter() - t0) * 1000.0

    ndcg = compute_ndcg(reranked, relevant)
    mrr = compute_mrr(reranked, relevant)
    recall = compute_recall(reranked, relevant)

    top_dois = [d for d in (extract_doc_doi(doc) for doc in reranked) if d]

    return {
        "query": query,
        "subject": query_item.get("subject", "unknown"),
        "difficulty": query_item.get("difficulty", "unknown"),
        "seed_doi": seed_doi,
        "ndcg_at_10": round(ndcg, 4),
        "mrr_at_10": round(mrr, 4),
        "recall_at_10": round(recall, 4),
        "latency_ms": round(latency_ms, 1),
        "num_results": len(reranked),
        "relevant_set_size": len(relevant),
        "seed_in_top10": (seed_doi in top_dois) if seed_doi else None,
        "top_dois": top_dois,
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    idx = int(len(sorted_vals) * p)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def aggregate_results(results: list[dict]) -> dict:
    valid = [r for r in results if "error" not in r]
    lats = [r["latency_ms"] for r in valid]
    return {
        "ndcg_at_10": round(_mean([r["ndcg_at_10"] for r in valid]), 4),
        "mrr_at_10": round(_mean([r["mrr_at_10"] for r in valid]), 4),
        "recall_at_10": round(_mean([r["recall_at_10"] for r in valid]), 4),
        "latency_ms_mean": round(_mean(lats), 1),
        "latency_ms_p95": round(_percentile(lats, 0.95), 1),
        "latency_ms_median": round(_percentile(lats, 0.50), 1),
        "num_queries_total": len(results),
        "num_queries_valid": len(valid),
        "num_errors": len(results) - len(valid),
    }


def by_subject_breakdown(results: list[dict]) -> dict[str, dict]:
    valid = [r for r in results if "error" not in r]
    buckets: dict[str, list[dict]] = {}
    for r in valid:
        buckets.setdefault(r["subject"], []).append(r)

    summary: dict[str, dict] = {}
    for subj, rows in sorted(buckets.items()):
        summary[subj] = {
            "n": len(rows),
            "ndcg_at_10": round(_mean([r["ndcg_at_10"] for r in rows]), 4),
            "mrr_at_10": round(_mean([r["mrr_at_10"] for r in rows]), 4),
            "recall_at_10": round(_mean([r["recall_at_10"] for r in rows]), 4),
            "latency_ms_mean": round(_mean([r["latency_ms"] for r in rows]), 1),
        }
    return summary


# ── Output ────────────────────────────────────────────────────────────────────

def print_summary(agg: dict, subject_summary: dict[str, dict]) -> None:
    print()
    print("=" * 72)
    print("BASELINE EVALUATION — Primo + Heuristic Reranker")
    print("=" * 72)
    print(f"{'Metric':<28} {'Value':>12}")
    print("-" * 42)
    print(f"{'NDCG@10':<28} {agg['ndcg_at_10']:>12.4f}")
    print(f"{'MRR@10':<28} {agg['mrr_at_10']:>12.4f}")
    print(f"{'Recall@10':<28} {agg['recall_at_10']:>12.4f}")
    print(f"{'Latency mean (ms)':<28} {agg['latency_ms_mean']:>12.1f}")
    print(f"{'Latency p50 (ms)':<28} {agg['latency_ms_median']:>12.1f}")
    print(f"{'Latency p95 (ms)':<28} {agg['latency_ms_p95']:>12.1f}")
    print(f"{'Queries evaluated':<28} {agg['num_queries_valid']:>12d}")
    print(f"{'Errors':<28} {agg['num_errors']:>12d}")

    print()
    print("=" * 72)
    print("BY SUBJECT AREA")
    print("=" * 72)
    hdr = f"{'Subject':<26} {'N':>4} {'NDCG@10':>9} {'MRR@10':>9} {'R@10':>9} {'ms':>9}"
    print(hdr)
    print("-" * 72)
    for subj, s in sorted(subject_summary.items(), key=lambda x: -x[1]["ndcg_at_10"]):
        print(
            f"{subj:<26} {s['n']:>4} {s['ndcg_at_10']:>9.4f} "
            f"{s['mrr_at_10']:>9.4f} {s['recall_at_10']:>9.4f} "
            f"{s['latency_ms_mean']:>9.1f}"
        )
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Skip Primo calls (test metric logic only)")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N queries (0 = all)")
    args = parser.parse_args()

    # Load eval queries
    if not EVAL_QUERIES_PATH.exists():
        print(f"ERROR: eval queries not found: {EVAL_QUERIES_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(EVAL_QUERIES_PATH) as f:
        all_queries: list[dict] = json.load(f)

    queries = all_queries[: args.limit] if args.limit else all_queries
    print(f"Loaded {len(queries)} / {len(all_queries)} eval queries")

    # Load OpenAlex reference cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    gt_cache: dict = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            gt_cache = json.load(f)
        print(f"OpenAlex cache: {len(gt_cache)} cached seed DOIs")

    # Authenticate (skip in dry-run)
    client: SFULibraryClient | None = None
    if not args.dry_run:
        config = load_config()
        client = SFULibraryClient(config=config)
        print("Authenticating with SFU Library Primo API…")
        if not client.ensure_authenticated():
            print("ERROR: Authentication failed. Check SFU_USERNAME/SFU_PASSWORD/SFU_MFA_SECRET.", file=sys.stderr)
            sys.exit(1)
        print("Authenticated. Starting evaluation…\n")
    else:
        print("[DRY-RUN] Skipping Primo auth. Results will show 0 for all metrics.\n")

    # ── Evaluation loop ──────────────────────────────────────────────────────
    results: list[dict] = []
    cache_dirty = False

    for i, query_item in enumerate(queries, 1):
        label = query_item["query"][:55]
        print(f"[{i:3d}/{len(queries)}] {label:<55}", end="", flush=True)

        try:
            r = run_query(client, query_item, gt_cache, dry_run=args.dry_run)
            results.append(r)
            cache_dirty = True
            print(
                f" NDCG={r['ndcg_at_10']:.3f}  MRR={r['mrr_at_10']:.3f}"
                f"  R@10={r['recall_at_10']:.3f}  {r['latency_ms']:>6.0f}ms"
            )
        except Exception as exc:
            print(f" ERROR: {exc}")
            results.append({
                "query": query_item.get("query", ""),
                "subject": query_item.get("subject", "unknown"),
                "difficulty": query_item.get("difficulty", "unknown"),
                "seed_doi": _normalize_doi(query_item.get("seed_doi", "")),
                "error": str(exc),
                "ndcg_at_10": 0.0,
                "mrr_at_10": 0.0,
                "recall_at_10": 0.0,
                "latency_ms": 0.0,
                "num_results": 0,
                "relevant_set_size": 0,
                "seed_in_top10": None,
                "top_dois": [],
            })

        # Flush cache every 10 queries
        if cache_dirty and i % 10 == 0:
            with open(CACHE_PATH, "w") as f:
                json.dump(gt_cache, f, indent=2)
            cache_dirty = False

    # Final cache flush
    if cache_dirty:
        with open(CACHE_PATH, "w") as f:
            json.dump(gt_cache, f, indent=2)
    print(f"\nOpenAlex cache saved → {CACHE_PATH}")

    # ── Aggregate + output ────────────────────────────────────────────────────
    agg = aggregate_results(results)
    subj = by_subject_breakdown(results)
    run_at = datetime.now(timezone.utc).isoformat()

    output = {
        "run_at": run_at,
        "pipeline": PIPELINE_ID,
        "description": PIPELINE_DESC,
        "aggregate": agg,
        "by_subject": subj,
        "per_query": results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved      → {RESULTS_PATH}")

    # Append to history (one JSON line per run — readable by eval_compare.py)
    history_entry = {
        "run_at": run_at,
        "pipeline": PIPELINE_ID,
        **agg,
    }
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(history_entry) + "\n")
    print(f"History appended   → {HISTORY_PATH}")

    print_summary(agg, subj)


if __name__ == "__main__":
    main()
