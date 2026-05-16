#!/usr/bin/env python3
"""LLM-judged NDCG@10 benchmark: replaces citation-count proxy with topical relevance.

Uses Claude Haiku to judge whether retrieved papers are actually relevant to each
query (0-3 scale, TREC-style). Pools top results from all retrieval methods before
judging so the LLM evaluates papers blind to which system found them — a fair,
unbiased comparison.

Why this beats the citation-count proxy:
  - Citation count = paper fame; LLM judgment = topical relevance to the query
  - No tautology: OpenAlex cite-sort and the judge are independent
  - Works across all 120 domain-tagged SFU queries including niche humanities topics
  - Cached to disk: API calls run once, then all analysis is free

Cost estimate (120 queries × ~30 unique papers):
  ~3,600 judgments × ~400 tokens each ≈ $1.50 in Claude Haiku API costs

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/benchmark_llm_judge.py
    python scripts/benchmark_llm_judge.py --skip-live --methods bm25,splade,rrf
    python scripts/benchmark_llm_judge.py --max-queries 10  # quick smoke test

Outputs:
    data/eval_results/llm_judge_cache.json     — all judgments (reused on reruns)
    data/eval_results/benchmark_llm_judge_DATETIME.json — full results
"""

import argparse
import json
import logging
import math
import os
import re
import sys
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
JUDGE_CACHE_FILE = ROOT / "data" / "eval_results" / "llm_judge_cache.json"
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
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

# 4-point TREC-style scale
RELEVANCE_LABELS = {
    3: "Perfectly relevant — directly addresses the query topic",
    2: "Highly relevant — substantially related, useful for the query",
    1: "Marginally relevant — tangentially related, limited usefulness",
    0: "Not relevant — unrelated to the query topic",
}

JUDGE_SYSTEM_PROMPT = """You are an expert academic librarian evaluating search results for a university library.

Your task: judge whether each paper is relevant to the given search query.

Use this 4-point scale:
3 = Perfectly relevant: directly about the query topic, highly useful
2 = Highly relevant: strongly related, would be useful for this query
1 = Marginally relevant: tangentially related but limited direct value
0 = Not relevant: unrelated to the query

Instructions:
- Evaluate based on topical relevance to the query, NOT paper quality or citation count
- A recent obscure paper can score 3 if it directly addresses the query
- A famous highly-cited paper scores 0 if it's off-topic

Respond ONLY with a JSON object: {"<paper_id>": <score>, ...}
No explanation, no markdown, just the JSON."""


# ── Judge cache ───────────────────────────────────────────────────────────────

def _load_judge_cache() -> dict:
    JUDGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if JUDGE_CACHE_FILE.exists():
        try:
            return json.loads(JUDGE_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_judge_cache(cache: dict):
    JUDGE_CACHE_FILE.write_text(json.dumps(cache, indent=2))


JUDGE_CACHE: dict = _load_judge_cache()


def _cache_key(query: str, openalex_id: str) -> str:
    return f"{query[:80]}||{openalex_id}"


# ── OpenSearch retrieval ──────────────────────────────────────────────────────

def fetch_abstracts(session: requests.Session, openalex_ids: list[str]) -> dict[str, dict]:
    """Fetch title+abstract for a batch of openalex_ids from OpenSearch."""
    if not openalex_ids:
        return {}
    body = {
        "size": len(openalex_ids),
        "query": {"terms": {"_id": openalex_ids}},
        "_source": ["title", "abstract"],
    }
    try:
        resp = session.post(f"{OPENSEARCH_URL}/{INDEX}/_search", json=body, timeout=15)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        return {
            h["_id"]: {
                "title": h["_source"].get("title", ""),
                "abstract": h["_source"].get("abstract", ""),
            }
            for h in hits
        }
    except Exception as e:
        log.warning("Abstract fetch failed: %s", e)
        return {}


def bm25f_search(session: requests.Session, query_text: str, k: int = 10) -> list[dict]:
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
        "_source": ["openalex_id", "title"],
    }
    try:
        resp = session.post(f"{OPENSEARCH_URL}/{INDEX}/_search", json=body, timeout=15)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        return [{"id": h["_id"], "openalex_id": h["_source"].get("openalex_id", h["_id"]),
                 "title": h["_source"].get("title", "")}
                for h in hits if h.get("_source", {}).get("title")]
    except Exception as e:
        log.warning("BM25F search failed: %s", e)
        return []


def splade_search(session: requests.Session, sparse_query: dict, k: int = 10) -> list[dict]:
    if not sparse_query:
        return []
    should = []
    for term, weight in sorted(sparse_query.items(), key=lambda x: -x[1])[:48]:
        should.append({"rank_feature": {"field": f"sparse_field.{term}", "boost": weight,
                                        "log": {"scaling_factor": 1}}})
    body = {"size": k, "query": {"bool": {"should": should}}, "_source": ["openalex_id", "title"]}
    try:
        resp = session.post(f"{OPENSEARCH_URL}/{INDEX}/_search", json=body, timeout=30)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        return [{"id": h["_id"], "openalex_id": h["_source"].get("openalex_id", h["_id"]),
                 "title": h["_source"].get("title", "")}
                for h in hits if h.get("_source", {}).get("title")]
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
    return [dict(docs[did], rrf_score=s) for did, s in sorted(scores.items(), key=lambda x: -x[1])]


def openalex_live_search(session: requests.Session, query_text: str,
                         sort: str = "relevance_score:desc", k: int = 10) -> list[dict]:
    params: dict = {
        "search": query_text, "per_page": k,
        "select": "id,display_name,cited_by_count,abstract_inverted_index",
    }
    if sort:
        params["sort"] = sort
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    else:
        params["mailto"] = OPENALEX_MAILTO

    for attempt in range(4):
        try:
            resp = session.get(f"{OPENALEX_BASE}/works", params=params,
                               headers={"User-Agent": "SFULibraryMCP-Eval/1.0"}, timeout=30)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            out = []
            for w in resp.json().get("results", []):
                raw_id = w.get("id", "")
                oid = raw_id.split("/")[-1] if "/" in raw_id else raw_id
                abstract = ""
                inv = w.get("abstract_inverted_index") or {}
                if inv:
                    positions = []
                    for word, pos_list in inv.items():
                        for p in pos_list:
                            positions.append((p, word))
                    abstract = " ".join(w for _, w in sorted(positions))
                title = w.get("display_name", "") or w.get("title", "")
                if title:
                    out.append({"id": oid, "openalex_id": oid, "title": title,
                                "abstract": abstract[:800], "cited_by_count": w.get("cited_by_count", 0)})
            time.sleep(0.15)
            return out
        except Exception as e:
            if attempt == 3:
                log.warning("OpenAlex search failed: %s", e)
            time.sleep(2 * (attempt + 1))
    return []


# ── LLM judge ────────────────────────────────────────────────────────────────

def _batch_judge(query: str, papers: list[dict], client) -> dict[str, int]:
    """
    Call Claude Haiku to judge a batch of papers for relevance to query.
    Returns {openalex_id: score} for all papers in the batch.
    """
    if not papers:
        return {}

    # Build paper descriptions for the prompt
    paper_descs = []
    for p in papers:
        title = p.get("title", "")[:200]
        abstract = p.get("abstract", "")[:500]
        desc = f'[{p["openalex_id"]}]\nTitle: {title}'
        if abstract:
            desc += f'\nAbstract snippet: {abstract}'
        paper_descs.append(desc)

    user_message = (
        f'Query: "{query}"\n\n'
        f'Rate each paper\'s relevance (0-3):\n\n'
        + "\n\n".join(paper_descs)
    )

    for attempt in range(4):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text.strip()
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                return {k: max(0, min(3, int(v))) for k, v in scores.items()}
            return {}
        except Exception as e:
            if attempt == 3:
                log.warning("LLM judge call failed: %s", e)
            time.sleep(2 * (attempt + 1))
    return {}


def judge_papers(query: str, papers: list[dict], client, batch_size: int = 8) -> dict[str, int]:
    """
    Judge all papers for a query, using the cache for already-judged papers.
    Returns {openalex_id: score 0-3}.
    """
    results: dict[str, int] = {}
    to_judge = []

    for p in papers:
        oid = p.get("openalex_id", p.get("id", ""))
        cache_k = _cache_key(query, oid)
        if cache_k in JUDGE_CACHE:
            results[oid] = JUDGE_CACHE[cache_k]
        else:
            to_judge.append(p)

    if not to_judge:
        return results

    # Batch API calls
    for i in range(0, len(to_judge), batch_size):
        batch = to_judge[i:i + batch_size]
        batch_scores = _batch_judge(query, batch, client)
        for p in batch:
            oid = p.get("openalex_id", p.get("id", ""))
            score = batch_scores.get(oid, -1)
            if score < 0:
                # Retry solo if batch extraction missed this paper
                solo_scores = _batch_judge(query, [p], client)
                score = solo_scores.get(oid, 0)
            results[oid] = score
            JUDGE_CACHE[_cache_key(query, oid)] = score

    _save_judge_cache(JUDGE_CACHE)
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_rel: list[float], k: int) -> float:
    def dcg(rels, k):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = dcg(sorted(ranked_rel, reverse=True), k)
    return dcg(ranked_rel, k) / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked_rel: list[float], k: int) -> float:
    for rank, rel in enumerate(ranked_rel[:k], start=1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked_rel: list[float], k: int, threshold: float = 1.0) -> float:
    hits = sum(1 for r in ranked_rel[:k] if r >= threshold)
    return hits / k


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-judged IR benchmark (TREC-style)")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=10,
                        help="Top-N results to retrieve and judge per method (default 10)")
    parser.add_argument("--methods", default="bm25,splade,rrf,openalex_relevance",
                        help="Comma-separated methods to run")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--skip-live", action="store_true",
                        help="Skip OpenAlex live API (local index only)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit number of queries (for smoke tests)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Retrieve docs but skip LLM judging (shows cache coverage)")
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY and not args.dry_run:
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        print("Add it to .env or export it:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("\nThen re-run. Or use --dry-run to test retrieval without judging.\n")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if not args.dry_run else None

    with open(args.queries) as f:
        eval_queries = json.load(f)
    if args.max_queries:
        eval_queries = eval_queries[:args.max_queries]
    log.info("Loaded %d eval queries", len(eval_queries))

    methods_requested = [m.strip() for m in args.methods.split(",")]
    local_methods = [m for m in methods_requested if m in ("bm25", "splade", "rrf")]
    live_methods = [m for m in methods_requested if m.startswith("openalex")]

    needs_splade = "splade" in methods_requested or "rrf" in methods_requested
    encoder = None
    if needs_splade:
        log.info("Loading SPLADE encoder...")
        sys.path.insert(0, str(ROOT / "scripts"))
        from benchmark_splade import SpladeQueryEncoder
        encoder = SpladeQueryEncoder(SPLADE_MODEL, device=args.device)

    session = requests.Session()

    # Phase 1: retrieve results from all methods
    log.info("Phase 1: Retrieving top-%d from: %s", args.top_n, ", ".join(methods_requested))
    query_data = []
    for qi, q in enumerate(eval_queries):
        query_text = q["query"]
        subject = q.get("subject", "Unknown")
        row: dict = {"query": query_text, "subject": subject}

        bm25_res = bm25f_search(session, query_text, k=args.top_n) if "bm25" in methods_requested or "rrf" in methods_requested else []
        splade_res = []
        if needs_splade and encoder:
            sparse = encoder.encode(query_text)
            splade_res = splade_search(session, sparse, k=args.top_n)

        if "bm25" in methods_requested:
            row["bm25"] = bm25_res
        if "splade" in methods_requested:
            row["splade"] = splade_res
        if "rrf" in methods_requested:
            row["rrf"] = rrf_fuse([bm25_res, splade_res])[:args.top_n]
        if "openalex_relevance" in methods_requested and not args.skip_live:
            row["openalex_relevance"] = openalex_live_search(
                session, query_text, sort="relevance_score:desc", k=args.top_n)
        if "openalex_topic" in methods_requested and not args.skip_live:
            row["openalex_topic"] = openalex_live_search(
                session, query_text, sort="cited_by_count:desc", k=args.top_n)

        query_data.append(row)
        if (qi + 1) % 20 == 0:
            log.info("  Retrieved %d/%d", qi + 1, len(eval_queries))

    # Phase 2: fetch abstracts for local index results
    log.info("Phase 2: Fetching abstracts from OpenSearch...")
    all_local_ids: set[str] = set()
    for row in query_data:
        for method in local_methods:
            for r in row.get(method, []):
                if r.get("id"):
                    all_local_ids.add(r["id"])

    abstracts = {}
    id_list = list(all_local_ids)
    for i in range(0, len(id_list), 50):
        batch_abstracts = fetch_abstracts(session, id_list[i:i + 50])
        abstracts.update(batch_abstracts)
    log.info("  Fetched abstracts for %d/%d local docs", len(abstracts), len(all_local_ids))

    # Attach abstracts to local results
    for row in query_data:
        for method in local_methods:
            for r in row.get(method, []):
                doc = abstracts.get(r["id"], {})
                if not r.get("abstract"):
                    r["abstract"] = doc.get("abstract", "")
                if not r.get("title") and doc.get("title"):
                    r["title"] = doc["title"]

    # Phase 3: pool unique papers per query, judge with LLM
    log.info("Phase 3: LLM judging (%s)...", "DRY RUN" if args.dry_run else "calling Claude Haiku")
    all_methods = [m for m in methods_requested
                   if m in row or (args.skip_live and m.startswith("openalex"))]

    ndcg_scores: dict[str, list[float]] = {m: [] for m in methods_requested}
    mrr_scores: dict[str, list[float]] = {m: [] for m in methods_requested}
    p_at_k_scores: dict[str, list[float]] = {m: [] for m in methods_requested}
    subject_ndcg: dict[str, dict[str, list[float]]] = {m: {} for m in methods_requested}
    per_query_detail = []

    total_cached = 0
    total_new = 0

    for qi, row in enumerate(query_data):
        query_text = row["query"]
        subject = row["subject"]

        # Pool unique papers from all methods
        seen_ids: set[str] = set()
        pool: list[dict] = []
        for method in methods_requested:
            for r in row.get(method, []):
                oid = r.get("openalex_id", r.get("id", ""))
                if oid and oid not in seen_ids:
                    seen_ids.add(oid)
                    pool.append(r)

        # Count cache hits before judging
        cached_ids = {p["openalex_id"] for p in pool
                      if _cache_key(query_text, p.get("openalex_id", "")) in JUDGE_CACHE}
        total_cached += len(cached_ids)
        total_new += len(pool) - len(cached_ids)

        if args.dry_run:
            log.info("  [%d] %s: %d pool (%d cached, %d new)",
                     qi + 1, subject, len(pool), len(cached_ids), len(pool) - len(cached_ids))
            continue

        # Judge the pool
        judge_scores = judge_papers(query_text, pool, client)

        # Compute metrics for each method
        detail_row: dict = {"query": query_text, "subject": subject,
                             "pool_size": len(pool), "judged": len(judge_scores)}
        for method in methods_requested:
            results = row.get(method, [])[:args.top_n]
            if not results:
                continue
            relevance = [float(judge_scores.get(r.get("openalex_id", r.get("id", "")), 0))
                         for r in results]
            ndcg = ndcg_at_k(relevance, args.k)
            mrr = mrr_at_k(relevance, args.k)
            p_k = precision_at_k(relevance, args.k, threshold=2.0)
            ndcg_scores[method].append(ndcg)
            mrr_scores[method].append(mrr)
            p_at_k_scores[method].append(p_k)
            subject_ndcg[method].setdefault(subject, []).append(ndcg)
            detail_row[f"{method}_ndcg"] = round(ndcg, 4)
            detail_row[f"{method}_mrr"] = round(mrr, 4)
            detail_row[f"{method}_p_at_k"] = round(p_k, 4)

        per_query_detail.append(detail_row)

        if (qi + 1) % 10 == 0:
            log.info("  Judged %d/%d queries | cached=%d new=%d",
                     qi + 1, len(eval_queries), total_cached, total_new)

    if args.dry_run:
        log.info("DRY RUN complete. Cached=%d, would need %d new API calls.",
                 total_cached, total_new)
        log.info("Estimated cost: ~$%.2f (Haiku at $0.80/M input tokens)",
                 total_new * 400 / 1_000_000 * 0.80)
        return

    # Print report
    method_labels = {
        "bm25":               "BM25F (local)",
        "splade":             "SPLADE (local)",
        "rrf":                "BM25F+SPLADE RRF (local)",
        "openalex_relevance": "OpenAlex relevance sort",
        "openalex_topic":     "OpenAlex cite-count sort",
    }

    print()
    print("=" * 95)
    print("  LLM-JUDGED IR BENCHMARK  —  TREC-style 0-3 relevance (Claude Haiku)")
    print(f"  {len(eval_queries)} queries  |  k={args.k}  |  top_n={args.top_n}")
    print(f"  LLM judgments: {total_cached} cached + {total_new} new = {total_cached + total_new} total")
    print("=" * 95)

    print(f"\n{'Method':<30} {'NDCG@10':>8} {'Median':>8} {'Std':>7} {'MRR@10':>8} {'P@10(≥2)':>9} {'N':>5}")
    print("-" * 85)
    for method in methods_requested:
        scores = ndcg_scores[method]
        if not scores:
            print(f"  {method_labels.get(method, method):<28}  (no results)")
            continue
        label = method_labels.get(method, method)
        print(f"  {label:<28}  {np.mean(scores):>8.4f} {np.median(scores):>8.4f} "
              f"{np.std(scores):>7.4f} {np.mean(mrr_scores[method]):>8.4f} "
              f"{np.mean(p_at_k_scores[method]):>9.4f} {len(scores):>5}")

    # Decision summary
    if "rrf" in ndcg_scores and ndcg_scores["rrf"]:
        rrf_m = np.mean(ndcg_scores["rrf"])
        print(f"\n{'=' * 55}")
        print("  WIRING DECISION (LLM-judged, unbiased)")
        print(f"{'=' * 55}")
        for cmp_method in ["openalex_relevance", "openalex_topic", "bm25", "splade"]:
            if ndcg_scores.get(cmp_method):
                cmp_m = np.mean(ndcg_scores[cmp_method])
                diff = rrf_m - cmp_m
                label = method_labels.get(cmp_method, cmp_method)
                verdict = "WIRE" if diff > 0.01 else ("SKIP" if diff < -0.01 else "NEUTRAL")
                print(f"  RRF vs {label:<28}: {diff:+.4f}  [{verdict}]")

    # Subject breakdown
    all_subjects = sorted({s for m in methods_requested for s in subject_ndcg.get(m, {})})
    if all_subjects:
        show_methods = methods_requested[:4]
        header = f"\n{'Subject':<45}" + "".join(f" {method_labels.get(m,m)[:9]:>9}" for m in show_methods)
        print(header)
        print("-" * (45 + 10 * len(show_methods)))
        for subj in all_subjects:
            row_str = f"{subj[:44]:<45}"
            for m in show_methods:
                sc = subject_ndcg.get(m, {}).get(subj, [])
                row_str += f" {np.mean(sc):>9.4f}" if sc else f" {'n/a':>9}"
            print(row_str)

    # Save
    output = args.output or (
        ROOT / "data" / "eval_results" / f"benchmark_llm_judge_{time.strftime('%Y%m%d_%H%M')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": "llm_judged_ndcg",
        "judge_model": "claude-haiku-4-5-20251001",
        "relevance_scale": RELEVANCE_LABELS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": args.k,
        "top_n": args.top_n,
        "num_queries": len(eval_queries),
        "judgments": {"cached": total_cached, "new": total_new},
        "summary": {
            method: {
                "mean_ndcg": round(float(np.mean(ndcg_scores[method])), 4) if ndcg_scores[method] else None,
                "median_ndcg": round(float(np.median(ndcg_scores[method])), 4) if ndcg_scores[method] else None,
                "std_ndcg": round(float(np.std(ndcg_scores[method])), 4) if ndcg_scores[method] else None,
                "mean_mrr": round(float(np.mean(mrr_scores[method])), 4) if mrr_scores[method] else None,
                "mean_p_at_k": round(float(np.mean(p_at_k_scores[method])), 4) if p_at_k_scores[method] else None,
                "label": method_labels.get(method, method),
            }
            for method in methods_requested
        },
        "subject_breakdown": {
            method: {subj: round(float(np.mean(sc)), 4) for subj, sc in subject_ndcg[method].items()}
            for method in methods_requested
        },
        "per_query": per_query_detail,
    }
    output.write_text(json.dumps(result, indent=2))
    log.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
