#!/usr/bin/env python3
"""LLM-judged NDCG@10 benchmark using Claude Code's built-in Haiku access.

Replaces the citation-count proxy (tautological, biased toward OpenAlex coverage)
with topical relevance judgments from Claude Haiku — no ANTHROPIC_API_KEY needed.

How it works:
  1. Retrieve top-N from each method (BM25F, SPLADE, RRF, OpenAlex live)
  2. Pool unique papers across all methods per query (TREC-style pooling)
  3. Fetch abstracts from OpenSearch for local results
  4. Call `claude -p --model claude-haiku-4-5-20251001` in parallel subprocesses
     to judge each paper 0-3 on topical relevance to the query
  5. Cache all judgments to disk — reruns are free
  6. Compute NDCG@10, MRR@10, P@10(rel≥2) per method and per subject

Why this beats citation count:
  - Citation count = paper fame; Haiku judgment = topical relevance
  - No tautology: cite-sort and the judge are completely independent
  - Works for niche SFU topics (Indigenous Studies, Film, etc.)
  - OpenAlex live and local RRF compete on equal footing

Performance:
  - 4 parallel subprocess workers → ~4-5 min for all 120 queries
  - ~$0 additional cost — uses Claude Code's existing session

Usage:
    python scripts/benchmark_llm_judge.py
    python scripts/benchmark_llm_judge.py --skip-live --workers 6
    python scripts/benchmark_llm_judge.py --max-queries 10  # quick test
"""

import argparse
import concurrent.futures
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DEFAULT_QUERIES = ROOT / "data" / "sfu_eval_queries.json"
OPENSEARCH_URL = os.environ.get(
    "SFU_OPENSEARCH_URL",
    "http://claudebox-sfu-library-mcp-training-opensearch:9200",
)
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

JUDGE_SYSTEM = (
    "You are a relevance scoring API. Output ONLY valid JSON. "
    "No markdown fences, no prose, no explanation."
)

JUDGE_SCALE = {3: "perfectly relevant (directly about the query)", 2: "highly relevant (strongly related)",
               1: "marginally relevant (tangential)", 0: "not relevant (unrelated)"}


# ── Judgment cache ────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    JUDGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if JUDGE_CACHE_FILE.exists():
        try:
            return json.loads(JUDGE_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    JUDGE_CACHE_FILE.write_text(json.dumps(cache, indent=2))


JUDGE_CACHE: dict = _load_cache()
_cache_lock = None  # set to threading.Lock() in main


def _cache_key(query: str, openalex_id: str) -> str:
    return f"{query[:80]}||{openalex_id}"


# ── Claude Haiku judge (subprocess) ──────────────────────────────────────────

def _call_haiku(prompt: str, timeout: int = 90) -> str:
    """Call claude CLI and return stdout text. Raises on timeout or non-zero exit."""
    result = subprocess.run(
        ["claude", "-p",
         "--model", "claude-haiku-4-5-20251001",
         "--output-format", "text",
         "--system-prompt", JUDGE_SYSTEM,
         "--no-session-persistence"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def _parse_json_scores(text: str) -> dict[str, int]:
    """Extract {id: score} from a response that may have markdown fences."""
    json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if not json_match:
        return {}
    try:
        raw = json.loads(json_match.group())
        return {str(k): max(0, min(3, int(v))) for k, v in raw.items()}
    except Exception:
        return {}


def _batch_judge_subprocess(query: str, papers: list[dict]) -> dict[str, int]:
    """Judge a batch of papers for a query using one claude subprocess call."""
    if not papers:
        return {}

    lines = [f'Score each paper 0-3 for relevance to: "{query}"\n',
             "3=perfectly relevant  2=highly relevant  1=marginal  0=unrelated\n"]
    for p in papers:
        title = (p.get("title") or "")[:200]
        abstract = (p.get("abstract") or "")[:400]
        oid = p.get("openalex_id") or p.get("id", "")
        lines.append(f'[{oid}]\nTitle: {title}')
        if abstract:
            lines.append(f'Abstract: {abstract}')
        lines.append("")
    lines.append('Reply ONLY with JSON: {"<id>": score, ...}')
    prompt = "\n".join(lines)

    for attempt in range(3):
        try:
            raw = _call_haiku(prompt)
            scores = _parse_json_scores(raw)
            if scores:
                return scores
        except subprocess.TimeoutExpired:
            log.warning("Haiku call timed out (attempt %d)", attempt + 1)
        except Exception as e:
            log.warning("Haiku call failed (attempt %d): %s", attempt + 1, e)
        time.sleep(1)
    return {}


def judge_query(query: str, papers: list[dict], batch_size: int = 12) -> dict[str, int]:
    """
    Judge all papers for a single query, using cache for previously-seen papers.
    Returns {openalex_id: score 0-3}.
    """
    import threading
    results: dict[str, int] = {}
    to_judge: list[dict] = []

    for p in papers:
        oid = p.get("openalex_id") or p.get("id", "")
        key = _cache_key(query, oid)
        if key in JUDGE_CACHE:
            results[oid] = JUDGE_CACHE[key]
        else:
            to_judge.append(p)

    for i in range(0, len(to_judge), batch_size):
        batch = to_judge[i:i + batch_size]
        batch_scores = _batch_judge_subprocess(query, batch)

        # Recover any papers the LLM missed (try solo)
        missed = [p for p in batch if (p.get("openalex_id") or p.get("id", "")) not in batch_scores]
        if missed and len(missed) < len(batch):
            for p in missed:
                solo = _batch_judge_subprocess(query, [p])
                batch_scores.update(solo)

        for p in batch:
            oid = p.get("openalex_id") or p.get("id", "")
            score = batch_scores.get(oid, 0)
            results[oid] = score
            JUDGE_CACHE[_cache_key(query, oid)] = score

    return results


# ── OpenSearch retrieval ──────────────────────────────────────────────────────

def fetch_abstracts(session: requests.Session, doc_ids: list[str]) -> dict[str, dict]:
    if not doc_ids:
        return {}
    body = {"size": len(doc_ids), "query": {"terms": {"_id": doc_ids}},
            "_source": ["title", "abstract"]}
    try:
        resp = session.post(f"{OPENSEARCH_URL}/{INDEX}/_search", json=body, timeout=15)
        resp.raise_for_status()
        return {h["_id"]: {"title": h["_source"].get("title", ""),
                            "abstract": h["_source"].get("abstract", "")}
                for h in resp.json().get("hits", {}).get("hits", [])}
    except Exception as e:
        log.warning("Abstract fetch failed: %s", e)
        return {}


def bm25f_search(session: requests.Session, query_text: str, k: int = 10) -> list[dict]:
    body = {"size": k, "query": {"multi_match": {"query": query_text,
            "fields": ["title^3", "abstract", "concepts^2"],
            "type": "most_fields", "tie_breaker": 0.5}},
            "_source": ["openalex_id", "title"]}
    try:
        resp = session.post(f"{OPENSEARCH_URL}/{INDEX}/_search", json=body, timeout=15)
        resp.raise_for_status()
        return [{"id": h["_id"], "openalex_id": h["_source"].get("openalex_id", h["_id"]),
                 "title": h["_source"].get("title", "")}
                for h in resp.json().get("hits", {}).get("hits", [])
                if h.get("_source", {}).get("title")]
    except Exception as e:
        log.warning("BM25F failed: %s", e)
        return []


def splade_search(session: requests.Session, sparse_query: dict, k: int = 10) -> list[dict]:
    if not sparse_query:
        return []
    should = [{"rank_feature": {"field": f"sparse_field.{t}", "boost": w, "log": {"scaling_factor": 4}}}
              for t, w in sorted(sparse_query.items(), key=lambda x: -x[1])[:64]]
    body = {"size": k, "query": {"bool": {"should": should}}, "_source": ["openalex_id", "title"]}
    try:
        resp = session.post(f"{OPENSEARCH_URL}/{INDEX}/_search", json=body, timeout=30)
        resp.raise_for_status()
        return [{"id": h["_id"], "openalex_id": h["_source"].get("openalex_id", h["_id"]),
                 "title": h["_source"].get("title", "")}
                for h in resp.json().get("hits", {}).get("hits", [])
                if h.get("_source", {}).get("title")]
    except Exception as e:
        log.warning("SPLADE failed: %s", e)
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
    return [dict(docs[did], rrf_score=s)
            for did, s in sorted(scores.items(), key=lambda x: -x[1])]


def _reconstruct_abstract(inv_index: dict) -> str:
    if not inv_index:
        return ""
    positions = []
    for word, pos_list in inv_index.items():
        for p in pos_list:
            positions.append((p, word))
    return " ".join(w for _, w in sorted(positions))


def openalex_live_search(session: requests.Session, query_text: str,
                          sort: str = "relevance_score:desc", k: int = 10) -> list[dict]:
    params: dict = {"search": query_text, "per_page": k,
                    "select": "id,display_name,cited_by_count,abstract_inverted_index"}
    if sort:
        params["sort"] = sort
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    else:
        params["mailto"] = OPENALEX_MAILTO
    for attempt in range(3):
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
                title = w.get("display_name", "") or ""
                abstract = _reconstruct_abstract(w.get("abstract_inverted_index") or {})[:600]
                if title:
                    out.append({"id": oid, "openalex_id": oid, "title": title, "abstract": abstract})
            time.sleep(0.15)
            return out
        except Exception as e:
            if attempt == 2:
                log.warning("OpenAlex search failed: %s", e)
            time.sleep(2 * (attempt + 1))
    return []


# ── Metrics ───────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_rel: list[float], k: int) -> float:
    def dcg(rels, k):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = dcg(sorted(ranked_rel, reverse=True), k)
    return dcg(ranked_rel, k) / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked_rel: list[float], k: int) -> float:
    for i, r in enumerate(ranked_rel[:k], 1):
        if r > 0:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_rel: list[float], k: int, threshold: float = 2.0) -> float:
    return sum(1 for r in ranked_rel[:k] if r >= threshold) / k


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-judged IR benchmark via Claude Code Haiku")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=10,
                        help="Docs to retrieve per method per query (default 10)")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="Papers per Haiku subprocess call (default 12)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel claude subprocess workers (default 4)")
    parser.add_argument("--methods", default="bm25,splade,rrf,openalex_relevance",
                        help="Comma-separated retrieval methods to evaluate")
    parser.add_argument("--skip-live", action="store_true",
                        help="Skip OpenAlex live API calls")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit queries (e.g. 10 for a smoke test)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Retrieve docs and show cache stats without calling Haiku")
    parser.add_argument("--k-param", type=int, default=60,
                        help="RRF k constant for BM25F+SPLADE fusion sweep (default 60)")
    args = parser.parse_args()

    with open(args.queries) as f:
        eval_queries = json.load(f)
    if args.max_queries:
        eval_queries = eval_queries[:args.max_queries]
    log.info("Loaded %d eval queries", len(eval_queries))

    methods = [m.strip() for m in args.methods.split(",")]
    local_methods = [m for m in methods if m in ("bm25", "splade", "rrf")]
    live_methods = [m for m in methods if m.startswith("openalex")]

    needs_splade = any(m in methods for m in ("splade", "rrf"))
    encoder = None
    if needs_splade:
        log.info("Loading SPLADE encoder...")
        sys.path.insert(0, str(ROOT / "scripts"))
        from benchmark_splade import SpladeQueryEncoder
        encoder = SpladeQueryEncoder(SPLADE_MODEL, device=args.device)

    session = requests.Session()

    # Phase 1: retrieve
    log.info("Phase 1: Retrieving top-%d from: %s", args.top_n, ", ".join(methods))
    query_data = []
    for qi, q in enumerate(eval_queries):
        qt = q["query"]
        row: dict = {"query": qt, "subject": q.get("subject", "Unknown")}

        bm25_res = bm25f_search(session, qt, k=args.top_n) if any(m in methods for m in ("bm25", "rrf")) else []
        splade_res = []
        if needs_splade and encoder:
            sparse = encoder.encode(qt)
            splade_res = splade_search(session, sparse, k=args.top_n)

        if "bm25" in methods:
            row["bm25"] = bm25_res
        if "splade" in methods:
            row["splade"] = splade_res
        if "rrf" in methods:
            row["rrf"] = rrf_fuse([bm25_res, splade_res], k_param=args.k_param)[:args.top_n]
        if "openalex_relevance" in methods and not args.skip_live:
            row["openalex_relevance"] = openalex_live_search(
                session, qt, sort="relevance_score:desc", k=args.top_n)
        if "openalex_topic" in methods and not args.skip_live:
            row["openalex_topic"] = openalex_live_search(
                session, qt, sort="cited_by_count:desc", k=args.top_n)

        query_data.append(row)
        if (qi + 1) % 20 == 0:
            log.info("  Retrieved %d/%d", qi + 1, len(eval_queries))

    # Phase 2: fetch abstracts for local results
    log.info("Phase 2: Fetching abstracts from OpenSearch...")
    all_local_ids: set[str] = set()
    for row in query_data:
        for m in local_methods:
            for r in row.get(m, []):
                if r.get("id"):
                    all_local_ids.add(r["id"])

    abstracts: dict[str, dict] = {}
    id_list = list(all_local_ids)
    for i in range(0, len(id_list), 50):
        abstracts.update(fetch_abstracts(session, id_list[i:i + 50]))
    log.info("  Abstracts: %d/%d fetched", len(abstracts), len(all_local_ids))

    for row in query_data:
        for m in local_methods:
            for r in row.get(m, []):
                doc = abstracts.get(r["id"], {})
                if not r.get("abstract"):
                    r["abstract"] = doc.get("abstract", "")
                if not r.get("title") and doc.get("title"):
                    r["title"] = doc["title"]

    # Phase 3: LLM judging (parallel subprocess workers)
    total_cached = sum(
        1 for row in query_data for m in methods
        for r in row.get(m, [])
        if _cache_key(row["query"], r.get("openalex_id", r.get("id", ""))) in JUDGE_CACHE
    )
    total_pool = sum(
        len({r.get("openalex_id", r.get("id", ""))
             for m in methods for r in row.get(m, [])})
        for row in query_data
    )
    need_new = total_pool - total_cached

    if args.dry_run:
        log.info("DRY RUN: %d total unique papers, %d cached, %d need judging",
                 total_pool, total_cached, need_new)
        log.info("Estimated Haiku calls: ~%d (batch_size=%d, workers=%d)",
                 math.ceil(need_new / args.batch_size), args.batch_size, args.workers)
        log.info("Estimated wall time:   ~%.0f min",
                 math.ceil(need_new / args.batch_size) / args.workers * 4.5 / 60)
        return

    log.info("Phase 3: Judging with Claude Haiku (%d workers, batch=%d)",
             args.workers, args.batch_size)
    log.info("  Cache: %d already judged, %d new calls needed", total_cached, need_new)

    import threading
    cache_lock = threading.Lock()

    def judge_one(row: dict) -> dict:
        qt = row["query"]
        # Pool unique papers
        seen: set[str] = set()
        pool: list[dict] = []
        for m in methods:
            for r in row.get(m, []):
                oid = r.get("openalex_id") or r.get("id", "")
                if oid and oid not in seen:
                    seen.add(oid)
                    pool.append(r)

        scores = judge_query(qt, pool, batch_size=args.batch_size)

        with cache_lock:
            _save_cache(JUDGE_CACHE)

        return {"query": qt, "subject": row["subject"], "scores": scores,
                "pool_size": len(pool)}

    ndcg_scores: dict[str, list[float]] = {m: [] for m in methods}
    mrr_scores: dict[str, list[float]] = {m: [] for m in methods}
    p_at_k_scores: dict[str, list[float]] = {m: [] for m in methods}
    subject_ndcg: dict[str, dict[str, list[float]]] = {m: {} for m in methods}
    per_query_detail = []
    done = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(judge_one, row): row for row in query_data}
        for future in concurrent.futures.as_completed(futures):
            row_orig = futures[future]
            try:
                judged = future.result()
            except Exception as e:
                log.warning("Judge failed for '%s': %s", row_orig["query"][:40], e)
                continue

            qt = judged["query"]
            subj = judged["subject"]
            scores = judged["scores"]
            detail: dict = {"query": qt, "subject": subj, "pool_size": judged["pool_size"]}

            for m in methods:
                results = row_orig.get(m, [])[:args.top_n]
                if not results:
                    continue
                relevance = [float(scores.get(r.get("openalex_id", r.get("id", "")), 0))
                             for r in results]
                ndcg = ndcg_at_k(relevance, args.k)
                mrr = mrr_at_k(relevance, args.k)
                p_k = precision_at_k(relevance, args.k)
                ndcg_scores[m].append(ndcg)
                mrr_scores[m].append(mrr)
                p_at_k_scores[m].append(p_k)
                subject_ndcg[m].setdefault(subj, []).append(ndcg)
                detail[f"{m}_ndcg"] = round(ndcg, 4)
                detail[f"{m}_mrr"] = round(mrr, 4)
                detail[f"{m}_p_at_k"] = round(p_k, 4)

            per_query_detail.append(detail)
            done += 1
            if done % 10 == 0:
                log.info("  Judged %d/%d queries", done, len(eval_queries))

    _save_cache(JUDGE_CACHE)

    # ── Report ────────────────────────────────────────────────────────────────
    METHOD_LABELS = {
        "bm25":               "BM25F (local)",
        "splade":             "SPLADE (local)",
        "rrf":                "BM25F+SPLADE RRF (local)",
        "openalex_relevance": "OpenAlex relevance sort",
        "openalex_topic":     "OpenAlex cite-count sort",
    }

    print()
    print("=" * 95)
    print("  LLM-JUDGED IR BENCHMARK  —  TREC 0-3 relevance via Claude Haiku")
    print(f"  {len(eval_queries)} queries  |  NDCG@{args.k}  |  top_n={args.top_n}  |  workers={args.workers}")
    print("=" * 95)

    print(f"\n{'Method':<32} {'NDCG@10':>8} {'Median':>8} {'Std':>7} "
          f"{'MRR@10':>8} {'P@10(≥2)':>9} {'N':>5}")
    print("-" * 88)
    for m in methods:
        sc = ndcg_scores[m]
        if not sc:
            print(f"  {METHOD_LABELS.get(m, m):<30}  (no results)")
            continue
        lbl = METHOD_LABELS.get(m, m)
        print(f"  {lbl:<30}  {np.mean(sc):>8.4f} {np.median(sc):>8.4f} "
              f"{np.std(sc):>7.4f} {np.mean(mrr_scores[m]):>8.4f} "
              f"{np.mean(p_at_k_scores[m]):>9.4f} {len(sc):>5}")

    # Wiring decision summary
    rrf_sc = ndcg_scores.get("rrf", [])
    if rrf_sc:
        rrf_m = np.mean(rrf_sc)
        print(f"\n{'=' * 58}")
        print("  WIRING DECISION (unbiased, LLM-judged topical relevance)")
        print(f"{'=' * 58}")
        for cmp_m in ["openalex_relevance", "openalex_topic", "bm25", "splade"]:
            cmp_sc = ndcg_scores.get(cmp_m, [])
            if not cmp_sc:
                continue
            diff = rrf_m - np.mean(cmp_sc)
            verdict = "WIRE ✓" if diff > 0.01 else ("SKIP ✗" if diff < -0.01 else "NEUTRAL")
            lbl = METHOD_LABELS.get(cmp_m, cmp_m)
            print(f"  RRF vs {lbl:<30}: {diff:+.4f}  [{verdict}]")

        # Per-query win rate vs best live baseline
        best_live = "openalex_relevance" if "openalex_relevance" in methods else None
        if best_live and ndcg_scores[best_live]:
            win = sum(1 for a, b in zip(ndcg_scores["rrf"], ndcg_scores[best_live]) if a > b + 0.01)
            lose = sum(1 for a, b in zip(ndcg_scores["rrf"], ndcg_scores[best_live]) if b > a + 0.01)
            tie = len(ndcg_scores["rrf"]) - win - lose
            print(f"\n  RRF vs OpenAlex-relevance: wins={win}  ties={tie}  losses={lose}")

    # Subject breakdown
    all_subjects = sorted({s for m in methods for s in subject_ndcg.get(m, {})})
    show = methods[:4]
    if all_subjects:
        print(f"\n{'Subject':<46}" + "".join(f" {METHOD_LABELS.get(m,m)[:9]:>9}" for m in show))
        print("-" * (46 + 10 * len(show)))
        for subj in all_subjects:
            row_str = f"{subj[:45]:<46}"
            for m in show:
                sc = subject_ndcg.get(m, {}).get(subj, [])
                row_str += f" {np.mean(sc):>9.4f}" if sc else f" {'n/a':>9}"
            print(row_str)

    # Top/bottom queries vs best baseline
    if best_live and per_query_detail:
        deltas = [(d, d.get("rrf_ndcg", 0) - d.get(f"{best_live}_ndcg", 0))
                  for d in per_query_detail if "rrf_ndcg" in d]
        deltas.sort(key=lambda x: -x[1])
        print(f"\n  Top 5 queries where RRF beats OpenAlex-relevance:")
        for d, delta in deltas[:5]:
            print(f"    {delta:+.4f}  [{d['subject']}]  {d['query'][:55]}")
        print(f"\n  Top 5 queries where OpenAlex-relevance beats RRF:")
        for d, delta in deltas[-5:]:
            print(f"    {delta:+.4f}  [{d['subject']}]  {d['query'][:55]}")

    # Save
    output = args.output or (
        ROOT / "data" / "eval_results" / f"benchmark_llm_judge_{time.strftime('%Y%m%d_%H%M')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": "llm_judged_ndcg",
        "judge": "claude-haiku-4-5-20251001 via claude CLI subprocess",
        "relevance_scale": JUDGE_SCALE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": args.k,
        "top_n": args.top_n,
        "num_queries": len(eval_queries),
        "workers": args.workers,
        "summary": {
            m: {
                "mean_ndcg": round(float(np.mean(ndcg_scores[m])), 4) if ndcg_scores[m] else None,
                "median_ndcg": round(float(np.median(ndcg_scores[m])), 4) if ndcg_scores[m] else None,
                "std_ndcg": round(float(np.std(ndcg_scores[m])), 4) if ndcg_scores[m] else None,
                "mean_mrr": round(float(np.mean(mrr_scores[m])), 4) if mrr_scores[m] else None,
                "mean_p_at_k": round(float(np.mean(p_at_k_scores[m])), 4) if p_at_k_scores[m] else None,
                "label": METHOD_LABELS.get(m, m),
            }
            for m in methods
        },
        "subject_breakdown": {
            m: {s: round(float(np.mean(sc)), 4) for s, sc in subject_ndcg[m].items()}
            for m in methods
        },
        "per_query": per_query_detail,
    }
    output.write_text(json.dumps(result, indent=2))
    log.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
