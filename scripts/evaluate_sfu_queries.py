#!/usr/bin/env python3
"""Evaluate embedding models on SFU-specific academic queries.

Unlike the general benchmark (benchmark_embeddings.py), this script tests
models specifically on the institutional domains where SFU has unique coverage:
  - Indigenous Studies
  - Canadian Studies
  - Criminology (SFU-unique program)
  - Interactive Arts & Technology / SIAT (SFU-unique)
  - Biomedical Physiology and Kinesiology (SFU-unique)
  - Health Sciences, Finance, History, etc.

These are the queries where off-the-shelf models fail most severely and where
a fine-tuned SFU model should show the largest improvement.

Usage:
    python scripts/evaluate_sfu_queries.py
    python scripts/evaluate_sfu_queries.py --custom-model models/sfu-academic-embed-v1
    python scripts/evaluate_sfu_queries.py --queries data/sfu_eval_queries.json
    python scripts/evaluate_sfu_queries.py --output results/sfu_eval_results.json
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
HEADERS = {"User-Agent": "SFULibraryMCP-Eval/1.0 (mailto:lib-systems@sfu.ca)"}
DEFAULT_EVAL_QUERIES = str(Path(__file__).parent.parent / "data/sfu_eval_queries.json")


def _load_dotenv() -> None:
    """Load .env so OPENALEX_API_KEY is available without exporting manually.

    .env is treated as the canonical source — values here override any
    pre-existing process env. Without this, a stale key baked into the
    container session at startup would silently win over a freshly-rotated
    key in the file (and worse, the stale key would be sent on every API
    request and end up in error logs).
    """
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k:
            os.environ[k] = v


_load_dotenv()
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()
if OPENALEX_API_KEY:
    logger.info("OpenAlex API key loaded from environment (premium quota)")
else:
    logger.info("No OPENALEX_API_KEY set; using anonymous polite pool (10 req/s shared)")
QUERY_CACHE_FILE = Path(__file__).parent.parent / "data/openalex_eval_cache.json"


def _load_query_cache() -> dict:
    if QUERY_CACHE_FILE.exists():
        try:
            return json.loads(QUERY_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_query_cache(cache: dict) -> None:
    QUERY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUERY_CACHE_FILE.write_text(json.dumps(cache))


_QUERY_CACHE = _load_query_cache()


def _reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    positions = []
    for word, pos_list in inv_index.items():
        for p in pos_list:
            positions.append((p, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def fetch_openalex_results(query: str, k: int = 50, use_cache: bool = True) -> list[dict]:
    """Fetch top-k OpenAlex results for a query via BM25.

    Cached on disk by (query, k) so re-runs across multiple models share fetches
    and avoid rate-limiting. Includes 429 retry-with-backoff.
    """
    cache_key = f"{k}::{query}"
    if use_cache and cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]

    params = {
        "search": query,
        "per_page": k,
        "sort": "relevance_score:desc",
        "select": "id,title,publication_year,cited_by_count,abstract_inverted_index,type",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    else:
        params["mailto"] = "lib-systems@sfu.ca"  # polite-pool fallback

    backoff = 5.0
    for attempt in range(5):
        try:
            resp = requests.get(
                f"{OPENALEX_BASE}/works",
                params=params,
                headers=HEADERS,
                timeout=30,
            )
            if resp.status_code == 429:
                logger.info("  429 from OpenAlex, backing off %.1fs (attempt %d/4)", backoff, attempt + 1)
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            # Strip api_key from any error string so it never lands in logs.
            err_str = str(e)
            if OPENALEX_API_KEY:
                err_str = err_str.replace(OPENALEX_API_KEY, "<api-key-redacted>")
            if attempt == 4:
                logger.warning("OpenAlex fetch failed for '%s': %s", query, err_str)
                return []
            time.sleep(backoff)
            backoff *= 2
    else:
        return []

    results = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.pop("abstract_inverted_index", None))
        w["abstract"] = abstract
        if w.get("title"):
            results.append(w)

    if use_cache and results:
        _QUERY_CACHE[cache_key] = results
        _save_query_cache(_QUERY_CACHE)
    return results


def compute_relevance_proxy(papers: list[dict]) -> list[float]:
    """Citation-count relevance proxy (same method as benchmark_embeddings.py).

    NDCG uses log(citation_count + 1) as the relevance grade.
    This rewards papers that are both returned AND highly cited.
    """
    scores = []
    for p in papers:
        score = math.log1p(p.get("cited_by_count", 0))
        scores.append(score)
    return scores


def fetch_seed_paper_text(doi: str) -> str:
    """Fetch title + abstract for a seed paper by DOI from OpenAlex.

    Returns an empty string on failure so the caller can fall back to the
    citation-only proxy rather than crashing.  Results are cached in the same
    on-disk cache as query results to avoid redundant API calls.
    """
    cache_key = f"doi::{doi}"
    if cache_key in _QUERY_CACHE:
        return _QUERY_CACHE[cache_key]

    params = {
        "filter": f"doi:{doi}",
        "select": "title,abstract_inverted_index",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    else:
        params["mailto"] = "lib-systems@sfu.ca"

    try:
        resp = requests.get(
            f"{OPENALEX_BASE}/works",
            params=params,
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            logger.warning("Seed paper not found for DOI: %s", doi)
            return ""
        w = results[0]
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        text = f"{w.get('title', '')} {abstract}".strip()
    except Exception as e:
        err = str(e)
        if OPENALEX_API_KEY:
            err = err.replace(OPENALEX_API_KEY, "<api-key-redacted>")
        logger.warning("Failed to fetch seed paper %s: %s", doi, err)
        text = ""

    _QUERY_CACHE[cache_key] = text
    _save_query_cache(_QUERY_CACHE)
    return text


def compute_mixed_relevance_proxy(
    papers: list[dict],
    seed_embedding: np.ndarray,
    paper_embeddings: np.ndarray,
) -> list[float]:
    """Mixed relevance proxy: 0.5 * log(cite+1) + 0.5 * cosine(paper, seed).

    The cosine term anchors relevance to a human-chosen seed paper rather than
    trusting citation count alone.  Citation count correlates with OpenAlex BM25
    score, creating a circular evaluation; the seed-paper cosine breaks that
    correlation for queries where a canonical paper is known.

    Both terms contribute equally.  Cosine ranges [-1, 1]; log1p(cite) ranges
    [0, ~15].  Equal weights therefore still skew toward citation for
    well-cited results — this is intentional.  When all candidates have low
    cosine similarity to the seed (subject mismatch), the proxy degrades
    gracefully to near-citation-only.
    """
    cosines = (paper_embeddings @ seed_embedding).tolist()
    scores = []
    for p, cos in zip(papers, cosines):
        scores.append(0.5 * math.log1p(p.get("cited_by_count", 0)) + 0.5 * cos)
    return scores


def ndcg_at_k(ranked_relevance: list[float], ideal_relevance: list[float], k: int) -> float:
    """Compute NDCG@k given ranked and ideal relevance lists."""
    def dcg(rels: list[float], k: int) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))

    dcg_val = dcg(ranked_relevance, k)
    ideal_dcg = dcg(sorted(ideal_relevance, reverse=True), k)
    return dcg_val / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr_at_k(ranked_relevance: list[float], ideal_relevance: list[float], k: int) -> float:
    """Mean Reciprocal Rank @k.  A document is 'relevant' if its score exceeds
    the median of the ideal list — same threshold used for the relevance proxy."""
    if not ideal_relevance:
        return 0.0
    threshold = sorted(ideal_relevance, reverse=True)[min(k - 1, len(ideal_relevance) - 1)]
    for rank, rel in enumerate(ranked_relevance[:k], start=1):
        if rel >= threshold:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_relevance: list[float], ideal_relevance: list[float], k: int) -> float:
    """Recall@k: fraction of the top-k ideal docs that appear in the top-k ranked list.

    Uses the same relevance threshold as mrr_at_k.
    """
    if not ideal_relevance:
        return 0.0
    threshold = sorted(ideal_relevance, reverse=True)[min(k - 1, len(ideal_relevance) - 1)]
    ideal_count = sum(1 for r in ideal_relevance[:k] if r >= threshold)
    if ideal_count == 0:
        return 0.0
    ranked_count = sum(1 for r in ranked_relevance[:k] if r >= threshold)
    return ranked_count / ideal_count


def encode_texts(model, texts: list[str]) -> np.ndarray:
    """Encode texts using a sentence-transformer model.

    Force numpy output — sentence-transformers >=5.0 may return Tensor objects
    or wrapper types if convert_to_numpy is not explicitly set.
    """
    result = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.array(result, dtype=np.float32)


def rerank_with_embedding(query: str, papers: list[dict], model) -> list[dict]:
    """Rerank papers by cosine similarity to the query embedding."""
    texts = [query] + [
        f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers
    ]
    embeddings = encode_texts(model, texts)
    query_emb = embeddings[0]
    paper_embs = embeddings[1:]
    scores = (paper_embs @ query_emb).tolist()

    # Sort by index to avoid dict comparison when scores are tied
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [papers[i] for i in ranked]


def rerank_with_rrf(query: str, papers: list[dict], model, k: int = 60) -> list[dict]:
    """Rerank by Reciprocal Rank Fusion of OpenAlex BM25 rank + embedding rank.

    RRF score:  Σ_r  1 / (k + rank_r(d))    (k=60 is the literature default)

    The input `papers` arrives in OpenAlex BM25 order (rank 0..N-1). We compute
    the embedding rank separately, then fuse. Documents that score well on
    either ranker bubble up; documents both rankers agree on bubble up the most.
    Crucially, neither ranker gets to fully suppress a result the other one
    rates highly — which is why RRF rescues queries where one signal is blind.
    """
    bm25_rank = {id(p): i for i, p in enumerate(papers)}

    embed_ranked = rerank_with_embedding(query, papers, model)
    embed_rank = {id(p): i for i, p in enumerate(embed_ranked)}

    fused = sorted(
        range(len(papers)),
        key=lambda i: (
            1.0 / (k + bm25_rank[id(papers[i])])
            + 1.0 / (k + embed_rank[id(papers[i])])
        ),
        reverse=True,
    )
    return [papers[i] for i in fused]


def evaluate_model(
    model_name: str,
    model,
    eval_queries: list[dict],
    k: int = 10,
    results_per_query: int = 50,
    delay: float = 0.5,
    fusion: str = "none",
    use_mixed_proxy: bool = False,
) -> dict:
    """Evaluate a model on the SFU eval query set. Returns per-query and aggregate metrics.

    fusion: "none" → embedding cosine alone (current behaviour)
            "rrf"  → reciprocal rank fusion of BM25 + embedding ranks

    use_mixed_proxy: when True and the query has a ``seed_doi`` field, relevance
        grades use the mixed citation+cosine proxy (M.3).  Falls back to
        citation-only when seed_doi is absent or the seed paper can't be fetched.
    """
    all_ndcg: list[float] = []
    all_mrr: list[float] = []
    all_recall: list[float] = []
    subject_ndcg: dict[str, list[float]] = {}
    query_results: list[dict] = []

    for qi, q_item in enumerate(eval_queries):
        query = q_item["query"]
        subject = q_item.get("subject", "Unknown")
        seed_doi = q_item.get("seed_doi", "").strip() if use_mixed_proxy else ""

        logger.info("  [%d/%d] '%s'", qi + 1, len(eval_queries), query[:60])

        # Fetch OpenAlex results (BM25 ranking)
        cache_key = f"{results_per_query}::{query}"
        was_cached = cache_key in _QUERY_CACHE
        papers = fetch_openalex_results(query, k=results_per_query)
        if not papers:
            logger.warning("  No results for query: %s", query)
            continue

        # Compute relevance grades — mixed proxy when seed_doi present, else citation-only
        use_seed = bool(seed_doi and model is not None)
        if use_seed:
            seed_text = fetch_seed_paper_text(seed_doi)
            use_seed = bool(seed_text)

        if use_seed:
            paper_texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers]
            all_texts = [seed_text] + paper_texts
            embs = encode_texts(model, all_texts)
            seed_emb = embs[0]
            paper_embs = embs[1:]
            relevance_scores = compute_mixed_relevance_proxy(papers, seed_emb, paper_embs)
        else:
            relevance_scores = compute_relevance_proxy(papers)

        if model is not None:
            if fusion == "rrf":
                reranked = rerank_with_rrf(query, papers, model)
            else:
                reranked = rerank_with_embedding(query, papers, model)
            if use_seed:
                # Re-embed in the reranked order (papers are the same objects — just reordered)
                reranked_texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in reranked]
                reranked_embs = encode_texts(model, reranked_texts)
                reranked_relevance = compute_mixed_relevance_proxy(reranked, seed_emb, reranked_embs)
            else:
                reranked_relevance = compute_relevance_proxy(reranked)
        else:
            # BM25 only — use OpenAlex rank order as-is
            reranked_relevance = relevance_scores

        ndcg = ndcg_at_k(reranked_relevance, relevance_scores, k)
        mrr = mrr_at_k(reranked_relevance, relevance_scores, k)
        rec = recall_at_k(reranked_relevance, relevance_scores, k)
        all_ndcg.append(ndcg)
        all_mrr.append(mrr)
        all_recall.append(rec)

        if subject not in subject_ndcg:
            subject_ndcg[subject] = []
        subject_ndcg[subject].append(ndcg)

        query_results.append({
            "query": query,
            "subject": subject,
            "ndcg_at_k": ndcg,
            "mrr_at_k": mrr,
            "recall_at_k": rec,
            "num_results": len(papers),
            "notes": q_item.get("notes", ""),
        })

        # Only delay if we actually hit the API
        if not was_cached:
            time.sleep(delay)

    # Aggregate metrics
    mean_ndcg = float(np.mean(all_ndcg)) if all_ndcg else 0.0
    median_ndcg = float(np.median(all_ndcg)) if all_ndcg else 0.0
    std_ndcg = float(np.std(all_ndcg)) if all_ndcg else 0.0
    mean_mrr = float(np.mean(all_mrr)) if all_mrr else 0.0
    mean_recall = float(np.mean(all_recall)) if all_recall else 0.0

    # Subject-level breakdown
    subject_breakdown = {
        subj: {
            "mean_ndcg": float(np.mean(scores)),
            "queries": len(scores),
        }
        for subj, scores in subject_ndcg.items()
    }

    return {
        "model": model_name,
        "mean_ndcg_at_k": mean_ndcg,
        "median_ndcg_at_k": median_ndcg,
        "std_ndcg_at_k": std_ndcg,
        "mean_mrr_at_k": mean_mrr,
        "mean_recall_at_k": mean_recall,
        "num_queries": len(all_ndcg),
        "k": k,
        "subject_breakdown": subject_breakdown,
        "per_query": sorted(query_results, key=lambda x: x["ndcg_at_k"]),
    }


def print_results_table(all_results: list[dict], k: int) -> None:
    """Print a comparison table of all evaluated models."""
    print(f"\n{'='*85}")
    print(f"SFU-SPECIFIC EMBEDDING MODEL EVALUATION  (NDCG@{k} | MRR@{k} | Recall@{k})")
    print(f"{'='*85}")

    # Summary table
    print(f"\n{'Model':<35} {'NDCG':>8} {'Median':>8} {'Std':>6} {'MRR':>8} {'Recall':>8} {'N':>5}")
    print("-" * 85)
    for r in sorted(all_results, key=lambda x: x["mean_ndcg_at_k"], reverse=True):
        print(
            f"{r['model']:<35} {r['mean_ndcg_at_k']:>8.4f} {r['median_ndcg_at_k']:>8.4f} "
            f"{r['std_ndcg_at_k']:>6.4f} {r.get('mean_mrr_at_k', 0.0):>8.4f} "
            f"{r.get('mean_recall_at_k', 0.0):>8.4f} {r['num_queries']:>5}"
        )

    # Subject breakdown (if multiple models)
    if len(all_results) >= 2:
        all_subjects = sorted({
            subj
            for r in all_results
            for subj in r["subject_breakdown"]
        })

        print(f"\n{'Subject':<40}", end="")
        for r in all_results:
            name = r["model"][:14]
            print(f" {name:>14}", end="")
        print()
        print("-" * (40 + 15 * len(all_results)))

        for subj in all_subjects:
            subj_short = subj[:39]
            print(f"{subj_short:<40}", end="")
            for r in all_results:
                bd = r["subject_breakdown"].get(subj, {})
                mean = bd.get("mean_ndcg", 0.0)
                print(f" {mean:>14.4f}", end="")
            print()

    # Per-query detail for the best model
    best = max(all_results, key=lambda x: x["mean_ndcg_at_k"])
    print(f"\nPer-query results for best model ({best['model']}):")
    print(f"{'Query':<55} {'NDCG':>6} {'Subject'}")
    print("-" * 80)
    for q in sorted(best["per_query"], key=lambda x: x["ndcg_at_k"], reverse=True):
        q_short = q["query"][:54]
        subj_short = q["subject"][:25]
        flag = " ← SFU-specific" if q.get("ndcg_at_k", 0) < 0.05 else ""
        print(f"{q_short:<55} {q['ndcg_at_k']:>6.4f} {subj_short}{flag}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on SFU-specific academic queries"
    )
    parser.add_argument("--queries", default=DEFAULT_EVAL_QUERIES,
                        help="Path to SFU eval queries JSON file")
    parser.add_argument("--custom-model", type=str, default=None,
                        help="Path to fine-tuned model to evaluate")
    parser.add_argument("--baseline-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Off-the-shelf baseline model to compare against")
    parser.add_argument("--bm25-only", action="store_true",
                        help="Evaluate BM25-only baseline (no embedding)")
    parser.add_argument("--k", type=int, default=10, help="NDCG@k cutoff")
    parser.add_argument("--results-per-query", type=int, default=50,
                        help="OpenAlex results to fetch per query")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full results to JSON file")
    parser.add_argument("--history", type=str,
                        default=str(Path(__file__).parent.parent / "results/sfu_eval_history.jsonl"),
                        help="Append-only history JSONL (M.4); one line per run")
    parser.add_argument("--delay", type=float, default=1.2,
                        help="Delay between OpenAlex API requests (seconds)")
    parser.add_argument("--fusion", choices=["none", "rrf", "both"], default="none",
                        help="Reranking strategy: 'none' = embedding only, "
                             "'rrf' = reciprocal rank fusion of BM25 + embedding, "
                             "'both' = run each custom model twice (with and without RRF)")
    parser.add_argument("--mixed-proxy", action="store_true",
                        help="Use mixed citation+cosine relevance proxy for queries that have "
                             "a seed_doi field (Phase M.3). Falls back to citation-only when "
                             "seed_doi is absent.")
    args = parser.parse_args()

    # Load eval queries
    queries_path = Path(args.queries)
    if not queries_path.exists():
        logger.error("Eval queries file not found: %s", queries_path)
        raise SystemExit(1)

    with queries_path.open() as f:
        eval_queries = json.load(f)
    logger.info("Loaded %d SFU eval queries", len(eval_queries))

    all_results = []

    # BM25-only baseline
    if args.bm25_only:
        logger.info("\nEvaluating: BM25-only (no embedding reranking)")
        result = evaluate_model(
            "BM25-only (OpenAlex)",
            model=None,
            eval_queries=eval_queries,
            k=args.k,
            results_per_query=args.results_per_query,
            delay=args.delay,
        )
        all_results.append(result)

    # Load models
    from sentence_transformers import SentenceTransformer

    models_to_eval: list[tuple[str, object]] = []

    # Baseline embedding model
    if args.baseline_model:
        logger.info("Loading baseline model: %s", args.baseline_model)
        baseline = SentenceTransformer(args.baseline_model)
        models_to_eval.append((Path(args.baseline_model).name or args.baseline_model, baseline))

    # Custom fine-tuned model
    if args.custom_model:
        custom_path = Path(args.custom_model)
        if not custom_path.exists():
            logger.error("Custom model not found: %s", custom_path)
        else:
            logger.info("Loading custom model: %s", custom_path)
            custom = SentenceTransformer(str(custom_path))
            models_to_eval.append((f"sfu-custom ({custom_path.name})", custom))

    fusion_modes = ["none", "rrf"] if args.fusion == "both" else [args.fusion]

    mixed_proxy = getattr(args, "mixed_proxy", False)
    seed_doi_count = sum(1 for q in eval_queries if q.get("seed_doi"))
    if mixed_proxy:
        logger.info("Mixed proxy enabled: %d/%d queries have seed_doi", seed_doi_count, len(eval_queries))

    for model_name, model in models_to_eval:
        for fusion in fusion_modes:
            label = model_name if fusion == "none" else f"{model_name} + RRF"
            logger.info("\nEvaluating: %s", label)
            result = evaluate_model(
                model_name=label,
                model=model,
                eval_queries=eval_queries,
                k=args.k,
                results_per_query=args.results_per_query,
                delay=args.delay,
                fusion=fusion,
                use_mixed_proxy=mixed_proxy,
            )
            all_results.append(result)

    if not all_results:
        logger.error("No models evaluated. Use --bm25-only or provide a --baseline-model.")
        raise SystemExit(1)

    # Print results
    print_results_table(all_results, args.k)

    # Save to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2))
        logger.info("Full results saved to %s", output_path)

    # Append to history JSONL (M.4) — one compact line per model per run
    import datetime
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as hf:
        for r in all_results:
            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "model": r["model"],
                "fusion": r.get("fusion", "none"),
                "ndcg10": r["mean_ndcg_at_k"],
                "mrr10": r.get("mean_mrr_at_k", 0.0),
                "recall10": r.get("mean_recall_at_k", 0.0),
                "num_queries": r["num_queries"],
                "k": r["k"],
                "per_subject": {s: v["mean_ndcg"] for s, v in r["subject_breakdown"].items()},
            }
            hf.write(json.dumps(entry) + "\n")
    logger.info("History appended to %s", history_path)

    # Print improvement summary if comparing custom vs baseline
    custom_results = [r for r in all_results if r["model"].startswith("sfu-custom")]
    baseline_results = [r for r in all_results if not r["model"].startswith("sfu-custom")
                        and r["model"] != "BM25-only (OpenAlex)"]
    if custom_results and baseline_results:
        custom_mean = custom_results[0]["mean_ndcg_at_k"]
        baseline_mean = baseline_results[0]["mean_ndcg_at_k"]
        if baseline_mean > 0:
            improvement_pct = 100 * (custom_mean - baseline_mean) / baseline_mean
            print(f"\nImprovement: {improvement_pct:+.1f}% over {baseline_results[0]['model']}")

        # Check key SFU-specific subjects
        print("\nKey SFU subject improvements:")
        for subj in ["Indigenous Studies", "Canadian Studies", "Criminology",
                     "Interactive Arts and Technology (SIAT)"]:
            cust_bd = custom_results[0]["subject_breakdown"].get(subj, {})
            base_bd = baseline_results[0]["subject_breakdown"].get(subj, {})
            if cust_bd and base_bd:
                c = cust_bd.get("mean_ndcg", 0)
                b = base_bd.get("mean_ndcg", 0)
                mult = c / b if b > 0 else float("inf")
                print(f"  {subj:<45} baseline={b:.4f}  custom={c:.4f}  ({mult:.1f}x)")


if __name__ == "__main__":
    main()
