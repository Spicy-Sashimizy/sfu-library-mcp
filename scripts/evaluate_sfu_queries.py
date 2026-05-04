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
import time
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
HEADERS = {"User-Agent": "SFULibraryMCP-Eval/1.0 (mailto:lib-systems@sfu.ca)"}
DEFAULT_EVAL_QUERIES = str(Path(__file__).parent.parent / "data/sfu_eval_queries.json")


def _reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    positions = []
    for word, pos_list in inv_index.items():
        for p in pos_list:
            positions.append((p, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def fetch_openalex_results(query: str, k: int = 50) -> list[dict]:
    """Fetch top-k OpenAlex results for a query via BM25."""
    try:
        resp = requests.get(
            f"{OPENALEX_BASE}/works",
            params={
                "search": query,
                "per_page": k,
                "sort": "relevance_score:desc",
                "select": "id,title,publication_year,cited_by_count,abstract_inverted_index,type",
            },
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("OpenAlex fetch failed for '%s': %s", query, e)
        return []

    results = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.pop("abstract_inverted_index", None))
        w["abstract"] = abstract
        if w.get("title"):
            results.append(w)
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


def ndcg_at_k(ranked_relevance: list[float], ideal_relevance: list[float], k: int) -> float:
    """Compute NDCG@k given ranked and ideal relevance lists."""
    def dcg(rels: list[float], k: int) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))

    dcg_val = dcg(ranked_relevance, k)
    ideal_dcg = dcg(sorted(ideal_relevance, reverse=True), k)
    return dcg_val / ideal_dcg if ideal_dcg > 0 else 0.0


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


def evaluate_model(
    model_name: str,
    model,
    eval_queries: list[dict],
    k: int = 10,
    results_per_query: int = 50,
    delay: float = 0.5,
) -> dict:
    """Evaluate a model on the SFU eval query set. Returns per-query and aggregate metrics."""
    all_ndcg: list[float] = []
    subject_ndcg: dict[str, list[float]] = {}
    query_results: list[dict] = []

    for qi, q_item in enumerate(eval_queries):
        query = q_item["query"]
        subject = q_item.get("subject", "Unknown")

        logger.info("  [%d/%d] '%s'", qi + 1, len(eval_queries), query[:60])

        # Fetch OpenAlex results (BM25 ranking)
        papers = fetch_openalex_results(query, k=results_per_query)
        if not papers:
            logger.warning("  No results for query: %s", query)
            continue

        # Get relevance scores at BM25 rank (ideal = by citation count)
        relevance_scores = compute_relevance_proxy(papers)

        if model is not None:
            # Rerank with embedding model
            reranked = rerank_with_embedding(query, papers, model)
            reranked_relevance = compute_relevance_proxy(reranked)
        else:
            # BM25 only — use OpenAlex rank order as-is
            reranked_relevance = relevance_scores

        ndcg = ndcg_at_k(reranked_relevance, relevance_scores, k)
        all_ndcg.append(ndcg)

        if subject not in subject_ndcg:
            subject_ndcg[subject] = []
        subject_ndcg[subject].append(ndcg)

        query_results.append({
            "query": query,
            "subject": subject,
            "ndcg_at_k": ndcg,
            "num_results": len(papers),
            "notes": q_item.get("notes", ""),
        })

        time.sleep(delay)

    # Aggregate metrics
    mean_ndcg = float(np.mean(all_ndcg)) if all_ndcg else 0.0
    median_ndcg = float(np.median(all_ndcg)) if all_ndcg else 0.0
    std_ndcg = float(np.std(all_ndcg)) if all_ndcg else 0.0

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
        "num_queries": len(all_ndcg),
        "k": k,
        "subject_breakdown": subject_breakdown,
        "per_query": sorted(query_results, key=lambda x: x["ndcg_at_k"]),
    }


def print_results_table(all_results: list[dict], k: int) -> None:
    """Print a comparison table of all evaluated models."""
    print(f"\n{'='*70}")
    print(f"SFU-SPECIFIC EMBEDDING MODEL EVALUATION  (NDCG@{k})")
    print(f"{'='*70}")

    # Summary table
    print(f"\n{'Model':<35} {'Mean NDCG':>10} {'Median':>8} {'Std':>6} {'Queries':>8}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: x["mean_ndcg_at_k"], reverse=True):
        print(f"{r['model']:<35} {r['mean_ndcg_at_k']:>10.4f} {r['median_ndcg_at_k']:>8.4f} "
              f"{r['std_ndcg_at_k']:>6.4f} {r['num_queries']:>8}")

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
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between OpenAlex API requests (seconds)")
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

    for model_name, model in models_to_eval:
        logger.info("\nEvaluating: %s", model_name)
        result = evaluate_model(
            model_name=model_name,
            model=model,
            eval_queries=eval_queries,
            k=args.k,
            results_per_query=args.results_per_query,
            delay=args.delay,
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
