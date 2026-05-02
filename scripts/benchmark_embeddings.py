#!/usr/bin/env python3
"""Phase 1 Benchmark: Compare embedding models for academic search reranking.

Evaluates off-the-shelf vs fine-tuned vs SPECTER2 on real academic queries
using OpenAlex results. Outputs NDCG@10 comparison table.

Usage:
    python scripts/benchmark_embeddings.py [--queries N] [--results-per-query N]
    python scripts/benchmark_embeddings.py --specter2-local   # also test SPECTER2 locally
    python scripts/benchmark_embeddings.py --custom-model path/to/model  # test fine-tuned model
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"

BENCHMARK_QUERIES = [
    "CRISPR gene editing salmon aquaculture",
    "machine learning protein folding prediction",
    "climate change impact British Columbia forests",
    "indigenous language revitalization Canada",
    "quantum computing error correction",
    "antibiotic resistance hospital acquired infections",
    "deep learning natural language processing transformers",
    "ocean acidification coral reef ecosystems",
    "mental health university students pandemic",
    "renewable energy grid storage batteries",
    "urban planning affordable housing Vancouver",
    "microplastics marine food chain",
    "autonomous vehicle computer vision safety",
    "epigenetics gene expression aging",
    "cybersecurity zero trust architecture",
    "wildfire smoke air quality health effects",
    "supply chain disruption resilience strategies",
    "social media misinformation democratic elections",
    "CRISPR diagnostics point of care testing",
    "carbon capture direct air technology",
    "neural network interpretability explainable AI",
    "biodiversity loss pollinator decline agriculture",
    "remote sensing deforestation satellite monitoring",
    "vaccine hesitancy public health communication",
    "blockchain decentralized finance regulation",
    "glacier retreat water resources Himalayas",
    "gut microbiome mental health connection",
    "federated learning privacy preserving machine learning",
    "sustainable fisheries management Pacific Northwest",
    "augmented reality education learning outcomes",
    "drug resistant tuberculosis treatment strategies",
    "dark matter detection particle physics",
    "precision agriculture drone remote sensing",
    "refugee integration social inclusion policy",
    "single cell RNA sequencing cancer heterogeneity",
    "smart city infrastructure internet of things",
    "post traumatic stress disorder novel therapies",
    "circular economy waste reduction manufacturing",
    "gravitational wave detection neutron stars",
    "childhood obesity prevention school programs",
    "natural language generation text summarization",
    "soil carbon sequestration regenerative agriculture",
    "wearable biosensors continuous health monitoring",
    "housing affordability crisis Canadian cities",
    "mRNA vaccine technology platform development",
    "deep sea mining environmental impact assessment",
    "attention mechanism transformer architecture efficiency",
    "forest fire prediction early warning systems",
    "antibiotic stewardship programs effectiveness",
    "quantum machine learning hybrid algorithms",
]


def fetch_openalex_results(query: str, per_page: int = 50) -> list[dict]:
    """Fetch search results from OpenAlex API."""
    try:
        resp = requests.get(
            f"{OPENALEX_BASE}/works",
            params={
                "search": query,
                "per_page": per_page,
                "select": "id,doi,title,publication_year,cited_by_count,type,authorships,abstract_inverted_index",
            },
            headers={"User-Agent": "SFULibraryMCP-Benchmark/1.0 (mailto:lib-systems@sfu.ca)"},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        for r in results:
            inv = r.pop("abstract_inverted_index", None)
            if inv:
                r["abstract"] = _reconstruct_abstract(inv)
            else:
                r["abstract"] = ""
        return results
    except Exception as e:
        logger.warning("OpenAlex fetch failed for '%s': %s", query, e)
        return []


def _reconstruct_abstract(inv_index: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inv_index:
        return ""
    word_positions = []
    for word, positions in inv_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def compute_bm25_proxy_scores(query: str, papers: list[dict]) -> list[float]:
    """Use position-based scoring as BM25 proxy (OpenAlex already BM25-ranks)."""
    n = len(papers)
    if n == 0:
        return []
    return [1.0 - (i / n) for i in range(n)]


_model_cache: dict = {}


def _get_or_load_model(model_name: str):
    """Load and cache a sentence-transformer model, handling SPECTER2 specially."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    from sentence_transformers import SentenceTransformer

    if model_name == "allenai/specter2":
        # SPECTER2 adapter has peft version issues — use base model directly
        # specter2_base is still SciBERT-based and academic-domain trained
        model = SentenceTransformer("allenai/specter2_base")
    else:
        model = SentenceTransformer(model_name)

    _model_cache[model_name] = model
    return model


def compute_embedding_scores(query: str, papers: list[dict], model_name: str) -> list[float]:
    """Compute semantic similarity scores using a sentence-transformer model."""
    try:
        model = _get_or_load_model(model_name)
        texts = [query] + [
            f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers
        ]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        query_emb = embeddings[0]
        doc_embs = embeddings[1:]
        return (doc_embs @ query_emb).tolist()
    except Exception as e:
        logger.error("Embedding scoring failed (%s): %s", model_name, e)
        return []


def compute_citation_relevance(papers: list[dict]) -> list[float]:
    """Use citation count as a proxy for relevance (self-supervised ground truth).

    Higher cited papers for a given query are more likely to be relevant.
    This serves as a cheap relevance proxy when human judgments aren't available.
    """
    counts = [p.get("cited_by_count", 0) for p in papers]
    max_count = max(counts) if counts else 1
    if max_count == 0:
        return [0.0] * len(papers)
    return [c / max_count for c in counts]


def dcg_at_k(scores: list[float], k: int) -> float:
    """Discounted cumulative gain at position k."""
    dcg = 0.0
    for i, s in enumerate(scores[:k]):
        dcg += s / math.log2(i + 2)
    return dcg


def ndcg_at_k(predicted_ranking: list[int], relevance: list[float], k: int) -> float:
    """NDCG@k given a predicted ranking (list of indices) and relevance scores."""
    predicted_scores = [relevance[i] for i in predicted_ranking[:k]]
    ideal_scores = sorted(relevance, reverse=True)[:k]

    dcg = dcg_at_k(predicted_scores, k)
    idcg = dcg_at_k(ideal_scores, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def rank_by_scores(scores: list[float]) -> list[int]:
    """Return indices sorted by descending score."""
    return [i for i, _ in sorted(enumerate(scores), key=lambda x: -x[1])]


def run_benchmark(
    num_queries: int = 50,
    results_per_query: int = 50,
    test_specter2_local: bool = False,
    custom_model: str | None = None,
    k: int = 10,
):
    """Run the full benchmark comparison."""
    queries = BENCHMARK_QUERIES[:num_queries]
    logger.info("Running benchmark with %d queries, %d results each", len(queries), results_per_query)

    models_to_test = {
        "bm25_only": None,
        "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    }

    if test_specter2_local:
        models_to_test["SPECTER2-base"] = "allenai/specter2"
        models_to_test["BGE-base-v1.5"] = "BAAI/bge-base-en-v1.5"

    if custom_model:
        models_to_test["custom-finetuned"] = custom_model

    results = {name: [] for name in models_to_test}
    query_results = []

    for qi, query in enumerate(queries):
        logger.info("[%d/%d] Query: %s", qi + 1, len(queries), query)

        papers = fetch_openalex_results(query, results_per_query)
        if len(papers) < 5:
            logger.warning("  Skipping — too few results (%d)", len(papers))
            continue

        relevance = compute_citation_relevance(papers)
        query_data = {"query": query, "num_papers": len(papers), "scores": {}}

        for name, model_name in models_to_test.items():
            if model_name is None:
                scores = compute_bm25_proxy_scores(query, papers)
            else:
                scores = compute_embedding_scores(query, papers, model_name)

            if not scores:
                logger.warning("  %s: no scores produced", name)
                continue

            ranking = rank_by_scores(scores)
            ndcg = ndcg_at_k(ranking, relevance, k)
            results[name].append(ndcg)
            query_data["scores"][name] = round(ndcg, 4)

        query_results.append(query_data)

        # Rate limit (OpenAlex polite pool)
        time.sleep(0.2)

    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS — NDCG@{k} (citation-based relevance proxy)")
    print("=" * 70)
    print(f"{'Model':<25} {'Mean NDCG@{}'.format(k):<15} {'Std Dev':<12} {'Queries':<10}")
    print("-" * 70)

    for name in models_to_test:
        scores = results[name]
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            print(f"{name:<25} {mean:<15.4f} {std:<12.4f} {len(scores):<10}")
        else:
            print(f"{name:<25} {'N/A':<15} {'N/A':<12} {0:<10}")

    print("=" * 70)

    # Delta analysis
    baseline_key = "bm25_only"
    baseline_scores = results.get(baseline_key, [])
    if baseline_scores:
        baseline_mean = np.mean(baseline_scores)
        print(f"\nDeltas vs {baseline_key} (mean NDCG@{k} = {baseline_mean:.4f}):")
        for name in models_to_test:
            if name == baseline_key:
                continue
            scores = results[name]
            if scores and len(scores) == len(baseline_scores):
                delta = np.mean(scores) - baseline_mean
                pct = (delta / baseline_mean) * 100 if baseline_mean > 0 else 0
                print(f"  {name}: {delta:+.4f} ({pct:+.1f}%)")

    # Decision recommendation
    minilm_scores = results.get("MiniLM-L6-v2", [])
    specter_scores = results.get("SPECTER2-base", [])
    if minilm_scores:
        minilm_mean = np.mean(minilm_scores)
        if specter_scores:
            specter_mean = np.mean(specter_scores)
            gap = specter_mean - minilm_mean
            gap_pct = abs(gap / specter_mean) * 100 if specter_mean > 0 else 0
            print(f"\nMiniLM vs SPECTER2 gap: {gap:+.4f} ({gap_pct:.1f}%)")
            if gap_pct < 3:
                print("RECOMMENDATION: Gap < 3% — use MiniLM off-the-shelf. No fine-tuning needed.")
            elif gap_pct < 5:
                print("RECOMMENDATION: Gap 3-5% — MiniLM acceptable. Fine-tuning optional.")
            else:
                print("RECOMMENDATION: Gap > 5% — fine-tuning recommended to close the gap.")
        else:
            print(f"\nMiniLM mean NDCG@{k}: {minilm_mean:.4f}")
            print("Run with --specter2-local to compare against SPECTER2.")

    # Save detailed results
    output_path = Path("benchmark_results.json")
    output_path.write_text(json.dumps(query_results, indent=2))
    print(f"\nDetailed per-query results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models for academic search")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries to test")
    parser.add_argument("--results-per-query", type=int, default=50, help="Results per query from OpenAlex")
    parser.add_argument("--specter2-local", action="store_true", help="Also test SPECTER2 locally")
    parser.add_argument("--custom-model", type=str, help="Path to custom fine-tuned model")
    parser.add_argument("--k", type=int, default=10, help="NDCG cutoff (default: 10)")
    args = parser.parse_args()

    run_benchmark(
        num_queries=args.queries,
        results_per_query=args.results_per_query,
        test_specter2_local=args.specter2_local,
        custom_model=args.custom_model,
        k=args.k,
    )


if __name__ == "__main__":
    main()
