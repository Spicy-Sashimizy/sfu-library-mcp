#!/usr/bin/env python3
"""Dense bi-encoder embedding evaluation — NDCG@10 comparison.

Evaluates one or more sentence-transformer bi-encoders against the LLM-judge
cache used as graded ground truth, on an apples-to-apples set of (query, doc)
candidates. This is the bi-encoder analogue of scripts/eval_cross_encoder.py:
same judge-cache harness, same OpenSearch _mget doc-text fetch, same NDCG@10 /
MRR@10 computation and qualifying-query filter. The ONLY difference is scoring:
the query and each candidate doc text are embedded SEPARATELY and ranked by
cosine similarity (embeddings are L2-normalized, so cosine == dot product),
exactly as src/lib/embedding.py does for the production semantic_similarity
rerank signal.

Method
──────
1. Group the LLM-judge cache (`{"<query>||<doc_id>": grade}`, grades 0-3) by
   query. Each query yields a candidate set of judged (doc_id, grade) pairs.
2. Keep only queries with >=2 judged candidates AND >=1 relevant (grade>=2).
3. Fetch each candidate doc's title+abstract from OpenSearch via batched `_mget`
   on `openalex_works` (docs keyed by OpenAlex id). Build doc text as
   "{title}. {abstract}". Candidate ids missing from the index are dropped, and
   the SAME fetched doc-text set is reused across all models.
4. For each model: embed the query and every candidate doc text with
   SentenceTransformer.encode(normalize_embeddings=True) (batched, on CUDA when
   available), score each candidate by cosine similarity to the query, rank each
   query's candidates by score, compute NDCG@10 (gain = 2^grade - 1, ideal-DCG
   from the grade-sorted order) and MRR@10 (first grade>=2). Average over
   qualifying queries.

After dropping missing docs a query may fall below the >=2-candidate / >=1-relevant
threshold; such queries are excluded from scoring and reported.

OpenSearch URL comes from $SFU_OPENSEARCH_URL.

Usage
─────
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    SFU_OPENSEARCH_URL=http://...:9200 \
    python scripts/eval_embedder.py \
        --judge-cache data/eval_results/llm_judge_cache.json \
        --models v4=models/sfu-academic-embed-v4-bge \
                 v5=models/sfu-academic-embed-v5 \
        --output data/eval_results/embedder_eval.json
"""
import argparse
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
INDEX = "openalex_works"
DEFAULT_JUDGE_CACHE = REPO_ROOT / "data/eval_results/llm_judge_cache.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/eval_results/embedder_eval.json"
RELEVANT_THRESHOLD = 2  # judge grade >= this == relevant
MIN_CANDIDATES = 2  # query must have >=2 judged candidates
K = 10  # NDCG@K / MRR@K cutoff
MGET_BATCH = 200
ENCODE_BATCH = 256

DEFAULT_MODELS = [
    "v4=models/sfu-academic-embed-v4-bge",
    "v5=models/sfu-academic-embed-v5",
]


def opensearch_url() -> str:
    return os.environ.get(
        "SFU_OPENSEARCH_URL",
        "http://claudebox-sfu-library-mcp-training-opensearch:9200",
    ).rstrip("/")


def doc_text(title: str, abstract: str) -> str:
    """Build doc text as "{title}. {abstract}" — mirrors build_ce_triplets.py."""
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if title and abstract:
        sep = " " if title.endswith((".", "?", "!")) else ". "
        return f"{title}{sep}{abstract}"
    return title or abstract


def parse_models(specs: list[str]) -> list[tuple[str, str]]:
    """Parse 'label=path' specs into ordered (label, path) pairs."""
    out: list[tuple[str, str]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--models entry must be label=path, got: {spec!r}")
        label, path = spec.split("=", 1)
        out.append((label.strip(), path.strip()))
    return out


def load_judge_grades(path: Path) -> dict[str, dict[str, int]]:
    """Return grades_by_query[query][doc_id] = grade (int)."""
    cache = json.loads(path.read_text())
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    for key, grade in cache.items():
        query, doc_id = key.rsplit("||", 1)
        if isinstance(grade, int):
            grades[query][doc_id] = grade
    return grades


def qualifies(cands: dict[str, int]) -> bool:
    """A query qualifies if it has >=MIN_CANDIDATES judged docs and >=1 relevant."""
    if len(cands) < MIN_CANDIDATES:
        return False
    return any(g >= RELEVANT_THRESHOLD for g in cands.values())


def mget_docs(
    session: requests.Session, url: str, doc_ids: list[str]
) -> tuple[dict[str, dict], list[str]]:
    """Fetch docs by _id via _mget. Returns (id -> _source, missing_ids)."""
    fetched: dict[str, dict] = {}
    missing: list[str] = []
    for i in range(0, len(doc_ids), MGET_BATCH):
        batch = doc_ids[i : i + MGET_BATCH]
        resp = session.post(
            f"{url}/{INDEX}/_mget",
            json={"ids": batch},
            params={"_source_includes": "title,abstract,openalex_id,doi,publication_year"},
            timeout=60,
        )
        resp.raise_for_status()
        for doc in resp.json().get("docs", []):
            if doc.get("found"):
                fetched[doc["_id"]] = doc.get("_source", {})
            else:
                missing.append(doc["_id"])
    return fetched, missing


def dcg(gains: list[int], k: int) -> float:
    """Standard DCG with gain = 2^grade - 1, log2(rank+1) discount."""
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_grades: list[int], k: int) -> float:
    """NDCG@k. ranked_grades = grades in model-ranked order; IDCG from sorted."""
    idcg = dcg(sorted(ranked_grades, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return dcg(ranked_grades, k) / idcg


def mrr_at_k(ranked_grades: list[int], k: int) -> float:
    """MRR@k — reciprocal rank of the first relevant (grade>=RELEVANT_THRESHOLD)."""
    for i, g in enumerate(ranked_grades[:k]):
        if g >= RELEVANT_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


def build_eval_set(
    grades_by_query: dict[str, dict[str, int]],
    pos_docs: dict[str, dict],
) -> tuple[dict[str, list[tuple[str, str, int]]], dict]:
    """Build per-query candidate lists of (doc_id, text, grade) using fetched docs.

    Re-applies the qualification check AFTER dropping docs missing from the index
    or with empty text. Returns (eval_set, stats).
    """
    eval_set: dict[str, list[tuple[str, str, int]]] = {}
    dropped_below_threshold = 0
    total_candidates = 0
    for query, cands in grades_by_query.items():
        if not qualifies(cands):
            continue  # was never in scope
        rows: list[tuple[str, str, int]] = []
        for doc_id, grade in cands.items():
            src = pos_docs.get(doc_id)
            if not src:
                continue  # missing from index
            text = doc_text(src.get("title", ""), src.get("abstract", ""))
            if not text:
                continue  # no usable text
            rows.append((doc_id, text, grade))
        # Re-check qualification on the surviving candidates.
        surviving_grades = {d: g for d, _, g in rows}
        if not qualifies(surviving_grades):
            dropped_below_threshold += 1
            continue
        eval_set[query] = rows
        total_candidates += len(rows)
    stats = {
        "queries_evaluated": len(eval_set),
        "queries_dropped_after_fetch": dropped_below_threshold,
        "total_candidates_scored": total_candidates,
    }
    return eval_set, stats


def evaluate_model(
    label: str,
    model_path: str,
    eval_set: dict[str, list[tuple[str, str, int]]],
) -> dict:
    """Score the shared eval set with one bi-encoder, return aggregate metrics.

    Embeds each query and all candidate doc texts separately (L2-normalized) and
    scores candidates by cosine similarity (== dot product on unit vectors), the
    same scoring src/lib/embedding.py uses for the production rerank signal.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("[%s] loading %s on %s", label, model_path, device)
    model = SentenceTransformer(model_path, device=device)

    # Unique doc texts across all queries -> embed once, reuse per query.
    queries = list(eval_set.keys())
    unique_texts: list[str] = []
    text_index: dict[str, int] = {}
    for rows in eval_set.values():
        for _doc_id, text, _grade in rows:
            if text not in text_index:
                text_index[text] = len(unique_texts)
                unique_texts.append(text)

    total_pairs = sum(len(rows) for rows in eval_set.values())
    logger.info("[%s] embedding %d queries + %d unique docs (%d (q,doc) pairs)...",
                label, len(queries), len(unique_texts), total_pairs)

    query_embs = model.encode(
        queries, batch_size=ENCODE_BATCH, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    doc_embs = model.encode(
        unique_texts, batch_size=ENCODE_BATCH, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    query_embs = np.asarray(query_embs)
    doc_embs = np.asarray(doc_embs)

    ndcgs: list[float] = []
    mrrs: list[float] = []
    for qi, query in enumerate(queries):
        rows = eval_set[query]
        q_emb = query_embs[qi]
        # Cosine similarity = dot product (both sides L2-normalized).
        qscores = [float(doc_embs[text_index[text]] @ q_emb) for _doc_id, text, _grade in rows]
        order = sorted(range(len(rows)), key=lambda i: qscores[i], reverse=True)
        ranked_grades = [rows[i][2] for i in order]
        ndcgs.append(ndcg_at_k(ranked_grades, K))
        mrrs.append(mrr_at_k(ranked_grades, K))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    n = len(ndcgs)
    return {
        "label": label,
        "model_path": model_path,
        "ndcg@10": sum(ndcgs) / n if n else 0.0,
        "mrr@10": sum(mrrs) / n if n else 0.0,
        "queries": n,
        "candidates_scored": total_pairs,
        "unique_docs_embedded": len(unique_texts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dense bi-encoder embedding NDCG@10 evaluation vs LLM-judge ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--judge-cache", default=str(DEFAULT_JUDGE_CACHE),
                        help="LLM judge cache JSON ({'<query>||<doc_id>': grade})")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Models as label=path (space-separated)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output results JSON")
    args = parser.parse_args()

    judge_path = Path(args.judge_cache)
    out_path = Path(args.output)
    models = parse_models(args.models)

    logger.info("Judge cache: %s", judge_path)
    grades_by_query = load_judge_grades(judge_path)
    qualifying = {q: c for q, c in grades_by_query.items() if qualifies(c)}
    logger.info("  queries in cache: %d", len(grades_by_query))
    logger.info("  queries qualifying (>=%d cands & >=1 grade>=%d): %d",
                MIN_CANDIDATES, RELEVANT_THRESHOLD, len(qualifying))

    # Collect all unique candidate doc ids across qualifying queries.
    all_ids = sorted({d for cands in qualifying.values() for d in cands})
    logger.info("Fetching %d unique candidate docs from OpenSearch (%s)...",
                len(all_ids), opensearch_url())
    session = requests.Session()
    pos_docs, missing = mget_docs(session, opensearch_url(), all_ids)
    logger.info("  fetched %d, missing (not in index) %d", len(pos_docs), len(missing))
    if missing:
        logger.warning("Missing ids sample: %s%s",
                       ", ".join(missing[:20]), " ..." if len(missing) > 20 else "")

    eval_set, build_stats = build_eval_set(qualifying, pos_docs)
    logger.info("Eval set: %d queries, %d candidates (dropped %d queries below "
                "threshold after fetch)",
                build_stats["queries_evaluated"],
                build_stats["total_candidates_scored"],
                build_stats["queries_dropped_after_fetch"])

    if not eval_set:
        raise SystemExit("No qualifying queries after fetch — nothing to evaluate.")

    results = [evaluate_model(label, path, eval_set) for label, path in models]

    # Determine winner + deltas vs base (the first 'v4' label, else first model).
    base_result = next((r for r in results if r["label"] == "v4"), results[0])
    base_ndcg = base_result["ndcg@10"]
    winner = max(results, key=lambda r: r["ndcg@10"])
    for r in results:
        r["ndcg@10_delta_vs_base"] = r["ndcg@10"] - base_ndcg

    out = {
        "config": {
            "judge_cache": str(judge_path),
            "index": INDEX,
            "opensearch_url": opensearch_url(),
            "relevant_threshold": RELEVANT_THRESHOLD,
            "min_candidates": MIN_CANDIDATES,
            "k": K,
            "gain": "2^grade - 1",
            "scoring": "bi-encoder cosine similarity (L2-normalized dot product)",
        },
        "dataset": {
            "queries_in_cache": len(grades_by_query),
            "queries_qualifying_pre_fetch": len(qualifying),
            "unique_candidate_ids": len(all_ids),
            "fetched": len(pos_docs),
            "missing_from_index": len(missing),
            **build_stats,
        },
        "base_label": base_result["label"],
        "winner": {
            "label": winner["label"],
            "ndcg@10": winner["ndcg@10"],
            "delta_vs_base": winner["ndcg@10"] - base_ndcg,
        },
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # ── Print table ──
    print("\n" + "=" * 82)
    print(f"DENSE BI-ENCODER EMBEDDING EVALUATION  (NDCG@{K} vs LLM-judge ground truth)")
    print("=" * 82)
    print(f"Queries evaluated: {build_stats['queries_evaluated']}   "
          f"Candidates: {build_stats['total_candidates_scored']}   "
          f"Missing from index: {len(missing)}   "
          f"Dropped after fetch: {build_stats['queries_dropped_after_fetch']}")
    print("-" * 82)
    print(f"{'model':<12} {'NDCG@10':>9} {'MRR@10':>9} {'#queries':>9} "
          f"{'#cands':>8} {'dNDCG/base':>11}")
    print("-" * 82)
    for r in results:
        marker = "  <-- WIN" if r["label"] == winner["label"] else ""
        delta = r["ndcg@10_delta_vs_base"]
        delta_str = f"{delta:+.4f}" if r["label"] != base_result["label"] else "  (base)"
        print(f"{r['label']:<12} {r['ndcg@10']:>9.4f} {r['mrr@10']:>9.4f} "
              f"{r['queries']:>9} {r['candidates_scored']:>8} {delta_str:>11}{marker}")
    print("-" * 82)
    print(f"WINNER: {winner['label']}  (NDCG@10={winner['ndcg@10']:.4f}, "
          f"delta vs base={winner['ndcg@10'] - base_ndcg:+.4f})")
    print("=" * 82)
    logger.info("Wrote results -> %s", out_path)


if __name__ == "__main__":
    main()
