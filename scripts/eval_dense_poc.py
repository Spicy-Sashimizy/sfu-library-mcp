#!/usr/bin/env python3
"""Dense-ANN retrieval POC evaluation: does a dense leg help, especially on
natural-language queries?

Compares two fused retrieval pipelines on the SFU eval queries + their
natural-language paraphrases, using the LLM-judge cache as graded ground truth:

    2-leg RRF = RRF(BM25F, SPLADE)          (both from full openalex_works)
    3-leg RRF = RRF(BM25F, SPLADE, dense)   (dense from the SUBSET openalex_works_dense)

Metrics (reusing eval_embedder.py / eval_cross_encoder.py machinery):
  - NDCG@10 with gain = 2^grade - 1 (judge grades 0-3; unjudged retrieved docs
    score grade 0).
  - MRR@10 (first retrieved doc with grade >= 2).
  - Recall@50 = (judged-relevant docs that appear in the fused top-50) /
    (all judged-relevant docs for the query). Relevant := grade >= 2.

Ground truth for a paraphrase is INHERITED from its original query via the
record's `judge_key` (= original_query[:80], the judge-cache key prefix). This is
the central fairness lever: keyword queries flatter lexical retrieval, so the
key signal is whether dense's lift is LARGER on the natural-language slice.

Results are broken down THREE ways: (a) keyword queries, (b) natural-language
paraphrases, (c) overall.

CAVEATS (also printed in the report):
  1. Dense indexes a SUBSET (judged docs + distractors) while BM25F/SPLADE search
     the full ~150M-doc index, so dense coverage is PARTIAL → its measured
     contribution is a LOWER BOUND.
  2. Paraphrase ground-truth reuse assumes a paraphrase preserves relevance.
  3. The judge cache reflects the ORIGINAL (keyword) query distribution, so even
     natural-language paraphrases are scored against keyword-seeded judgments.

Usage
─────
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    SFU_OPENSEARCH_URL=http://...:9200 \
    python scripts/eval_dense_poc.py \
        --judge-cache data/eval_results/llm_judge_cache.json \
        --diverse-queries data/eval_results/diverse_queries.json \
        --output data/eval_results/dense_poc_eval.json \
        [--top-k 50] [--limit N]
"""
import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_dense_poc")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_JUDGE_CACHE = REPO_ROOT / "data/eval_results/llm_judge_cache.json"
DEFAULT_DIVERSE = REPO_ROOT / "data/eval_results/diverse_queries.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/eval_results/dense_poc_eval.json"
DENSE_MODEL = str(REPO_ROOT / "models/sfu-academic-embed-v5")
SPLADE_ONNX = str(REPO_ROOT / "models/splade_onnx")

RELEVANT_THRESHOLD = 2
K_NDCG = 10
K_RECALL = 50
RRF_K = 60
JUDGE_KEY_LEN = 80


def opensearch_url() -> str:
    return os.environ.get(
        "SFU_OPENSEARCH_URL",
        "http://claudebox-sfu-library-mcp-training-opensearch:9200",
    ).rstrip("/")


# ── Metrics (identical to eval_embedder.py / eval_cross_encoder.py) ─────────────

def dcg(gains: list[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_grades: list[int], ideal_grades: list[int], k: int) -> float:
    """NDCG@k. ideal_grades = ALL judged grades for the query (the true ideal set).

    Using the full judged-grade pool for IDCG (not just the retrieved grades) is
    the standard convention when a retriever may miss relevant docs — it penalizes
    a pipeline that fails to surface known-relevant docs.
    """
    idcg = dcg(sorted(ideal_grades, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return dcg(ranked_grades, k) / idcg


def mrr_at_k(ranked_grades: list[int], k: int) -> float:
    for i, g in enumerate(ranked_grades[:k]):
        if g >= RELEVANT_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


# ── RRF (mirrors federated_search._rrf_merge, k=60), keyed by openalex_id ───────

def doc_id(doc: dict) -> str:
    """Stable id for fusion/grading. Dense hits carry openalex_id; lexical hits
    from openalex_works expose it via _source too. Fall back to doi/title."""
    return doc.get("openalex_id") or doc.get("doi") or doc.get("title") or ""


def rrf_fuse(ranked_lists: list[list[dict]], top_k: int) -> list[str]:
    """Reciprocal Rank Fusion of N ranked lists. Returns fused doc_ids (top_k)."""
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, doc in enumerate(lst, start=1):
            k = doc_id(doc)
            if k:
                scores[k] += 1.0 / (RRF_K + rank)
    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return ranked[:top_k]


# ── Ground truth ───────────────────────────────────────────────────────────────

def load_judge_grades(path: Path) -> dict[str, dict[str, int]]:
    """grades_by_key[query[:80]][doc_id] = grade. Keyed by the cache-key prefix."""
    cache = json.loads(path.read_text())
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    for key, grade in cache.items():
        qkey, did = key.rsplit("||", 1)
        if isinstance(grade, int):
            grades[qkey][did] = grade
    return grades


def grades_for(ids: list[str], gradebook: dict[str, int]) -> list[int]:
    """Grades for ranked ids; unjudged retrieved docs score grade 0."""
    return [gradebook.get(i, 0) for i in ids]


# ── Scoring one fused ranking ───────────────────────────────────────────────────

def score_ranking(fused_ids: list[str], gradebook: dict[str, int]) -> tuple[float, float, float]:
    """Return (ndcg@10, mrr@10, recall@50) for a fused id ranking."""
    ranked_grades = grades_for(fused_ids, gradebook)
    ideal = list(gradebook.values())
    ndcg = ndcg_at_k(ranked_grades, ideal, K_NDCG)
    mrr = mrr_at_k(ranked_grades, K_NDCG)
    relevant = {i for i, g in gradebook.items() if g >= RELEVANT_THRESHOLD}
    if relevant:
        hit = sum(1 for i in fused_ids[:K_RECALL] if i in relevant)
        recall = hit / len(relevant)
    else:
        recall = 0.0
    return ndcg, mrr, recall


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense-ANN retrieval POC eval (2-leg vs 3-leg RRF)")
    parser.add_argument("--judge-cache", default=str(DEFAULT_JUDGE_CACHE))
    parser.add_argument("--diverse-queries", default=str(DEFAULT_DIVERSE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=K_RECALL,
                        help="hits retrieved per leg before fusion")
    parser.add_argument("--dense-index", default="openalex_works_dense")
    parser.add_argument("--limit", type=int, default=0, help="limit #records (0=all)")
    args = parser.parse_args()

    from lib.opensearch_retriever import OpenSearchRetriever

    url = opensearch_url()
    retriever = OpenSearchRetriever(url=url, index="openalex_works",
                                    splade_model_path=SPLADE_ONNX, timeout=30)

    gradebooks = load_judge_grades(Path(args.judge_cache))
    records = json.loads(Path(args.diverse_queries).read_text())
    if args.limit:
        records = records[: args.limit]

    # Per-slice accumulators.
    slices = ("keyword", "natural", "overall")
    acc: dict[str, dict[str, list[float]]] = {
        s: {"ndcg2": [], "ndcg3": [], "mrr2": [], "mrr3": [],
            "rec2": [], "rec3": [], "dense_overlap": []}
        for s in slices
    }
    per_query: list[dict] = []
    skipped_no_gt = 0

    for i, rec in enumerate(records):
        qtext = rec["paraphrase"]
        qtype = rec["query_type"]
        key = rec.get("judge_key", rec["original_query"][:JUDGE_KEY_LEN])
        gradebook = gradebooks.get(key, {})
        # A query must have >=1 known-relevant doc to be scorable.
        if not any(g >= RELEVANT_THRESHOLD for g in gradebook.values()):
            skipped_no_gt += 1
            continue

        bm25 = retriever.search(qtext, top_k=args.top_k, mode="bm25f")
        splade = retriever.search(qtext, top_k=args.top_k, mode="splade")
        dense = retriever.dense_search(qtext, top_k=args.top_k,
                                       dense_index=args.dense_index,
                                       dense_model_path=DENSE_MODEL)

        fused2 = rrf_fuse([bm25, splade], args.top_k)
        fused3 = rrf_fuse([bm25, splade, dense], args.top_k)

        n2, m2, r2 = score_ranking(fused2, gradebook)
        n3, m3, r3 = score_ranking(fused3, gradebook)

        # How many of the dense hits are judged-relevant (diagnostic)?
        relevant = {d for d, g in gradebook.items() if g >= RELEVANT_THRESHOLD}
        dense_ids = [doc_id(d) for d in dense]
        dense_rel_hits = sum(1 for d in dense_ids if d in relevant)

        for s in (qtype, "overall"):
            acc[s]["ndcg2"].append(n2)
            acc[s]["ndcg3"].append(n3)
            acc[s]["mrr2"].append(m2)
            acc[s]["mrr3"].append(m3)
            acc[s]["rec2"].append(r2)
            acc[s]["rec3"].append(r3)
            acc[s]["dense_overlap"].append(dense_rel_hits / max(1, len(relevant)))

        per_query.append({
            "paraphrase": qtext, "query_type": qtype, "judge_key": key,
            "ndcg2": round(n2, 4), "ndcg3": round(n3, 4),
            "recall2": round(r2, 4), "recall3": round(r3, 4),
            "dense_rel_hits": dense_rel_hits, "n_relevant": len(relevant),
        })
        if (i + 1) % 25 == 0:
            logger.info("  scored %d/%d records", i + 1, len(records))

    def summarize(s: str) -> dict:
        a = acc[s]
        n = len(a["ndcg2"])
        return {
            "n_queries": n,
            "ndcg@10_2leg": round(mean(a["ndcg2"]), 4),
            "ndcg@10_3leg": round(mean(a["ndcg3"]), 4),
            "ndcg@10_delta": round(mean(a["ndcg3"]) - mean(a["ndcg2"]), 4),
            "mrr@10_2leg": round(mean(a["mrr2"]), 4),
            "mrr@10_3leg": round(mean(a["mrr3"]), 4),
            "recall@50_2leg": round(mean(a["rec2"]), 4),
            "recall@50_3leg": round(mean(a["rec3"]), 4),
            "recall@50_delta": round(mean(a["rec3"]) - mean(a["rec2"]), 4),
            "avg_dense_relevant_coverage": round(mean(a["dense_overlap"]), 4),
        }

    summary = {s: summarize(s) for s in slices}

    out = {
        "config": {
            "judge_cache": args.judge_cache,
            "diverse_queries": args.diverse_queries,
            "dense_index": args.dense_index,
            "opensearch_url": url,
            "top_k_per_leg": args.top_k,
            "rrf_k": RRF_K,
            "ndcg_k": K_NDCG,
            "recall_k": K_RECALL,
            "relevant_threshold": RELEVANT_THRESHOLD,
            "gain": "2^grade - 1",
            "comparison": "2-leg RRF(BM25F+SPLADE) vs 3-leg RRF(+dense)",
        },
        "dataset": {
            "records_in_file": len(records),
            "scored": len(per_query),
            "skipped_no_ground_truth": skipped_no_gt,
        },
        "summary": summary,
        "caveats": [
            "Dense indexes a SUBSET (judged docs + distractors) while BM25F/SPLADE "
            "search the full ~150M-doc index, so dense coverage is partial -> its "
            "contribution is a LOWER BOUND.",
            "Paraphrase ground-truth reuse assumes a paraphrase preserves relevance.",
            "The judge cache reflects the original (keyword) query distribution.",
        ],
        "per_query": per_query,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    # ── Print report ──
    print("\n" + "=" * 88)
    print("DENSE-ANN RETRIEVAL POC  —  2-leg RRF(BM25F+SPLADE)  vs  3-leg RRF(+dense)")
    print("=" * 88)
    print(f"Records scored: {len(per_query)}   skipped (no relevant ground truth): {skipped_no_gt}")
    print(f"top_k/leg={args.top_k}  RRF_k={RRF_K}  NDCG@{K_NDCG}  Recall@{K_RECALL}  "
          f"dense_index={args.dense_index}")
    print("-" * 88)
    hdr = (f"{'slice':<10} {'n':>4} {'NDCG2':>8} {'NDCG3':>8} {'dNDCG':>8} "
           f"{'Rec2':>7} {'Rec3':>7} {'dRec':>7} {'dnsCov':>7}")
    print(hdr)
    print("-" * 88)
    for s in slices:
        v = summary[s]
        print(f"{s:<10} {v['n_queries']:>4} {v['ndcg@10_2leg']:>8.4f} {v['ndcg@10_3leg']:>8.4f} "
              f"{v['ndcg@10_delta']:>+8.4f} {v['recall@50_2leg']:>7.4f} {v['recall@50_3leg']:>7.4f} "
              f"{v['recall@50_delta']:>+7.4f} {v['avg_dense_relevant_coverage']:>7.4f}")
    print("-" * 88)
    print("dNDCG/dRec = 3-leg minus 2-leg.  dnsCov = avg fraction of a query's judged-relevant")
    print("docs that the dense leg alone surfaced in its top-k.")
    print("=" * 88)
    logger.info("Wrote results -> %s", args.output)


if __name__ == "__main__":
    main()
