#!/usr/bin/env python3
"""Full-PIPELINE eval comparator: does the dense-ANN leg's retrieval-recall gain
SURVIVE the production rerank path and improve the FINAL reranked top-10?

This is the "entire workflow" test the retrieval-only dense POC
(scripts/eval_dense_poc.py) could not answer. The POC measured fused retrieval
(NDCG@10 / Recall@50 of the RRF top-50) but stopped before reranking. Production
search (src/lib/tools.py::_maybe_rerank) applies TWO more stages after fusion:

    Stage 1  embedding rerank   rerank_results(..., use_embedding=True, use_rrf=...)
    Stage 2  cross-encoder      rerank_with_crossencoder(..., limit) on top _CE_CANDIDATE_POOL

so a retrieval-level gain can be amplified, preserved, or WASHED OUT by the
rerankers (they re-score the fused candidate pool; a recall win only helps the
final top-10 if the rerankers also rank those newly-surfaced docs highly).

This harness mirrors that exact shipped path. For each query:
    retrieve enabled legs (bm25f / splade / dense), top-50 each
    -> RRF fuse (k=60), KEEPING the doc objects (rerankers need title+abstract)
    -> Stage 1 embedding rerank   (v5 embedder via SFU_EMBEDDING_MODEL_PATH)
    -> Stage 2 cross-encoder rerank (models/sfu-cross-encoder-v1), pool=_CE_CANDIDATE_POOL
    -> score the FINAL top-10 against the judge cache (NDCG@10, MRR@10)
sliced keyword / natural / overall.

It also emits the RETRIEVAL-ONLY (pre-rerank) NDCG@10/MRR@10 of the fused top-10
for reference, so the report can show how much the retrieval-only delta shrank
(or held) after reranking.

CAVEATS (printed in the report — identical to the POC):
  1. The dense index is a 600K SUBSET (judged docs + distractors) while
     BM25F/SPLADE search the full ~150M-doc index, so dense had GUARANTEED
     judged-doc coverage and far fewer distractors. Absolute deltas are therefore
     OPTIMISTIC / an upper bound, NOT production magnitudes. The DIRECTIONAL
     result (does dense survive reranking; keyword vs natural) is the robust take.
  2. Paraphrase ground-truth reuse assumes a paraphrase preserves relevance.
  3. The judge cache reflects the ORIGINAL (keyword) query distribution, so it
     likely UNDERSTATES dense's natural-language advantage.

Usage
─────
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    SFU_OPENSEARCH_URL=http://...:9200 \
    python scripts/eval_pipeline.py \
        --output data/eval_results/pipeline_dense_comparison.json \
        [--rerank embed+ce] [--limit N]

By default it runs the two headline configs (A = bm25f+splade, B = +dense),
both with the full production rerank path (embed+ce).
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
logger = logging.getLogger("eval_pipeline")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_JUDGE_CACHE = REPO_ROOT / "data/eval_results/llm_judge_cache.json"
DEFAULT_DIVERSE = REPO_ROOT / "data/eval_results/diverse_queries.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/eval_results/pipeline_dense_comparison.json"

# v5 dense bi-encoder — used both for the dense retrieval leg AND, to mirror the
# real shipped pipeline (task: "the embedder is v5 via SFU_EMBEDDING_MODEL_PATH"),
# for the Stage-1 embedding rerank. The deployed SFU_EMBEDDING_MODEL_PATH points
# at a model dir absent from this checkout, so we wire v5 explicitly here.
DENSE_MODEL = str(REPO_ROOT / "models/sfu-academic-embed-v5")
SPLADE_ONNX = str(REPO_ROOT / "models/splade_onnx")

RELEVANT_THRESHOLD = 2
K_NDCG = 10
RRF_K = 60          # matches eval_dense_poc.py and federated_search RRF
JUDGE_KEY_LEN = 80


def opensearch_url() -> str:
    return os.environ.get(
        "SFU_OPENSEARCH_URL",
        "http://claudebox-sfu-library-mcp-training-opensearch:9200",
    ).rstrip("/")


# ── Metrics (identical to eval_dense_poc.py) ────────────────────────────────────

def dcg(gains: list[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_grades: list[int], ideal_grades: list[int], k: int) -> float:
    """NDCG@k. ideal_grades = ALL judged grades for the query (the true ideal set)."""
    idcg = dcg(sorted(ideal_grades, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return dcg(ranked_grades, k) / idcg


def mrr_at_k(ranked_grades: list[int], k: int) -> float:
    for i, g in enumerate(ranked_grades[:k]):
        if g >= RELEVANT_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


# ── RRF over DOC OBJECTS (k=60) ─────────────────────────────────────────────────
# Unlike the POC's rrf_fuse (which returns ids), the pipeline must keep the doc
# bodies so the rerankers can score title+abstract. Keyed by openalex_id.

def doc_id(doc: dict) -> str:
    """Stable id for fusion/grading (matches the judge-cache key suffix)."""
    return doc.get("openalex_id") or doc.get("doi") or doc.get("title") or ""


def rrf_fuse_docs(ranked_lists: list[list[dict]], top_k: int) -> list[dict]:
    """Reciprocal Rank Fusion of N ranked lists. Returns fused DOC OBJECTS (top_k),
    deduped by doc_id, in fused-score order. The retained object is the first one
    seen for that id (legs return the same _source fields)."""
    scores: dict[str, float] = defaultdict(float)
    first_doc: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst, start=1):
            k = doc_id(doc)
            if not k:
                continue
            scores[k] += 1.0 / (RRF_K + rank)
            first_doc.setdefault(k, doc)
    ranked_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [first_doc[i] for i in ranked_ids[:top_k]]


# ── Ground truth ────────────────────────────────────────────────────────────────

def load_judge_grades(path: Path) -> dict[str, dict[str, int]]:
    cache = json.loads(path.read_text())
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    for key, grade in cache.items():
        qkey, did = key.rsplit("||", 1)
        if isinstance(grade, int):
            grades[qkey][did] = grade
    return grades


def score_top10(ranked_docs: list[dict], gradebook: dict[str, int]) -> tuple[float, float]:
    """Return (NDCG@10, MRR@10) for a ranked list of doc objects."""
    ranked_grades = [gradebook.get(doc_id(d), 0) for d in ranked_docs]
    ndcg = ndcg_at_k(ranked_grades, list(gradebook.values()), K_NDCG)
    mrr = mrr_at_k(ranked_grades, K_NDCG)
    return ndcg, mrr


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ── Production rerank path (mirrors tools.py::_maybe_rerank) ─────────────────────

def apply_rerank(docs: list[dict], query: str, limit: int, rerank: str,
                 embed_model_path: str) -> list[dict]:
    """Mirror tools.py::_maybe_rerank for rerank in {none, embed, embed+ce}.

    none    -> fused order, truncated to `limit`.
    embed   -> Stage-1 embedding rerank only (use_embedding=True, use_rrf=True
               to match production SFU_FEATURE_RRF_ENABLED=true).
    embed+ce-> Stage-1 embedding rerank (widened to >= _CE_CANDIDATE_POOL so the
               CE can promote docs from rank 11-20), THEN cross-encoder Stage 2.
    """
    if rerank == "none" or not docs:
        return docs[:limit]

    from lib.reranker import (rerank_results, rerank_with_crossencoder,
                              _CE_CANDIDATE_POOL)

    ce_enabled = rerank == "embed+ce"
    # Production widens Stage-1's output so the CE sees a pool > limit (item #4).
    stage1_limit = max(limit, _CE_CANDIDATE_POOL) if ce_enabled else limit
    reranked = rerank_results(
        docs,
        query,
        stage1_limit,
        use_embedding=True,
        use_rrf=True,                       # SFU_FEATURE_RRF_ENABLED=true in prod
        embedding_model_path=embed_model_path,
    )
    if ce_enabled:
        reranked = rerank_with_crossencoder(reranked, query, limit)
    return reranked[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-pipeline dense eval (A vs B, reranked final-output)")
    parser.add_argument("--judge-cache", default=str(DEFAULT_JUDGE_CACHE))
    parser.add_argument("--diverse-queries", default=str(DEFAULT_DIVERSE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=50,
                        help="hits retrieved per leg before fusion")
    parser.add_argument("--rerank", choices=["none", "embed", "embed+ce"],
                        default="embed+ce", help="rerank path for both configs")
    parser.add_argument("--dense-index", default="openalex_works_dense")
    parser.add_argument("--splade-onnx", default=SPLADE_ONNX,
                        help="ONNX export dir used to encode SPLADE QUERIES. Must "
                             "match the model the index's sparse_field was encoded "
                             "with — after a re-encode pass that's "
                             "models/splade_onnx_fp16 (the indexer's doc encoder), "
                             "NOT the default models/splade_onnx.")
    parser.add_argument("--limit", type=int, default=0, help="limit #records (0=all)")
    args = parser.parse_args()

    from lib.opensearch_retriever import OpenSearchRetriever

    url = opensearch_url()
    retriever = OpenSearchRetriever(url=url, index="openalex_works",
                                    splade_model_path=args.splade_onnx, timeout=60)

    gradebooks = load_judge_grades(Path(args.judge_cache))
    records = json.loads(Path(args.diverse_queries).read_text())
    if args.limit:
        records = records[: args.limit]

    # Two headline configs. Both run the SAME rerank path; they differ only in
    # whether the dense leg is fused in before reranking.
    CONFIGS = {
        "A_current":  {"legs": ["bm25f", "splade"],          "label": "bm25f+splade (current)"},
        "B_plusdense": {"legs": ["bm25f", "splade", "dense"], "label": "bm25f+splade+dense"},
    }

    slices = ("keyword", "natural", "overall")
    # acc[cfg][slice] = {"final_ndcg": [...], "final_mrr": [...], "retr_ndcg": [...], "retr_mrr": [...]}
    acc = {
        cfg: {s: defaultdict(list) for s in slices}
        for cfg in CONFIGS
    }
    per_query: list[dict] = []
    skipped_no_gt = 0

    for i, rec in enumerate(records):
        qtext = rec["paraphrase"]
        qtype = rec["query_type"]
        key = rec.get("judge_key", rec["original_query"][:JUDGE_KEY_LEN])
        gradebook = gradebooks.get(key, {})
        if not any(g >= RELEVANT_THRESHOLD for g in gradebook.values()):
            skipped_no_gt += 1
            continue

        # Retrieve each leg ONCE; reuse across configs.
        legs = {
            "bm25f": retriever.search(qtext, top_k=args.top_k, mode="bm25f"),
            "splade": retriever.search(qtext, top_k=args.top_k, mode="splade"),
            "dense": retriever.dense_search(qtext, top_k=args.top_k,
                                            dense_index=args.dense_index,
                                            dense_model_path=DENSE_MODEL),
        }

        q_record: dict = {"paraphrase": qtext, "query_type": qtype, "judge_key": key}
        for cfg, spec in CONFIGS.items():
            fused = rrf_fuse_docs([legs[l] for l in spec["legs"]], args.top_k)

            # Retrieval-only reference: fused top-10 before any reranking.
            retr_ndcg, retr_mrr = score_top10(fused[:K_NDCG], gradebook)

            # Production rerank path -> FINAL top-10.
            final_docs = apply_rerank(fused, qtext, K_NDCG, args.rerank, DENSE_MODEL)
            final_ndcg, final_mrr = score_top10(final_docs, gradebook)

            for s in (qtype, "overall"):
                acc[cfg][s]["final_ndcg"].append(final_ndcg)
                acc[cfg][s]["final_mrr"].append(final_mrr)
                acc[cfg][s]["retr_ndcg"].append(retr_ndcg)
                acc[cfg][s]["retr_mrr"].append(retr_mrr)

            q_record[f"{cfg}_final_ndcg"] = round(final_ndcg, 4)
            q_record[f"{cfg}_final_mrr"] = round(final_mrr, 4)
            q_record[f"{cfg}_retr_ndcg"] = round(retr_ndcg, 4)

        per_query.append(q_record)
        if (i + 1) % 10 == 0:
            logger.info("  processed %d/%d records (%d scored)",
                        i + 1, len(records), len(per_query))

    def summarize(cfg: str, s: str) -> dict:
        a = acc[cfg][s]
        return {
            "n_queries": len(a["final_ndcg"]),
            "final_ndcg@10": round(mean(a["final_ndcg"]), 4),
            "final_mrr@10": round(mean(a["final_mrr"]), 4),
            "retrieval_ndcg@10": round(mean(a["retr_ndcg"]), 4),
            "retrieval_mrr@10": round(mean(a["retr_mrr"]), 4),
        }

    summary = {cfg: {s: summarize(cfg, s) for s in slices} for cfg in CONFIGS}

    # A-vs-B deltas (B minus A) on the FINAL reranked output and on retrieval-only.
    deltas = {}
    for s in slices:
        a, b = summary["A_current"][s], summary["B_plusdense"][s]
        deltas[s] = {
            "final_ndcg@10_delta": round(b["final_ndcg@10"] - a["final_ndcg@10"], 4),
            "final_mrr@10_delta": round(b["final_mrr@10"] - a["final_mrr@10"], 4),
            "retrieval_ndcg@10_delta": round(b["retrieval_ndcg@10"] - a["retrieval_ndcg@10"], 4),
            "retrieval_mrr@10_delta": round(b["retrieval_mrr@10"] - a["retrieval_mrr@10"], 4),
        }

    out = {
        "config": {
            "judge_cache": args.judge_cache,
            "diverse_queries": args.diverse_queries,
            "dense_index": args.dense_index,
            "splade_query_onnx": args.splade_onnx,
            "dense_embed_model": DENSE_MODEL,
            "rerank_model_embed": DENSE_MODEL,
            "rerank_model_crossencoder": str(REPO_ROOT / "models/sfu-cross-encoder-v1"),
            "opensearch_url": url,
            "top_k_per_leg": args.top_k,
            "rrf_k": RRF_K,
            "ndcg_k": K_NDCG,
            "rerank_path": args.rerank,
            "use_rrf_in_embed_stage": True,
            "relevant_threshold": RELEVANT_THRESHOLD,
            "gain": "2^grade - 1",
            "configs": {cfg: CONFIGS[cfg]["label"] for cfg in CONFIGS},
            "comparison": ("A=RRF(bm25f+splade) vs B=RRF(bm25f+splade+dense), "
                           "BOTH through production rerank (embed -> cross-encoder); "
                           "headline = FINAL reranked top-10."),
        },
        "dataset": {
            "records_in_file": len(records),
            "scored": len(per_query),
            "skipped_no_ground_truth": skipped_no_gt,
        },
        "summary": summary,
        "delta_B_minus_A": deltas,
        "caveats": [
            "Dense index is a 600K SUBSET (judged docs + distractors) vs the full "
            "~150M-doc lexical index, so dense had GUARANTEED judged-doc coverage and "
            "far fewer distractors. Absolute deltas are OPTIMISTIC / an upper bound, "
            "NOT production magnitudes; the DIRECTIONAL result (does dense survive "
            "reranking; keyword vs natural) is the robust takeaway.",
            "Paraphrase ground-truth reuse assumes a paraphrase preserves relevance.",
            "The keyword-seeded judge cache likely UNDERSTATES dense's "
            "natural-language advantage.",
        ],
        "per_query": per_query,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    # ── Print report ──
    print("\n" + "=" * 96)
    print("FULL-PIPELINE DENSE EVAL  —  A=RRF(bm25f+splade)  vs  B=RRF(+dense)  "
          f"[rerank={args.rerank}]")
    print("=" * 96)
    print(f"Records scored: {len(per_query)}   skipped (no relevant ground truth): {skipped_no_gt}")
    print(f"top_k/leg={args.top_k}  RRF_k={RRF_K}  NDCG@{K_NDCG}  "
          f"embed=v5  CE=sfu-cross-encoder-v1  dense_index={args.dense_index}")
    print("-" * 96)
    print("FINAL reranked top-10 (the headline — full production rerank path):")
    hdr = (f"  {'slice':<9} {'A_NDCG':>8} {'B_NDCG':>8} {'dNDCG':>8}   "
           f"{'A_MRR':>8} {'B_MRR':>8} {'dMRR':>8}")
    print(hdr)
    print("  " + "-" * 70)
    for s in slices:
        a, b, d = summary["A_current"][s], summary["B_plusdense"][s], deltas[s]
        print(f"  {s:<9} {a['final_ndcg@10']:>8.4f} {b['final_ndcg@10']:>8.4f} "
              f"{d['final_ndcg@10_delta']:>+8.4f}   "
              f"{a['final_mrr@10']:>8.4f} {b['final_mrr@10']:>8.4f} "
              f"{d['final_mrr@10_delta']:>+8.4f}")
    print("-" * 96)
    print("RETRIEVAL-ONLY fused top-10 (pre-rerank reference — compare delta shrinkage):")
    print(hdr.replace("NDCG", "rNDC").replace("MRR", "rMRR"))
    print("  " + "-" * 70)
    for s in slices:
        a, b, d = summary["A_current"][s], summary["B_plusdense"][s], deltas[s]
        print(f"  {s:<9} {a['retrieval_ndcg@10']:>8.4f} {b['retrieval_ndcg@10']:>8.4f} "
              f"{d['retrieval_ndcg@10_delta']:>+8.4f}   "
              f"{a['retrieval_mrr@10']:>8.4f} {b['retrieval_mrr@10']:>8.4f} "
              f"{d['retrieval_mrr@10_delta']:>+8.4f}")
    print("-" * 96)
    print("d* = B minus A.  FINAL block = does the dense recall gain SURVIVE reranking.")
    print("CAVEAT: dense is a 600K judged-coverage subset vs 150M lexical -> deltas are")
    print("an OPTIMISTIC upper bound; the DIRECTION (survives? keyword vs natural) is robust.")
    print("=" * 96)
    logger.info("Wrote results -> %s", args.output)


if __name__ == "__main__":
    main()
