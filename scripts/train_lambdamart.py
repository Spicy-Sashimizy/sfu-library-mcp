#!/usr/bin/env python3
"""Phase N — LambdaMART learned reranker (Tier 2 stacking).

Trains a LightGBM ``lambdarank`` model that re-scores reranker candidates at serve
time (lib.reranker.rerank_with_lambdamart). Features come from the SHARED module
lib.lambdamart_features.doc_features, so what the model trains on is exactly what it
scores in production.

Two data paths:
  --dataset data/lambdamart_dataset.jsonl   PRIMARY. Built by
      scripts/build_lambdamart_dataset.py from the LLM-judged relevance grades
      (TREC 0-3). True graded labels — the right signal.
  --log logs/query_log.jsonl                FALLBACK. The structured query log
      written by the MCP server. No human labels, so it uses the citation-count
      proxy (same weak proxy as the old eval); only useful before judged data
      exists for the logged queries.

Usage:
  # Inspect a dataset without training:
  python scripts/train_lambdamart.py --dataset data/lambdamart_dataset.jsonl --dry-run

  # Train (needs lightgbm):
  python scripts/train_lambdamart.py --dataset data/lambdamart_dataset.jsonl \
      --out models/lambdamart_v1.txt --embed-model models/sfu-academic-embed-v5

  # Cross-validated NDCG@10 vs the semantic baseline (needs lightgbm):
  python scripts/train_lambdamart.py --dataset data/lambdamart_dataset.jsonl --eval

Minimum data
  Judged-grade training:  the 120-query judge cache (~4k pairs) is usable but small
  (LightGBM lambdarank is noisy below ~1k pairs). Treat --eval deltas as provisional
  and keep SFU_FEATURE_LAMBDAMART_ENABLED off until a CV win is confirmed.
"""

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_lambdamart")

from lib.lambdamart_features import FEATURE_NAMES, N_FEATURES, doc_features  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data/lambdamart_dataset.jsonl"
DEFAULT_OUT = REPO_ROOT / "models/lambdamart_v1.txt"
EVAL_OUT = REPO_ROOT / "data/eval_results/lambdamart_eval.json"

RELEVANT_THRESHOLD = 2
K = 10
N_FOLDS = 5

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "lambdarank_truncation_level": 10,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 5,
    "verbose": -1,
}
NUM_BOOST_ROUND = 200


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_dataset(path: Path):
    """Load the builder's JSONL into grouped arrays.

    Returns (groups) where groups[qid] = {"query", "X": [[...]], "y": [...], "ids": [...]}.
    """
    if not path.exists():
        logger.error("Dataset not found: %s. Run scripts/build_lambdamart_dataset.py first.", path)
        return {}
    groups: dict[int, dict] = defaultdict(lambda: {"query": "", "X": [], "y": [], "ids": []})
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            g = groups[row["qid"]]
            g["query"] = row.get("query", "")
            g["X"].append(row["features"])
            g["y"].append(int(row["grade"]))
            g["ids"].append(row.get("workid", ""))
    return dict(groups)


def load_query_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        logger.error("Query log not found: %s", log_path)
        return []
    entries = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def groups_from_query_log(entries: list[dict], embed_model_path: str | None):
    """Fallback path: build groups from the query log using the citation proxy label.

    The log stores compact result rows (no abstract text, no embedding), so the
    embed_cosine feature is 0 here and the label is log1p(cited_by_count). This is a
    stopgap until judged grades exist for logged queries.
    """
    from lib.reranker import _normalize_for_rerank
    groups: dict[int, dict] = {}
    for qid, entry in enumerate(entries):
        results = entry.get("results", [])
        if not results:
            continue
        g = {"query": entry.get("query", ""), "X": [], "y": [], "ids": []}
        for r in results:
            year = r.get("year")
            doc = {
                "title": r.get("title", ""),
                "abstract": "x" * (r.get("abstract_len", 0) or 0),
                "date": str(year) if year else "",
                "type": r.get("type", ""),
                "doi": r.get("doi", ""),
                "authors": [],
                "is_oa": False, "oa_url": "",
            }
            norm = _normalize_for_rerank(doc)
            g["X"].append(doc_features(norm, r.get("cited_by_count", 0), 0.0))
            g["y"].append(int(round(math.log1p(r.get("cited_by_count") or 0))))
            g["ids"].append(r.get("doi", ""))
        if g["X"]:
            groups[qid] = g
    return groups


# ── Metrics (ported from scripts/eval_cross_encoder.py) ────────────────────────

def dcg(gains: list[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_grades: list[int], k: int) -> float:
    idcg = dcg(sorted(ranked_grades, reverse=True), k)
    return dcg(ranked_grades, k) / idcg if idcg else 0.0


def mrr_at_k(ranked_grades: list[int], k: int) -> float:
    for i, g in enumerate(ranked_grades[:k]):
        if g >= RELEVANT_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


# ── Training ───────────────────────────────────────────────────────────────────

def _build_lgb_dataset(lgb, np, groups: dict[int, dict]):
    X, y, group_sizes = [], [], []
    for qid in sorted(groups):
        g = groups[qid]
        X.extend(g["X"])
        y.extend(g["y"])
        group_sizes.append(len(g["X"]))
    return (
        lgb.Dataset(np.array(X, dtype=np.float32), label=np.array(y, dtype=np.float32),
                    group=group_sizes, feature_name=FEATURE_NAMES),
        len(X),
    )


def train(groups: dict[int, dict], out_path: Path, embed_model_path: str):
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        logger.error("lightgbm not installed. Run: sudo .venv/bin/pip install lightgbm")
        return

    train_set, n_pairs = _build_lgb_dataset(lgb, np, groups)
    logger.info("Dataset: %d (query, doc) pairs from %d queries", n_pairs, len(groups))
    if n_pairs < 1000:
        logger.warning("Only %d pairs — lambdarank is noisy below ~1,000; treat results as provisional.",
                       n_pairs)

    logger.info("Training LambdaMART (%d rounds)...", NUM_BOOST_ROUND)
    model = lgb.train(LGB_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND, valid_sets=[train_set])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    importances = dict(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain").tolist()))
    logger.info("Model saved to %s", out_path)
    logger.info("Feature importances (gain): %s", importances)

    # Sidecar consumed by the GUI reranker-signal radar AND a parity guard: records
    # which embedding checkpoint f5 was trained against (see reranker Risk 3).
    sidecar = out_path.with_suffix(".feature_importance.json")
    sidecar.write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "importance_gain": importances,
        "embed_model_path": embed_model_path or "",
        "n_pairs": n_pairs,
        "n_queries": len(groups),
    }, indent=2))
    logger.info("Feature-importance sidecar: %s", sidecar)


def evaluate(groups: dict[int, dict]):
    """K-fold CV NDCG@10/MRR@10: LambdaMART vs the embed_cosine (semantic) baseline."""
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        logger.error("lightgbm not installed. Run: sudo .venv/bin/pip install lightgbm")
        return

    embed_idx = FEATURE_NAMES.index("embed_cosine")
    qids = sorted(groups)
    per_q = {"lambdamart": [], "baseline": []}

    for fold in range(N_FOLDS):
        test_qids = [q for i, q in enumerate(qids) if i % N_FOLDS == fold]
        train_qids = [q for q in qids if q not in set(test_qids)]
        if not test_qids or not train_qids:
            continue
        train_groups = {q: groups[q] for q in train_qids}
        train_set, _ = _build_lgb_dataset(lgb, np, train_groups)
        model = lgb.train(LGB_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND)

        for q in test_qids:
            g = groups[q]
            X = np.array(g["X"], dtype=np.float32)
            grades = g["y"]
            # LambdaMART order
            lm_scores = model.predict(X).tolist()
            lm_order = [grades[i] for i in sorted(range(len(grades)), key=lambda i: lm_scores[i], reverse=True)]
            per_q["lambdamart"].append(ndcg_at_k(lm_order, K))
            # Baseline: rank by the semantic feature (the dominant production signal)
            base_scores = [row[embed_idx] for row in g["X"]]
            base_order = [grades[i] for i in sorted(range(len(grades)), key=lambda i: base_scores[i], reverse=True)]
            per_q["baseline"].append(ndcg_at_k(base_order, K))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    lm_ndcg = _mean(per_q["lambdamart"])
    base_ndcg = _mean(per_q["baseline"])
    delta = lm_ndcg - base_ndcg
    n = len(per_q["lambdamart"])

    result = {
        "metric": "ndcg_at_10",
        "n_queries": n,
        "folds": N_FOLDS,
        "lambdamart_ndcg": round(lm_ndcg, 4),
        "baseline_embed_cosine_ndcg": round(base_ndcg, 4),
        "delta": round(delta, 4),
        "winner": "lambdamart" if delta > 0 else "baseline",
        "note": "Cross-validated; small judged set — provisional. Ship only on a positive delta.",
    }
    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.write_text(json.dumps(result, indent=2))

    print("\nLambdaMART CV evaluation (NDCG@10):")
    print(f"  queries evaluated:        {n}")
    print(f"  LambdaMART:               {lm_ndcg:.4f}")
    print(f"  baseline (embed_cosine):  {base_ndcg:.4f}")
    print(f"  delta:                    {delta:+.4f}  -> winner: {result['winner']}")
    print(f"  written to {EVAL_OUT}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _report(groups: dict[int, dict]):
    n_pairs = sum(len(g["X"]) for g in groups.values())
    print(f"\nGroups: {len(groups)} queries, {n_pairs} (query, doc) pairs")
    if groups:
        grade_dist = Counter(y for g in groups.values() for y in g["y"])
        print(f"  Grade/label distribution: {dict(sorted(grade_dist.items()))}")
        avg = n_pairs / len(groups)
        print(f"  Avg candidates/query: {avg:.1f}")
    if n_pairs < 1000:
        print(f"  ⚠  Below ~1,000 pairs — lambdarank deltas will be provisional.")


def main():
    parser = argparse.ArgumentParser(description="LambdaMART reranker training (Phase N)")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="Primary: JSONL from build_lambdamart_dataset.py (judged grades)")
    parser.add_argument("--log", default="",
                        help="Fallback: query log JSONL (citation-proxy labels)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--embed-model", default=str(REPO_ROOT / "models/sfu-academic-embed-v5"),
                        help="Embedding checkpoint the dataset's embed_cosine was built with (recorded in sidecar)")
    parser.add_argument("--eval", action="store_true", help="K-fold CV NDCG@10 vs baseline; no model saved")
    parser.add_argument("--dry-run", action="store_true", help="Report dataset stats; no training")
    args = parser.parse_args()

    if args.log:
        entries = load_query_log(Path(args.log))
        if not entries:
            print(f"No entries in query log {args.log}.")
            print("Enable logging: SFU_FEATURE_QUERY_LOG_ENABLED=true SFU_QUERY_LOG_PATH=/path/log.jsonl")
            sys.exit(1)
        embed = args.embed_model if Path(args.embed_model).exists() else None
        groups = groups_from_query_log(entries, embed)
        print(f"Source: query log ({args.log}) — citation-proxy labels")
    else:
        groups = load_dataset(Path(args.dataset))
        if not groups:
            sys.exit(1)
        print(f"Source: judged dataset ({args.dataset})")

    _report(groups)

    if args.dry_run:
        print("\nDry run — no training.")
        return
    if args.eval:
        evaluate(groups)
        return
    train(groups, Path(args.out), args.embed_model)


if __name__ == "__main__":
    main()
