#!/usr/bin/env python3
"""Phase O.3 — LambdaMART learned reranker (Tier 2 stacking).

Reads the structured query log written by the MCP server (SFU_QUERY_LOG_PATH),
extracts per-(query, doc) feature vectors, and trains a LightGBM LambdaMART
ranker that replaces (or post-processes) the heuristic weighted sum in reranker.py.

STATUS: Framework complete.  Needs ~2,000+ logged queries with implicit or
explicit click signals before training produces reliable deltas.  Use
`--dry-run` to verify feature extraction against the current log without
training.

Usage:
    # Check how many queries are logged and preview features:
    python scripts/train_lambdamart.py --log logs/query_log.jsonl --dry-run

    # Train a ranker (needs lightgbm):
    python scripts/train_lambdamart.py --log logs/query_log.jsonl --out models/lambdamart_v1.txt

    # Evaluate against the SFU eval set after training:
    python scripts/train_lambdamart.py --log logs/query_log.jsonl --eval

Feature vector (per query-doc pair)
-------------------------------------
  f0   bm25_rank_score         1 / (1 + original rank from OpenAlex)
  f1   log_citations           log(1 + cited_by_count)
  f2   recency                 max(0, 1 - 0.05 * (current_year - pub_year))
  f3   has_doi                 1 if doi present else 0
  f4   type_score              article=1.0, book=0.7, conf=0.6, else 0.5
  f5   abstract_present        1 if abstract_len > 50 else 0
  f6   embed_cosine            cosine(encode(query), encode(title+abstract))
                               (requires SFU embedding model)

Relevance labels (without click data)
---------------------------------------
  log(1 + cited_by_count)   — citation-count proxy, same as Phase M eval
  Once real clicks accumulate:
    clicked_at_rank_1  → grade 3
    clicked_at_rank_2-5 → grade 2
    appeared_not_clicked → grade 0
    Ignored (not in top-10) → grade 0

Minimum data requirements
--------------------------
  Citation-proxy training:  ~500 queries (log already captures enough signal)
  Click-signal training:    ~2,000 queries with non-trivial click diversity
  Eval gain over v4-bge+RRF: expected +1-3% NDCG@10 at 2k queries (literature estimate)
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CURRENT_YEAR = 2026
REPO_ROOT = Path(__file__).parent.parent


# ── Feature extraction ────────────────────────────────────────────────────────

def _type_score(doc_type: str) -> float:
    t = (doc_type or "").lower()
    if t in ("article", "journal-article", "review"):
        return 1.0
    if t in ("book", "book-chapter"):
        return 0.7
    if t in ("proceedings-article", "dissertation"):
        return 0.6
    return 0.5


def extract_features(query: str, results: list[dict], embed_model=None) -> list[list[float]]:
    """Return a feature matrix (n_results × n_features) for one query."""
    embeddings = None
    if embed_model is not None:
        try:
            texts = [f"{r.get('title', '')} {r.get('abstract', '')[:300]}" for r in results]
            q_emb = embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            d_embs = embed_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            embeddings = (d_embs @ q_emb).tolist()
        except Exception as e:
            logger.warning("Embedding failed for query '%s': %s", query[:40], e)

    rows = []
    n = len(results)
    for i, r in enumerate(results):
        bm25_rank_score = 1.0 / (1 + i)
        log_citations = math.log1p(r.get("cited_by_count") or 0)
        year = r.get("year") or r.get("publication_year") or r.get("pub_year")
        recency = max(0.0, 1.0 - 0.05 * (CURRENT_YEAR - year)) if year else 0.0
        has_doi = 1.0 if r.get("doi") else 0.0
        type_sc = _type_score(r.get("type") or "")
        abstract_present = 1.0 if (r.get("abstract_len") or 0) > 50 else 0.0
        embed_cosine = embeddings[i] if embeddings else 0.0
        rows.append([bm25_rank_score, log_citations, recency, has_doi, type_sc, abstract_present, embed_cosine])
    return rows


def relevance_label(r: dict) -> float:
    """Citation-count relevance grade (same proxy as eval harness)."""
    return math.log1p(r.get("cited_by_count") or 0)


# ── Data loading ──────────────────────────────────────────────────────────────

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


def build_dataset(entries: list[dict], embed_model=None):
    """Build LambdaMART training arrays from logged queries.

    Returns (X, y, qids):
      X    float[n_pairs × n_features]
      y    float[n_pairs] relevance grades
      qids int[n_pairs]   query group id (same id for all docs of one query)
    """
    X, y, qids = [], [], []
    for qid, entry in enumerate(entries):
        query = entry.get("query", "")
        results = entry.get("results", [])
        if not results:
            continue
        features = extract_features(query, results, embed_model)
        labels = [relevance_label(r) for r in results]
        for feat_row, label in zip(features, labels):
            X.append(feat_row)
            y.append(label)
            qids.append(qid)
    return X, y, qids


# ── Training ──────────────────────────────────────────────────────────────────

def train(log_path: Path, out_path: Path, embed_model_path: str = "", use_embed: bool = True):
    entries = load_query_log(log_path)
    if not entries:
        logger.error("No usable entries in query log. Run the MCP server with query logging enabled first.")
        return

    embed_model = None
    if use_embed:
        try:
            from sentence_transformers import SentenceTransformer
            path = embed_model_path or str(REPO_ROOT / "models/sfu-academic-embed-v4-bge")
            embed_model = SentenceTransformer(path)
            logger.info("Embedding model loaded: %s", path)
        except Exception as e:
            logger.warning("Embedding model unavailable (%s); training without embed feature", e)

    X, y, qids = build_dataset(entries, embed_model)
    logger.info("Dataset: %d (query, doc) pairs from %d queries", len(X), len(entries))

    if len(X) < 100:
        logger.warning(
            "Only %d training pairs — LambdaMART is unreliable below ~1,000. "
            "Accumulate more query logs first.", len(X)
        )

    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        logger.error("lightgbm not installed. Run: pip install lightgbm")
        return

    import numpy as np
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)

    # Group sizes: number of docs per query
    from collections import Counter
    group_counts = Counter(qids)
    groups = [group_counts[i] for i in range(max(qids) + 1)]

    train_data = lgb.Dataset(X_arr, label=y_arr, group=groups, feature_name=[
        "bm25_rank_score", "log_citations", "recency",
        "has_doi", "type_score", "abstract_present", "embed_cosine",
    ])

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "lambdarank_truncation_level": 10,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "verbose": -1,
    }

    logger.info("Training LambdaMART (200 rounds)...")
    model = lgb.train(params, train_data, num_boost_round=200, valid_sets=[train_data])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    logger.info("Model saved to %s", out_path)
    logger.info("Feature importances: %s", dict(zip(
        ["bm25_rank", "log_citations", "recency", "has_doi", "type", "abstract", "embed"],
        model.feature_importance(importance_type="gain").tolist()
    )))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LambdaMART reranker training (Phase O.3)")
    parser.add_argument("--log", default=str(REPO_ROOT / "logs/query_log.jsonl"),
                        help="Path to query log JSONL (written by MCP server)")
    parser.add_argument("--out", default=str(REPO_ROOT / "models/lambdamart_v1.txt"),
                        help="Output path for trained LightGBM model")
    parser.add_argument("--embed-model", default="",
                        help="Path to SFU embedding model (default: models/sfu-academic-embed-v4-bge)")
    parser.add_argument("--no-embed", action="store_true",
                        help="Skip embedding cosine feature (faster, lower quality)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse log and report statistics without training")
    args = parser.parse_args()

    log_path = Path(args.log)
    entries = load_query_log(log_path)

    if not entries:
        print(f"No entries found at {log_path}")
        print("Enable query logging: SFU_FEATURE_QUERY_LOG_ENABLED=true  SFU_QUERY_LOG_PATH=/path/to/log.jsonl")
        sys.exit(1)

    print(f"\nQuery log: {log_path}")
    print(f"  Total queries logged:  {len(entries)}")
    queries_with_results = sum(1 for e in entries if e.get("results"))
    print(f"  Queries with results:  {queries_with_results}")
    handlers = {}
    for e in entries:
        h = e.get("handler", "unknown")
        handlers[h] = handlers.get(h, 0) + 1
    for h, c in sorted(handlers.items()):
        print(f"  Handler '{h}':  {c}")

    total_pairs = sum(len(e.get("results", [])) for e in entries)
    print(f"  Total (query, doc) pairs:  {total_pairs}")

    threshold_msg = ""
    if len(entries) < 500:
        threshold_msg = f"  ⚠  Need ~500 for citation-proxy training ({500 - len(entries)} more)"
    elif len(entries) < 2000:
        threshold_msg = f"  ⚠  Need ~2,000 for click-signal training ({2000 - len(entries)} more)"
    else:
        threshold_msg = "  ✓  Sufficient for click-signal LambdaMART training"
    print(threshold_msg)

    if args.dry_run:
        print("\nDry run complete — no model trained.")
        return

    train(log_path, Path(args.out), args.embed_model, not args.no_embed)


if __name__ == "__main__":
    main()
