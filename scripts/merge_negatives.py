#!/usr/bin/env python3
"""Q3.1: Merge SPLADE + BM25F hard-negative legs and filter false negatives.

BM25F and SPLADE find different "looks-relevant-but-isn't" papers (they overlap
only ~7.5%). Pooling both legs gives the union of both failure modes — the
hardest possible negative signal for training.

CRITICAL false-negative filter: ~70% of academic BM25 top-k hits are actually
relevant (NV-Retriever 2024). Any pooled candidate the LLM judge scored >=2 is
truly relevant and MUST be dropped — keeping it would teach the model to push
down good results. We use the existing judge cache at
data/eval_results/llm_judge_cache.json, keyed "{query}||{doc_id}" -> int score
(0-3). Candidates with no judge entry are kept as negatives (unjudged = assumed
hard negative), which is the standard hard-negative-mining convention.

Each leg file is JSONL produced by scripts/mine_hard_negatives.py with fields:
    query, subject, retriever, doc_id, rank, score, title, abstract, doi, publication_year

Output: one merged JSONL record per surviving (query, doc_id) pair, with the
retrievers that surfaced it and their per-leg ranks recorded.

Usage:
    python scripts/merge_negatives.py \
        --inputs data/training/hard_negatives_splade.jsonl \
                 data/training/hard_negatives_bm25.jsonl \
        --judge-cache data/eval_results/llm_judge_cache.json \
        --min-judge-score 2 \
        --output data/training/hard_negatives_rrf_pool.jsonl
"""
import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_judge_cache(path: Path) -> dict[str, int]:
    """Return {"{query}||{doc_id}": int_score} from the LLM judge cache."""
    raw = json.loads(path.read_text())
    cache: dict[str, int] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)):
            cache[k] = int(v)
    return cache


def judge_key(query: str, doc_id: str) -> str:
    return f"{query}||{doc_id}"


def merge(
    input_paths: list[Path],
    judge_cache: dict[str, int],
    min_judge_score: int,
    output_path: Path,
) -> dict:
    # Pool by (query, doc_id). Track which legs surfaced each candidate and ranks.
    pool: dict[tuple[str, str], dict] = {}
    per_leg_loaded: Counter = Counter()

    for path in input_paths:
        records = load_jsonl(path)
        for r in records:
            query = r["query"]
            doc_id = r["doc_id"]
            leg = r.get("retriever", path.stem)
            per_leg_loaded[leg] += 1
            key = (query, doc_id)
            if key not in pool:
                pool[key] = {
                    "query": query,
                    "subject": r.get("subject", ""),
                    "doc_id": doc_id,
                    "title": r.get("title", ""),
                    "abstract": r.get("abstract", ""),
                    "doi": r.get("doi", ""),
                    "publication_year": r.get("publication_year"),
                    "retrievers": [],
                    "ranks": {},
                }
            entry = pool[key]
            if leg not in entry["retrievers"]:
                entry["retrievers"].append(leg)
            # keep best (lowest) rank seen for this leg
            prev = entry["ranks"].get(leg)
            if prev is None or r.get("rank", 1e9) < prev:
                entry["ranks"][leg] = r.get("rank")
            # fill any missing text from whichever leg has it
            if not entry["title"] and r.get("title"):
                entry["title"] = r["title"]
            if not entry["abstract"] and r.get("abstract"):
                entry["abstract"] = r["abstract"]

    pooled_total = len(pool)

    # False-negative filter: drop any candidate the judge scored >= min_judge_score.
    kept: list[dict] = []
    filtered_false_neg = 0
    unjudged_kept = 0
    judged_kept = 0
    overlap_count = 0
    filtered_score_dist: Counter = Counter()

    for (query, doc_id), entry in pool.items():
        if len(entry["retrievers"]) >= 2:
            overlap_count += 1
        score = judge_cache.get(judge_key(query, doc_id))
        if score is not None and score >= min_judge_score:
            filtered_false_neg += 1
            filtered_score_dist[score] += 1
            continue
        entry["judge_score"] = score  # None if unjudged
        if score is None:
            unjudged_kept += 1
        else:
            judged_kept += 1
        kept.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    # Atomic write: temp sibling -> fsync -> atomic os.replace (never a partial pool file).
    with tmp_path.open("w") as out:
        for entry in kept:
            out.write(json.dumps(entry) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp_path, output_path)

    return {
        "per_leg_loaded": dict(per_leg_loaded),
        "pooled_unique": pooled_total,
        "overlap_both_legs": overlap_count,
        "filtered_false_negatives": filtered_false_neg,
        "filtered_score_dist": dict(sorted(filtered_score_dist.items())),
        "final_pool_size": len(kept),
        "kept_judged_true_negatives": judged_kept,
        "kept_unjudged": unjudged_kept,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool SPLADE+BM25F negatives and filter LLM-judge false negatives"
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Per-leg JSONL files from mine_hard_negatives.py")
    parser.add_argument("--judge-cache",
                        default=str(REPO_ROOT / "data/eval_results/llm_judge_cache.json"))
    parser.add_argument("--min-judge-score", type=int, default=2,
                        help="Drop candidates judged >= this score (default 2 = relevant)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            logger.error("Input not found: %s", p)
            raise SystemExit(1)

    judge_path = Path(args.judge_cache)
    judge_cache = load_judge_cache(judge_path) if judge_path.exists() else {}
    logger.info("Loaded judge cache: %d scored entries from %s", len(judge_cache), judge_path)

    stats = merge(
        input_paths=input_paths,
        judge_cache=judge_cache,
        min_judge_score=args.min_judge_score,
        output_path=Path(args.output),
    )

    logger.info("=== merge_negatives summary ===")
    logger.info("  per-leg candidates loaded:  %s", stats["per_leg_loaded"])
    logger.info("  pooled unique (q,doc):      %d", stats["pooled_unique"])
    logger.info("  surfaced by BOTH legs:      %d", stats["overlap_both_legs"])
    logger.info("  false negatives filtered:   %d (judge >= %d)",
                stats["filtered_false_negatives"], args.min_judge_score)
    logger.info("    filtered score breakdown: %s", stats["filtered_score_dist"])
    logger.info("  FINAL pool size:            %d", stats["final_pool_size"])
    logger.info("    kept judged true-negs:    %d", stats["kept_judged_true_negatives"])
    logger.info("    kept unjudged:            %d", stats["kept_unjudged"])
    logger.info("  output:                     %s", stats["output"])


if __name__ == "__main__":
    main()
