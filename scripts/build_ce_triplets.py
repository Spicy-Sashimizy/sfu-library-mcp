#!/usr/bin/env python3
"""Q3.2 — Build query-doc cross-encoder training triplets.

This produces the *correct* Q3.2 training data: (anchor, positive, negative)
triplets where `anchor` is an SFU eval query, `positive` is a document the LLM
judge scored relevant (>=2) for that query, and `negative` is a mined hard
negative for the same query (a doc a retriever ranked highly but the judge
scored <2 / null). This is what `scripts/train_cross_encoder.py`'s
`build_input_examples` expects (keys `anchor`/`positive`/`negative`).

It is NOT the citation-pair triplets, and it is NOT the raw
`hard_negatives_rrf_pool.jsonl` (that pool has no `anchor`/`positive` keys and
would abort training).

Pipeline
────────
1. Group the LLM-judge cache (`{"<query>||<doc_id>": score}`) by query.
   Positives = doc_ids with score >= 2.
2. Fetch positive docs' title+abstract from the local OpenSearch index via
   `_mget` on `openalex_works` (docs keyed by OpenAlex id, e.g. W7066536488).
   Positive text = "{title}. {abstract}" — mirrors the negatives' text.
3. Group the mined hard-negative pool by query → that query's negatives.
4. For each query, pair every positive with up to --max-neg-per-pos sampled
   (shuffled) negatives. Queries with no score>=2 positive are skipped.

OpenSearch URL comes from $SFU_OPENSEARCH_URL (same as mine_hard_negatives.py).

Usage
─────
    SFU_OPENSEARCH_URL=http://...:9200 \
    python scripts/build_ce_triplets.py \
        --judge-cache data/eval_results/llm_judge_cache.json \
        --neg-pool data/training/hard_negatives_rrf_pool.jsonl \
        --output data/training/hard_negatives_triplets.jsonl \
        --max-neg-per-pos 5
"""
import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
INDEX = "openalex_works"
DEFAULT_JUDGE_CACHE = REPO_ROOT / "data/eval_results/llm_judge_cache.json"
DEFAULT_NEG_POOL = REPO_ROOT / "data/training/hard_negatives_rrf_pool.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data/training/hard_negatives_triplets.jsonl"
RELEVANT_THRESHOLD = 2  # judge score >= this == relevant positive
MGET_BATCH = 200


def opensearch_url() -> str:
    return os.environ.get(
        "SFU_OPENSEARCH_URL",
        "http://claudebox-sfu-library-mcp-training-opensearch:9200",
    ).rstrip("/")


def doc_text(title: str, abstract: str) -> str:
    """Build doc text as "{title}. {abstract}" — mirrors the negatives' text."""
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if title and abstract:
        # avoid doubling a period if the title already ends with one
        sep = " " if title.endswith((".", "?", "!")) else ". "
        return f"{title}{sep}{abstract}"
    return title or abstract


def load_judge_positives(path: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """Return (positives_by_query, all_scores_by_query).

    positives_by_query[query] = {doc_id with score >= RELEVANT_THRESHOLD}
    all_scores_by_query[query][doc_id] = score (for provenance/metadata).
    """
    cache = json.loads(path.read_text())
    positives: dict[str, set[str]] = defaultdict(set)
    scores: dict[str, dict[str, int]] = defaultdict(dict)
    for key, score in cache.items():
        query, doc_id = key.rsplit("||", 1)
        scores[query][doc_id] = score
        if isinstance(score, int) and score >= RELEVANT_THRESHOLD:
            positives[query].add(doc_id)
    return positives, scores


def load_neg_pool(path: Path) -> dict[str, list[dict]]:
    """Group mined hard-negative rows by query."""
    by_query: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_query[row["query"]].append(row)
    return by_query


def mget_docs(
    session: requests.Session, url: str, doc_ids: list[str]
) -> tuple[dict[str, dict], list[str]]:
    """Fetch docs by _id via _mget. Returns (id -> {title, abstract, ...}, missing_ids)."""
    fetched: dict[str, dict] = {}
    missing: list[str] = []
    for i in range(0, len(doc_ids), MGET_BATCH):
        batch = doc_ids[i : i + MGET_BATCH]
        resp = session.post(
            f"{url}/{INDEX}/_mget",
            json={"ids": batch},
            params={
                "_source_includes": "title,abstract,openalex_id,doi,publication_year"
            },
            timeout=60,
        )
        resp.raise_for_status()
        for doc in resp.json().get("docs", []):
            if doc.get("found"):
                fetched[doc["_id"]] = doc.get("_source", {})
            else:
                missing.append(doc["_id"])
    return fetched, missing


def build_triplets(
    positives_by_query: dict[str, set[str]],
    scores_by_query: dict[str, dict[str, int]],
    neg_by_query: dict[str, list[dict]],
    pos_docs: dict[str, dict],
    max_neg_per_pos: int,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    """Emit (anchor, positive, negative) triplets. Returns (triplets, stats)."""
    triplets: list[dict] = []
    queries_emitted = 0
    queries_skipped_no_pos = 0
    queries_skipped_no_negtext = 0
    pos_used: set[str] = set()
    neg_per_pos_counts: list[int] = []

    for query, pos_ids in positives_by_query.items():
        if not pos_ids:
            queries_skipped_no_pos += 1
            continue

        subject = ""
        negs = neg_by_query.get(query, [])
        if negs:
            subject = negs[0].get("subject", "") or ""

        # Build usable negative records (need non-empty text).
        neg_records = []
        for n in negs:
            ntext = doc_text(n.get("title", ""), n.get("abstract", ""))
            if ntext:
                neg_records.append({
                    "doc_id": n.get("doc_id"),
                    "text": ntext,
                    "judge_score": n.get("judge_score"),
                })

        emitted_for_query = 0
        for pos_id in sorted(pos_ids):
            src = pos_docs.get(pos_id)
            if not src:
                continue  # missing from index — counted as missing elsewhere
            ptext = doc_text(src.get("title", ""), src.get("abstract", ""))
            if not ptext:
                continue
            pos_used.add(pos_id)

            if not neg_records:
                # positive-only pair (train_cross_encoder handles missing negative)
                triplets.append({
                    "anchor": query,
                    "positive": ptext,
                    "negative": "",
                    "subject": subject,
                    "metadata": {
                        "positive_id": pos_id,
                        "positive_judge_score": scores_by_query.get(query, {}).get(pos_id),
                        "negative_id": None,
                        "negative_judge_score": None,
                    },
                })
                emitted_for_query += 1
                neg_per_pos_counts.append(0)
                continue

            sampled = neg_records[:]
            rng.shuffle(sampled)
            sampled = sampled[:max_neg_per_pos]
            neg_per_pos_counts.append(len(sampled))
            for neg in sampled:
                triplets.append({
                    "anchor": query,
                    "positive": ptext,
                    "negative": neg["text"],
                    "subject": subject,
                    "metadata": {
                        "positive_id": pos_id,
                        "positive_judge_score": scores_by_query.get(query, {}).get(pos_id),
                        "negative_id": neg["doc_id"],
                        "negative_judge_score": neg["judge_score"],
                    },
                })
                emitted_for_query += 1

        if emitted_for_query:
            queries_emitted += 1
        else:
            queries_skipped_no_negtext += 1

    avg_neg = (sum(neg_per_pos_counts) / len(neg_per_pos_counts)) if neg_per_pos_counts else 0.0
    stats = {
        "triplets": len(triplets),
        "queries_emitted": queries_emitted,
        "queries_skipped_no_positive": queries_skipped_no_pos,
        "queries_skipped_no_emit": queries_skipped_no_negtext,
        "unique_positives_used": len(pos_used),
        "avg_neg_per_positive": round(avg_neg, 3),
    }
    return triplets, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q3.2 — Build (anchor, positive, negative) cross-encoder triplets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--judge-cache", default=str(DEFAULT_JUDGE_CACHE),
                        help="LLM judge cache JSON ({'<query>||<doc_id>': score})")
    parser.add_argument("--neg-pool", default=str(DEFAULT_NEG_POOL),
                        help="Mined hard-negative pool JSONL (one row per neg)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output triplets JSONL")
    parser.add_argument("--max-neg-per-pos", type=int, default=5,
                        help="Cap negatives paired with each positive")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    judge_path = Path(args.judge_cache)
    neg_path = Path(args.neg_pool)
    out_path = Path(args.output)
    rng = random.Random(args.seed)

    logger.info("Judge cache: %s", judge_path)
    positives_by_query, scores_by_query = load_judge_positives(judge_path)
    logger.info("  queries in cache: %d, queries with >=1 positive: %d",
                len(scores_by_query),
                sum(1 for v in positives_by_query.values() if v))

    logger.info("Neg pool: %s", neg_path)
    neg_by_query = load_neg_pool(neg_path)
    logger.info("  queries in neg pool: %d, total neg rows: %d",
                len(neg_by_query), sum(len(v) for v in neg_by_query.values()))

    # Collect all unique positive ids to fetch.
    all_pos_ids = sorted({d for ids in positives_by_query.values() for d in ids})
    logger.info("Fetching %d unique positive docs from OpenSearch (%s)...",
                len(all_pos_ids), opensearch_url())
    session = requests.Session()
    pos_docs, missing = mget_docs(session, opensearch_url(), all_pos_ids)
    logger.info("  fetched %d, missing %d", len(pos_docs), len(missing))
    if missing:
        logger.warning("Missing positive ids (not in index): %s%s",
                       ", ".join(missing[:20]),
                       " ..." if len(missing) > 20 else "")

    triplets, stats = build_triplets(
        positives_by_query, scores_by_query, neg_by_query,
        pos_docs, args.max_neg_per_pos, rng,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        for t in triplets:
            out.write(json.dumps(t) + "\n")

    # ── Validation summary ──
    logger.info("=== Triplet build complete ===")
    logger.info("  output:                  %s", out_path)
    logger.info("  triplets:                %d", stats["triplets"])
    logger.info("  queries covered:         %d", stats["queries_emitted"])
    logger.info("  queries skipped (no pos):%d", stats["queries_skipped_no_positive"])
    logger.info("  queries skipped (no emit):%d", stats["queries_skipped_no_emit"])
    logger.info("  unique positives fetched/used: %d / %d",
                len(pos_docs), stats["unique_positives_used"])
    logger.info("  avg negs/positive:       %.3f", stats["avg_neg_per_positive"])
    logger.info("  missing OpenSearch ids:  %d", len(missing))

    # Spot-check: confirm a positive's text is non-empty.
    if triplets:
        sample = triplets[0]
        logger.info("--- sample triplet (truncated) ---")
        logger.info("  anchor:   %s", sample["anchor"][:90])
        logger.info("  positive: %s", sample["positive"][:120])
        logger.info("  negative: %s", (sample["negative"] or "<none>")[:120])
        logger.info("  subject:  %s", sample["subject"])
        logger.info("  metadata: %s", json.dumps(sample["metadata"]))
        spot_id = sample["metadata"]["positive_id"]
        logger.info("  SPOT-CHECK positive id %s text non-empty: %s",
                    spot_id, bool(sample["positive"].strip()))


if __name__ == "__main__":
    main()
