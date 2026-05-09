#!/usr/bin/env python3
"""Phase K: BM25-mined hard negatives for the existing clean triplets.

For each triplet (anchor, positive, subject), queries OpenAlex with the anchor
text, finds the highest-ranked result with a different primary subject (and that
isn't the positive), and emits a new triplet with that hard negative.

Doubles the strategies 1 & 3 triplet count. Results are cached in
data/hard_negatives_cache.json to avoid redundant API calls.

Usage:
    python -m scripts.mine_hard_negatives --max-total 5000
    python -m scripts.mine_hard_negatives --append-to data/sfu_training_triplets.clean.jsonl
"""
import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
CLEAN_TRIPLETS = REPO_ROOT / "data/sfu_training_triplets.clean.jsonl"
CACHE_FILE = REPO_ROOT / "data/hard_negatives_cache.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/hard_negative_triplets.jsonl"

OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")
HEADERS = {"User-Agent": "SFULibraryMCP/1.0 (mailto:lib-systems@sfu.ca)"}

# Strategies eligible for hard-negative augmentation
ELIGIBLE_STRATEGIES = {"citation_pairs", "provider_aware", "unknown"}


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _is_valid(text: str, min_tokens: int = 5) -> bool:
    return bool(text) and len(text.split()) >= min_tokens


def _load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def load_corpus(path: Path) -> list[dict]:
    triplets = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    return triplets


def _reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    positions = []
    for word, pos_list in inv_index.items():
        for p in pos_list:
            positions.append((p, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def fetch_openalex_for_anchor(anchor_text: str, top_k: int = 20) -> list[dict]:
    """Fetch top_k OpenAlex results for an anchor text. Returns list of works."""
    # Use first 100 chars of anchor as the query (avoid overly long BM25 queries)
    query = anchor_text[:150].split(".")[0].strip()
    if not query:
        return []

    params: dict = {
        "search": query,
        "per_page": top_k,
        "sort": "relevance_score:desc",
        "select": "id,title,publication_year,cited_by_count,concepts,abstract_inverted_index",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    else:
        params["mailto"] = "lib-systems@sfu.ca"

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
                logger.info("  429 from OpenAlex, backing off %.0fs", backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt == 4:
                logger.warning("OpenAlex fetch failed: %s", e)
                return []
            time.sleep(backoff)
            backoff *= 2
    else:
        return []

    works = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.pop("abstract_inverted_index", None))
        w["abstract"] = abstract
        if w.get("title"):
            works.append(w)
    return works


def _get_primary_subject(work: dict) -> str:
    """Extract primary subject label from OpenAlex concepts."""
    concepts = work.get("concepts", [])
    if not concepts:
        return ""
    # Highest-level concept (level 0 = field of study)
    top = sorted(concepts, key=lambda c: (-c.get("level", 99), -c.get("score", 0)))
    for c in top:
        if c.get("display_name"):
            return c["display_name"]
    return ""


def _work_to_text(work: dict) -> str:
    title = work.get("title", "")
    abstract = work.get("abstract", "")
    return f"{title}. {abstract}".strip()


def find_hard_negative(
    anchor_text: str,
    anchor_subject: str,
    positive_text: str,
    cache: dict,
    top_k: int = 20,
) -> str | None:
    """Return a hard negative paper text, or None if none found."""
    cache_key = _sha1(anchor_text)
    if cache_key in cache:
        return cache[cache_key] or None

    results = fetch_openalex_for_anchor(anchor_text, top_k=top_k)

    pos_sha = _sha1(positive_text[:200])
    hard_neg = None
    for work in results:
        text = _work_to_text(work)
        if not _is_valid(text, min_tokens=10):
            continue
        if _sha1(text[:200]) == pos_sha:
            continue

        # Accept if primary concept doesn't match anchor subject
        primary = _get_primary_subject(work).lower()
        if anchor_subject.lower() not in primary and primary not in anchor_subject.lower():
            hard_neg = text
            break

    # Cache even if None (to avoid re-fetching)
    cache[cache_key] = hard_neg or ""
    return hard_neg


def run(
    triplets: list[dict],
    output_path: Path,
    cache: dict,
    max_total: int,
    seed: int,
    append: bool,
    save_every: int = 50,
) -> int:
    rng = random.Random(seed)
    eligible = [t for t in triplets if t.get("strategy", "unknown") in ELIGIBLE_STRATEGIES]
    rng.shuffle(eligible)
    logger.info("Eligible triplets for hard-neg augmentation: %d / %d", len(eligible), len(triplets))

    count = 0
    seen_pairs: set[tuple[str, str]] = set()
    mode = "a" if append else "w"

    with output_path.open(mode) as out:
        for i, triplet in enumerate(eligible):
            if count >= max_total:
                break

            if (i + 1) % save_every == 0:
                logger.info("  [%d/%d] hard negs found: %d", i + 1, len(eligible), count)
                _save_cache(cache, CACHE_FILE)

            anchor = triplet.get("anchor", "")
            positive = triplet.get("positive", "")
            subject = triplet.get("subject", "Unknown")

            if not _is_valid(anchor, min_tokens=5):
                continue

            hard_neg = find_hard_negative(
                anchor_text=anchor,
                anchor_subject=subject,
                positive_text=positive,
                cache=cache,
            )
            if not hard_neg:
                continue

            pair_key = (_sha1(anchor), _sha1(hard_neg))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            new_triplet = {
                "anchor": anchor,
                "positive": positive,
                "negative": hard_neg[:4000],
                "strategy": f"{triplet.get('strategy', 'unknown')}_hard_neg",
                "subject": subject,
                "metadata": {
                    **triplet.get("metadata", {}),
                    "negative_source": "openalex_bm25",
                    "original_strategy": triplet.get("strategy", "unknown"),
                },
            }
            out.write(json.dumps(new_triplet) + "\n")
            count += 1

            # Polite delay between API calls
            time.sleep(0.5)

    _save_cache(cache, CACHE_FILE)
    return count


def main() -> None:
    global CACHE_FILE
    parser = argparse.ArgumentParser(
        description="Phase K: Mine hard negatives from OpenAlex for existing triplets"
    )
    parser.add_argument("--input", default=str(CLEAN_TRIPLETS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--append-to", default=None,
                        help="Append to existing JSONL instead of --output")
    parser.add_argument("--max-total", type=int, default=5000,
                        help="Max hard-negative triplets to generate")
    parser.add_argument("--top-k", type=int, default=20,
                        help="OpenAlex results to fetch per anchor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-file", default=str(CACHE_FILE))
    args = parser.parse_args()

    CACHE_FILE = Path(args.cache_file)
    cache = _load_cache(CACHE_FILE)
    logger.info("Hard-neg cache: %d entries", len(cache))

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    output_path = Path(args.append_to) if args.append_to else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    triplets = load_corpus(input_path)
    logger.info("Loaded %d triplets from %s", len(triplets), input_path)

    count = run(
        triplets=triplets,
        output_path=output_path,
        cache=cache,
        max_total=args.max_total,
        seed=args.seed,
        append=bool(args.append_to),
    )

    logger.info("Done. Wrote %d hard-negative triplets to %s", count, output_path)


if __name__ == "__main__":
    main()
