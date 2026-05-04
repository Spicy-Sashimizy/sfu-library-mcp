#!/usr/bin/env python3
"""Phase I-bis B: Re-mine the existing 8,724-triplet corpus for additional triplets.

For each paper text in the clean corpus, generate 2-3 template queries from
its title, find k-nearest same-subject papers by sentence-BERT cosine similarity
using the current fine-tuned model as additional positives, and pair with a
different-subject paper as negative.

Zero API calls. CPU-only (~30 min first run).

Usage:
    python -m scripts.remine_corpus --output data/remined_triplets.jsonl
    python -m scripts.remine_corpus --append-to data/sfu_training_triplets.clean.jsonl
"""
import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
CLEAN_TRIPLETS = REPO_ROOT / "data/sfu_training_triplets.clean.jsonl"
DEFAULT_MODEL = str(REPO_ROOT / "models/sfu-academic-embed-v3")
DEFAULT_OUTPUT = REPO_ROOT / "data/remined_triplets.jsonl"

PAPER_QUERY_TEMPLATES = [
    "{title_words} research",
    "papers about {title_words}",
    "{subject} {title_words}",
    "scholarly articles on {title_words}",
    "studies on {title_words}",
    "{title_words} academic literature",
]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _is_valid(text: str, min_tokens: int = 5) -> bool:
    return bool(text) and len(text.split()) >= min_tokens


def _extract_title(text: str) -> str:
    """Extract the title from a paper text (first sentence / period-delimited)."""
    text = text.strip()
    # First period that ends a sentence (not a middle-of-word period)
    for i, ch in enumerate(text):
        if ch == "." and i > 10:
            candidate = text[:i].strip()
            # Reject if title is longer than 200 chars (probably no title separator)
            if len(candidate) <= 200:
                return candidate
    return text[:150].strip()


def _generate_paper_queries(title: str, subject: str, rng: random.Random) -> list[str]:
    """Generate 2-3 short search queries from a paper title and subject."""
    # Take first 5-7 content words from title as a key phrase
    words = [w for w in title.split() if len(w) > 3][:7]
    title_words = " ".join(words).rstrip(".,;:").lower()
    if not title_words:
        return []

    templates = rng.sample(PAPER_QUERY_TEMPLATES, min(3, len(PAPER_QUERY_TEMPLATES)))
    queries = []
    for tmpl in templates:
        try:
            q = tmpl.format(title_words=title_words, subject=subject).strip()
            if len(q.split()) >= 3:
                queries.append(q)
        except KeyError:
            pass
    return queries[:3]


def load_corpus(path: Path) -> list[dict]:
    triplets = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    return triplets


def extract_papers(triplets: list[dict]) -> tuple[list[str], list[str]]:
    """Extract unique paper texts (positive field) with their subjects."""
    seen: dict[str, str] = {}  # sha1 -> subject
    text_map: dict[str, str] = {}  # sha1 -> text
    for t in triplets:
        pos = t.get("positive", "")
        subj = t.get("subject", "Unknown")
        if _is_valid(pos, min_tokens=10):
            h = _sha1(pos)
            if h not in seen:
                seen[h] = subj
                text_map[h] = pos
    hashes = list(seen.keys())
    texts = [text_map[h] for h in hashes]
    subjects = [seen[h] for h in hashes]
    logger.info("Extracted %d unique paper texts across %d subjects",
                len(texts), len(set(subjects)))
    return texts, subjects


def encode_papers(texts: list[str], model_path: str) -> np.ndarray:
    """Encode paper texts with a SentenceTransformer model. Returns (N, D) float32."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed; run: pip install sentence-transformers")
        sys.exit(1)

    logger.info("Loading model: %s", model_path)
    model = SentenceTransformer(model_path)

    logger.info("Encoding %d papers (batch_size=256, this may take a while)...", len(texts))
    embs = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.array(embs, dtype=np.float32)


def mine_triplets(
    texts: list[str],
    subjects: list[str],
    embeddings: np.ndarray,
    max_total: int,
    top_k_positives: int = 5,
    seed: int = 42,
) -> Iterator[dict]:
    """For each paper, generate queries and pair with kNN same-subject positives."""
    rng = random.Random(seed)
    subject_to_idxs: dict[str, list[int]] = {}
    for i, subj in enumerate(subjects):
        subject_to_idxs.setdefault(subj, []).append(i)

    seen_pairs: set[tuple[str, str]] = set()
    count = 0

    # Shuffle paper order for variety
    paper_order = list(range(len(texts)))
    rng.shuffle(paper_order)

    for idx in paper_order:
        if count >= max_total:
            break

        text = texts[idx]
        subject = subjects[idx]
        title = _extract_title(text)
        if not title or len(title.split()) < 3:
            continue

        paper_rng = random.Random(_sha1(text)[:8])
        queries = _generate_paper_queries(title, subject, paper_rng)
        if not queries:
            continue

        # Find top-k same-subject papers by cosine (embs already normalized)
        same_idxs = [i for i in subject_to_idxs.get(subject, []) if i != idx]
        if not same_idxs:
            continue

        query_emb = embeddings[idx]  # use paper itself as proxy for query
        same_embs = embeddings[same_idxs]
        sims = same_embs @ query_emb
        top_k = min(top_k_positives, len(same_idxs))
        top_pos_local = np.argsort(sims)[::-1][:top_k]
        pos_candidates = [same_idxs[i] for i in top_pos_local]

        # Negative: random paper from a different subject
        other_subjects = [s for s in subject_to_idxs if s != subject]
        if not other_subjects:
            continue
        neg_subj = rng.choice(other_subjects)
        neg_idx = rng.choice(subject_to_idxs[neg_subj])
        neg_text = texts[neg_idx]

        for pos_idx in pos_candidates:
            if count >= max_total:
                break
            pos_text = texts[pos_idx]

            for query in queries:
                if count >= max_total:
                    break

                pair_key = (_sha1(query), _sha1(pos_text))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                yield {
                    "anchor": query,
                    "positive": pos_text[:4000],
                    "negative": neg_text[:4000],
                    "strategy": "corpus_remine",
                    "subject": subject,
                    "metadata": {
                        "source": "corpus_remine",
                        "anchor_paper_sha1": _sha1(text)[:12],
                        "positive_sha1": _sha1(pos_text)[:12],
                        "cosine_sim": float(embeddings[idx] @ embeddings[pos_idx]),
                    },
                }
                count += 1


def run_quality_check(path: Path) -> None:
    """Basic validity check on output file."""
    total = 0
    short_anchor = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            total += 1
            if len(t.get("anchor", "").split()) < 3:
                short_anchor += 1
    logger.info("Quality check: %d triplets, %d with short anchors", total, short_anchor)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase I-bis B: Re-mine corpus for additional zero-API triplets"
    )
    parser.add_argument("--input", default=str(CLEAN_TRIPLETS),
                        help="Clean triplets JSONL to mine from")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output JSONL file for remined triplets")
    parser.add_argument("--append-to", default=None,
                        help="Append directly to an existing JSONL instead of --output")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="SentenceTransformer model path for similarity")
    parser.add_argument("--max-total", type=int, default=5000,
                        help="Maximum triplets to generate")
    parser.add_argument("--top-k-positives", type=int, default=5,
                        help="kNN positives per anchor paper")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_path = Path(args.append_to) if args.append_to else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading corpus from %s", input_path)
    triplets = load_corpus(input_path)
    logger.info("Loaded %d triplets", len(triplets))

    texts, subjects = extract_papers(triplets)

    embeddings = encode_papers(texts, args.model)

    mode = "a" if args.append_to else "w"
    count = 0
    with output_path.open(mode) as out:
        for triplet in mine_triplets(
            texts, subjects, embeddings,
            max_total=args.max_total,
            top_k_positives=args.top_k_positives,
            seed=args.seed,
        ):
            out.write(json.dumps(triplet) + "\n")
            count += 1

    logger.info("Wrote %d remined triplets to %s", count, output_path)
    run_quality_check(output_path if not args.append_to else output_path)


if __name__ == "__main__":
    main()
