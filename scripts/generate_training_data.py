#!/usr/bin/env python3
"""Generate training data for fine-tuning an academic embedding model.

Produces (query, positive_doc, negative_doc) triplets from:
1. OpenAlex citation pairs (self-supervised, no labeling needed)
2. Synthetic queries generated from paper titles/abstracts
3. Related works pairs from OpenAlex metadata

Output: JSON lines file suitable for sentence-transformers training.

Usage:
    python scripts/generate_training_data.py --output data/training_triplets.jsonl
    python scripts/generate_training_data.py --citation-pairs 5000 --synthetic-pairs 2000
    python scripts/generate_training_data.py --topics "machine learning,climate change,genomics"
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
HEADERS = {"User-Agent": "SFULibraryMCP-Training/1.0 (mailto:lib-systems@sfu.ca)"}

ACADEMIC_TOPICS = [
    "machine learning", "climate change", "genomics", "public health",
    "renewable energy", "artificial intelligence", "neuroscience",
    "biodiversity", "quantum computing", "materials science",
    "epidemiology", "ocean science", "cybersecurity", "education",
    "sustainable agriculture", "drug discovery", "robotics",
    "social media", "urban planning", "environmental science",
    "cancer research", "natural language processing", "ecology",
    "data science", "psychology", "economics", "political science",
    "chemistry", "physics", "mathematics", "sociology", "anthropology",
    "computer vision", "bioinformatics", "marine biology",
]

QUERY_TEMPLATES = [
    "{topic} recent advances",
    "how does {topic} work",
    "{topic} review paper",
    "{topic} applications in {domain}",
    "challenges in {topic}",
    "{topic} methods comparison",
    "impact of {topic} on {domain}",
    "{topic} future directions",
    "systematic review {topic}",
    "{topic} experimental results",
]

DOMAINS = [
    "healthcare", "agriculture", "education", "industry",
    "environment", "society", "technology", "policy",
]


def _reconstruct_abstract(inv_index: dict) -> str:
    if not inv_index:
        return ""
    word_positions = []
    for word, positions in inv_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def _openalex_get(path: str, params: dict) -> dict | None:
    try:
        resp = requests.get(
            f"{OPENALEX_BASE}{path}",
            params=params,
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("OpenAlex request failed: %s", e)
        return None


def fetch_works_by_topic(topic: str, per_page: int = 50) -> list[dict]:
    """Fetch highly-cited works for a topic."""
    data = _openalex_get("/works", {
        "search": topic,
        "per_page": per_page,
        "sort": "cited_by_count:desc",
        "select": "id,doi,title,publication_year,cited_by_count,abstract_inverted_index,referenced_works",
    })
    if not data:
        return []

    results = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.pop("abstract_inverted_index", None))
        if abstract and w.get("title"):
            w["abstract"] = abstract
            results.append(w)
    return results


def fetch_work_references(work_id: str, per_page: int = 25) -> list[dict]:
    """Fetch referenced works for a given work (citation-based pairs)."""
    openalex_id = work_id.replace("https://openalex.org/", "")
    data = _openalex_get(f"/works", {
        "filter": f"cited_by:{openalex_id}",
        "per_page": per_page,
        "select": "id,title,abstract_inverted_index,cited_by_count",
    })
    if not data:
        return []

    results = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.pop("abstract_inverted_index", None))
        if abstract and w.get("title"):
            w["abstract"] = abstract
            results.append(w)
    return results


def generate_citation_pairs(
    topics: list[str],
    pairs_per_topic: int = 200,
    max_total: int = 5000,
) -> list[dict]:
    """Generate (anchor, positive, negative) triplets from citation relationships.

    If paper A references paper B, then:
    - anchor = A's title + abstract
    - positive = B's title + abstract
    - negative = random unrelated paper's title + abstract
    """
    logger.info("Generating citation-based training pairs...")
    all_papers = []
    pairs = []

    for ti, topic in enumerate(topics):
        if len(pairs) >= max_total:
            break

        logger.info("  [%d/%d] Topic: %s", ti + 1, len(topics), topic)
        works = fetch_works_by_topic(topic, per_page=50)
        if not works:
            continue

        all_papers.extend(works)

        for work in works[:10]:
            if len(pairs) >= max_total:
                break

            refs = fetch_work_references(work["id"], per_page=15)
            if not refs:
                continue

            anchor_text = f"{work['title']} {work['abstract']}"

            for ref in refs:
                if len(pairs) >= max_total:
                    break

                positive_text = f"{ref['title']} {ref['abstract']}"
                pairs.append({
                    "anchor": anchor_text,
                    "positive": positive_text,
                    "source": "citation",
                    "anchor_id": work["id"],
                    "positive_id": ref["id"],
                })

            time.sleep(0.15)

        time.sleep(0.2)

    # Add hard negatives (random papers from different topics)
    if all_papers and pairs:
        logger.info("  Adding hard negatives to %d pairs...", len(pairs))
        for pair in pairs:
            neg_paper = random.choice(all_papers)
            while neg_paper["id"] == pair.get("anchor_id") or neg_paper["id"] == pair.get("positive_id"):
                neg_paper = random.choice(all_papers)
            pair["negative"] = f"{neg_paper['title']} {neg_paper['abstract']}"

    logger.info("  Generated %d citation pairs", len(pairs))
    return pairs


def generate_synthetic_query_pairs(
    topics: list[str],
    pairs_per_topic: int = 100,
    max_total: int = 2000,
) -> list[dict]:
    """Generate (synthetic_query, paper) pairs using template-based queries.

    Creates natural-language search queries that should match specific papers.
    """
    logger.info("Generating synthetic query-paper pairs...")
    pairs = []
    all_papers = []

    for ti, topic in enumerate(topics):
        if len(pairs) >= max_total:
            break

        logger.info("  [%d/%d] Topic: %s", ti + 1, len(topics), topic)
        works = fetch_works_by_topic(topic, per_page=30)
        if not works:
            continue

        all_papers.extend(works)

        for work in works:
            if len(pairs) >= max_total:
                break

            title = work.get("title", "")
            abstract = work.get("abstract", "")
            positive_text = f"{title} {abstract}"

            # Generate diverse queries from the paper's content
            key_phrases = _extract_key_phrases(title, abstract)
            for phrase in key_phrases[:3]:
                pairs.append({
                    "anchor": phrase,
                    "positive": positive_text,
                    "source": "synthetic_keyphrase",
                })

            # Template-based queries
            template = random.choice(QUERY_TEMPLATES)
            domain = random.choice(DOMAINS)
            query = template.format(topic=topic, domain=domain)
            pairs.append({
                "anchor": query,
                "positive": positive_text,
                "source": "synthetic_template",
            })

        time.sleep(0.2)

    # Add hard negatives
    if all_papers and pairs:
        logger.info("  Adding hard negatives to %d synthetic pairs...", len(pairs))
        for pair in pairs:
            neg_paper = random.choice(all_papers)
            pair["negative"] = f"{neg_paper['title']} {neg_paper['abstract']}"

    logger.info("  Generated %d synthetic pairs", len(pairs))
    return pairs


def _extract_key_phrases(title: str, abstract: str) -> list[str]:
    """Extract key phrases from title/abstract for synthetic query generation.

    Uses simple heuristics — first noun phrase from title, key sentence fragments.
    """
    phrases = []

    # Title-based queries
    if title:
        phrases.append(title)
        words = title.split()
        if len(words) > 4:
            mid = len(words) // 2
            phrases.append(" ".join(words[:mid]))
            phrases.append(" ".join(words[mid:]))

    # Abstract sentence fragments
    if abstract:
        sentences = abstract.split(". ")
        if sentences:
            first = sentences[0].strip()
            if 10 < len(first) < 200:
                phrases.append(first)
        if len(sentences) > 1:
            last = sentences[-1].strip()
            if 10 < len(last) < 200:
                phrases.append(last)

    return phrases


def generate_related_works_pairs(
    topics: list[str],
    max_total: int = 3000,
) -> list[dict]:
    """Generate pairs from OpenAlex 'related_works' metadata."""
    logger.info("Generating related-works pairs...")
    pairs = []

    for ti, topic in enumerate(topics):
        if len(pairs) >= max_total:
            break

        logger.info("  [%d/%d] Topic: %s", ti + 1, len(topics), topic)
        works = fetch_works_by_topic(topic, per_page=25)

        for work in works:
            if len(pairs) >= max_total:
                break

            referenced = work.get("referenced_works", [])
            if not referenced:
                continue

            # Fetch a few referenced works with abstracts
            for ref_id in referenced[:3]:
                ref_short = ref_id.replace("https://openalex.org/", "")
                data = _openalex_get(f"/works/{ref_short}", {
                    "select": "id,title,abstract_inverted_index",
                })
                if not data or not data.get("title"):
                    continue

                ref_abstract = _reconstruct_abstract(data.get("abstract_inverted_index"))
                if not ref_abstract:
                    continue

                pairs.append({
                    "anchor": f"{work['title']} {work['abstract']}",
                    "positive": f"{data['title']} {ref_abstract}",
                    "source": "related_works",
                })

                time.sleep(0.1)

        time.sleep(0.2)

    logger.info("  Generated %d related-works pairs", len(pairs))
    return pairs


def save_training_data(pairs: list[dict], output_path: Path):
    """Save training triplets as JSON lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for pair in pairs:
            line = {
                "anchor": pair["anchor"],
                "positive": pair["positive"],
                "source": pair.get("source", "unknown"),
            }
            if "negative" in pair:
                line["negative"] = pair["negative"]
            f.write(json.dumps(line) + "\n")

    logger.info("Saved %d training pairs to %s", len(pairs), output_path)

    # Stats
    sources = {}
    for p in pairs:
        s = p.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1

    print(f"\nTraining Data Summary:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  With negatives: {sum(1 for p in pairs if 'negative' in p)}")
    print(f"  Sources:")
    for source, count in sorted(sources.items()):
        print(f"    {source}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Generate training data for academic embedding model")
    parser.add_argument("--output", type=str, default="data/training_triplets.jsonl")
    parser.add_argument("--citation-pairs", type=int, default=5000)
    parser.add_argument("--synthetic-pairs", type=int, default=2000)
    parser.add_argument("--related-pairs", type=int, default=3000)
    parser.add_argument("--topics", type=str, default=None, help="Comma-separated list of topics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    topics = args.topics.split(",") if args.topics else ACADEMIC_TOPICS
    topics = [t.strip() for t in topics]

    all_pairs = []

    # Strategy 1: Citation-based pairs
    citation_pairs = generate_citation_pairs(topics, max_total=args.citation_pairs)
    all_pairs.extend(citation_pairs)

    # Strategy 2: Synthetic query-paper pairs
    synthetic_pairs = generate_synthetic_query_pairs(topics, max_total=args.synthetic_pairs)
    all_pairs.extend(synthetic_pairs)

    # Strategy 3: Related works pairs
    related_pairs = generate_related_works_pairs(topics, max_total=args.related_pairs)
    all_pairs.extend(related_pairs)

    # Shuffle
    random.shuffle(all_pairs)

    save_training_data(all_pairs, Path(args.output))


if __name__ == "__main__":
    main()
