#!/usr/bin/env python3
"""Phase I-bis D: Fetch papers from arXiv as an independent training data source.

arXiv API has no daily cap (only polite-use limits), no key required.
Maps SFU subject areas to arXiv category codes and generates triplets from
same-subject papers (positive) vs cross-subject papers (negative).

Usage:
    python -m scripts.arxiv_fetcher --output data/arxiv_triplets.jsonl
    python -m scripts.arxiv_fetcher --subjects "Computer Science" "Mathematics" --max-per-subject 200
"""
import argparse
import hashlib
import json
import logging
import random
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data/arxiv_triplets.jsonl"
ARXIV_CACHE = REPO_ROOT / "data/arxiv_cache.json"

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# SFU subject → arXiv category codes (primary mapping)
SUBJECT_TO_ARXIV: dict[str, list[str]] = {
    "Computer Science": ["cs.LG", "cs.AI", "cs.CV", "cs.NLP", "cs.IR", "cs.SE"],
    "Mathematics": ["math.NA", "math.ST", "math.CO", "math.PR", "math.OC"],
    "Physics": ["physics.comp-ph", "cond-mat", "quant-ph", "astro-ph"],
    "Statistics": ["stat.ML", "stat.ME", "stat.AP", "stat.TH"],
    "Biology": ["q-bio.GN", "q-bio.NC", "q-bio.PE", "q-bio.QM"],
    "Economics": ["econ.GN", "econ.EM", "econ.TH"],
    "Engineering": ["eess.SP", "eess.IV", "cs.RO", "cs.SY"],
    "Environmental Science": ["physics.ao-ph", "astro-ph.EP"],
    "Psychology": ["q-bio.NC", "cs.HC"],
    "Information Science": ["cs.IR", "cs.DL", "cs.DM"],
    "Biomedical Engineering": ["q-bio.BM", "cs.CE"],
    "Chemistry": ["physics.chem-ph", "q-bio.BM"],
    "Neuroscience": ["q-bio.NC", "cs.NE"],
    "Social Sciences": ["cs.SI", "econ.GN"],
    "Education": ["cs.HC", "cs.CY"],
    "Business": ["econ.GN", "cs.GT"],
    "Health Sciences": ["q-bio.GN", "q-bio.TO"],
    "Kinesiology": ["q-bio.NC", "physics.med-ph"],
    "Linguistics": ["cs.CL", "cs.AI"],
    "Philosophy": ["cs.AI", "math.LO"],
}

QUERY_TEMPLATES = [
    "{title_words} research",
    "papers on {title_words}",
    "{subject} {title_words}",
    "{title_words} methods",
    "study of {title_words}",
]


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


def fetch_arxiv(category: str, max_results: int = 100, start: int = 0) -> list[dict]:
    """Fetch papers from arXiv API for a category. Returns list of paper dicts."""
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": min(max_results, 100),  # arXiv caps at 100 per request
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"

    backoff = 3.0
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SFULibraryMCP/1.0 (mailto:lib-systems@sfu.ca)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
            break
        except Exception as e:
            if attempt == 3:
                logger.warning("arXiv fetch failed for %s: %s", category, e)
                return []
            time.sleep(backoff)
            backoff *= 2

    root = ET.fromstring(content)
    papers = []
    for entry in root.findall("atom:entry", NS):
        title_el = entry.find("atom:title", NS)
        summary_el = entry.find("atom:summary", NS)
        id_el = entry.find("atom:id", NS)
        published_el = entry.find("atom:published", NS)

        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
        abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
        arxiv_id = (id_el.text or "").strip() if id_el is not None else ""
        published = (published_el.text or "")[:10] if published_el is not None else ""

        authors = []
        for author in entry.findall("atom:author", NS):
            name_el = author.find("atom:name", NS)
            if name_el is not None:
                authors.append(name_el.text or "")

        if title and abstract:
            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published,
                "category": category,
                "full_text": f"{title}. {abstract}",
            })

    return papers


def fetch_subject_papers(subject: str, categories: list[str], max_per_subject: int, cache: dict) -> list[dict]:
    """Fetch papers for a subject via its arXiv categories, using cache."""
    cache_key = f"subject::{subject}::max{max_per_subject}"
    if cache_key in cache:
        return cache[cache_key]

    papers: list[dict] = []
    per_category = max(1, max_per_subject // len(categories))

    for cat in categories:
        cat_key = f"arxiv::{cat}::n{per_category}"
        if cat_key in cache:
            cat_papers = cache[cat_key]
        else:
            logger.info("  Fetching arXiv %s (%d papers)", cat, per_category)
            cat_papers = fetch_arxiv(cat, max_results=per_category)
            cache[cat_key] = cat_papers
            time.sleep(3)  # polite-use delay

        for p in cat_papers:
            p["subject"] = subject
        papers.extend(cat_papers)
        if len(papers) >= max_per_subject:
            break

    papers = papers[:max_per_subject]
    cache[cache_key] = papers
    return papers


def _extract_title_words(title: str, n: int = 6) -> str:
    stop = {"a", "an", "the", "of", "in", "on", "and", "or", "for", "to", "with", "from", "by"}
    words = [w for w in title.split() if w.lower() not in stop and len(w) > 2][:n]
    return " ".join(words).lower()


def _generate_queries(title: str, subject: str, rng: random.Random) -> list[str]:
    title_words = _extract_title_words(title)
    if not title_words:
        return []
    templates = rng.sample(QUERY_TEMPLATES, min(3, len(QUERY_TEMPLATES)))
    queries = []
    for tmpl in templates:
        q = tmpl.format(title_words=title_words, subject=subject).strip()
        if len(q.split()) >= 3:
            queries.append(q)
    return queries[:3]


def generate_triplets(
    subject_papers: dict[str, list[dict]],
    max_total: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    seen_pairs: set[tuple[str, str]] = set()
    triplets: list[dict] = []

    subjects = list(subject_papers.keys())
    paper_order = [(s, p) for s, papers in subject_papers.items() for p in papers]
    rng.shuffle(paper_order)

    for subject, paper in paper_order:
        if len(triplets) >= max_total:
            break

        full_text = paper.get("full_text", "")
        title = paper.get("title", "")
        if not _is_valid(full_text, min_tokens=15) or not title:
            continue

        paper_rng = random.Random(_sha1(full_text)[:8])
        queries = _generate_queries(title, subject, paper_rng)
        if not queries:
            continue

        # Positive: another paper from the same subject
        same_papers = [p for p in subject_papers.get(subject, []) if p is not paper]
        if not same_papers:
            continue

        # Negative: a paper from a different subject
        other_subjects = [s for s in subjects if s != subject]
        if not other_subjects:
            continue
        neg_subj = rng.choice(other_subjects)
        neg_paper = rng.choice(subject_papers[neg_subj])
        neg_text = neg_paper.get("full_text", "")

        for query in queries:
            if len(triplets) >= max_total:
                break

            pos_paper = rng.choice(same_papers)
            pos_text = pos_paper.get("full_text", "")

            if not _is_valid(pos_text, min_tokens=10) or not _is_valid(neg_text, min_tokens=10):
                continue

            pair_key = (_sha1(query), _sha1(pos_text))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            triplets.append({
                "anchor": query,
                "positive": pos_text[:4000],
                "negative": neg_text[:4000],
                "strategy": "arxiv",
                "subject": subject,
                "metadata": {
                    "source": "arxiv",
                    "anchor_arxiv_id": paper.get("arxiv_id", ""),
                    "positive_arxiv_id": pos_paper.get("arxiv_id", ""),
                    "category": paper.get("category", ""),
                },
            })

    return triplets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase I-bis D: arXiv paper fetcher for training data"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--append-to", default=None,
                        help="Append to existing JSONL instead of --output")
    parser.add_argument("--subjects", nargs="+",
                        default=list(SUBJECT_TO_ARXIV.keys()),
                        help="SFU subject areas to fetch (default: all mapped)")
    parser.add_argument("--max-per-subject", type=int, default=150,
                        help="Max papers to fetch per subject")
    parser.add_argument("--max-total", type=int, default=10000,
                        help="Max triplets to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-file", default=str(ARXIV_CACHE))
    args = parser.parse_args()

    cache_path = Path(args.cache_file)
    cache = _load_cache(cache_path)
    logger.info("Cache: %d entries", len(cache))

    output_path = Path(args.append_to) if args.append_to else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subject_papers: dict[str, list[dict]] = {}
    for subject in args.subjects:
        categories = SUBJECT_TO_ARXIV.get(subject)
        if not categories:
            logger.warning("No arXiv mapping for subject: %s — skipping", subject)
            continue
        logger.info("Fetching papers for: %s", subject)
        papers = fetch_subject_papers(subject, categories, args.max_per_subject, cache)
        if papers:
            subject_papers[subject] = papers
            logger.info("  %d papers for %s", len(papers), subject)
        _save_cache(cache, cache_path)

    if not subject_papers:
        logger.error("No papers fetched. Check network and subject names.")
        sys.exit(1)

    logger.info("Generating triplets from %d subjects...", len(subject_papers))
    triplets = generate_triplets(subject_papers, args.max_total, args.seed)
    logger.info("Generated %d triplets", len(triplets))

    mode = "a" if args.append_to else "w"
    with output_path.open(mode) as out:
        for t in triplets:
            out.write(json.dumps(t) + "\n")

    _save_cache(cache, cache_path)
    logger.info("Wrote %d arXiv triplets to %s", len(triplets), output_path)


if __name__ == "__main__":
    main()
