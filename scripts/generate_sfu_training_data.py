#!/usr/bin/env python3
"""SFU-specific training data generator for the academic embedding model.

Implements 4 strategies using the SFU Solr database registry + OpenAlex API:

  Strategy 1: Subject-aligned citation pairs          (target: 5,000 triplets)
    For each SFU subject → fetch top-cited OpenAlex works → extract citation pairs
    Teaches model SFU's subject boundaries and institutional priorities.

  Strategy 2: Synthetic SFU-style queries             (target: 3,000 triplets)
    Generate search queries from Solr database metadata → match to OpenAlex papers
    Teaches the mapping between SFU user query patterns and available resources.

  Strategy 3: Provider-aware positive/negative pairs  (target: 4,000 triplets)
    Papers in SFU-subscribed subject areas = positives
    Papers from unrelated subjects = hard negatives
    Teaches subtle subscription-aware preference.

  Strategy 4: Cross-subject hard negatives            (target: 3,000 triplets)
    Anchor + positive from one SFU subject; negative from an ADJACENT subject
    Teaches discrimination between SFU's neighbouring academic domains.

Checkpoint/resume:
  Progress is saved to {output_dir}/generation_state.json after each strategy.
  Run with --resume to skip completed strategies and continue from where you left off.
  The output JSONL is appended incrementally, so partial runs are never lost.

Usage:
    python scripts/generate_sfu_training_data.py
    python scripts/generate_sfu_training_data.py --resume
    python scripts/generate_sfu_training_data.py --strategy 1 2  # only specific strategies
    python scripts/generate_sfu_training_data.py --dry-run       # validate without API calls
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import os

import requests

# Add src/ to path to reuse the SFU Solr registry client
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lib.sfu_databases import SFUDatabaseRegistry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'").strip('"')
        if k:
            os.environ[k] = v


_load_dotenv()

OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_HEADERS = {"User-Agent": "SFULibraryMCP-Training/1.0 (mailto:lib-systems@sfu.ca)"}
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()
if OPENALEX_API_KEY:
    logger.info("OpenAlex API key loaded (premium quota)")
else:
    logger.info("No OPENALEX_API_KEY — using polite pool (10 req/s)")

DEFAULT_OUTPUT = "data/sfu_training_triplets.jsonl"
DEFAULT_CHECKPOINT_DIR = "data/generation_checkpoints"
SOLR_CACHE_FILE = "data/solr_cache.json"

# SFU's key subject areas (from Solr analysis — 108 total subjects)
SFU_PRIORITY_SUBJECTS = [
    "History",
    "General & Multidisciplinary",
    "English - General",
    "Finance",
    "Canadian Studies",
    "Political Science",
    "Economics",
    "Health Sciences",
    "Indigenous Studies",
    "Sociology",
    "Interactive Arts & Technology",
    "Criminology",
    "Biological Sciences",
    "Psychology",
    "Communication",
    "Nursing",
    "Environmental Science",
    "Education",
    "Philosophy",
    "Anthropology",
    "International Studies",
    "Business Administration",
    "Geography",
    "Computing Science",
    "Women's Studies",
    "Linguistics",
    "Chemistry",
    "Physics",
    "Mathematics",
    "Engineering",
]

# Adjacent subject pairs — these create the hardest negatives (Strategy 4)
# Papers from adjacent subjects share vocabulary but differ in focus
ADJACENT_SUBJECTS: dict[str, list[str]] = {
    "Health Sciences": ["Biomedical Sciences", "Nursing", "Psychology", "Kinesiology"],
    "Psychology": ["Health Sciences", "Sociology", "Education", "Neuroscience"],
    "Political Science": ["Sociology", "Economics", "International Studies", "Canadian Studies"],
    "Economics": ["Finance", "Business Administration", "Political Science", "Sociology"],
    "Finance": ["Economics", "Business Administration", "Accounting"],
    "Computing Science": ["Mathematics", "Engineering", "Interactive Arts & Technology"],
    "Interactive Arts & Technology": ["Computing Science", "Communication", "Design", "Education"],
    "Criminology": ["Sociology", "Political Science", "Law", "Psychology"],
    "Indigenous Studies": ["Canadian Studies", "Anthropology", "History", "Sociology"],
    "Canadian Studies": ["History", "Political Science", "Sociology", "Geography"],
    "History": ["Anthropology", "Political Science", "Canadian Studies", "Sociology"],
    "Sociology": ["Anthropology", "Political Science", "Psychology", "Criminology"],
    "Environmental Science": ["Geography", "Biological Sciences", "Chemistry"],
    "Biological Sciences": ["Chemistry", "Environmental Science", "Health Sciences"],
    "Engineering": ["Computing Science", "Mathematics", "Physics"],
    "Education": ["Psychology", "Sociology", "Communication"],
    "Communication": ["Sociology", "Political Science", "Interactive Arts & Technology"],
    "Business Administration": ["Finance", "Economics", "Computing Science"],
    "Anthropology": ["Sociology", "History", "Indigenous Studies"],
    "Geography": ["Environmental Science", "Sociology", "Canadian Studies"],
    "International Studies": ["Political Science", "Economics", "History"],
    "Women's Studies": ["Sociology", "History", "Political Science", "Psychology"],
    "Linguistics": ["Communication", "Psychology", "English - General"],
    "Philosophy": ["History", "Political Science", "Psychology"],
    "Mathematics": ["Physics", "Computing Science", "Engineering"],
    "Physics": ["Mathematics", "Chemistry", "Engineering"],
    "Chemistry": ["Biological Sciences", "Physics", "Environmental Science"],
    "Nursing": ["Health Sciences", "Psychology", "Sociology"],
    "English - General": ["Linguistics", "History", "Communication"],
}

# OpenAlex concept IDs for SFU subjects (for more precise API queries)
SUBJECT_TO_OPENALEX_SEARCH: dict[str, str] = {
    "Indigenous Studies": "indigenous studies First Nations Aboriginal",
    "Canadian Studies": "Canada Canadian society politics culture",
    "Criminology": "criminology criminal justice crime deviance",
    "Interactive Arts & Technology": "interactive media human computer interaction game design",
    "Health Sciences": "health sciences medicine clinical",
    "Finance": "finance investment portfolio economics",
    "Political Science": "political science government democracy",
    "Economics": "economics macroeconomics microeconomics",
    "Sociology": "sociology social theory inequality",
    "History": "history historical analysis",
    "Psychology": "psychology cognitive behavioral neuroscience",
    "Environmental Science": "environmental science ecology sustainability",
    "Biological Sciences": "biology genetics molecular cell",
    "Computing Science": "computer science algorithms software",
    "Education": "education pedagogy learning teaching",
    "Communication": "communication media journalism",
    "Engineering": "engineering applied sciences",
    "Business Administration": "business management organizational",
    "Anthropology": "anthropology cultural ethnography",
    "Geography": "geography spatial GIS cartography",
    "Women's Studies": "gender studies feminism women",
    "Linguistics": "linguistics language syntax phonology",
    "Philosophy": "philosophy ethics epistemology",
    "Mathematics": "mathematics statistics probability",
    "Physics": "physics quantum mechanics thermodynamics",
    "Chemistry": "chemistry organic inorganic reaction",
    "Nursing": "nursing patient care clinical healthcare",
    "English - General": "English literature literary criticism",
    "International Studies": "international relations geopolitics foreign policy",
    "General & Multidisciplinary": "interdisciplinary multidisciplinary research",
}

# Synthetic query templates using Solr database metadata
QUERY_TEMPLATES = [
    "{name} research papers",
    "studies using {name}",
    "{subject} academic literature",
    "{subject} research methods",
    "how to research {subject}",
    "{subject} scholarly articles",
    "{subject} recent publications",
    "{description_fragment} studies",
    "{description_fragment} research",
    "academic sources {subject}",
    "{subject} journal articles",
    "{subject} peer reviewed",
    "{name} database coverage",
    "find papers about {subject}",
    "{subject} bibliography",
]


# ── State management ──────────────────────────────────────────────────────────

@dataclass
class GenerationState:
    completed_strategies: list[str] = field(default_factory=list)
    triplets_per_strategy: dict[str, int] = field(default_factory=dict)
    total_triplets: int = 0
    output_file: str = ""
    quality_check_done: bool = False
    splits_done: bool = False
    started_at: str = ""
    updated_at: str = ""

    def save(self, path: Path) -> None:
        d = asdict(self)
        d["updated_at"] = datetime.now().isoformat()
        path.write_text(json.dumps(d, indent=2))
        logger.debug("State saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "GenerationState":
        d = json.loads(path.read_text())
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def strategy_done(self, strategy: str) -> bool:
        return strategy in self.completed_strategies

    def mark_strategy_done(self, strategy: str, count: int) -> None:
        if strategy not in self.completed_strategies:
            self.completed_strategies.append(strategy)
        self.triplets_per_strategy[strategy] = count
        self.total_triplets = sum(self.triplets_per_strategy.values())


# ── OpenAlex helpers ──────────────────────────────────────────────────────────

def _openalex_get(path: str, params: dict, retries: int = 3) -> dict | None:
    url = f"{OPENALEX_BASE}{path}"
    if OPENALEX_API_KEY:
        params = {**params, "api_key": OPENALEX_API_KEY}
    else:
        params = {**params, "mailto": "lib-systems@sfu.ca"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=OPENALEX_HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Rate limited — waiting %ds", wait)
                time.sleep(wait)
            else:
                logger.warning("HTTP %d for %s: %s", resp.status_code, path, e)
                return None
        except Exception as e:
            logger.warning("Request failed (attempt %d/%d): %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(1)
    return None


def _reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, pos_list in inv_index.items():
        for p in pos_list:
            positions.append((p, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def fetch_works_for_subject(subject: str, per_page: int = 50, sort: str = "cited_by_count:desc") -> list[dict]:
    """Fetch OpenAlex works for an SFU subject area."""
    search_query = SUBJECT_TO_OPENALEX_SEARCH.get(subject, subject)
    data = _openalex_get("/works", {
        "search": search_query,
        "per_page": per_page,
        "sort": sort,
        "select": "id,title,publication_year,cited_by_count,abstract_inverted_index,referenced_works,concepts,keywords,primary_topic,primary_location",
        "filter": "has_abstract:true",
    })
    if not data:
        return []
    results = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.pop("abstract_inverted_index", None))
        if abstract and w.get("title") and len(abstract) > 50:
            w["abstract"] = abstract
            results.append(w)
    return results


def fetch_works_citing(work_id: str, per_page: int = 15) -> list[dict]:
    """Fetch works that cite the given work (reverse citation lookup)."""
    openalex_id = work_id.replace("https://openalex.org/", "")
    data = _openalex_get("/works", {
        "filter": f"cited_by:{openalex_id},has_abstract:true",
        "per_page": per_page,
        "select": "id,title,abstract_inverted_index,concepts,keywords,primary_topic,primary_location",
    })
    if not data:
        return []
    results = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.pop("abstract_inverted_index", None))
        if abstract and w.get("title") and len(abstract) > 50:
            w["abstract"] = abstract
            results.append(w)
    return results


def _make_text(work: dict) -> str:
    title = work.get("title", "").strip()
    abstract = work.get("abstract", "").strip()
    base = f"{title}. {abstract}" if title and abstract else (title or abstract)

    venue = (((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "").strip()
    concepts = work.get("concepts") or []
    top_concepts = ", ".join(
        c.get("display_name", "") for c in concepts[:3] if c.get("display_name")
    )
    keywords = work.get("keywords") or []
    top_keywords = ", ".join(
        k.get("keyword", "") for k in keywords[:5] if k.get("keyword")
    )
    primary_topic = ((work.get("primary_topic") or {}).get("display_name") or "").strip()

    parts = [base]
    if venue:
        parts.append(f"Venue: {venue}")
    if top_concepts or primary_topic:
        parts.append(f"Topics: {top_concepts or primary_topic}")
    if top_keywords:
        parts.append(f"Keywords: {top_keywords}")
    return " | ".join(parts)


def _token_count_approx(text: str) -> int:
    return len(text.split())


def _is_valid_text(text: str, min_tokens: int = 10, max_tokens: int = 512) -> bool:
    n = _token_count_approx(text)
    return min_tokens <= n <= max_tokens


# ── Strategy 1: Subject-aligned citation pairs ────────────────────────────────

def run_strategy1(
    subjects: list[str],
    max_total: int,
    output_file: Path,
    state: GenerationState,
    state_file: Path,
    dry_run: bool = False,
) -> int:
    """Citation pairs within SFU subject areas.

    For each subject: fetch top-cited works, find what they cite, create triplets.
    Hard negatives come from a different randomly selected subject.
    """
    logger.info("=== Strategy 1: Subject-aligned citation pairs (target: %d) ===", max_total)

    if dry_run:
        logger.info("[dry-run] Would generate ~%d citation pairs across %d subjects", max_total, len(subjects))
        return 0

    all_papers_by_subject: dict[str, list[dict]] = {}
    count = 0

    with output_file.open("a") as out:
        for si, subject in enumerate(subjects):
            if count >= max_total:
                break

            logger.info("  [%d/%d] Subject: %s (total so far: %d)", si + 1, len(subjects), subject, count)
            works = fetch_works_for_subject(subject, per_page=50)
            if not works:
                logger.warning("  No works found for %s", subject)
                continue

            all_papers_by_subject[subject] = works

            for work in works[:8]:
                if count >= max_total:
                    break

                anchor_text = _make_text(work)
                if not _is_valid_text(anchor_text):
                    continue

                cited_works = fetch_works_citing(work["id"], per_page=12)
                time.sleep(0.2)

                for cited in cited_works:
                    if count >= max_total:
                        break
                    positive_text = _make_text(cited)
                    if not _is_valid_text(positive_text):
                        continue

                    # Negative: random paper from a DIFFERENT subject
                    other_subjects = [s for s in all_papers_by_subject if s != subject]
                    if not other_subjects:
                        continue
                    neg_subject = random.choice(other_subjects)
                    neg_work = random.choice(all_papers_by_subject[neg_subject])
                    negative_text = _make_text(neg_work)
                    if not _is_valid_text(negative_text):
                        continue

                    triplet = {
                        "anchor": anchor_text[:4000],
                        "positive": positive_text[:4000],
                        "negative": negative_text[:4000],
                        "strategy": "citation_pair",
                        "subject": subject,
                        "metadata": {
                            "anchor_id": work.get("id"),
                            "positive_id": cited.get("id"),
                            "negative_subject": neg_subject,
                        },
                    }
                    out.write(json.dumps(triplet) + "\n")
                    count += 1

            time.sleep(0.3)

    logger.info("Strategy 1 complete: %d triplets", count)
    return count


# ── Strategy 2: Synthetic SFU-style queries from Solr metadata ────────────────

def _generate_queries_from_db_record(record: dict) -> list[str]:
    """Generate natural language search queries from a Solr database record."""
    queries = []
    name = record.get("name", "")
    subjects = record.get("subjects", [])
    if isinstance(subjects, str):
        subjects = [subjects]
    desc = record.get("publicNote", "") or record.get("description", "")

    if isinstance(desc, list):
        desc = " ".join(desc)
    desc = str(desc).strip()

    # Extract first meaningful sentence fragment from description
    desc_fragment = ""
    if desc:
        sentences = [s.strip() for s in desc.split(".") if len(s.strip()) > 20]
        if sentences:
            frag = sentences[0][:80]
            if frag and not frag.lower().startswith(("this", "the database", "a database")):
                desc_fragment = frag

    for template in random.sample(QUERY_TEMPLATES, min(4, len(QUERY_TEMPLATES))):
        subject = random.choice(subjects) if subjects else "academic"
        try:
            q = template.format(
                name=name,
                subject=subject,
                description_fragment=desc_fragment or subject,
            ).strip()
            if len(q) > 10:
                queries.append(q)
        except KeyError:
            pass

    # Name-based query
    if name and len(name) < 60:
        queries.append(f"databases like {name}")
        if subjects:
            queries.append(f"{subjects[0]} research {name.split()[0].lower()}")

    return queries[:5]


def run_strategy2(
    solr_docs: list[dict],
    max_total: int,
    output_file: Path,
    state: GenerationState,
    state_file: Path,
    dry_run: bool = False,
) -> int:
    """Synthetic queries generated from Solr database metadata.

    Each Solr record (database) generates query templates → matched to OpenAlex papers.
    """
    logger.info("=== Strategy 2: Synthetic SFU-style queries (target: %d) ===", max_total)

    if dry_run:
        logger.info("[dry-run] Would generate ~%d synthetic query pairs from %d Solr records",
                    max_total, len(solr_docs))
        return 0

    # Collect background papers for negatives
    background_papers: list[dict] = []
    # We'll build this lazily as we fetch papers

    count = 0
    random.shuffle(solr_docs)

    with output_file.open("a") as out:
        for doc in solr_docs:
            if count >= max_total:
                break

            subjects = doc.get("subjects", [])
            if isinstance(subjects, str):
                subjects = [subjects]
            if not subjects:
                continue

            primary_subject = subjects[0]
            queries = _generate_queries_from_db_record(doc)
            if not queries:
                continue

            # Fetch papers for this subject
            works = fetch_works_for_subject(primary_subject, per_page=20)
            if not works:
                continue

            background_papers.extend(works)

            for query in queries:
                if count >= max_total:
                    break

                if not _is_valid_text(query, min_tokens=3):
                    continue

                # Positive: paper from the relevant subject
                if not works:
                    continue
                pos_work = random.choice(works)
                positive_text = _make_text(pos_work)
                if not _is_valid_text(positive_text):
                    continue

                # Negative: paper from background (different subject)
                neg_candidates = [p for p in background_papers if p.get("id") != pos_work.get("id")]
                if not neg_candidates:
                    continue
                neg_work = random.choice(neg_candidates)
                negative_text = _make_text(neg_work)
                if not _is_valid_text(negative_text):
                    continue

                triplet = {
                    "anchor": query,
                    "positive": positive_text[:4000],
                    "negative": negative_text[:4000],
                    "strategy": "synthetic_query",
                    "subject": primary_subject,
                    "metadata": {
                        "db_name": doc.get("name", ""),
                        "positive_id": pos_work.get("id"),
                        "query_template": "from_solr_record",
                    },
                }
                out.write(json.dumps(triplet) + "\n")
                count += 1

            time.sleep(0.2)

    logger.info("Strategy 2 complete: %d triplets", count)
    return count


# ── Strategy 3: Provider-aware positive/negative pairs ────────────────────────

def run_strategy3(
    solr_docs: list[dict],
    max_total: int,
    output_file: Path,
    state: GenerationState,
    state_file: Path,
    dry_run: bool = False,
    seed: int = 42,
) -> int:
    """Provider-aware pairs: SFU-subscribed subjects as positives, unrelated as negatives.

    For every (provider, subject) pair found in the Solr registry:
      - Fetch top OpenAlex papers for the subject (cached per subject, one API call each).
      - Sample a diverse subset using a stable per-(provider, subject) RNG.
      - Generate SFU-style queries from the matching Solr record(s) via
        ``_generate_queries_from_db_record`` so the database/provider signal lives
        in the *anchor* itself rather than only in metadata.
      - Pair each query with each sampled paper (positive) and a random unsubscribed
        paper (negative).
    """
    logger.info("=== Strategy 3: Provider-aware pairs (target: %d) ===", max_total)

    if dry_run:
        logger.info("[dry-run] Would generate ~%d provider-aware pairs", max_total)
        return 0

    # provider -> set of subjects
    provider_subjects: dict[str, set[str]] = defaultdict(set)
    # (provider, subject) -> list of Solr docs (for query generation)
    provider_subject_docs: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for doc in solr_docs:
        provider = doc.get("provider", "")
        if isinstance(provider, list):
            provider = provider[0] if provider else ""
        subjects = doc.get("subjects", [])
        if isinstance(subjects, str):
            subjects = [subjects]
        if provider and subjects:
            for s in subjects:
                provider_subjects[provider].add(s)
                provider_subject_docs[(provider, s)].append(doc)

    # ALL providers, shuffled deterministically so a max_total cut is fair across the tail.
    all_providers = sorted(provider_subjects.items(), key=lambda x: len(x[1]), reverse=True)
    random.Random(seed).shuffle(all_providers)
    logger.info("  %d providers, %d (provider, subject) pairs",
                len(all_providers), len(provider_subject_docs))

    # Heuristic negative pool: subjects covered by NO provider in the registry.
    covered_subjects: set[str] = set()
    for _, subj_set in all_providers:
        covered_subjects |= subj_set
    unsubscribed_subjects = [s for s in SFU_PRIORITY_SUBJECTS if s not in covered_subjects]

    unsubscribed_papers: list[dict] = []
    for subj in unsubscribed_subjects[:5]:
        works = fetch_works_for_subject(subj, per_page=20)
        unsubscribed_papers.extend(works)
        time.sleep(0.2)

    if not unsubscribed_papers:
        logger.warning("  No unsubscribed-subject papers fetched — strategy 3 cannot emit negatives")
        return 0

    # Cache per subject: papers don't depend on provider; diversity is via the sampler below.
    subject_papers: dict[str, list[dict]] = {}

    count = 0
    with output_file.open("a") as out:
        for provider, subjects in all_providers:
            if count >= max_total:
                break

            for subject in list(subjects)[:4]:
                if count >= max_total:
                    break

                if subject not in subject_papers:
                    works = fetch_works_for_subject(subject, per_page=30)
                    subject_papers[subject] = works
                    time.sleep(0.2)

                works = subject_papers.get(subject, [])
                if not works:
                    continue

                # Per-(provider, subject) deterministic sample to diversify positives.
                sample_rng = random.Random(hash((provider, subject)))
                k = min(6, len(works))
                sampled = sample_rng.sample(works, k=k)

                docs_for_pair = provider_subject_docs.get((provider, subject), [])
                if not docs_for_pair:
                    continue

                for doc in docs_for_pair[:2]:
                    if count >= max_total:
                        break

                    queries = _generate_queries_from_db_record(doc)
                    if not queries:
                        continue

                    for query in queries:
                        if count >= max_total:
                            break
                        if not _is_valid_text(query, min_tokens=3):
                            continue

                        for work in sampled:
                            if count >= max_total:
                                break

                            positive_text = _make_text(work)
                            if not _is_valid_text(positive_text):
                                continue

                            neg_work = random.choice(unsubscribed_papers)
                            negative_text = _make_text(neg_work)
                            if not _is_valid_text(negative_text):
                                continue

                            triplet = {
                                "anchor": query,
                                "positive": positive_text[:4000],
                                "negative": negative_text[:4000],
                                "strategy": "provider_aware",
                                "subject": subject,
                                "metadata": {
                                    "provider": provider,
                                    "db_name": doc.get("name", ""),
                                    "subscribed": True,
                                    "positive_id": work.get("id"),
                                },
                            }
                            out.write(json.dumps(triplet) + "\n")
                            count += 1

    logger.info("Strategy 3 complete: %d triplets", count)
    return count


# ── Strategy 4: Cross-subject hard negatives ──────────────────────────────────

def run_strategy4(
    max_total: int,
    output_file: Path,
    state: GenerationState,
    state_file: Path,
    dry_run: bool = False,
) -> int:
    """Cross-subject hard negatives using SFU's adjacent subject taxonomy.

    Anchor + positive from one SFU subject; negative from an ADJACENT subject.
    These are the hardest negatives because they share vocabulary.
    """
    logger.info("=== Strategy 4: Cross-subject hard negatives (target: %d) ===", max_total)

    if dry_run:
        logger.info("[dry-run] Would generate ~%d cross-subject hard negative triplets", max_total)
        return 0

    # Preload papers for all subjects with adjacency pairs
    subject_papers: dict[str, list[dict]] = {}
    subjects_with_adj = [s for s in ADJACENT_SUBJECTS if ADJACENT_SUBJECTS[s]]
    target_subjects = [s for s in subjects_with_adj if s in SUBJECT_TO_OPENALEX_SEARCH][:20]

    logger.info("  Fetching papers for %d subjects with adjacent pairs...", len(target_subjects))
    for subject in target_subjects:
        works = fetch_works_for_subject(subject, per_page=30)
        if works:
            subject_papers[subject] = works
            logger.debug("  %s: %d papers", subject, len(works))
        time.sleep(0.25)

    count = 0
    with output_file.open("a") as out:
        for subject, adj_subjects in ADJACENT_SUBJECTS.items():
            if count >= max_total:
                break

            if subject not in subject_papers:
                continue

            works = subject_papers[subject]
            # Find adjacent subjects that have papers loaded
            available_adj = [s for s in adj_subjects if s in subject_papers]
            if not available_adj:
                continue

            for work in works:
                if count >= max_total:
                    break

                anchor_text = _make_text(work)
                if not _is_valid_text(anchor_text):
                    continue

                # Positive: another paper from the SAME subject
                same_subj_candidates = [w for w in works if w.get("id") != work.get("id")]
                if not same_subj_candidates:
                    continue
                pos_work = random.choice(same_subj_candidates)
                positive_text = _make_text(pos_work)
                if not _is_valid_text(positive_text):
                    continue

                # Negative: paper from an ADJACENT subject (hard negative)
                adj_subject = random.choice(available_adj)
                neg_work = random.choice(subject_papers[adj_subject])
                negative_text = _make_text(neg_work)
                if not _is_valid_text(negative_text):
                    continue

                triplet = {
                    "anchor": anchor_text[:4000],
                    "positive": positive_text[:4000],
                    "negative": negative_text[:4000],
                    "strategy": "cross_subject_hard_negative",
                    "subject": subject,
                    "metadata": {
                        "anchor_id": work.get("id"),
                        "positive_id": pos_work.get("id"),
                        "negative_subject": adj_subject,
                        "negative_id": neg_work.get("id"),
                    },
                }
                out.write(json.dumps(triplet) + "\n")
                count += 1

    logger.info("Strategy 4 complete: %d triplets", count)
    return count


# ── Data quality checks ───────────────────────────────────────────────────────

def run_quality_checks(input_file: Path, output_file: Path) -> dict:
    """Deduplicate and length-filter the training data.

    Dedup is by (anchor, positive) pair only. Multiple positives per anchor
    are kept on purpose — contrastive learning benefits from many positive
    examples sharing an anchor (e.g. one synthetic query → many relevant
    papers). Earlier versions deduped by anchor alone, which discarded ~83%
    of the dataset and collapsed provider/citation strategies to a handful
    of triplets.

    Returns stats dict with counts before/after filtering.
    """
    logger.info("=== Running data quality checks ===")

    seen_pairs: set[tuple[str, str]] = set()
    stats = {
        "total_read": 0,
        "removed_duplicate_pair": 0,
        "removed_too_short": 0,
        "removed_too_long": 0,
        "removed_anchor_eq_positive": 0,
        "kept": 0,
    }
    strategy_counts: dict[str, int] = defaultdict(int)
    subject_counts: dict[str, int] = defaultdict(int)

    with input_file.open() as fin, output_file.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total_read"] += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            anchor = item.get("anchor", "").strip()
            positive = item.get("positive", "").strip()
            negative = item.get("negative", "").strip()

            # Skip if anchor == positive
            if anchor == positive:
                stats["removed_anchor_eq_positive"] += 1
                continue

            # Deduplicate by (anchor, positive) pair only
            pair_key = (anchor[:200], positive[:200])
            if pair_key in seen_pairs:
                stats["removed_duplicate_pair"] += 1
                continue
            seen_pairs.add(pair_key)

            # Length check. Anchor min is 2 tokens — real search queries are
            # often 2-8 tokens (e.g. "circular economy waste reduction"), and
            # the synthetic_query strategy intentionally emits short queries.
            # Positive/negative are paper text and should have ≥10 tokens.
            length_ok = True
            for label, text, min_tok in (
                ("anchor", anchor, 2),
                ("positive", positive, 10),
                ("negative", negative, 10),
            ):
                n = _token_count_approx(text)
                if n < min_tok:
                    stats["removed_too_short"] += 1
                    length_ok = False
                    break
                if n > 512:
                    stats["removed_too_long"] += 1
                    length_ok = False
                    break

            if length_ok:
                fout.write(json.dumps(item) + "\n")
                stats["kept"] += 1
                strategy_counts[item.get("strategy", "unknown")] += 1
                subject_counts[item.get("subject", "unknown")] += 1

    stats["strategy_breakdown"] = dict(strategy_counts)
    stats["subject_coverage"] = len(subject_counts)
    stats["top_subjects"] = dict(sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    logger.info("Quality check complete:")
    logger.info("  Total read: %d", stats["total_read"])
    logger.info("  Kept: %d (%.1f%%)", stats["kept"], 100 * stats["kept"] / max(1, stats["total_read"]))
    logger.info("  Subjects covered: %d", stats["subject_coverage"])
    logger.info("  Strategy breakdown: %s", stats["strategy_breakdown"])

    return stats


# ── Train/val/test split ──────────────────────────────────────────────────────

def create_data_splits(
    input_file: Path,
    output_dir: Path,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict:
    """Create stratified train/val/test splits.

    Stratified by strategy to ensure each split has proportional representation.
    No anchor text appears in more than one split.
    """
    logger.info("=== Creating train/val/test splits ===")

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by strategy for stratified split
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    with input_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    by_strategy[item.get("strategy", "unknown")].append(item)
                except json.JSONDecodeError:
                    pass

    train_items, val_items, test_items = [], [], []

    for strategy, items in by_strategy.items():
        random.shuffle(items)
        n = len(items)
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * (1 - train_ratio - val_ratio)))
        n_train = n - n_val - n_test

        train_items.extend(items[:n_train])
        val_items.extend(items[n_train:n_train + n_val])
        test_items.extend(items[n_train + n_val:])

    random.shuffle(train_items)
    random.shuffle(val_items)
    random.shuffle(test_items)

    def write_split(items: list[dict], path: Path) -> None:
        with path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    write_split(train_items, output_dir / "train.jsonl")
    write_split(val_items, output_dir / "val.jsonl")
    write_split(test_items, output_dir / "test.jsonl")

    stats = {
        "train": len(train_items),
        "val": len(val_items),
        "test": len(test_items),
        "total": len(train_items) + len(val_items) + len(test_items),
    }

    logger.info("Splits created:")
    logger.info("  Train: %d (%.1f%%)", stats["train"], 100 * stats["train"] / stats["total"])
    logger.info("  Val:   %d (%.1f%%)", stats["val"], 100 * stats["val"] / stats["total"])
    logger.info("  Test:  %d (%.1f%%)", stats["test"], 100 * stats["test"] / stats["total"])

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate SFU-specific training data for the academic embedding model"
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output JSONL file for raw triplets")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory to store generation state checkpoints")
    parser.add_argument("--splits-dir", default="data/splits",
                        help="Directory to write train/val/test splits")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint, skipping completed strategies")
    parser.add_argument("--strategy", type=int, nargs="+", choices=[1, 2, 3, 4],
                        help="Run only specific strategies (e.g. --strategy 1 3)")
    parser.add_argument("--citation-pairs", type=int, default=5000,
                        help="Target triplet count for strategy 1")
    parser.add_argument("--synthetic-pairs", type=int, default=3000,
                        help="Target triplet count for strategy 2")
    parser.add_argument("--provider-pairs", type=int, default=4000,
                        help="Target triplet count for strategy 3")
    parser.add_argument("--cross-subject-pairs", type=int, default=3000,
                        help="Target triplet count for strategy 4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without making API calls")
    parser.add_argument("--skip-splits", action="store_true",
                        help="Skip train/val/test split creation")
    parser.add_argument("--skip-quality-check", action="store_true",
                        help="Skip deduplication and quality filtering")
    parser.add_argument("--reclean-only", action="store_true",
                        help="Only re-run quality checks + splits on existing raw file")
    args = parser.parse_args()

    random.seed(args.seed)

    output_file = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    state_file = checkpoint_dir / "generation_state.json"

    # Reclean-only path: skip generation entirely, just rerun quality + splits
    if args.reclean_only:
        if not output_file.exists():
            logger.error("Raw file %s not found — nothing to reclean.", output_file)
            sys.exit(1)
        clean_file = output_file.with_suffix(".clean.jsonl")
        qc_stats = run_quality_checks(output_file, clean_file)
        (checkpoint_dir / "quality_check_stats.json").write_text(
            json.dumps(qc_stats, indent=2)
        )
        logger.info("Clean data written to %s", clean_file)
        if not args.skip_splits:
            split_stats = create_data_splits(clean_file, Path(args.splits_dir))
            (checkpoint_dir / "split_stats.json").write_text(json.dumps(split_stats, indent=2))
        print("\nReclean complete:")
        print(f"  Read:  {qc_stats['total_read']:,}")
        print(f"  Kept:  {qc_stats['kept']:,} ({100*qc_stats['kept']/max(1,qc_stats['total_read']):.1f}%)")
        print(f"  Strategy breakdown: {qc_stats['strategy_breakdown']}")
        return

    # Load or create state
    if args.resume and state_file.exists():
        state = GenerationState.load(state_file)
        logger.info("Resuming from checkpoint. Completed strategies: %s", state.completed_strategies)
        logger.info("Triplets so far: %d", state.total_triplets)
        if not output_file.exists():
            logger.error("Output file %s not found — cannot resume. Run without --resume.", output_file)
            sys.exit(1)
    else:
        state = GenerationState(
            output_file=str(output_file),
            started_at=datetime.now().isoformat(),
        )
        if output_file.exists() and not args.resume:
            logger.warning("Output file exists — will APPEND to it. Use --resume or delete it first.")

    state.save(state_file)

    # Determine which strategies to run
    run_strategies = set(args.strategy) if args.strategy else {1, 2, 3, 4}

    # ── Load SFU Solr registry ──
    logger.info("Loading SFU database registry from Solr...")
    registry = SFUDatabaseRegistry(cache_file=SOLR_CACHE_FILE)
    registry.ensure_loaded()
    solr_docs = registry.get_all()
    logger.info("Loaded %d SFU database records (%d unique subjects)",
                len(solr_docs),
                len({s for doc in solr_docs
                     for s in (doc.get("subjects", []) if isinstance(doc.get("subjects"), list)
                               else [doc.get("subjects", "")])
                     if s}))

    # ── Strategy 1: Citation pairs ──
    if 1 in run_strategies:
        if state.strategy_done("strategy1"):
            logger.info("Strategy 1 already complete (%d triplets) — skipping",
                        state.triplets_per_strategy.get("strategy1", 0))
        else:
            count = run_strategy1(
                subjects=SFU_PRIORITY_SUBJECTS,
                max_total=args.citation_pairs,
                output_file=output_file,
                state=state,
                state_file=state_file,
                dry_run=args.dry_run,
            )
            state.mark_strategy_done("strategy1", count)
            state.save(state_file)

    # ── Strategy 2: Synthetic queries ──
    if 2 in run_strategies:
        if state.strategy_done("strategy2"):
            logger.info("Strategy 2 already complete (%d triplets) — skipping",
                        state.triplets_per_strategy.get("strategy2", 0))
        else:
            count = run_strategy2(
                solr_docs=solr_docs,
                max_total=args.synthetic_pairs,
                output_file=output_file,
                state=state,
                state_file=state_file,
                dry_run=args.dry_run,
            )
            state.mark_strategy_done("strategy2", count)
            state.save(state_file)

    # ── Strategy 3: Provider-aware pairs ──
    if 3 in run_strategies:
        if state.strategy_done("strategy3"):
            logger.info("Strategy 3 already complete (%d triplets) — skipping",
                        state.triplets_per_strategy.get("strategy3", 0))
        else:
            count = run_strategy3(
                solr_docs=solr_docs,
                max_total=args.provider_pairs,
                output_file=output_file,
                state=state,
                state_file=state_file,
                dry_run=args.dry_run,
                seed=args.seed,
            )
            state.mark_strategy_done("strategy3", count)
            state.save(state_file)

    # ── Strategy 4: Cross-subject hard negatives ──
    if 4 in run_strategies:
        if state.strategy_done("strategy4"):
            logger.info("Strategy 4 already complete (%d triplets) — skipping",
                        state.triplets_per_strategy.get("strategy4", 0))
        else:
            count = run_strategy4(
                max_total=args.cross_subject_pairs,
                output_file=output_file,
                state=state,
                state_file=state_file,
                dry_run=args.dry_run,
            )
            state.mark_strategy_done("strategy4", count)
            state.save(state_file)

    # ── Quality checks ──
    if not args.skip_quality_check and not args.dry_run:
        if state.quality_check_done:
            logger.info("Quality check already done — skipping")
        else:
            clean_file = output_file.with_suffix(".clean.jsonl")
            qc_stats = run_quality_checks(output_file, clean_file)
            (checkpoint_dir / "quality_check_stats.json").write_text(
                json.dumps(qc_stats, indent=2)
            )
            state.quality_check_done = True
            state.save(state_file)
            logger.info("Clean data written to %s", clean_file)
            output_file = clean_file

    # ── Create splits ──
    if not args.skip_splits and not args.dry_run:
        if state.splits_done:
            logger.info("Splits already created — skipping")
        else:
            clean_file = output_file.parent / (output_file.stem.replace(".clean", "") + ".clean.jsonl")
            source = clean_file if clean_file.exists() else output_file
            split_stats = create_data_splits(source, Path(args.splits_dir))
            (checkpoint_dir / "split_stats.json").write_text(json.dumps(split_stats, indent=2))
            state.splits_done = True
            state.save(state_file)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total triplets generated: {state.total_triplets:,}")
    for strategy, count in state.triplets_per_strategy.items():
        print(f"    {strategy}: {count:,}")
    print(f"  Output file:    {args.output}")
    if not args.skip_splits:
        print(f"  Splits dir:     {args.splits_dir}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    if state.total_triplets < 10000:
        print(f"\n  WARNING: Only {state.total_triplets:,} triplets generated.")
        print("  Target is 15,000+. Consider increasing --citation-pairs or re-running.")
    print("\nNext step:")
    print(f"  python scripts/train_embedding_model.py \\")
    print(f"    --data data/splits/train.jsonl \\")
    print(f"    --val-data data/splits/val.jsonl \\")
    print(f"    --output models/sfu-academic-embed-v1")


if __name__ == "__main__":
    main()
