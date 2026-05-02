"""Two-stage re-ranking for search results.

Scores documents on multiple signals and re-orders them to surface
the most useful results. Designed to run after the initial API ranking.
"""

import re
import logging
from datetime import datetime

logger = logging.getLogger("sfu_library_mcp")

# Current year for recency scoring
_CURRENT_YEAR = datetime.now().year

# Scoring weights (must sum to 1.0)
_WEIGHTS = {
    "title_relevance": 0.35,
    "recency": 0.20,
    "fulltext_available": 0.20,
    "type_match": 0.15,
    "completeness": 0.10,
}


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score_title_relevance(doc: dict, query_tokens: set[str]) -> float:
    """Score 0-1 based on query term overlap with title."""
    pnx = doc.get("pnx", {})
    titles = pnx.get("display", {}).get("title", [])
    title = titles[0] if titles else ""
    title_tokens = _tokenize(title)
    if not query_tokens or not title_tokens:
        return 0.0
    overlap = query_tokens & title_tokens
    return len(overlap) / len(query_tokens)


def _score_recency(doc: dict) -> float:
    """Score 0-1 based on publication year. Current year = 1.0, decays 0.05/yr."""
    pnx = doc.get("pnx", {})
    dates = pnx.get("display", {}).get("creationdate", [])
    date_str = dates[0] if dates else ""
    if not date_str:
        return 0.0
    try:
        year = int(date_str[:4])
    except (ValueError, IndexError):
        return 0.0
    age = _CURRENT_YEAR - year
    return max(0.0, 1.0 - age * 0.05)


def _score_fulltext(doc: dict) -> float:
    """Score 1.0 if full text available, 0.0 otherwise."""
    pnx = doc.get("pnx", {})
    delivery = pnx.get("delivery", {})
    links = pnx.get("links", {})

    availability = delivery.get("availability", [""])[0] if delivery.get("availability") else ""
    if "available" in availability.lower():
        return 1.0

    if links.get("linktorsrc") or links.get("linktohtml") or links.get("linktopdf"):
        return 1.0

    return 0.0


def _score_type_match(doc: dict) -> float:
    """Score based on resource type. Peer-reviewed articles score highest."""
    pnx = doc.get("pnx", {})
    types = pnx.get("display", {}).get("type", [])
    doc_type = types[0].lower() if types else ""

    if doc_type in ("article", "journal_article", "review"):
        return 1.0
    elif doc_type in ("book", "book_chapter"):
        return 0.7
    elif doc_type in ("conference_proceeding", "dissertation"):
        return 0.6
    return 0.5


def _score_completeness(doc: dict) -> float:
    """Score 0-1 based on metadata completeness (DOI, authors, date)."""
    pnx = doc.get("pnx", {})
    addata = pnx.get("addata", {})
    display = pnx.get("display", {})

    score = 0.0
    # Has DOI
    if addata.get("doi") and addata["doi"][0]:
        score += 0.34
    # Has authors
    if display.get("creator") and display["creator"][0]:
        score += 0.33
    # Has date
    if display.get("creationdate") and display["creationdate"][0]:
        score += 0.33

    return min(score, 1.0)


def rerank_results(docs: list[dict], query: str, limit: int) -> list[dict]:
    """Re-rank search results using weighted multi-signal scoring.

    Args:
        docs: List of document dicts (with pnx structure).
        query: The original search query for title relevance scoring.
        limit: Maximum number of results to return.

    Returns:
        Re-ranked list of docs, sorted by composite score descending.
    """
    if not docs:
        return []

    query_tokens = _tokenize(query)

    scored: list[tuple[float, int, dict]] = []
    for idx, doc in enumerate(docs):
        title_score = _score_title_relevance(doc, query_tokens)
        recency_score = _score_recency(doc)
        fulltext_score = _score_fulltext(doc)
        type_score = _score_type_match(doc)
        completeness_score = _score_completeness(doc)

        composite = (
            _WEIGHTS["title_relevance"] * title_score
            + _WEIGHTS["recency"] * recency_score
            + _WEIGHTS["fulltext_available"] * fulltext_score
            + _WEIGHTS["type_match"] * type_score
            + _WEIGHTS["completeness"] * completeness_score
        )

        # Use negative idx as tiebreaker to preserve original Primo order
        scored.append((composite, -idx, doc))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return [doc for _, _, doc in scored[:limit]]
