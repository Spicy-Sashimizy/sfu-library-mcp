"""Two-stage re-ranking for search results.

Scores documents on multiple signals and re-orders them to surface
the most useful results. Designed to run after the initial API ranking.
"""

import re
import logging
from datetime import datetime

logger = logging.getLogger("sfu_library_mcp")

_CURRENT_YEAR = datetime.now().year

# Scoring weights (must sum to 1.0)
# When semantic embedding is enabled, weights are redistributed
_WEIGHTS_NO_EMBEDDING = {
    "title_relevance": 0.35,
    "recency": 0.20,
    "fulltext_available": 0.20,
    "type_match": 0.15,
    "completeness": 0.10,
}

_WEIGHTS_WITH_EMBEDDING = {
    "semantic_similarity": 0.35,
    "title_relevance": 0.15,
    "recency": 0.15,
    "fulltext_available": 0.15,
    "type_match": 0.10,
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
    if addata.get("doi") and addata["doi"][0]:
        score += 0.34
    if display.get("creator") and display["creator"][0]:
        score += 0.33
    if display.get("creationdate") and display["creationdate"][0]:
        score += 0.33

    return min(score, 1.0)


def _extract_doc_text(doc: dict) -> str:
    """Extract title + description text from a document for embedding."""
    pnx = doc.get("pnx", {})
    display = pnx.get("display", {})
    titles = display.get("title", [])
    descriptions = display.get("description", [])
    title = titles[0] if titles else ""
    desc = descriptions[0] if descriptions else ""
    return f"{title} {desc}".strip()


def _compute_semantic_scores(query: str, docs: list[dict], model_path: str | None = None) -> list[float] | None:
    """Compute semantic similarity scores using local embedding model.

    Returns None if embedding model is unavailable (graceful fallback).
    """
    try:
        from lib.embedding import compute_similarity
    except ImportError:
        logger.debug("Embedding module not available, skipping semantic scoring")
        return None

    doc_texts = [_extract_doc_text(doc) for doc in docs]
    if not any(doc_texts):
        return None

    scores = compute_similarity(query, doc_texts, model_path)
    if not scores:
        return None

    return scores


def rerank_results(
    docs: list[dict],
    query: str,
    limit: int,
    use_embedding: bool = True,
    embedding_model_path: str | None = None,
) -> list[dict]:
    """Re-rank search results using weighted multi-signal scoring.

    Args:
        docs: List of document dicts (with pnx structure).
        query: The original search query for title relevance scoring.
        limit: Maximum number of results to return.
        use_embedding: Whether to use local embedding model for semantic scoring.
        embedding_model_path: Custom path to embedding model (None = default).

    Returns:
        Re-ranked list of docs, sorted by composite score descending.
    """
    if not docs:
        return []

    query_tokens = _tokenize(query)

    # Try semantic scoring if enabled
    semantic_scores = None
    if use_embedding:
        semantic_scores = _compute_semantic_scores(query, docs, embedding_model_path)
        if semantic_scores:
            logger.debug("Semantic reranking active (%d docs scored)", len(semantic_scores))

    weights = _WEIGHTS_WITH_EMBEDDING if semantic_scores else _WEIGHTS_NO_EMBEDDING

    scored: list[tuple[float, int, dict]] = []
    for idx, doc in enumerate(docs):
        title_score = _score_title_relevance(doc, query_tokens)
        recency_score = _score_recency(doc)
        fulltext_score = _score_fulltext(doc)
        type_score = _score_type_match(doc)
        completeness_score = _score_completeness(doc)

        composite = (
            weights["title_relevance"] * title_score
            + weights["recency"] * recency_score
            + weights["fulltext_available"] * fulltext_score
            + weights["type_match"] * type_score
            + weights["completeness"] * completeness_score
        )

        if semantic_scores:
            composite += weights["semantic_similarity"] * semantic_scores[idx]

        scored.append((composite, -idx, doc))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return [doc for _, _, doc in scored[:limit]]
