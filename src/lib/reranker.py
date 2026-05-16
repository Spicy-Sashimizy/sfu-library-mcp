"""Two-stage re-ranking for search results.

Scores documents on multiple signals and re-orders them to surface
the most useful results. Designed to run after the initial API ranking.
Handles both PNX (Primo) and OpenAlex flat doc shapes via _normalize_for_rerank.
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
    # TODO(Phase P.9): title_relevance is a patch for lacking query ownership; remove once
    # SPLADE ships and redistribute 0.15 to semantic_similarity — docs/SPLADE_OPENSEARCH_INTEGRATION_PLAN.md §P.9
    "title_relevance": 0.15,
    "recency": 0.15,
    "fulltext_available": 0.15,
    "type_match": 0.10,
    "completeness": 0.10,
}


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _normalize_for_rerank(doc: dict) -> dict:
    """Return a shape-stable view of a doc for scoring.

    Detects PNX (Primo) vs OpenAlex flat shape and extracts:
      title, abstract, year (int|None), type, doi, authors (list), has_fulltext (bool)
    """
    if doc.get("pnx"):
        pnx = doc["pnx"]
        display = pnx.get("display", {})
        addata = pnx.get("addata", {})
        links = pnx.get("links", {})
        delivery = pnx.get("delivery", {})

        titles = display.get("title", [])
        title = titles[0] if titles else ""

        descriptions = display.get("description", [])
        abstract = descriptions[0] if descriptions else ""

        dates = display.get("creationdate", [])
        date_str = dates[0] if dates else ""
        year: int | None = None
        if date_str:
            try:
                year = int(date_str[:4])
            except (ValueError, IndexError):
                pass

        types = display.get("type", [])
        doc_type = types[0].lower() if types else ""

        dois = addata.get("doi", [])
        doi = dois[0] if dois else ""

        authors = display.get("creator", [])

        availability = delivery.get("availability", [""])[0] if delivery.get("availability") else ""
        has_fulltext = (
            "available" in availability.lower()
            or bool(links.get("linktorsrc"))
            or bool(links.get("linktohtml"))
            or bool(links.get("linktopdf"))
        )
    elif "authorships" in doc:
        # Raw OpenAlex API shape (used in scripts/mining, not in production search path)
        title = doc.get("title") or ""
        abstract = doc.get("abstract") or ""
        year = doc.get("publication_year")
        doc_type = (doc.get("type") or "").lower()
        doi = doc.get("doi") or ""
        authors = [
            a["author"]["display_name"]
            for a in doc.get("authorships", [])
            if (a.get("author") or {}).get("display_name")
        ]
        oa = doc.get("open_access") or {}
        primary_loc = doc.get("primary_location") or {}
        has_fulltext = (
            bool(oa.get("is_oa"))
            or bool(primary_loc.get("pdf_url"))
            or bool(primary_loc.get("landing_page_url"))
        )
    else:
        # Normalized OpenAlex shape — output of normalize_work() in openalex.py.
        # This is the production path: tools.py passes normalize_work() output to reranker.
        # Keys differ from raw API: "date" not "publication_year", "authors" flat strings,
        # "is_oa" and "oa_url" not nested under "open_access"/"primary_location".
        title = doc.get("title") or ""
        abstract = doc.get("abstract") or ""
        date_str = doc.get("date") or ""
        year: int | None = None
        if date_str:
            try:
                year = int(str(date_str)[:4])
            except (ValueError, IndexError):
                pass
        doc_type = (doc.get("type") or "").lower()
        doi = doc.get("doi") or ""
        authors = doc.get("authors") or []
        has_fulltext = bool(doc.get("is_oa")) or bool(doc.get("oa_url"))

    return {
        "title": title,
        "abstract": abstract,
        "year": year,
        "type": doc_type,
        "doi": doi,
        "authors": authors,
        "has_fulltext": has_fulltext,
    }


def _score_title_relevance(norm: dict, query_tokens: set[str]) -> float:
    """Score 0-1 based on query term overlap with title."""
    title_tokens = _tokenize(norm["title"])
    if not query_tokens or not title_tokens:
        return 0.0
    overlap = query_tokens & title_tokens
    return len(overlap) / len(query_tokens)


def _score_recency(norm: dict) -> float:
    """Score 0-1 based on publication year. Current year = 1.0, decays 0.05/yr."""
    year = norm["year"]
    if year is None:
        return 0.0
    age = _CURRENT_YEAR - year
    return max(0.0, 1.0 - age * 0.05)


def _score_fulltext(norm: dict) -> float:
    """Score 1.0 if full text available, 0.0 otherwise."""
    return 1.0 if norm["has_fulltext"] else 0.0


def _score_type_match(norm: dict) -> float:
    """Score based on resource type. Peer-reviewed articles score highest."""
    doc_type = norm["type"]
    if doc_type in ("article", "journal_article", "review"):
        return 1.0
    elif doc_type in ("book", "book_chapter"):
        return 0.7
    elif doc_type in ("conference_proceeding", "dissertation"):
        return 0.6
    return 0.5


def _score_completeness(norm: dict) -> float:
    """Score 0-1 based on metadata completeness (DOI, authors, date)."""
    score = 0.0
    if norm["doi"]:
        score += 0.34
    if norm["authors"]:
        score += 0.33
    if norm["year"] is not None:
        score += 0.33
    return min(score, 1.0)


def _extract_doc_text(norm: dict) -> str:
    """Extract title + abstract text from a normalized doc for embedding."""
    return f"{norm['title']} {norm['abstract']}".strip()


def _compute_semantic_scores(query: str, docs: list[dict], model_path: str | None = None) -> list[float] | None:
    """Compute semantic similarity scores using local embedding model.

    Returns None if embedding model is unavailable (graceful fallback).
    """
    try:
        from lib.embedding import compute_similarity
    except ImportError:
        logger.debug("Embedding module not available, skipping semantic scoring")
        return None

    norms = [_normalize_for_rerank(doc) for doc in docs]
    doc_texts = [_extract_doc_text(n) for n in norms]
    if not any(doc_texts):
        return None

    scores = compute_similarity(query, doc_texts, model_path)
    if not scores:
        return None

    return scores


# RRF k for Primo-position + embedding fusion (distinct from OpenSearch BM25F+SPLADE RRF
# in federated_search.py which uses its own _RRF_K tuned separately via Q1.1 sweep).
_EMBEDDING_RRF_K = 60


def _compute_rrf_scores(semantic_scores: list[float], n_docs: int) -> list[float]:
    """Fuse original Primo doc order with embedding rank via Reciprocal Rank Fusion.

    Primo returns docs in BM25-derived relevance order, so the input position
    is itself a lexical-relevance signal. RRF combines two ranked lists:

        rrf(d) = 1 / (k + rank_lexical(d))  +  1 / (k + rank_embedding(d))

    Returns scores normalised to [0, 1] so they slot into the existing weighted
    sum in place of raw cosine similarity. Documents that rank well on either
    signal — or both — surface; neither ranker can fully suppress a doc the
    other rates highly. This is the property that rescues queries where one
    signal is blind (e.g. niche topics where embeddings fail but lexical hits).
    """
    embed_order = sorted(range(n_docs), key=lambda i: semantic_scores[i], reverse=True)
    embed_rank = [0] * n_docs
    for rank, idx in enumerate(embed_order):
        embed_rank[idx] = rank

    scores = [
        1.0 / (_EMBEDDING_RRF_K + i) + 1.0 / (_EMBEDDING_RRF_K + embed_rank[i])
        for i in range(n_docs)
    ]
    max_possible = 2.0 / _EMBEDDING_RRF_K  # both rankings put doc at position 0
    return [s / max_possible for s in scores]


_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None  # None = not yet tried; False = unavailable


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
            logger.info("CrossEncoder loaded: %s", _CROSS_ENCODER_MODEL)
        except Exception as e:
            logger.warning("CrossEncoder unavailable (%s), skipping second-pass rerank", e)
            _cross_encoder = False
    return _cross_encoder if _cross_encoder is not False else None


def rerank_with_crossencoder(docs: list[dict], query: str, limit: int) -> list[dict]:
    """Second-pass reranker: score top candidates with a cross-encoder.

    Runs only on the top-20 inputs to bound latency (~50-100ms on CPU).
    Falls back silently to `docs[:limit]` if the model is unavailable or errors.
    """
    ce = _get_cross_encoder()
    if ce is None:
        return docs[:limit]

    candidates = docs[:20]
    norms = [_normalize_for_rerank(d) for d in candidates]
    texts = [f"{n['title']} {n['abstract']}".strip() or n["title"] for n in norms]
    pairs = [(query, t) for t in texts]

    try:
        scores = ce.predict(pairs, show_progress_bar=False).tolist()
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        result = [d for _, d in ranked[:limit]]
        logger.info("crossencoder: scored %d candidates -> %d results", len(candidates), len(result))
        return result
    except Exception as e:
        logger.warning("CrossEncoder scoring failed (%s), using pre-CE order", e)
        return docs[:limit]


def rerank_results(
    docs: list[dict],
    query: str,
    limit: int,
    use_embedding: bool = True,
    use_rrf: bool = False,
    embedding_model_path: str | None = None,
) -> list[dict]:
    """Re-rank search results using weighted multi-signal scoring.

    Args:
        docs: List of document dicts (with pnx structure), in Primo's
            relevance-derived order (used as the lexical signal for RRF).
        query: The original search query for title relevance scoring.
        limit: Maximum number of results to return.
        use_embedding: Whether to use local embedding model for semantic scoring.
        use_rrf: Whether to fuse original doc order with embedding rank via
            Reciprocal Rank Fusion. Replaces raw cosine in the semantic weight
            slot. No-op when use_embedding is False or no embedding scores
            could be computed.
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

    # Optionally fuse with original lexical rank
    semantic_signal = semantic_scores
    if use_rrf and semantic_scores:
        semantic_signal = _compute_rrf_scores(semantic_scores, len(docs))
        logger.debug("RRF fusion active (k=%d)", _EMBEDDING_RRF_K)

    weights = _WEIGHTS_WITH_EMBEDDING if semantic_signal else _WEIGHTS_NO_EMBEDDING

    scored: list[tuple[float, int, dict]] = []
    for idx, doc in enumerate(docs):
        norm = _normalize_for_rerank(doc)
        title_score = _score_title_relevance(norm, query_tokens)
        recency_score = _score_recency(norm)
        fulltext_score = _score_fulltext(norm)
        type_score = _score_type_match(norm)
        completeness_score = _score_completeness(norm)

        composite = (
            weights["title_relevance"] * title_score
            + weights["recency"] * recency_score
            + weights["fulltext_available"] * fulltext_score
            + weights["type_match"] * type_score
            + weights["completeness"] * completeness_score
        )

        if semantic_signal:
            composite += weights["semantic_similarity"] * semantic_signal[idx]

        scored.append((composite, -idx, doc))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    result = [doc for _, _, doc in scored[:limit]]
    logger.info("rerank: rrf=%s embed=%s docs=%d -> %d", use_rrf, bool(semantic_scores), len(docs), limit)
    return result
