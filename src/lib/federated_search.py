"""Federated search router — Option C (Federated Hybrid).

Routes queries between live OpenAlex API and local OpenSearch index:
  - Fresh queries (within federated_recency_days, or explicit temporal cues) → LIVE_API
  - Historical queries → LOCAL_INDEX
  - When federated_both_enabled or both sources requested → BOTH (with RRF fusion + DOI dedup)

DOI deduplication uses normalized DOI strings (no https://doi.org/ prefix) as keys.
RRF fusion formula: score = Σ 1 / (k + rank_i) where k=60.
"""

import logging
import re
from datetime import date, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger("sfu_library_mcp")

_TEMPORAL_CUES = re.compile(
    r"\b(recent|latest|new|2025|2026|current|this year|last year|emerging)\b",
    re.IGNORECASE,
)

_RRF_K = 60


class SearchSource(str, Enum):
    LIVE_API = "live_api"
    LOCAL_INDEX = "local_index"
    LOCAL_RRF = "local_rrf"  # BM25F + SPLADE on local OpenSearch, fused via RRF
    BOTH = "both"  # live API + local (RRF if local_rrf_enabled else single-mode)


def _normalize_doi(doi: str) -> str:
    """Strip URL prefix from DOI for deduplication."""
    if not doi:
        return ""
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if doi.startswith(prefix):
            return doi[len(prefix):]
    return doi


def _rrf_merge(
    primary: list[dict],
    secondary: list[dict],
    top_k: int,
) -> list[dict]:
    """Merge two ranked lists with Reciprocal Rank Fusion, deduplicating by DOI.

    primary docs (typically live API) take precedence for metadata when DOI overlaps.
    """
    scores: dict[str, float] = {}
    by_doi: dict[str, dict] = {}

    def _key(doc: dict) -> str:
        return _normalize_doi(doc.get("doi", "")) or doc.get("openalex_id", "") or doc.get("title", "")

    for rank, doc in enumerate(primary, start=1):
        k = _key(doc)
        if k:
            scores[k] = scores.get(k, 0.0) + 1.0 / (_RRF_K + rank)
            if k not in by_doi:
                by_doi[k] = doc

    for rank, doc in enumerate(secondary, start=1):
        k = _key(doc)
        if k:
            scores[k] = scores.get(k, 0.0) + 1.0 / (_RRF_K + rank)
            if k not in by_doi:
                by_doi[k] = doc

    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    merged = []
    for k in ranked[:top_k]:
        doc = dict(by_doi[k])
        doc["rrf_score"] = scores[k]
        merged.append(doc)
    return merged


class FederatedSearchRouter:
    """Route academic queries between live OpenAlex API and local OpenSearch.

    Instantiated with references to both backends so it can be tested with mocks.
    """

    def __init__(
        self,
        openalex_client: Any,
        opensearch_retriever: Any,
        recency_days: int = 30,
        local_rrf_enabled: bool = True,
    ):
        self._openalex = openalex_client
        self._opensearch = opensearch_retriever
        self.recency_days = recency_days
        # When True, historical queries dispatch BM25F + SPLADE on the local
        # index and RRF-fuse. The 120-query Phase P.11 eval showed RRF beats
        # either retriever alone (+5.1% NDCG@10 vs BM25, avg overlap only 7.5%).
        self.local_rrf_enabled = local_rrf_enabled

    def route(self, query: str, filters: dict) -> SearchSource:
        """Determine which backend(s) to query.

        Rules (evaluated in order):
        1. from_publication_date within recency_days → LIVE_API
        2. Temporal cue words in query → LIVE_API
        3. Everything else → LOCAL_RRF (when local_rrf_enabled) else LOCAL_INDEX
        """
        from_date_str = filters.get("from_publication_date", "")
        if from_date_str:
            try:
                from_date = date.fromisoformat(from_date_str[:10])
                cutoff = date.today() - timedelta(days=self.recency_days)
                if from_date >= cutoff:
                    return SearchSource.LIVE_API
            except ValueError:
                pass

        if _TEMPORAL_CUES.search(query):
            return SearchSource.LIVE_API

        return SearchSource.LOCAL_RRF if self.local_rrf_enabled else SearchSource.LOCAL_INDEX

    def search(
        self,
        query: str,
        filters: dict,
        top_k: int = 50,
        force_source: SearchSource | None = None,
    ) -> list[dict]:
        """Dispatch to one or both backends and return a merged, deduplicated list.

        Args:
            query: raw query string
            filters: OpenAlex-style filter dict (e.g. from_publication_date, type)
            top_k: maximum results to return
            force_source: override automatic routing when set
        """
        source = force_source or self.route(query, filters)

        live_results: list[dict] = []
        local_results: list[dict] = []

        if source in (SearchSource.LIVE_API, SearchSource.BOTH):
            try:
                data = self._openalex.search_works(
                    query, filters=filters, per_page=top_k
                )
                live_results = data.get("results", [])
            except Exception:
                logger.exception("FederatedSearchRouter: live API search failed")

        if source in (SearchSource.LOCAL_INDEX, SearchSource.BOTH):
            try:
                local_results = self._opensearch.search(query, top_k=top_k)
            except Exception:
                logger.exception("FederatedSearchRouter: local OpenSearch search failed")

        if source == SearchSource.LOCAL_RRF:
            bm25f_results: list[dict] = []
            splade_results: list[dict] = []
            try:
                bm25f_results = self._opensearch.search(query, top_k=top_k, mode="bm25f")
            except Exception:
                logger.exception("FederatedSearchRouter: BM25F search failed")
            try:
                splade_results = self._opensearch.search(query, top_k=top_k, mode="splade")
            except Exception:
                logger.exception("FederatedSearchRouter: SPLADE search failed")
            return _rrf_merge(bm25f_results, splade_results, top_k)

        if source == SearchSource.LIVE_API:
            return live_results[:top_k]
        if source == SearchSource.LOCAL_INDEX:
            return local_results[:top_k]

        # BOTH — RRF merge of live + local
        return _rrf_merge(live_results, local_results, top_k)
