"""Federated search router — Option C (Federated Hybrid).

Routes queries between live OpenAlex API and local OpenSearch index:
  - Fresh queries (within federated_recency_days, or explicit temporal cues) → LIVE_API
  - Zero-coverage subjects (no local index data) → LIVE_API (Q2.1)
  - Soft-live subjects (OpenAlex beats local RRF) → LIVE_API unless overridden (Q2.2)
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

# Q2.1 — Zero-coverage subjects. These 15 subjects have 0.0 local NDCG@10 in the
# 120-query LLM-judged benchmark; the local index holds no useful documents for
# them, so a LOCAL_RRF route would return blank results. Always route live.
ALWAYS_LIVE_SUBJECTS = {
    "Theatre", "Music", "Urban Studies", "Applied Legal Studies",
    "Publishing", "Management & Organizational Studies", "Accounting",
    "Forensics", "Statistics & Actuarial Science",
    "Sustainable Energy Engineering (SEE)",
    "Sustainable Community Development",
    "Visual Arts", "Public Policy",
    "Molecular Biology & Biochemistry", "Global Health",
}

# Q2.2 — Soft-live subjects. These have real local coverage, but OpenAlex live
# scores higher than local RRF (Anthropology: RRF 0.898 vs OpenAlex 0.931).
# Prefer live by default, but allow the caller to override back to local via
# the `prefer_local` flag (e.g. for offline/air-gapped runs or when live is down).
SOFT_LIVE_SUBJECTS = {
    "Anthropology",
}

# Q2.3 — Lightweight query→subject heuristic.
#
# A full subject classifier is out of scope for this fix (no model/training data
# wired into the serving path). Instead we map a small set of high-signal keyword
# phrases to the subjects that actually matter for routing — i.e. the entries in
# ALWAYS_LIVE_SUBJECTS / SOFT_LIVE_SUBJECTS. Only those subjects change routing;
# detecting any other subject would route LOCAL anyway, so we don't bother.
#
# Matching is substring/word based and case-insensitive. Longer/more specific
# phrases are listed first per subject so the most decisive cue wins. This is a
# pragmatic heuristic, NOT a classifier: it will miss paraphrases and only covers
# the routing-relevant subjects. Returns "" when nothing matches (default LOCAL).
_SUBJECT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Theatre", ("theatre", "theater", "playwright", "stagecraft", "dramaturgy", "scenography")),
    ("Music", ("music", "musical", "composer", "symphony", "orchestra", "string quartet", "opera", "ethnomusicology")),
    ("Urban Studies", ("urban studies", "urban planning", "city planning", "gentrification", "urbanism")),
    ("Applied Legal Studies", ("legal studies", "paralegal", "law and society", "access to justice")),
    ("Publishing", ("publishing industry", "book publishing", "scholarly publishing", "editorial workflow")),
    ("Management & Organizational Studies", ("organizational behavior", "organisational behaviour", "management studies", "organizational studies", "human resource management")),
    ("Accounting", ("accounting", "auditing", "financial reporting", "bookkeeping", "gaap")),
    ("Forensics", ("forensic", "forensics", "criminalistics")),
    ("Statistics & Actuarial Science", ("actuarial", "actuarial science", "biostatistics")),
    ("Sustainable Energy Engineering (SEE)", ("sustainable energy engineering", "renewable energy engineering", "photovoltaic engineering")),
    ("Sustainable Community Development", ("sustainable community development", "community development", "sustainable communities")),
    ("Visual Arts", ("visual arts", "painting", "sculpture", "printmaking", "studio art")),
    ("Public Policy", ("public policy", "policy analysis", "policy evaluation", "governance policy")),
    ("Molecular Biology & Biochemistry", ("molecular biology", "biochemistry", "enzyme kinetics", "protein folding", "gene expression")),
    ("Global Health", ("global health", "public health", "epidemiology", "health systems")),
    ("Anthropology", ("anthropology", "anthropological", "ethnography", "kinship", "zooarchaeology", "ethnographic")),
]


def detect_subject(query: str) -> str:
    """Heuristically map a raw query to a routing-relevant SFU subject name.

    Returns the matched subject (one of ALWAYS_LIVE_SUBJECTS / SOFT_LIVE_SUBJECTS)
    or "" if no high-signal keyword matched. This is intentionally conservative:
    it exists only to wake the subject-aware routing rules (Q2.1/Q2.2) for the
    zero-coverage and soft-live subjects, not to classify every query.
    """
    if not query:
        return ""
    lowered = query.lower()
    for subject, keywords in _SUBJECT_KEYWORDS:
        for kw in keywords:
            # word-ish boundary check for short tokens to avoid e.g. "law" in "flaw"
            if " " in kw:
                if kw in lowered:
                    return subject
            elif re.search(rf"\b{re.escape(kw)}\b", lowered):
                return subject
    return ""


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
        # Set by search() when the local cluster fails on every leg attempted;
        # the handler reads this to emit a degradation notice (Q-LOW item #6).
        self.last_degraded = False

    def route(
        self,
        query: str,
        filters: dict,
        subject_hint: str = "",
        prefer_local: bool = False,
    ) -> SearchSource:
        """Determine which backend(s) to query.

        Rules (evaluated in order):
        1. from_publication_date within recency_days → LIVE_API
        2. Temporal cue words in query → LIVE_API
        3. subject_hint in ALWAYS_LIVE_SUBJECTS (zero local coverage) → LIVE_API (Q2.1)
        4. subject_hint in SOFT_LIVE_SUBJECTS → LIVE_API unless prefer_local (Q2.2)
        5. Everything else → LOCAL_RRF (when local_rrf_enabled) else LOCAL_INDEX

        Args:
            query: raw query string
            filters: OpenAlex-style filter dict (e.g. from_publication_date, type)
            subject_hint: detected SFU subject area for the query (e.g. "Theatre").
                Empty string means no subject was detected and subject-aware rules
                are skipped. NOTE: no caller currently passes this — see module note.
            prefer_local: when True, overrides the SOFT_LIVE_SUBJECTS preference and
                keeps soft-live subjects on the local route. Has no effect on the
                hard ALWAYS_LIVE_SUBJECTS list (those have no usable local data).
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

        # Q2.1 — zero-coverage subjects: route live before attempting local
        # retrieval, otherwise the local index returns blank results.
        if subject_hint in ALWAYS_LIVE_SUBJECTS:
            return SearchSource.LIVE_API

        # Q2.2 — soft-live subjects: prefer live, but allow an explicit override
        # back to the local route.
        if subject_hint in SOFT_LIVE_SUBJECTS and not prefer_local:
            return SearchSource.LIVE_API

        return SearchSource.LOCAL_RRF if self.local_rrf_enabled else SearchSource.LOCAL_INDEX

    def search(
        self,
        query: str,
        filters: dict,
        top_k: int = 50,
        force_source: SearchSource | None = None,
        subject_hint: str = "",
        prefer_local: bool = False,
    ) -> list[dict]:
        """Dispatch to one or both backends and return a merged, deduplicated list.

        Args:
            query: raw query string
            filters: OpenAlex-style filter dict (e.g. from_publication_date, type)
            top_k: maximum results to return
            force_source: override automatic routing when set
            subject_hint: detected SFU subject area, forwarded to route() for the
                Q2.1/Q2.2 subject-aware rules.
            prefer_local: forwarded to route() — overrides the soft-live preference.
        """
        source = force_source or self.route(
            query, filters, subject_hint=subject_hint, prefer_local=prefer_local
        )

        live_results: list[dict] = []
        local_results: list[dict] = []
        # Track local-leg failures so we can surface a degradation notice rather
        # than silently returning "no results" when the cluster is down. Mirrors
        # the S2-fallback notice pattern in tools._s2_fallback.
        self.last_degraded = False

        if source in (SearchSource.LIVE_API, SearchSource.BOTH):
            try:
                data = self._openalex.search_works(
                    query, filters=filters, per_page=top_k
                )
                live_results = data.get("results", [])
            except Exception:
                # The router coordinates two independent backends and must never
                # crash the calling handler; the underlying clients already narrow
                # their own exceptions (see opensearch_retriever._http).
                logger.exception("FederatedSearchRouter: live API search failed")

        if source in (SearchSource.LOCAL_INDEX, SearchSource.BOTH):
            try:
                local_results = self._opensearch.search(
                    query, top_k=top_k, filters=filters
                )
            except Exception:
                logger.exception("FederatedSearchRouter: local OpenSearch search failed")
                if source == SearchSource.LOCAL_INDEX:
                    self.last_degraded = True

        if source == SearchSource.LOCAL_RRF:
            bm25f_results: list[dict] = []
            splade_results: list[dict] = []
            bm25f_ok = splade_ok = False
            try:
                bm25f_results = self._opensearch.search(
                    query, top_k=top_k, mode="bm25f", filters=filters
                )
                bm25f_ok = True
            except Exception:
                logger.exception("FederatedSearchRouter: BM25F search failed")
            try:
                splade_results = self._opensearch.search(
                    query, top_k=top_k, mode="splade", filters=filters
                )
                splade_ok = True
            except Exception:
                logger.exception("FederatedSearchRouter: SPLADE search failed")
            # Both local legs errored → the local cluster is degraded.
            if not bm25f_ok and not splade_ok:
                self.last_degraded = True
            return _rrf_merge(bm25f_results, splade_results, top_k)

        if source == SearchSource.LIVE_API:
            return live_results[:top_k]
        if source == SearchSource.LOCAL_INDEX:
            return local_results[:top_k]

        # BOTH — RRF merge of live + local
        return _rrf_merge(live_results, local_results, top_k)
