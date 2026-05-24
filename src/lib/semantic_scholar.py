"""Semantic Scholar Graph API client — citations, references, TLDRs."""

import logging
import time

import requests

from lib.retry import CircuitBreaker

logger = logging.getLogger("sfu_library_mcp")

S2_BASE = "https://api.semanticscholar.org/graph/v1"

_PAPER_FIELDS = (
    "title,authors,year,abstract,citationCount,influentialCitationCount,"
    "openAccessPdf,tldr,externalIds,publicationDate"
)
_CITATION_FIELDS = "title,authors,year,citationCount,externalIds"


def _normalize_paper(paper: dict) -> dict:
    authors = [a.get("name", "") for a in (paper.get("authors") or [])]
    ext = paper.get("externalIds") or {}
    doi = ext.get("DOI", "")
    tldr_raw = paper.get("tldr")
    tldr = (
        tldr_raw.get("text", "") if isinstance(tldr_raw, dict) else (tldr_raw or "")
    )
    oa_pdf = (paper.get("openAccessPdf") or {}).get("url", "")
    return {
        "s2_id": paper.get("paperId", ""),
        "title": paper.get("title", ""),
        "authors": authors,
        "year": paper.get("year"),
        "publication_date": paper.get("publicationDate", ""),
        "abstract": paper.get("abstract", "") or "",
        "doi": doi,
        "citation_count": paper.get("citationCount", 0),
        "influential_citation_count": paper.get("influentialCitationCount", 0),
        "open_access_pdf": oa_pdf,
        "tldr": tldr,
    }


class SemanticScholarClient:
    """Client for the Semantic Scholar Graph API.

    Free tier: ~100 req/5 min unauthenticated.
    With API key: ~1 req/s sustained.
    """

    def __init__(
        self,
        api_key: str = "",
        timeout: int = 30,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._breaker = CircuitBreaker(
            threshold=circuit_breaker_threshold,
            timeout=circuit_breaker_timeout,
        )

    def _headers(self) -> dict:
        h = {"User-Agent": "SFULibraryMCP/1.0"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _get(self, path: str, params: dict) -> dict | None:
        if not self._breaker.can_proceed():
            logger.warning("Semantic Scholar circuit breaker OPEN — skipping request")
            return None

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = requests.get(
                    f"{S2_BASE}{path}",
                    params=params,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                if resp.status_code == 429:
                    # S2 free tier: 100 req / 5 min — use a longer base delay on 429
                    delay = min(self._retry_base_delay * (4 ** attempt), 120.0)
                    logger.warning(
                        "Semantic Scholar rate limited (429), retry %d/%d after %.1fs",
                        attempt + 1, self._max_retries, delay,
                    )
                    if attempt < self._max_retries:
                        time.sleep(delay)
                        continue
                    self._breaker.record_failure()
                    return None
                resp.raise_for_status()
                self._breaker.record_success()
                return resp.json()
            except requests.Timeout as e:
                last_exc = e
                delay = min(self._retry_base_delay * (2 ** attempt), 60.0)
                logger.warning(
                    "Semantic Scholar request timed out, retry %d/%d after %.1fs",
                    attempt + 1, self._max_retries, delay,
                )
                # Record the breaker failure ONCE per logical request — only on
                # the final exhausted attempt. Recording per-attempt over-counts
                # (max_retries=3 → 4 failures) and trips a threshold-5 breaker
                # after a single failing request.
                if attempt >= self._max_retries:
                    self._breaker.record_failure()
                else:
                    time.sleep(delay)
            except requests.HTTPError as e:
                last_exc = e
                delay = min(self._retry_base_delay * (2 ** attempt), 60.0)
                logger.warning(
                    "Semantic Scholar HTTP error %s, retry %d/%d after %.1fs",
                    e, attempt + 1, self._max_retries, delay,
                )
                if attempt >= self._max_retries:
                    self._breaker.record_failure()
                else:
                    time.sleep(delay)
            except Exception as e:
                logger.error("Semantic Scholar request failed: %s", e)
                self._breaker.record_failure()
                return None

        logger.error("Semantic Scholar request failed after %d retries: %s", self._max_retries, last_exc)
        return None

    def search_papers(
        self,
        query: str,
        fields: str = _PAPER_FIELDS,
        limit: int = 10,
    ) -> list[dict]:
        data = self._get("/paper/search", {"query": query, "fields": fields, "limit": limit})
        if not data:
            return []
        return [_normalize_paper(p) for p in data.get("data", [])]

    def get_paper(self, paper_id: str, fields: str = _PAPER_FIELDS) -> dict | None:
        """Get a paper by S2 ID, DOI, or arXiv ID."""
        data = self._get(f"/paper/{paper_id}", {"fields": fields})
        return _normalize_paper(data) if data and data.get("paperId") else None

    def get_paper_by_doi(self, doi: str, fields: str = _PAPER_FIELDS) -> dict | None:
        return self.get_paper(f"DOI:{doi}", fields)

    def get_citations(
        self,
        paper_id: str,
        limit: int = 20,
        fields: str = _CITATION_FIELDS,
    ) -> list[dict]:
        """Get papers that cite the given paper."""
        data = self._get(
            f"/paper/{paper_id}/citations",
            {"fields": fields, "limit": limit},
        )
        if not data:
            return []
        return [_normalize_paper(item.get("citingPaper", {})) for item in data.get("data", [])]

    def get_references(
        self,
        paper_id: str,
        limit: int = 20,
        fields: str = _CITATION_FIELDS,
    ) -> list[dict]:
        """Get papers referenced by the given paper."""
        data = self._get(
            f"/paper/{paper_id}/references",
            {"fields": fields, "limit": limit},
        )
        if not data:
            return []
        return [_normalize_paper(item.get("citedPaper", {})) for item in data.get("data", [])]

    def get_tldr(self, paper_id: str) -> str:
        """Get an AI-generated one-sentence summary of a paper."""
        data = self._get(f"/paper/{paper_id}", {"fields": "tldr"})
        if not data:
            return ""
        tldr = data.get("tldr")
        return tldr.get("text", "") if isinstance(tldr, dict) else (tldr or "")
