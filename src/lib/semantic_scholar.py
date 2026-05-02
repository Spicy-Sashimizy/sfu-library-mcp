"""Semantic Scholar Graph API client — citations, references, TLDRs."""

import logging

import requests

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

    def __init__(self, api_key: str = "", timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        h = {"User-Agent": "SFULibraryMCP/1.0"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _get(self, path: str, params: dict) -> dict | None:
        try:
            resp = requests.get(
                f"{S2_BASE}{path}",
                params=params,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 429:
                logger.warning("Semantic Scholar rate limited (429)")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            logger.error("Semantic Scholar request timed out")
            return None
        except Exception as e:
            logger.error("Semantic Scholar request failed: %s", e)
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
