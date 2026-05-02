"""OpenAlex open academic search client + CrossRef DOI enrichment."""

import logging
from typing import Any

import requests

logger = logging.getLogger("sfu_library_mcp")

OPENALEX_BASE = "https://api.openalex.org"
CROSSREF_BASE = "https://api.crossref.org/works"

_USER_AGENT = "SFULibraryMCP/1.0 (mailto:lib-systems@sfu.ca)"


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize_doi(doi: str | None) -> str:
    """Strip URL prefix and return a bare DOI string."""
    if not doi:
        return ""
    if doi.startswith("https://doi.org/"):
        return doi[len("https://doi.org/"):]
    if doi.startswith("http://doi.org/"):
        return doi[len("http://doi.org/"):]
    return doi


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Rebuild abstract text from OpenAlex inverted-index format."""
    if not inverted_index:
        return ""
    try:
        pos_word: dict[int, str] = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                pos_word[pos] = word
        return " ".join(pos_word[p] for p in sorted(pos_word))
    except Exception:
        return ""


def normalize_work(work: dict) -> dict:
    """Normalize an OpenAlex work dict to the shared WorkMetadata format.

    The output is compatible with citations.py formatters (same keys as
    extract_metadata() output) plus OpenAlex-specific extras.
    """
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    oa_info = work.get("open_access") or {}
    biblio = work.get("biblio") or {}
    ids = work.get("ids") or {}
    host_venue = work.get("host_venue") or {}  # legacy field, some records only

    # Authors
    authors: list[str] = []
    for auth in work.get("authorships") or []:
        name = (auth.get("author") or {}).get("display_name", "")
        if name:
            authors.append(name)

    # Date
    pub_date = work.get("publication_date", "")
    pub_year = work.get("publication_year")
    date = pub_date or (str(pub_year) if pub_year else "")

    # DOI
    doi = normalize_doi(work.get("doi"))

    # ISBN (books)
    isbn_raw = ids.get("isbn", "")
    isbn = (isbn_raw[0] if isinstance(isbn_raw, list) else isbn_raw) or ""

    # ISSN
    issn = source.get("issn_l") or ""

    # Pages
    first_page = biblio.get("first_page", "") or ""
    last_page = biblio.get("last_page", "") or ""
    pages = f"{first_page}-{last_page}" if first_page and last_page else first_page

    # Publisher
    publisher = (
        source.get("host_organization_name")
        or host_venue.get("publisher")
        or ""
    )

    # Topics for display
    topics = [
        t.get("display_name", "")
        for t in (work.get("topics") or work.get("concepts") or [])[:5]
        if t.get("display_name")
    ]

    # Abstract
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    # Best OA URL
    oa_url = (
        oa_info.get("oa_url")
        or location.get("pdf_url")
        or location.get("landing_page_url")
        or ""
    ) or ""

    work_type = work.get("type", "") or ""
    if "article" in work_type:
        resource_type = "article"
    elif "book" in work_type:
        resource_type = "book"
    else:
        resource_type = work_type or "other"

    return {
        # Citation-compatible fields (matches extract_metadata() output schema)
        "title": work.get("title") or "",
        "authors": authors,
        "creators": authors,
        "contributors": [],
        "date": date,
        "publisher": publisher,
        "type": work_type,
        "source": source.get("display_name") or host_venue.get("display_name") or "",
        "isbn": isbn,
        "issn": issn,
        "doi": doi,
        "volume": biblio.get("volume", "") or "",
        "issue": biblio.get("issue", "") or "",
        "spage": first_page,
        "epage": last_page,
        "pages": pages,
        "record_id": work.get("id", ""),
        "is_cdi": False,
        "resource_type": resource_type,
        # OpenAlex-specific extras
        "openalex_id": work.get("id", ""),
        "is_oa": bool(oa_info.get("is_oa")),
        "oa_url": oa_url,
        "cited_by_count": work.get("cited_by_count", 0),
        "abstract": abstract,
        "topics": topics,
    }


# ── OpenAlex client ───────────────────────────────────────────────────────────

class OpenAlexClient:
    """Client for the OpenAlex open academic search API (CC0).

    Auth priority: api_key (100 req/s) > mailto polite pool (10 req/s) > anonymous (1 req/s).
    """

    def __init__(self, mailto: str = "", api_key: str = "", timeout: int = 30):
        self.mailto = mailto
        self.api_key = api_key
        self.timeout = timeout

    def _params(self, extra: dict) -> dict:
        p = dict(extra)
        if self.api_key:
            p.setdefault("api_key", self.api_key)
        elif self.mailto:
            p.setdefault("mailto", self.mailto)
        return p

    def _get(self, path: str, params: dict) -> dict | None:
        try:
            resp = requests.get(
                f"{OPENALEX_BASE}{path}",
                params=self._params(params),
                headers={"User-Agent": _USER_AGENT},
                timeout=self.timeout,
            )
            if resp.status_code == 429:
                logger.warning("OpenAlex rate limited (429)")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            logger.error("OpenAlex request timed out")
            return None
        except Exception as e:
            logger.error("OpenAlex request failed: %s", e)
            return None

    def search_works(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        sort: str = "relevance_score:desc",
        page: int = 1,
        per_page: int = 10,
    ) -> dict:
        """Search works. filters keys are OpenAlex filter names, values are filter values."""
        params: dict[str, Any] = {
            "search": query,
            "per_page": per_page,
            "page": page,
        }
        if sort:
            params["sort"] = sort
        if filters:
            params["filter"] = ",".join(f"{k}:{v}" for k, v in filters.items())
        data = self._get("/works", params)
        if not data:
            return {"results": [], "meta": {"count": 0}}
        return {
            "results": [normalize_work(w) for w in data.get("results", [])],
            "meta": data.get("meta", {}),
        }

    def get_work_by_doi(self, doi: str) -> dict | None:
        """Fetch a single work by DOI."""
        doi_norm = normalize_doi(doi)
        if not doi_norm:
            return None
        data = self._get(f"/works/https://doi.org/{doi_norm}", {})
        return normalize_work(data) if data and "id" in data else None

    def get_work(self, openalex_id: str) -> dict | None:
        """Fetch a single work by OpenAlex ID (full URL or bare ID)."""
        if openalex_id.startswith("https://openalex.org/"):
            bare = openalex_id.split("/")[-1]
        else:
            bare = openalex_id
        data = self._get(f"/works/{bare}", {})
        return normalize_work(data) if data and "id" in data else None

    def search_by_author(
        self, author_name: str, per_page: int = 10
    ) -> dict:
        """Search works filtered to a specific author display name."""
        return self.search_works(
            query=author_name,
            filters={"authorships.author.display_name.search": author_name},
            per_page=per_page,
        )


# ── CrossRef enrichment ───────────────────────────────────────────────────────

def fetch_crossref_work(doi: str, timeout: int = 10) -> dict | None:
    """Fetch metadata from CrossRef by DOI.

    Returns a WorkMetadata-compatible dict or None on failure.
    Free API, no auth required.
    """
    if not doi:
        return None
    doi_norm = normalize_doi(doi)
    try:
        resp = requests.get(
            f"{CROSSREF_BASE}/{doi_norm}",
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug("CrossRef returned %d for DOI %s", resp.status_code, doi_norm)
            return None
        work = resp.json().get("message", {})

        authors = []
        for author in work.get("author", []):
            family = author.get("family", "")
            given = author.get("given", "")
            if family and given:
                authors.append(f"{family}, {given}")
            elif family:
                authors.append(family)

        date_parts = work.get("published-print", work.get("published-online", {}))
        date = ""
        if date_parts and date_parts.get("date-parts"):
            parts = date_parts["date-parts"][0]
            date = str(parts[0]) if parts else ""

        container = work.get("container-title", [])
        journal = container[0] if container else ""

        pages = work.get("page", "")
        spage, epage = "", ""
        if "-" in pages:
            p = pages.split("-", 1)
            spage, epage = p[0].strip(), p[1].strip()

        return {
            "title": work.get("title", [""])[0] if work.get("title") else "",
            "authors": authors,
            "creators": authors,
            "contributors": [],
            "date": date,
            "publisher": work.get("publisher", ""),
            "type": work.get("type", ""),
            "source": journal,
            "isbn": "",
            "issn": work.get("ISSN", [""])[0] if work.get("ISSN") else "",
            "doi": doi_norm,
            "volume": work.get("volume", ""),
            "issue": work.get("issue", ""),
            "spage": spage,
            "epage": epage,
            "pages": pages,
            "record_id": "",
            "is_cdi": False,
            "resource_type": "article" if "article" in work.get("type", "") else "other",
        }
    except requests.Timeout:
        logger.debug("CrossRef timed out for DOI %s", doi_norm)
        return None
    except Exception as e:
        logger.debug("CrossRef lookup failed for DOI %s: %s", doi_norm, e)
        return None
