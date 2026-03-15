"""SFU Library API client — unauthenticated Primo REST API access."""

import json
import logging
from typing import Any

import requests

from lib.config import ServerConfig, load_config
from lib.retry import retry_with_backoff, CircuitBreaker
from lib.validators import sanitize_search_query, validate_api_response
from lib.cache import ResponseCache

logger = logging.getLogger("sfu_library_mcp")


class SFULibraryClient:
    """Client for searching SFU Library via the public Primo REST API."""

    BASE_URL = "https://sfu-primo.hosted.exlibrisgroup.com"
    SEARCH_PATH = "/primo_library/libweb/webservices/rest/primo-explore/v1/pnxs"

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or load_config()

        # Circuit breaker for API calls
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout,
        )

        # Response cache
        self.cache = ResponseCache(
            ttl=self.config.cache_ttl,
            max_size=self.config.cache_max_size,
            max_memory_mb=self.config.cache_max_memory_mb,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        field: str = "any",
        precision: str = "contains",
        sort: str = "rank",
        tab: str = "default_tab",
        scope: str = "default_scope",
    ) -> dict | None:
        """Search the library using the public Primo REST API."""
        sanitized_query = sanitize_search_query(query)
        if not sanitized_query:
            logger.warning("Empty query after sanitization")
            return None

        # Check cache
        if self.config.features.get("cache_enabled"):
            cache_key = self.cache.make_key("search", sanitized_query, limit, offset, field, sort, tab, scope)
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for search: %s", sanitized_query[:50])
                return cached

        url = f"{self.BASE_URL}{self.SEARCH_PATH}"

        params = {
            "q": f"{field},{precision},{sanitized_query}",
            "vid": "SFUL",
            "inst": "01SFUL",
            "tab": tab,
            "scope": scope,
            "lang": "en_US",
            "offset": offset,
            "limit": limit,
            "sort": sort,
            "skipDelivery": "Y",
            "blendFacetsSeparately": "true",
            "pcAvailability": "false",
            "getMore": 0,
            "rtaLinks": "true",
            "newspapersActive": "true",
            "newspapersSearch": "false",
        }

        result = self._make_api_request(url, params)

        # Cache on success
        if result is not None and self.config.features.get("cache_enabled"):
            self.cache.put(cache_key, result)

        return result

    def get_item_details(self, doc_id: str, context: str = "L") -> dict | None:
        """Get detailed information about a specific item."""
        # Check cache
        if self.config.features.get("cache_enabled"):
            cache_key = self.cache.make_key("item", doc_id, context)
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit for item: %s", doc_id)
                return cached

        url = f"{self.BASE_URL}{self.SEARCH_PATH}/{doc_id}"

        params = {
            "vid": "SFUL",
            "inst": "01SFUL",
            "lang": "en_US",
            "context": context,
        }

        result = self._make_api_request(url, params)

        if result is not None and self.config.features.get("cache_enabled"):
            self.cache.put(cache_key, result)

        return result

    def _make_api_request(self, url: str, params: dict) -> dict | None:
        """Make an unauthenticated API request."""
        headers = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"{self.BASE_URL}/primo-explore/search?vid=SFUL",
            "Origin": self.BASE_URL,
        }

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.search_timeout,
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    return data
                logger.warning("Unexpected response type: %s", type(data))
                return data
            elif response.status_code == 429:
                logger.warning("Rate limited (429)")
                return None
            else:
                logger.warning("API returned status %d", response.status_code)
                return None
        except requests.Timeout:
            logger.error("API request timed out after %ds", self.config.search_timeout)
            return None
        except requests.ConnectionError as e:
            logger.error("API connection failed: %s", e)
            return None
        except Exception as e:
            logger.error("API request failed: %s", e)
            return None


def fetch_crossref_metadata(doi: str, timeout: int = 10) -> dict | None:
    """Fetch metadata from CrossRef API by DOI.

    Free, no authentication required. Returns a normalized metadata dict
    compatible with extract_metadata() output, or None on failure.
    """
    if not doi:
        return None

    # Normalize DOI — strip URL prefix if present
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    elif doi.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/"):]

    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "SFULibraryMCP/1.0",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            logger.debug("CrossRef returned %d for DOI %s", response.status_code, doi)
            return None

        data = response.json()
        work = data.get("message", {})

        # Extract authors
        authors = []
        for author in work.get("author", []):
            family = author.get("family", "")
            given = author.get("given", "")
            if family and given:
                authors.append(f"{family}, {given}")
            elif family:
                authors.append(family)

        # Extract date
        date_parts = work.get("published-print", work.get("published-online", {}))
        date = ""
        if date_parts and date_parts.get("date-parts"):
            parts = date_parts["date-parts"][0]
            date = str(parts[0]) if parts else ""

        # Extract journal
        journal = ""
        container = work.get("container-title", [])
        if container:
            journal = container[0]

        # Extract pages
        pages = work.get("page", "")
        spage, epage = "", ""
        if "-" in pages:
            parts = pages.split("-", 1)
            spage, epage = parts[0].strip(), parts[1].strip()

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
            "doi": doi,
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
        logger.debug("CrossRef request timed out for DOI %s", doi)
        return None
    except Exception as e:
        logger.debug("CrossRef lookup failed for DOI %s: %s", doi, e)
        return None
