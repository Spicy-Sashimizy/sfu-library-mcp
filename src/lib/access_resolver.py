"""Access resolution cascade: OA → Unpaywall → SFU subscription → DOI fallback."""

import logging

import requests

from lib.sfu_databases import SFUDatabaseRegistry, extract_domain

logger = logging.getLogger("sfu_library_mcp")

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


class AccessResolver:
    """Resolve the best access URL for an academic article.

    Cascade order:
    1. OpenAlex OA flag (is_oa=True with oa_url)
    2. Unpaywall check by DOI
    3. SFU Database Registry — domain match → EZProxy or direct URL
    4. SFU Database Registry — provider/publisher fuzzy match
    5. DOI link as universal fallback
    """

    def __init__(
        self,
        registry: SFUDatabaseRegistry,
        unpaywall_email: str = "",
        timeout: int = 15,
    ):
        self.registry = registry
        self.unpaywall_email = unpaywall_email
        self.timeout = timeout

    def resolve(
        self,
        doi: str = "",
        source_url: str = "",
        publisher: str = "",
        is_oa: bool = False,
        oa_url: str = "",
    ) -> dict:
        """Return the best access URL and resolution metadata.

        Returns a dict with:
            access_url  : str  — best URL for the user to click
            access_type : str  — "oa" | "unpaywall" | "ezproxy" | "direct" | "doi_fallback"
            doi_url     : str  — https://doi.org/{doi} (always present when DOI known)
            proxy_needed: bool — True if EZProxy wrapping was applied
            db_name     : str  — matched SFU database name (empty if no Solr match)
        """
        doi_url = f"https://doi.org/{doi}" if doi else ""

        # 1. OpenAlex open-access flag
        if is_oa and oa_url:
            return {
                "access_url": oa_url,
                "access_type": "oa",
                "doi_url": doi_url,
                "proxy_needed": False,
                "db_name": "",
            }

        # 2. Unpaywall
        if doi:
            uw = self._check_unpaywall(doi)
            if uw:
                return {
                    "access_url": uw["url"],
                    "access_type": "unpaywall",
                    "doi_url": doi_url,
                    "proxy_needed": False,
                    "db_name": "",
                }

        # 3. SFU subscription — domain match
        domain = extract_domain(source_url)
        if domain:
            records = self.registry.lookup_by_domain(domain)
            if records:
                rec = records[0]
                proxy = _is_proxy(rec)
                article_url = source_url or doi_url
                return {
                    "access_url": self.registry.ezproxy_url(article_url) if proxy else article_url,
                    "access_type": "ezproxy" if proxy else "direct",
                    "doi_url": doi_url,
                    "proxy_needed": proxy,
                    "db_name": rec.get("name", ""),
                }

        # 4. SFU subscription — publisher/provider fuzzy match
        if publisher:
            records = self.registry.lookup_by_provider(publisher)
            if records:
                rec = records[0]
                proxy = _is_proxy(rec)
                article_url = source_url or doi_url
                return {
                    "access_url": self.registry.ezproxy_url(article_url) if proxy else article_url,
                    "access_type": "ezproxy" if proxy else "direct",
                    "doi_url": doi_url,
                    "proxy_needed": proxy,
                    "db_name": rec.get("name", ""),
                }

        # 5. DOI fallback
        return {
            "access_url": doi_url,
            "access_type": "doi_fallback",
            "doi_url": doi_url,
            "proxy_needed": False,
            "db_name": "",
        }

    def _check_unpaywall(self, doi: str) -> dict | None:
        """Query Unpaywall for a free full-text copy. Returns None if not found."""
        if not self.unpaywall_email:
            return None
        try:
            resp = requests.get(
                f"{UNPAYWALL_BASE}/{doi}",
                params={"email": self.unpaywall_email},
                timeout=self.timeout,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            best = resp.json().get("best_oa_location")
            if best and best.get("url"):
                return {
                    "url": best["url"],
                    "version": best.get("version", ""),
                    "host_type": best.get("host_type", ""),
                }
        except Exception as e:
            logger.debug("Unpaywall check failed for DOI %s: %s", doi, e)
        return None


def _is_proxy(rec: dict) -> bool:
    return rec.get("proxy") is True or str(rec.get("proxy", "")).lower() == "true"
