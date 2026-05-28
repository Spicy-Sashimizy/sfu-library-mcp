"""SFU Database Registry client — public Solr endpoint, no auth required."""

import json
import logging
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

logger = logging.getLogger("sfu_library_mcp")

SOLR_URL = "https://databases.lib.sfu.ca/solr/sfu_databases/select"
EZPROXY_BASE = "https://proxy.lib.sfu.ca/login?url="

_PROVIDER_STRIP = re.compile(
    r"\b(inc\.?|ltd\.?|llc\.?|publishing|press|group|corporation|corp\.?"
    r"|co\.?|gmbh|s\.?a\.?|plc|limited|verlag)\b",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text).strip()


def extract_domain(url: str) -> str | None:
    """Return the bare domain (no www.) from a URL, or None."""
    if not url:
        return None
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        domain = (parsed.hostname or "").lower().strip(".")
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or None
    except Exception:
        return None


def normalize_provider(name: str) -> str:
    """Lowercase, strip corporate suffixes, collapse whitespace."""
    if not name:
        return ""
    name = name.lower().strip()
    name = _PROVIDER_STRIP.sub("", name)
    name = re.sub(r"[.,;:]+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


class SFUDatabaseRegistry:
    """Client for the SFU Database Registry via public Solr endpoint.

    Fetches all ~764 records on first access and caches them in memory
    and optionally on disk. Provides search/filter and access-resolution helpers.
    """

    def __init__(
        self,
        cache_ttl: int = 86400,
        cache_file: str | None = None,
        timeout: int = 30,
    ):
        self.cache_ttl = cache_ttl
        self.cache_file = Path(cache_file) if cache_file else None
        self.timeout = timeout

        self._docs: list[dict] = []
        self._domain_index: dict[str, list[dict]] = {}
        self._provider_index: dict[str, list[dict]] = {}
        self._loaded_at: float = 0.0
        self._lock = threading.Lock()

    def _is_stale(self) -> bool:
        return time.time() - self._loaded_at > self.cache_ttl

    def ensure_loaded(self) -> None:
        """Load registry if not loaded or cache has expired."""
        with self._lock:
            if self._docs and not self._is_stale():
                return
            self._load()

    def refresh(self) -> None:
        """Force a fresh fetch from Solr."""
        with self._lock:
            self._loaded_at = 0.0
            self._load()

    def _load(self) -> None:
        """Fetch from Solr with disk-cache fallback (caller must hold lock)."""
        docs = self._fetch_from_solr()
        if not docs and self.cache_file and self.cache_file.is_file():
            docs = self._load_disk_cache()
            if docs:
                logger.warning("Solr unreachable — serving from disk cache")
        if not docs:
            logger.error("Failed to load SFU database registry from any source")
            return
        self._docs = docs
        self._build_indices()
        self._loaded_at = time.time()
        if self.cache_file:
            self._save_disk_cache(docs)

    def _fetch_from_solr(self) -> list[dict]:
        try:
            resp = requests.get(
                SOLR_URL,
                params={"q": "*:*", "rows": "1000", "wt": "json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            docs = resp.json().get("response", {}).get("docs", [])
            for doc in docs:
                if doc.get("publicNote"):
                    doc["publicNote"] = _strip_html(doc["publicNote"])
            logger.info("Loaded %d SFU database records from Solr", len(docs))
            return docs
        except Exception as e:
            logger.warning("Solr fetch failed: %s", e)
            return []

    def _load_disk_cache(self) -> list[dict]:
        try:
            data = json.loads(self.cache_file.read_text())
            docs = data.get("docs", [])
            logger.info("Loaded %d records from disk cache %s", len(docs), self.cache_file)
            return docs
        except Exception as e:
            logger.warning("Disk cache read failed: %s", e)
            return []

    def _save_disk_cache(self, docs: list[dict]) -> None:
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(
                json.dumps({"loaded_at": time.time(), "docs": docs})
            )
        except Exception as e:
            logger.warning("Disk cache write failed: %s", e)

    def _build_indices(self) -> None:
        """Build domain and provider lookup indices (caller must hold lock)."""
        domain_idx: dict[str, list[dict]] = defaultdict(list)
        provider_idx: dict[str, list[dict]] = defaultdict(list)

        for doc in self._docs:
            url = doc.get("url", "")
            if isinstance(url, list):
                url = url[0] if url else ""
            domain = extract_domain(url)
            if domain:
                doc["_domain"] = domain
                domain_idx[domain].append(doc)

            provider_raw = doc.get("provider", "")
            if isinstance(provider_raw, list):
                provider_raw = provider_raw[0] if provider_raw else ""
            if provider_raw:
                norm = normalize_provider(provider_raw)
                if norm:
                    provider_idx[norm].append(doc)

        self._domain_index = dict(domain_idx)
        self._provider_index = dict(provider_idx)

    # ── Public API ──────────────────────────────────────────────────

    def get_all(self) -> list[dict]:
        self.ensure_loaded()
        return list(self._docs)

    def search(
        self,
        query: str = "",
        subject: str = "",
        content_type: str = "",
        free_only: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        """Search/filter the registry. All filters are ANDed."""
        self.ensure_loaded()
        q_lower = query.lower() if query else ""
        results = []
        for doc in self._docs:
            if free_only:
                is_free = doc.get("free") is True or str(doc.get("free", "")).lower() == "true"
                if not is_free:
                    continue
            if subject:
                subjects = doc.get("subjects", [])
                if isinstance(subjects, str):
                    subjects = [subjects]
                if not any(subject.lower() in s.lower() for s in subjects):
                    continue
            if content_type:
                ctypes = doc.get("contentTypes", [])
                if isinstance(ctypes, str):
                    ctypes = [ctypes]
                if not any(content_type.lower() in ct.lower() for ct in ctypes):
                    continue
            if q_lower:
                name = doc.get("name", "").lower()
                desc = doc.get("description", "").lower()
                alt_names = " ".join(
                    n.lower() for n in (doc.get("names") or []) if isinstance(n, str)
                )
                if q_lower not in name and q_lower not in desc and q_lower not in alt_names:
                    continue
            results.append(doc)
            if len(results) >= limit:
                break
        return results

    def lookup_by_domain(self, domain: str) -> list[dict]:
        """Find databases whose URL matches the given domain."""
        self.ensure_loaded()
        records = self._domain_index.get(domain)
        if records:
            return records
        # Try parent domain (e.g., journals.sagepub.com → sagepub.com)
        parts = domain.split(".")
        if len(parts) > 2:
            parent = ".".join(parts[-2:])
            return self._domain_index.get(parent, [])
        return []

    def lookup_by_provider(self, publisher: str) -> list[dict]:
        """Find databases whose provider fuzzy-matches the given publisher name."""
        self.ensure_loaded()
        pub_norm = normalize_provider(publisher)
        if not pub_norm:
            return []
        if pub_norm in self._provider_index:
            return self._provider_index[pub_norm]
        for pn, recs in self._provider_index.items():
            if pub_norm in pn or pn in pub_norm:
                return recs
        return []

    def ezproxy_url(self, article_url: str) -> str:
        """Wrap an article URL with SFU EZProxy."""
        return f"{EZPROXY_BASE}{quote(article_url, safe='')}"

    @property
    def record_count(self) -> int:
        self.ensure_loaded()
        return len(self._docs)
