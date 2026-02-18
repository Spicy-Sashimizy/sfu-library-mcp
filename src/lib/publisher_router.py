"""Publisher-based download tier routing.

Learns per-domain tier preferences at runtime so known browser-only publishers
(Wiley, Springer, etc.) skip direct HTTP tiers on subsequent attempts, while
known direct-OK publishers (SAGE, etc.) skip the expensive Playwright tier.

Session-scoped and thread-safe — one instance shared across all downloads.
"""

import logging
import threading
from enum import Enum
from urllib.parse import urlparse

from lib.downloader import unwrap_proxied_hostname

logger = logging.getLogger("sfu_library_mcp")


class DomainClass(Enum):
    """Classification of a publisher domain's download tier compatibility."""
    UNKNOWN = "unknown"
    DIRECT_OK = "direct_ok"
    BROWSER_ONLY = "browser_only"


# Tiers considered "direct" (HTTP-based, no headless browser)
DIRECT_TIERS = frozenset({"curl_cffi", "requests"})


class PublisherRouter:
    """Session-scoped router that learns which download tier works per publisher domain.

    Thread-safe: all mutations go through a lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # domain -> DomainClass
        self._classifications: dict[str, DomainClass] = {}
        # domain -> set of tier names that have succeeded
        self._successes: dict[str, set[str]] = {}
        # domain -> set of tier names that have failed
        self._failures: dict[str, set[str]] = {}

    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract the registrable domain from a URL.

        Strips subdomains (e.g. onlinelibrary.wiley.com -> wiley.com).
        Unwraps hostname-based proxy URLs first
        (e.g. onlinelibrary-wiley-com.proxy.lib.sfu.ca -> wiley.com).
        """
        try:
            hostname = urlparse(url).hostname or ""
        except Exception:
            return "unknown"
        if not hostname:
            return "unknown"
        # Unwrap hostname-based proxy
        hostname = unwrap_proxied_hostname(hostname)
        parts = hostname.lower().split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return hostname

    def get_domain_class(self, url: str) -> DomainClass:
        """Return the current classification for a URL's domain."""
        domain = self.extract_domain(url)
        with self._lock:
            return self._classifications.get(domain, DomainClass.UNKNOWN)

    def get_tier_order(self, url: str, configured_tiers: list[str]) -> list[str]:
        """Return reordered tier list based on learned domain preferences.

        Args:
            url: The download URL (used to extract domain).
            configured_tiers: The base tier order from config (e.g. ["curl_cffi", "playwright", "requests"]).

        Returns:
            Reordered tier list. All configured tiers are always included (as fallbacks).
        """
        domain = self.extract_domain(url)
        with self._lock:
            classification = self._classifications.get(domain, DomainClass.UNKNOWN)

        if classification == DomainClass.UNKNOWN:
            # First encounter — use default order
            return list(configured_tiers)

        if classification == DomainClass.DIRECT_OK:
            # Direct tiers first, browser last
            direct = [t for t in configured_tiers if t in DIRECT_TIERS]
            browser = [t for t in configured_tiers if t not in DIRECT_TIERS]
            return direct + browser

        if classification == DomainClass.BROWSER_ONLY:
            # Browser first, direct tiers last (still included as fallback)
            browser = [t for t in configured_tiers if t not in DIRECT_TIERS]
            direct = [t for t in configured_tiers if t in DIRECT_TIERS]
            return browser + direct

        return list(configured_tiers)

    def record_success(self, url: str, tier_name: str) -> None:
        """Record a validated PDF download success for a domain+tier.

        Only call this after PDF magic bytes have been confirmed.
        A direct tier success immediately promotes domain to DIRECT_OK.
        """
        domain = self.extract_domain(url)
        with self._lock:
            if domain not in self._successes:
                self._successes[domain] = set()
            self._successes[domain].add(tier_name)

            if tier_name in DIRECT_TIERS:
                old = self._classifications.get(domain, DomainClass.UNKNOWN)
                self._classifications[domain] = DomainClass.DIRECT_OK
                if old != DomainClass.DIRECT_OK:
                    logger.info(
                        "PublisherRouter: %s reclassified %s -> DIRECT_OK (tier=%s)",
                        domain, old.value, tier_name,
                    )

    def record_failure(self, url: str, tier_name: str) -> None:
        """Record a download tier failure (exception, 403, timeout, etc.).

        A direct tier failure with no prior direct success → BROWSER_ONLY.
        """
        domain = self.extract_domain(url)
        with self._lock:
            if domain not in self._failures:
                self._failures[domain] = set()
            self._failures[domain].add(tier_name)

            # Only reclassify on direct tier failures
            if tier_name in DIRECT_TIERS:
                has_direct_success = bool(
                    self._successes.get(domain, set()) & DIRECT_TIERS
                )
                if not has_direct_success:
                    old = self._classifications.get(domain, DomainClass.UNKNOWN)
                    if old != DomainClass.BROWSER_ONLY:
                        self._classifications[domain] = DomainClass.BROWSER_ONLY
                        logger.info(
                            "PublisherRouter: %s reclassified %s -> BROWSER_ONLY (tier=%s failed)",
                            domain, old.value, tier_name,
                        )

    def get_stats(self) -> dict:
        """Return current router state for diagnostics."""
        with self._lock:
            return {
                "classifications": {
                    d: c.value for d, c in self._classifications.items()
                },
                "successes": {d: list(s) for d, s in self._successes.items()},
                "failures": {d: list(f) for d, f in self._failures.items()},
            }
