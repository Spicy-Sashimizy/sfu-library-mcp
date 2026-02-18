"""Download rate limiter for anti-detection safety nets.

Enforces session budgets, hourly caps, per-domain limits, and humanized
inter-download delays to prevent triggering publisher bot detection.
"""

import logging
import random
import threading
import time
from urllib.parse import urlparse

from lib.config import ServerConfig
from lib.downloader import unwrap_proxied_hostname

logger = logging.getLogger("sfu_library_mcp")


class RateLimitExceeded(Exception):
    """Raised when download rate limits are exceeded."""
    pass


class DownloadRateLimiter:
    """Thread-safe download rate limiter with session, hourly, and per-domain budgets."""

    def __init__(self, config: ServerConfig):
        self._lock = threading.Lock()
        self._session_budget = config.download_session_budget
        self._hourly_limit = config.download_hourly_limit
        self._per_domain_hourly_limit = config.download_per_domain_hourly_limit
        self._min_delay = config.download_min_delay
        self._max_delay = config.download_max_delay

        # Counters
        self._session_count = 0
        self._hourly_timestamps: list[float] = []
        self._domain_timestamps: dict[str, list[float]] = {}

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract the registrable domain from a URL.

        Unwraps hostname-based proxy URLs first so rate limits apply
        to the actual publisher domain, not the proxy domain.
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

    def _prune_old_timestamps(self, timestamps: list[float], window: float = 3600.0) -> list[float]:
        """Remove timestamps older than the window (default 1 hour)."""
        cutoff = time.time() - window
        return [t for t in timestamps if t > cutoff]

    def acquire(self, url: str) -> None:
        """Check all budgets and sleep for humanized delay. Raises RateLimitExceeded if any limit hit."""
        with self._lock:
            # Check session budget
            if self._session_count >= self._session_budget:
                raise RateLimitExceeded(
                    f"Session download budget exceeded ({self._session_budget} downloads). "
                    f"Restart the server to reset."
                )

            # Prune and check hourly limit
            self._hourly_timestamps = self._prune_old_timestamps(self._hourly_timestamps)
            if len(self._hourly_timestamps) >= self._hourly_limit:
                raise RateLimitExceeded(
                    f"Hourly download limit exceeded ({self._hourly_limit} downloads/hour). "
                    f"Wait before downloading more."
                )

            # Prune and check per-domain limit
            domain = self._extract_domain(url)
            if domain in self._domain_timestamps:
                self._domain_timestamps[domain] = self._prune_old_timestamps(
                    self._domain_timestamps[domain]
                )
            domain_count = len(self._domain_timestamps.get(domain, []))
            if domain_count >= self._per_domain_hourly_limit:
                raise RateLimitExceeded(
                    f"Per-domain hourly limit exceeded for {domain} "
                    f"({self._per_domain_hourly_limit} downloads/hour from this domain). "
                    f"Try a different source or wait."
                )

            # Calculate delay
            delay = random.uniform(self._min_delay, self._max_delay)

        # Sleep outside the lock so other threads aren't blocked
        logger.info("Rate limiter: sleeping %.1fs before download from %s", delay, self._extract_domain(url))
        time.sleep(delay)

    def record_download(self, url: str) -> None:
        """Record a successful download. Call after PDF validated."""
        with self._lock:
            now = time.time()
            self._session_count += 1
            self._hourly_timestamps.append(now)
            domain = self._extract_domain(url)
            if domain not in self._domain_timestamps:
                self._domain_timestamps[domain] = []
            self._domain_timestamps[domain].append(now)
            logger.info(
                "Rate limiter: recorded download #%d (domain=%s, hourly=%d)",
                self._session_count, domain, len(self._hourly_timestamps),
            )

    def get_budget_status(self) -> dict:
        """Return remaining budgets for display to user."""
        with self._lock:
            self._hourly_timestamps = self._prune_old_timestamps(self._hourly_timestamps)
            domain_counts = {}
            for domain, timestamps in self._domain_timestamps.items():
                self._domain_timestamps[domain] = self._prune_old_timestamps(timestamps)
                count = len(self._domain_timestamps[domain])
                if count > 0:
                    domain_counts[domain] = {
                        "used": count,
                        "remaining": max(0, self._per_domain_hourly_limit - count),
                    }
            return {
                "session_remaining": max(0, self._session_budget - self._session_count),
                "session_used": self._session_count,
                "session_budget": self._session_budget,
                "hourly_remaining": max(0, self._hourly_limit - len(self._hourly_timestamps)),
                "hourly_used": len(self._hourly_timestamps),
                "hourly_limit": self._hourly_limit,
                "per_domain": domain_counts,
            }
