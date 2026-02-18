"""PDF download, caching, and text extraction for library articles.

Provides ArticleDownloader that resolves full-text URLs from PNX records,
downloads PDFs through EZProxy using a tiered strategy (curl_cffi → Playwright
→ requests), caches them locally, optionally copies to the host Downloads
folder, and extracts text via pdftotext.
"""

import concurrent.futures
import hashlib
import logging
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from lib.citations import extract_full_text_links, extract_metadata
from lib.config import ServerConfig
from lib.publisher_router import PublisherRouter
from lib.rate_limiter import DownloadRateLimiter, RateLimitExceeded
from lib.retry import CircuitBreaker
from lib.stealth import (
    CURL_IMPERSONATE_VERSION,
    STEALTH_LAUNCH_ARGS,
    apply_stealth,
    get_curl_extra_fingerprints,
    get_stealth_context_options,
)

logger = logging.getLogger("sfu_library_mcp")

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
}


class DownloadError(Exception):
    """Raised when a PDF download fails."""
    pass


class PDFTextExtractionError(Exception):
    """Raised when text extraction from a PDF fails."""
    pass


@dataclass
class FetchResult:
    """Result from a single tier fetch attempt."""
    content: bytes
    status_code: int
    content_type: str
    tier_used: str
    url: str


class ArticleDownloader:
    """Downloads and caches PDFs, extracts text for LLM consumption."""

    def __init__(self, config: ServerConfig, cookies: dict | None = None,
                 rate_limiter: DownloadRateLimiter | None = None,
                 publisher_router: PublisherRouter | None = None):
        self.config = config
        self.cookies = cookies or {}
        self._session: requests.Session | None = None
        self.rate_limiter = rate_limiter
        self._publisher_router = publisher_router
        self._circuit_breaker = CircuitBreaker(threshold=3, timeout=120.0)

        # Probe which download tiers are actually importable at startup
        self._available_tiers: set[str] = {"requests"}  # always available
        try:
            import curl_cffi  # noqa: F401
            self._available_tiers.add("curl_cffi")
        except ImportError:
            logger.warning("curl_cffi not installed — Tier 1 (TLS impersonation) unavailable")
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
            self._available_tiers.add("playwright")
        except ImportError:
            logger.warning("playwright not installed — Tier 2 (headless browser) unavailable")

        configured = config.download_tiers
        available = [t for t in configured if t in self._available_tiers]
        skipped = [t for t in configured if t not in self._available_tiers]
        if skipped:
            logger.warning("Download tiers unavailable (missing packages): %s", skipped)
        logger.info("Active download tiers: %s", available)

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            for name, value in self.cookies.items():
                self._session.cookies.set(name, value)
        return self._session

    def update_cookies(self, cookies: dict) -> None:
        """Update session cookies (e.g. after re-auth)."""
        self.cookies = cookies
        self._session = None  # Force recreation

    def _cache_path(self, record_id: str) -> Path:
        """Deterministic cache path based on record ID hash."""
        h = hashlib.sha256(record_id.encode()).hexdigest()[:16]
        cache_dir = Path(self.config.download_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Cache path for %s: %s/%s.pdf", record_id, cache_dir, h)
        return cache_dir / f"{h}.pdf"

    def resolve_pdf_url(self, item: dict) -> str | None:
        """Resolve the best PDF URL from a PNX record.

        Priority: pdf_links > doi_url > source_links > html_links.
        Non-open-access URLs are wrapped with EZProxy prefix.
        """
        links = extract_full_text_links(item)
        if not links:
            logger.info("resolve_pdf_url: no links extracted from record")
            return None

        is_oa = links.get("open_access", False)
        url = None
        link_type = None

        # Priority order
        if links.get("pdf_links"):
            url = links["pdf_links"][0]
            link_type = "pdf_links"
        elif links.get("doi_url"):
            url = links["doi_url"]
            link_type = "doi_url"
        elif links.get("source_links"):
            url = links["source_links"][0]
            link_type = "source_links"
        elif links.get("html_links"):
            url = links["html_links"][0]
            link_type = "html_links"

        if not url:
            logger.info("resolve_pdf_url: links dict present but no usable URL found")
            return None

        logger.info("resolve_pdf_url: selected %s -> %s (open_access=%s)", link_type, url, is_oa)

        # Return the raw URL — download_pdf() handles EZProxy wrapping
        # as a fallback strategy when direct access fails.
        return url

    def resolve_all_pdf_urls(self, item: dict) -> list[str]:
        """Resolve ALL available PDF URLs from a PNX record, in priority order.

        Returns a deduplicated list of URLs: pdf_links, doi_url, source_links,
        html_links. Each URL is raw (no EZProxy wrapping) — download_pdf()
        handles proxy fallback per-URL.
        """
        links = extract_full_text_links(item)
        if not links:
            logger.info("resolve_all_pdf_urls: no links extracted from record")
            return []

        urls: list[str] = []
        seen: set[str] = set()

        def _add(url_or_list, label: str):
            items = url_or_list if isinstance(url_or_list, list) else [url_or_list]
            for u in items:
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
                    logger.debug("resolve_all_pdf_urls: added %s -> %s", label, u)

        # Priority order: pdf_links first, then DOI, source, html
        _add(links.get("pdf_links", []), "pdf_links")
        if links.get("doi_url"):
            _add(links["doi_url"], "doi_url")
        _add(links.get("source_links", []), "source_links")
        _add(links.get("html_links", []), "html_links")

        logger.info("resolve_all_pdf_urls: found %d URLs", len(urls))
        return urls

    # ─── Tiered fetch methods ──────────────────────────────────

    def _fetch_curl_cffi(self, url: str, cookies: dict | None = None) -> FetchResult:
        """Tier 1: Fetch using curl_cffi with Chrome TLS impersonation."""
        from curl_cffi import requests as cffi_requests

        merged_cookies = {**self.cookies, **(cookies or {})}
        logger.info("_fetch_curl_cffi: fetching %s", url)
        parsed = urlparse(url)
        headers = {**BROWSER_HEADERS, "Referer": f"{parsed.scheme}://{parsed.hostname}/"}
        resp = cffi_requests.get(
            url,
            headers=headers,
            cookies=merged_cookies,
            impersonate=CURL_IMPERSONATE_VERSION,
            extra_fp=get_curl_extra_fingerprints(),
            timeout=self.config.download_timeout,
            allow_redirects=True,
        )
        resp.raise_for_status()
        return FetchResult(
            content=resp.content,
            status_code=resp.status_code,
            content_type=resp.headers.get("Content-Type", "unknown"),
            tier_used="curl_cffi",
            url=str(resp.url),
        )

    def _fetch_playwright(self, url: str, cookies: dict | None = None) -> FetchResult:
        """Tier 2: Fetch using Playwright headless Chromium with stealth.

        Runs sync Playwright in a separate thread to avoid crashing when
        called inside an asyncio event loop (MCP server runs async).
        """
        merged_cookies = {**self.cookies, **(cookies or {})}
        logger.info("_fetch_playwright: fetching %s", url)
        pw_timeout = self.config.playwright_timeout

        def _run() -> FetchResult:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=STEALTH_LAUNCH_ARGS,
                )
                context = browser.new_context(
                    **get_stealth_context_options(BROWSER_HEADERS["User-Agent"]),
                )

                # Inject cookies — Playwright requires domain/path
                if merged_cookies:
                    parsed = urlparse(url)
                    pw_cookies = []
                    for name, value in merged_cookies.items():
                        pw_cookies.append({
                            "name": name,
                            "value": value,
                            "domain": parsed.hostname,
                            "path": "/",
                        })
                    context.add_cookies(pw_cookies)

                page = context.new_page()
                apply_stealth(page)
                pdf_content = None
                pdf_content_type = "unknown"

                def handle_response(response):
                    nonlocal pdf_content, pdf_content_type
                    ct = response.headers.get("content-type", "")
                    if "application/pdf" in ct or response.url.endswith(".pdf"):
                        try:
                            pdf_content = response.body()
                            pdf_content_type = ct
                        except Exception:
                            pass

                page.on("response", handle_response)

                try:
                    resp = page.goto(url, timeout=pw_timeout * 1000, wait_until="networkidle")
                except Exception as e:
                    browser.close()
                    raise DownloadError(f"Playwright navigation failed: {e}")

                # If we caught a PDF via response interception, use that
                if pdf_content and pdf_content[:5] == b"%PDF-":
                    result = FetchResult(
                        content=pdf_content,
                        status_code=200,
                        content_type=pdf_content_type,
                        tier_used="playwright",
                        url=page.url,
                    )
                    browser.close()
                    return result

                # Otherwise, get the page body (may be a PDF loaded directly)
                if resp:
                    body = resp.body()
                    ct = resp.headers.get("content-type", "unknown")
                    status = resp.status
                else:
                    body = b""
                    ct = "unknown"
                    status = 0

                browser.close()

                if status >= 400:
                    raise DownloadError(f"Playwright HTTP {status} for {url}")

                return FetchResult(
                    content=body,
                    status_code=status,
                    content_type=ct,
                    tier_used="playwright",
                    url=url,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            return future.result(timeout=pw_timeout + 15)

    def _fetch_requests(self, url: str, cookies: dict | None = None) -> FetchResult:
        """Tier 3: Fetch using requests (legacy fallback)."""
        merged_cookies = {**self.cookies, **(cookies or {})}
        logger.info("_fetch_requests: fetching %s", url)

        parsed = urlparse(url)
        session = requests.Session()
        session.headers.update(BROWSER_HEADERS)
        session.headers["Referer"] = f"{parsed.scheme}://{parsed.hostname}/"
        for name, value in merged_cookies.items():
            session.cookies.set(name, value)

        resp = session.get(url, timeout=self.config.download_timeout, stream=True)
        resp.raise_for_status()

        content = b""
        for chunk in resp.iter_content(chunk_size=8192):
            content += chunk

        return FetchResult(
            content=content,
            status_code=resp.status_code,
            content_type=resp.headers.get("Content-Type", "unknown"),
            tier_used="requests",
            url=resp.url,
        )

    def _tiered_fetch(self, url: str, cookies: dict | None = None) -> FetchResult:
        """Iterate through configured tiers in order, returning first success."""
        # Circuit breaker check
        if not self._circuit_breaker.can_proceed():
            raise DownloadError(
                "Downloads paused — too many recent failures (circuit breaker open)"
            )

        tier_methods = {
            "curl_cffi": self._fetch_curl_cffi,
            "playwright": self._fetch_playwright,
            "requests": self._fetch_requests,
        }

        # Only attempt tiers whose packages are actually importable
        base_tiers = [t for t in self.config.download_tiers if t in self._available_tiers]
        if not base_tiers:
            raise DownloadError(
                "No download tiers available. Install curl_cffi and/or playwright, "
                "then restart the server."
            )

        # Consult publisher router for optimized tier ordering
        if self._publisher_router:
            tiers = self._publisher_router.get_tier_order(url, base_tiers)
            if tiers != base_tiers:
                logger.info(
                    "_tiered_fetch: router reordered tiers %s -> %s for %s",
                    base_tiers, tiers, url,
                )
        else:
            tiers = base_tiers

        last_error = None
        attempt_index = 0

        for tier_name in tiers:
            method = tier_methods.get(tier_name)
            if method is None:
                logger.warning("_tiered_fetch: unknown tier '%s', skipping", tier_name)
                continue

            # Inter-tier delay to prevent rapid-fire cascade
            if attempt_index > 0:
                delay = random.uniform(1.5, 3.0)
                logger.info("_tiered_fetch: waiting %.1fs before trying tier '%s'", delay, tier_name)
                time.sleep(delay)

            attempt_index += 1

            try:
                result = method(url, cookies)
                logger.info("_tiered_fetch: tier '%s' succeeded for %s", tier_name, url)
                self._circuit_breaker.record_success()
                return result
            except Exception as e:
                logger.warning("_tiered_fetch: tier '%s' failed for %s: %s", tier_name, url, e)
                last_error = e
                # Record failure in publisher router
                if self._publisher_router:
                    self._publisher_router.record_failure(url, tier_name)

        self._circuit_breaker.record_failure()
        raise DownloadError(
            f"All download tiers failed for {url}. Last error: {last_error}"
        )

    # ─── Login page detection ─────────────────────────────────

    @staticmethod
    def _is_login_page(content: bytes) -> bool:
        """Detect if response body is an authentication/login page.

        Checks the first 2 KB for known login page signatures from
        EZProxy, CAS, and common publisher paywalls.
        """
        head = content[:2048].lower()
        signatures = [
            b"<title>authentication required</title>",
            b"<title>login</title>",
            b"cas " + b"\xe2\x80\x93" + b" central authentication service",
            b"proxy.lib.sfu.ca/login",
            b"id=\"username\"",
        ]
        return any(sig in head for sig in signatures)

    # ─── PDF validation ────────────────────────────────────────

    def _write_and_validate_pdf(self, resp, cache_path: Path, record_id: str) -> str | None:
        """Write response content to cache and validate PDF magic bytes.

        Returns None on success, or an error string on failure.
        Works with requests.Response objects (legacy path).
        """
        redirect_count = len(resp.history)
        content_type = resp.headers.get("Content-Type", "unknown")
        logger.info(
            "download_pdf: HTTP 200, content_type=%s, redirects=%d, final_url=%s",
            content_type, redirect_count, resp.url,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        with open(cache_path, "rb") as f:
            header = f.read(200)
        if not header[:5] == b"%PDF-":
            logger.warning(
                "Downloaded content is not a PDF for %s: content_type=%s final_url=%s body_preview=%r",
                record_id, content_type, resp.url, header[:200],
            )
            cache_path.unlink(missing_ok=True)
            return (
                f"Downloaded content is not a PDF (Content-Type: {content_type}, "
                f"final URL: {resp.url}). May be a login page or HTML redirect."
            )
        return None

    def _write_and_validate_fetch_result(self, fetch_result: FetchResult, cache_path: Path, record_id: str) -> str | None:
        """Write FetchResult content to cache and validate PDF magic bytes.

        Returns None on success, or an error string on failure.
        """
        logger.info(
            "download_pdf: tier=%s, status=%d, content_type=%s, url=%s",
            fetch_result.tier_used, fetch_result.status_code,
            fetch_result.content_type, fetch_result.url,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(fetch_result.content)

        if not fetch_result.content[:5] == b"%PDF-":
            logger.warning(
                "Downloaded content is not a PDF for %s: content_type=%s tier=%s url=%s body_preview=%r",
                record_id, fetch_result.content_type, fetch_result.tier_used,
                fetch_result.url, fetch_result.content[:200],
            )
            cache_path.unlink(missing_ok=True)
            return (
                f"Downloaded content is not a PDF (Content-Type: {fetch_result.content_type}, "
                f"tier: {fetch_result.tier_used}, URL: {fetch_result.url}). "
                f"May be a login page or HTML redirect."
            )
        return None

    # ─── Main download methods ─────────────────────────────────

    def download_pdf(
        self,
        url: str,
        record_id: str,
        metadata: dict | None = None,
        copy_to_host: bool = True,
    ) -> dict:
        """Download a PDF from the given URL using tiered strategy.

        Returns dict with keys: success, container_path, host_path,
        size_bytes, tier_used, error.
        """
        cache_path = self._cache_path(record_id)
        result = {
            "success": False,
            "container_path": None,
            "host_path": None,
            "size_bytes": 0,
            "tier_used": None,
            "error": None,
        }

        # Check cache first
        if cache_path.exists() and cache_path.stat().st_size > 0:
            logger.info("Cache hit for %s: %s", record_id, cache_path)
            result["success"] = True
            result["container_path"] = str(cache_path)
            result["size_bytes"] = cache_path.stat().st_size
            result["tier_used"] = "cache"
            if copy_to_host:
                result["host_path"] = self._copy_to_host(cache_path, record_id, metadata)
            return result

        # Log cookie state for diagnostics
        cookie_names = list(self.cookies.keys())
        has_proxy_cookies = any(
            "ezproxy" in n.lower() or "proxy" in n.lower()
            for n in cookie_names
        )
        logger.info(
            "download_pdf: cookies=%s, has_proxy_cookies=%s",
            cookie_names, has_proxy_cookies,
        )
        if not has_proxy_cookies:
            logger.warning("download_pdf: no proxy cookies found — auth may fail for proxied URLs")

        logger.info("Downloading PDF from %s for record %s", url, record_id)

        # Rate limiter check
        if self.rate_limiter:
            try:
                self.rate_limiter.acquire(url)
            except RateLimitExceeded as e:
                result["error"] = str(e)
                return result

        # Helper: check if an EZProxy fallback would be useful
        url_already_proxied = self.config.ezproxy_prefix in url
        def _should_try_ezproxy() -> bool:
            """Don't try EZProxy if URL is already proxied or we have no proxy cookies."""
            if url_already_proxied:
                return False
            if not has_proxy_cookies:
                logger.warning(
                    "download_pdf: skipping EZProxy fallback — no proxy cookies available. "
                    "Re-authenticate to establish EZProxy session."
                )
                return False
            return True

        # Helper: attempt EZProxy fetch with full validation
        def _try_ezproxy(direct_error: str) -> FetchResult | None:
            """Try downloading through EZProxy. Returns FetchResult on success, None on failure."""
            if not _should_try_ezproxy():
                return None
            proxied_url = self.config.ezproxy_prefix + url
            logger.info("download_pdf: direct failed (%s), retrying via EZProxy: %s", direct_error, proxied_url)
            try:
                proxy_result = self._tiered_fetch(proxied_url)
            except DownloadError as retry_err:
                logger.error("download_pdf: EZProxy fallback also failed: %s", retry_err)
                result["error"] = (
                    f"Both direct and EZProxy downloads failed. "
                    f"Direct: {direct_error}. EZProxy: {retry_err}"
                )
                return None
            if self._is_login_page(proxy_result.content):
                logger.warning("download_pdf: EZProxy returned login page for %s", record_id)
                result["error"] = (
                    f"Both direct and EZProxy downloads failed. "
                    f"Direct: {direct_error}. "
                    f"EZProxy: returned a login page (session may be expired)."
                )
                return None
            proxy_validation = self._write_and_validate_fetch_result(proxy_result, cache_path, record_id)
            if proxy_validation:
                result["error"] = (
                    f"Both direct and EZProxy downloads failed. "
                    f"Direct: {direct_error}. EZProxy: {proxy_validation}"
                )
                return None
            return proxy_result

        # ── Phase 1: Try direct URL ──────────────────────────────
        fetch_result = None
        direct_error = None
        try:
            fetch_result = self._tiered_fetch(url)
        except DownloadError as e:
            direct_error = str(e)

        # Detect login page on direct fetch
        if fetch_result and self._is_login_page(fetch_result.content):
            logger.warning(
                "download_pdf: direct fetch returned login/auth page for %s (tier=%s, url=%s)",
                record_id, fetch_result.tier_used, fetch_result.url,
            )
            direct_error = "Direct URL returned a login/auth page"
            fetch_result = None

        # Validate PDF on direct fetch
        if fetch_result:
            validation_error = self._write_and_validate_fetch_result(fetch_result, cache_path, record_id)
            if validation_error:
                direct_error = validation_error
                fetch_result = None
            elif self._publisher_router and fetch_result.tier_used in ("curl_cffi", "requests"):
                # Confirmed PDF via direct tier — whitelist this domain
                self._publisher_router.record_success(url, fetch_result.tier_used)

        # ── Phase 2: If direct failed, try EZProxy fallback ──────
        if not fetch_result and direct_error:
            fetch_result = _try_ezproxy(direct_error)
            if not fetch_result:
                # _try_ezproxy already set result["error"] if it ran,
                # otherwise set the direct error
                if not result["error"]:
                    result["error"] = direct_error
                return result

        size = cache_path.stat().st_size
        logger.info("Download success for %s: %d bytes at %s (tier: %s)", record_id, size, cache_path, fetch_result.tier_used)
        result["success"] = True
        result["container_path"] = str(cache_path)
        result["size_bytes"] = size
        result["tier_used"] = fetch_result.tier_used

        # Record successful download for rate limiting
        if self.rate_limiter:
            self.rate_limiter.record_download(url)

        if copy_to_host:
            result["host_path"] = self._copy_to_host(cache_path, record_id, metadata)

        return result

    def download_from_direct_url(
        self,
        url: str,
        cookies: dict | None = None,
        filename: str | None = None,
    ) -> dict:
        """Download a PDF from a direct URL (e.g. from capture server or MCP tool).

        Uses tiered fetch with optional extra cookies (e.g. from Chrome extension).

        Returns dict with keys: success, container_path, size_bytes, tier_used, error.
        """
        # Use URL hash as record ID for caching
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_path = self._cache_path(url_hash)

        result = {
            "success": False,
            "container_path": None,
            "size_bytes": 0,
            "tier_used": None,
            "error": None,
            "filename": filename,
        }

        # Check cache first
        if cache_path.exists() and cache_path.stat().st_size > 0:
            logger.info("Cache hit for direct URL %s: %s", url[:80], cache_path)
            result["success"] = True
            result["container_path"] = str(cache_path)
            result["size_bytes"] = cache_path.stat().st_size
            result["tier_used"] = "cache"
            return result

        logger.info("download_from_direct_url: %s", url)

        # Rate limiter check
        if self.rate_limiter:
            try:
                self.rate_limiter.acquire(url)
            except RateLimitExceeded as e:
                result["error"] = str(e)
                return result

        try:
            fetch_result = self._tiered_fetch(url, cookies)
        except DownloadError as e:
            result["error"] = str(e)
            return result

        # Detect login page before writing to cache
        if self._is_login_page(fetch_result.content):
            logger.warning(
                "download_from_direct_url: response is a login/auth page (tier=%s, url=%s)",
                fetch_result.tier_used, fetch_result.url,
            )
            result["error"] = (
                "Download returned a login page instead of a PDF. "
                "EZProxy session may have expired — try re-authenticating."
            )
            return result

        validation_error = self._write_and_validate_fetch_result(fetch_result, cache_path, url_hash)
        if validation_error:
            result["error"] = validation_error
            return result

        size = cache_path.stat().st_size
        logger.info("Direct URL download success: %d bytes at %s (tier: %s)", size, cache_path, fetch_result.tier_used)
        result["success"] = True
        result["container_path"] = str(cache_path)
        result["size_bytes"] = size
        result["tier_used"] = fetch_result.tier_used

        # Record successful download for rate limiting
        if self.rate_limiter:
            self.rate_limiter.record_download(url)

        return result

    # ─── Text extraction ───────────────────────────────────────

    def extract_text(self, pdf_path: str, max_chars: int | None = None) -> str:
        """Extract text from a PDF using pdftotext.

        Args:
            pdf_path: Path to the PDF file.
            max_chars: Maximum characters to return (default: config value).

        Returns:
            Extracted text content.

        Raises:
            PDFTextExtractionError: If extraction fails.
        """
        if max_chars is None:
            max_chars = self.config.max_pdf_text_chars

        if not os.path.exists(pdf_path):
            raise PDFTextExtractionError(f"PDF file not found: {pdf_path}")

        logger.info("Extracting text from %s", pdf_path)
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", pdf_path, "-"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            raise PDFTextExtractionError(
                "pdftotext not found. Install poppler-utils: apt-get install poppler-utils"
            )
        except subprocess.TimeoutExpired:
            raise PDFTextExtractionError("Text extraction timed out after 60s")

        if result.returncode != 0:
            raise PDFTextExtractionError(f"pdftotext failed: {result.stderr}")

        text = result.stdout
        logger.info("Text extraction completed: %d chars", len(text))

        if len(text) > max_chars:
            logger.warning("Text truncated from %d to %d chars", len(text), max_chars)
            text = text[:max_chars] + f"\n\n[... Text truncated at {max_chars:,} characters. Full PDF available at {pdf_path}]"

        return text

    # ─── Host copy ─────────────────────────────────────────────

    def _copy_to_host(self, pdf_path: Path, record_id: str, metadata: dict | None) -> str | None:
        """Copy PDF to host Downloads folder with a readable filename.

        Returns the host path on success, None on failure.
        """
        host_dir = Path(self.config.host_download_dir)
        if not host_dir.exists():
            logger.warning("Host download dir not available: %s", host_dir)
            return None

        # Build readable filename from metadata
        if metadata:
            author = ""
            authors = metadata.get("authors") or metadata.get("creators", [])
            if authors:
                first_author = authors[0].split("$$")[0].strip()
                # Take last name only
                if "," in first_author:
                    author = first_author.split(",")[0].strip()
                else:
                    parts = first_author.split()
                    author = parts[-1] if parts else ""

            year = metadata.get("date", "")[:4]
            title = metadata.get("title", "")[:80]
            # Sanitize for filesystem
            title = re.sub(r'[<>:"/\\|?*]', '', title).strip()

            parts = [p for p in [author, year, title] if p]
            filename = " - ".join(parts) + ".pdf" if parts else f"{record_id}.pdf"
        else:
            filename = f"{record_id}.pdf"

        # Sanitize filename length
        if len(filename) > 200:
            filename = filename[:196] + ".pdf"

        dest = host_dir / filename
        try:
            shutil.copy2(pdf_path, dest)
            logger.info("Copied PDF to host: %s", dest)
            return str(dest)
        except OSError as e:
            logger.error("Failed to copy PDF to host: %s", e)
            return None
