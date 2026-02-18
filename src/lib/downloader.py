"""PDF download, caching, and text extraction for library articles.

Provides ArticleDownloader that resolves full-text URLs from PNX records,
downloads PDFs through EZProxy using a tiered strategy (curl_cffi → Playwright
→ requests), caches them locally, optionally copies to the host Downloads
folder, and extracts text via pdftotext.
"""

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
                 rate_limiter: DownloadRateLimiter | None = None):
        self.config = config
        self.cookies = cookies or {}
        self._session: requests.Session | None = None
        self.rate_limiter = rate_limiter
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

        # Wrap with EZProxy if not open access and not already wrapped
        if not is_oa and self.config.ezproxy_prefix not in url:
            original_url = url
            url = self.config.ezproxy_prefix + url
            logger.info("resolve_pdf_url: EZProxy wrapped %s -> %s", original_url, url)

        return url

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
        """Tier 2: Fetch using Playwright headless Chromium with stealth."""
        from playwright.sync_api import sync_playwright

        merged_cookies = {**self.cookies, **(cookies or {})}
        logger.info("_fetch_playwright: fetching %s", url)

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
                from urllib.parse import urlparse
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
                resp = page.goto(url, timeout=self.config.playwright_timeout * 1000, wait_until="networkidle")
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
        tiers = [t for t in self.config.download_tiers if t in self._available_tiers]
        if not tiers:
            raise DownloadError(
                "No download tiers available. Install curl_cffi and/or playwright, "
                "then restart the server."
            )

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

        self._circuit_breaker.record_failure()
        raise DownloadError(
            f"All download tiers failed for {url}. Last error: {last_error}"
        )

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
        ezproxy_cookies = [n for n in cookie_names if "ezproxy" in n.lower()]
        logger.info(
            "download_pdf: cookies=%s, ezproxy_cookies=%s",
            cookie_names, ezproxy_cookies or "NONE",
        )
        if not ezproxy_cookies:
            logger.warning("download_pdf: no EZProxy cookies found — auth may fail for proxied URLs")

        logger.info("Downloading PDF from %s for record %s", url, record_id)

        # Rate limiter check
        if self.rate_limiter:
            try:
                self.rate_limiter.acquire(url)
            except RateLimitExceeded as e:
                result["error"] = str(e)
                return result

        # Try tiered fetch
        try:
            fetch_result = self._tiered_fetch(url)
        except DownloadError as e:
            # On failure, retry through EZProxy if not already proxied
            if self.config.ezproxy_prefix not in url:
                proxied_url = self.config.ezproxy_prefix + url
                logger.info("download_pdf: tiered fetch failed, retrying through EZProxy: %s", proxied_url)
                try:
                    fetch_result = self._tiered_fetch(proxied_url)
                except DownloadError as retry_err:
                    logger.error("download_pdf: EZProxy retry also failed: %s", retry_err)
                    result["error"] = (
                        f"All tiers failed (retried via EZProxy, also failed). "
                        f"Original URL: {url}. Last error: {retry_err}"
                    )
                    return result
            else:
                result["error"] = str(e)
                return result

        # Write and validate
        validation_error = self._write_and_validate_fetch_result(fetch_result, cache_path, record_id)
        if validation_error:
            # If not already proxied, retry with EZProxy
            if self.config.ezproxy_prefix not in url:
                proxied_url = self.config.ezproxy_prefix + url
                logger.info("download_pdf: not a PDF, retrying through EZProxy: %s", proxied_url)
                try:
                    fetch_result = self._tiered_fetch(proxied_url)
                    validation_error = self._write_and_validate_fetch_result(fetch_result, cache_path, record_id)
                    if validation_error:
                        result["error"] = validation_error
                        return result
                except DownloadError as e:
                    result["error"] = validation_error
                    return result
            else:
                result["error"] = validation_error
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
