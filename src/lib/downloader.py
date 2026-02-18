"""PDF download, caching, and text extraction for library articles.

Provides ArticleDownloader that resolves full-text URLs from PNX records,
downloads PDFs through EZProxy, caches them locally, optionally copies
to the host Downloads folder, and extracts text via pdftotext.
"""

import hashlib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import requests

from lib.citations import extract_full_text_links, extract_metadata
from lib.config import ServerConfig

logger = logging.getLogger("sfu_library_mcp")


class DownloadError(Exception):
    """Raised when a PDF download fails."""
    pass


class PDFTextExtractionError(Exception):
    """Raised when text extraction from a PDF fails."""
    pass


class ArticleDownloader:
    """Downloads and caches PDFs, extracts text for LLM consumption."""

    def __init__(self, config: ServerConfig, cookies: dict | None = None):
        self.config = config
        self.cookies = cookies or {}
        self._session: requests.Session | None = None

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

    def download_pdf(
        self,
        url: str,
        record_id: str,
        metadata: dict | None = None,
        copy_to_host: bool = True,
    ) -> dict:
        """Download a PDF from the given URL.

        Returns dict with keys: success, container_path, host_path,
        size_bytes, error.
        """
        cache_path = self._cache_path(record_id)
        result = {
            "success": False,
            "container_path": None,
            "host_path": None,
            "size_bytes": 0,
            "error": None,
        }

        # Check cache first
        if cache_path.exists() and cache_path.stat().st_size > 0:
            logger.info("Cache hit for %s: %s", record_id, cache_path)
            result["success"] = True
            result["container_path"] = str(cache_path)
            result["size_bytes"] = cache_path.stat().st_size
            if copy_to_host:
                result["host_path"] = self._copy_to_host(cache_path, record_id, metadata)
            return result

        # Log cookie state for diagnostics
        cookie_names = list(self.session.cookies.keys())
        ezproxy_cookies = [n for n in cookie_names if "ezproxy" in n.lower()]
        logger.info(
            "download_pdf: cookies=%s, ezproxy_cookies=%s",
            cookie_names, ezproxy_cookies or "NONE",
        )
        if not ezproxy_cookies:
            logger.warning("download_pdf: no EZProxy cookies found — auth may fail for proxied URLs")

        logger.info("Downloading PDF from %s for record %s", url, record_id)
        try:
            resp = self.session.get(url, timeout=self.config.download_timeout, stream=True)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            result["error"] = f"Download timed out after {self.config.download_timeout}s"
            logger.error("Download timeout for %s: %s", record_id, url)
            return result
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            content_type = (e.response.headers.get("Content-Type", "unknown")
                           if e.response is not None else "unknown")
            final_url = e.response.url if e.response is not None else url
            redirect_count = len(e.response.history) if e.response is not None else 0
            logger.error(
                "Download HTTP error for %s: status=%s content_type=%s final_url=%s redirects=%d",
                record_id, status, content_type, final_url, redirect_count,
            )
            result["error"] = (
                f"HTTP error {status} (Content-Type: {content_type}, "
                f"final URL: {final_url})"
            )
            return result
        except requests.exceptions.RequestException as e:
            result["error"] = f"Download failed: {e}"
            logger.error("Download failed for %s: %s", record_id, e)
            return result

        # Log successful HTTP response diagnostics
        redirect_count = len(resp.history)
        content_type = resp.headers.get("Content-Type", "unknown")
        logger.info(
            "download_pdf: HTTP 200, content_type=%s, redirects=%d, final_url=%s",
            content_type, redirect_count, resp.url,
        )

        # Write to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Validate PDF magic bytes
        with open(cache_path, "rb") as f:
            header = f.read(200)
        if not header[:5] == b"%PDF-":
            logger.warning(
                "Downloaded content is not a PDF for %s: content_type=%s final_url=%s body_preview=%r",
                record_id, content_type, resp.url, header[:200],
            )
            cache_path.unlink(missing_ok=True)
            result["error"] = (
                f"Downloaded content is not a PDF (Content-Type: {content_type}, "
                f"final URL: {resp.url}). May be a login page or HTML redirect."
            )
            return result

        size = cache_path.stat().st_size
        logger.info("Download success for %s: %d bytes at %s", record_id, size, cache_path)
        result["success"] = True
        result["container_path"] = str(cache_path)
        result["size_bytes"] = size

        if copy_to_host:
            result["host_path"] = self._copy_to_host(cache_path, record_id, metadata)

        return result

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
