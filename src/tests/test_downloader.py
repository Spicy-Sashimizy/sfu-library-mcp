"""Unit tests for the downloader module."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import requests

from lib.config import ServerConfig
from lib.downloader import (
    ArticleDownloader,
    DownloadError,
    FetchResult,
    PDFTextExtractionError,
)


@pytest.fixture
def dl_config():
    return ServerConfig(
        download_dir="/tmp/test-dl-cache",
        host_download_dir="/tmp/test-host-downloads",
        download_timeout=30,
        max_pdf_text_chars=1000,
        ezproxy_prefix="https://proxy.lib.sfu.ca/login?url=",
        download_tiers=["curl_cffi", "playwright", "requests"],
    )


@pytest.fixture
def downloader(dl_config):
    return ArticleDownloader(dl_config, cookies={"session": "abc123"})


class TestFetchResult:
    """Verify FetchResult dataclass creation and fields."""

    def test_create_fetch_result(self):
        fr = FetchResult(
            content=b"%PDF-1.4 test",
            status_code=200,
            content_type="application/pdf",
            tier_used="curl_cffi",
            url="https://example.com/test.pdf",
        )
        assert fr.content == b"%PDF-1.4 test"
        assert fr.status_code == 200
        assert fr.content_type == "application/pdf"
        assert fr.tier_used == "curl_cffi"
        assert fr.url == "https://example.com/test.pdf"

    def test_fetch_result_fields(self):
        fr = FetchResult(
            content=b"", status_code=404,
            content_type="text/html", tier_used="requests",
            url="https://example.com/missing",
        )
        assert fr.status_code == 404
        assert fr.tier_used == "requests"


class TestCachePath:
    def test_pdf_cache_path_deterministic(self, downloader):
        """Same record_id should always produce the same path."""
        path1 = downloader._cache_path("record_abc")
        path2 = downloader._cache_path("record_abc")
        assert path1 == path2

    def test_pdf_cache_path_different_ids(self, downloader):
        """Different record_ids should produce different paths."""
        path1 = downloader._cache_path("record_abc")
        path2 = downloader._cache_path("record_xyz")
        assert path1 != path2


class TestEZProxyWrapping:
    def test_wrap_ezproxy_adds_prefix(self, downloader, sample_article_record):
        """Non-OA URLs should get EZProxy prefix."""
        # Make it non-OA by removing openaccess
        sample_article_record["pnx"]["links"]["openaccess"] = []
        url = downloader.resolve_pdf_url(sample_article_record)
        assert url is not None
        assert url.startswith("https://proxy.lib.sfu.ca/login?url=")

    def test_wrap_ezproxy_idempotent(self, downloader, sample_article_record):
        """Already-wrapped URL should not be double-wrapped."""
        sample_article_record["pnx"]["links"]["openaccess"] = []
        sample_article_record["pnx"]["links"]["linktopdf"] = [
            "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf"
        ]
        url = downloader.resolve_pdf_url(sample_article_record)
        assert url is not None
        assert url.count("proxy.lib.sfu.ca") == 1


class TestResolvePdfUrl:
    def test_resolve_pdf_url_priority_pdf(self, downloader, sample_article_record):
        """pdf_links should be chosen first."""
        url = downloader.resolve_pdf_url(sample_article_record)
        assert url is not None
        assert "article.pdf" in url

    def test_resolve_pdf_url_priority_doi(self, downloader, sample_article_record):
        """doi_url should be chosen when no pdf_links."""
        sample_article_record["pnx"]["links"]["linktopdf"] = []
        url = downloader.resolve_pdf_url(sample_article_record)
        assert url is not None
        assert "doi.org" in url

    def test_resolve_pdf_url_priority_source(self, downloader, sample_pnx_record):
        """source_links should be chosen when no pdf or doi."""
        url = downloader.resolve_pdf_url(sample_pnx_record)
        assert url is not None
        assert "fulltext" in url

    def test_resolve_pdf_url_wraps_non_oa(self, downloader, sample_pnx_record):
        """Non-open-access URLs should be wrapped with EZProxy."""
        url = downloader.resolve_pdf_url(sample_pnx_record)
        assert url is not None
        assert url.startswith("https://proxy.lib.sfu.ca/login?url=")

    def test_resolve_pdf_url_no_wrap_oa(self, downloader, sample_article_record):
        """Open access URLs should NOT be wrapped."""
        url = downloader.resolve_pdf_url(sample_article_record)
        assert url is not None
        # OA record has openaccess set, so should not wrap
        assert "proxy.lib.sfu.ca" not in url

    def test_resolve_pdf_url_returns_none(self, downloader):
        """Empty links dict should return None."""
        item = {
            "pnx": {
                "links": {},
                "addata": {},
            }
        }
        url = downloader.resolve_pdf_url(item)
        assert url is None


class TestTieredFetch:
    """Test the _tiered_fetch orchestrator."""

    def test_first_tier_succeeds(self, downloader):
        """When first tier succeeds, return immediately."""
        expected = FetchResult(
            content=b"%PDF-1.4 data", status_code=200,
            content_type="application/pdf", tier_used="curl_cffi",
            url="https://example.com/test.pdf",
        )
        with patch.object(downloader, "_fetch_curl_cffi", return_value=expected) as mock_cffi:
            result = downloader._tiered_fetch("https://example.com/test.pdf")
        assert result.tier_used == "curl_cffi"
        mock_cffi.assert_called_once()

    def test_fallback_to_second_tier(self, downloader):
        """When first tier fails, should try second tier."""
        expected = FetchResult(
            content=b"%PDF-1.4 data", status_code=200,
            content_type="application/pdf", tier_used="playwright",
            url="https://example.com/test.pdf",
        )
        with patch.object(downloader, "_fetch_curl_cffi", side_effect=Exception("TLS error")), \
             patch.object(downloader, "_fetch_playwright", return_value=expected) as mock_pw:
            result = downloader._tiered_fetch("https://example.com/test.pdf")
        assert result.tier_used == "playwright"
        mock_pw.assert_called_once()

    def test_fallback_to_third_tier(self, downloader):
        """When first two tiers fail, should try third."""
        expected = FetchResult(
            content=b"%PDF-1.4 data", status_code=200,
            content_type="application/pdf", tier_used="requests",
            url="https://example.com/test.pdf",
        )
        with patch.object(downloader, "_fetch_curl_cffi", side_effect=Exception("fail")), \
             patch.object(downloader, "_fetch_playwright", side_effect=Exception("fail")), \
             patch.object(downloader, "_fetch_requests", return_value=expected) as mock_req:
            result = downloader._tiered_fetch("https://example.com/test.pdf")
        assert result.tier_used == "requests"
        mock_req.assert_called_once()

    def test_all_tiers_fail_raises(self, downloader):
        """When all tiers fail, should raise DownloadError."""
        with patch.object(downloader, "_fetch_curl_cffi", side_effect=Exception("fail1")), \
             patch.object(downloader, "_fetch_playwright", side_effect=Exception("fail2")), \
             patch.object(downloader, "_fetch_requests", side_effect=Exception("fail3")):
            with pytest.raises(DownloadError, match="All download tiers failed"):
                downloader._tiered_fetch("https://example.com/test.pdf")

    def test_respects_tier_order(self, downloader):
        """Should use configured tier order."""
        downloader.config.download_tiers = ["requests", "curl_cffi"]
        expected = FetchResult(
            content=b"%PDF-1.4", status_code=200,
            content_type="application/pdf", tier_used="requests",
            url="https://example.com/test.pdf",
        )
        with patch.object(downloader, "_fetch_requests", return_value=expected) as mock_req, \
             patch.object(downloader, "_fetch_curl_cffi") as mock_cffi:
            result = downloader._tiered_fetch("https://example.com/test.pdf")
        assert result.tier_used == "requests"
        mock_req.assert_called_once()
        mock_cffi.assert_not_called()

    def test_skips_unknown_tier(self, downloader):
        """Unknown tier names should be skipped gracefully."""
        downloader.config.download_tiers = ["nonexistent", "requests"]
        expected = FetchResult(
            content=b"%PDF-1.4", status_code=200,
            content_type="application/pdf", tier_used="requests",
            url="https://example.com/test.pdf",
        )
        with patch.object(downloader, "_fetch_requests", return_value=expected):
            result = downloader._tiered_fetch("https://example.com/test.pdf")
        assert result.tier_used == "requests"

    def test_passes_cookies_to_tier(self, downloader):
        """Cookies should be forwarded to tier methods."""
        extra_cookies = {"extra": "cookie"}
        expected = FetchResult(
            content=b"%PDF-1.4", status_code=200,
            content_type="application/pdf", tier_used="curl_cffi",
            url="https://example.com/test.pdf",
        )
        with patch.object(downloader, "_fetch_curl_cffi", return_value=expected) as mock_cffi:
            downloader._tiered_fetch("https://example.com/test.pdf", cookies=extra_cookies)
        mock_cffi.assert_called_once_with("https://example.com/test.pdf", extra_cookies)


class TestCurlCffiFetch:
    """Test curl_cffi tier with mocked curl_cffi library."""

    @patch("lib.downloader.ArticleDownloader._fetch_curl_cffi")
    def test_curl_cffi_returns_fetch_result(self, mock_method, downloader):
        """curl_cffi tier should return a FetchResult."""
        mock_method.return_value = FetchResult(
            content=b"%PDF-1.4 data", status_code=200,
            content_type="application/pdf", tier_used="curl_cffi",
            url="https://example.com/test.pdf",
        )
        result = downloader._fetch_curl_cffi("https://example.com/test.pdf")
        assert result.tier_used == "curl_cffi"
        assert result.content == b"%PDF-1.4 data"


class TestPlaywrightFetch:
    """Test Playwright tier with mocked playwright library."""

    @patch("lib.downloader.ArticleDownloader._fetch_playwright")
    def test_playwright_returns_fetch_result(self, mock_method, downloader):
        """Playwright tier should return a FetchResult."""
        mock_method.return_value = FetchResult(
            content=b"%PDF-1.4 data", status_code=200,
            content_type="application/pdf", tier_used="playwright",
            url="https://example.com/test.pdf",
        )
        result = downloader._fetch_playwright("https://example.com/test.pdf")
        assert result.tier_used == "playwright"


class TestDownloadPdf:
    def test_download_pdf_success_via_tiered_fetch(self, downloader, tmp_path):
        """Successful tiered download should write PDF and return success."""
        downloader.config.download_dir = str(tmp_path)

        fetch_result = FetchResult(
            content=b"%PDF-1.4 fake content",
            status_code=200,
            content_type="application/pdf",
            tier_used="curl_cffi",
            url="https://example.com/test.pdf",
        )
        with patch.object(downloader, "_tiered_fetch", return_value=fetch_result):
            result = downloader.download_pdf(
                "https://example.com/test.pdf", "rec_001", copy_to_host=False
            )
        assert result["success"] is True
        assert result["container_path"] is not None
        assert result["size_bytes"] > 0
        assert result["tier_used"] == "curl_cffi"

    def test_download_pdf_not_a_pdf(self, downloader, tmp_path):
        """HTML response should be rejected with enriched error."""
        downloader.config.download_dir = str(tmp_path)

        fetch_result = FetchResult(
            content=b"<html>Login page</html>",
            status_code=200,
            content_type="text/html",
            tier_used="curl_cffi",
            url="https://proxy.lib.sfu.ca/login",
        )
        # URL contains ezproxy prefix so no EZProxy retry
        with patch.object(downloader, "_tiered_fetch", return_value=fetch_result):
            result = downloader.download_pdf(
                "https://proxy.lib.sfu.ca/login?url=https://example.com/test.pdf",
                "rec_002",
                copy_to_host=False,
            )
        assert result["success"] is False
        assert "not a PDF" in result["error"]
        assert "Content-Type" in result["error"]

    def test_download_pdf_all_tiers_fail(self, downloader):
        """All tiers failing should return error."""
        with patch.object(downloader, "_tiered_fetch", side_effect=DownloadError("All download tiers failed for url. Last error: timeout")):
            result = downloader.download_pdf(
                "https://proxy.lib.sfu.ca/login?url=https://example.com/nope.pdf",
                "rec_003",
                copy_to_host=False,
            )
        assert result["success"] is False
        assert "All download tiers failed" in result["error"]

    def test_download_pdf_cache_hit(self, downloader, tmp_path):
        """Cached PDF should return without any fetch."""
        downloader.config.download_dir = str(tmp_path)
        cache_path = downloader._cache_path("rec_cached")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"%PDF-1.4 cached content")

        with patch.object(downloader, "_tiered_fetch") as mock_fetch:
            result = downloader.download_pdf(
                "https://example.com/cached.pdf", "rec_cached", copy_to_host=False
            )
        assert result["success"] is True
        assert result["tier_used"] == "cache"
        mock_fetch.assert_not_called()

    def test_download_pdf_reports_tier_used(self, downloader, tmp_path):
        """Result should include which tier succeeded."""
        downloader.config.download_dir = str(tmp_path)
        fetch_result = FetchResult(
            content=b"%PDF-1.4 data", status_code=200,
            content_type="application/pdf", tier_used="playwright",
            url="https://example.com/test.pdf",
        )
        with patch.object(downloader, "_tiered_fetch", return_value=fetch_result):
            result = downloader.download_pdf(
                "https://example.com/test.pdf", "rec_tier", copy_to_host=False
            )
        assert result["tier_used"] == "playwright"


class TestEZProxyFallback:
    """Verify EZProxy retry on tiered fetch failure."""

    def test_ezproxy_retry_on_all_tiers_fail(self, downloader, tmp_path):
        """Tiered fetch failure on non-proxied URL should retry with EZProxy."""
        downloader.config.download_dir = str(tmp_path)

        fetch_result_ok = FetchResult(
            content=b"%PDF-1.4 retried content",
            status_code=200,
            content_type="application/pdf",
            tier_used="curl_cffi",
            url="https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf",
        )
        # First call fails, second (EZProxy) succeeds
        with patch.object(
            downloader, "_tiered_fetch",
            side_effect=[DownloadError("All failed"), fetch_result_ok],
        ):
            result = downloader.download_pdf(
                "https://example.com/article.pdf", "rec_fallback", copy_to_host=False
            )
        assert result["success"] is True

    def test_no_ezproxy_retry_when_already_proxied(self, downloader):
        """Already-proxied URL should not retry with EZProxy."""
        with patch.object(
            downloader, "_tiered_fetch",
            side_effect=DownloadError("All failed"),
        ):
            result = downloader.download_pdf(
                "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf",
                "rec_no_retry",
                copy_to_host=False,
            )
        assert result["success"] is False

    def test_ezproxy_retry_also_fails(self, downloader):
        """Both original and EZProxy retry failing should report error."""
        with patch.object(
            downloader, "_tiered_fetch",
            side_effect=[DownloadError("fail1"), DownloadError("fail2")],
        ):
            result = downloader.download_pdf(
                "https://example.com/article.pdf", "rec_both_fail", copy_to_host=False
            )
        assert result["success"] is False
        assert "retried via EZProxy" in result["error"]
        assert "also failed" in result["error"]


class TestDownloadFromDirectUrl:
    """Test the download_from_direct_url method."""

    def test_success(self, downloader, tmp_path):
        downloader.config.download_dir = str(tmp_path)
        fetch_result = FetchResult(
            content=b"%PDF-1.4 direct content",
            status_code=200,
            content_type="application/pdf",
            tier_used="curl_cffi",
            url="https://example.com/direct.pdf",
        )
        with patch.object(downloader, "_tiered_fetch", return_value=fetch_result):
            result = downloader.download_from_direct_url("https://example.com/direct.pdf")
        assert result["success"] is True
        assert result["size_bytes"] > 0
        assert result["tier_used"] == "curl_cffi"

    def test_with_extra_cookies(self, downloader, tmp_path):
        downloader.config.download_dir = str(tmp_path)
        fetch_result = FetchResult(
            content=b"%PDF-1.4 data", status_code=200,
            content_type="application/pdf", tier_used="requests",
            url="https://example.com/test.pdf",
        )
        with patch.object(downloader, "_tiered_fetch", return_value=fetch_result) as mock_fetch:
            downloader.download_from_direct_url(
                "https://example.com/test.pdf",
                cookies={"extra": "value"},
            )
        mock_fetch.assert_called_once_with("https://example.com/test.pdf", {"extra": "value"})

    def test_failure(self, downloader, tmp_path):
        downloader.config.download_dir = str(tmp_path)
        with patch.object(downloader, "_tiered_fetch", side_effect=DownloadError("all failed")):
            result = downloader.download_from_direct_url("https://example.com/fail.pdf")
        assert result["success"] is False
        assert "all failed" in result["error"]

    def test_not_a_pdf(self, downloader, tmp_path):
        downloader.config.download_dir = str(tmp_path)
        fetch_result = FetchResult(
            content=b"<html>Not a PDF</html>",
            status_code=200,
            content_type="text/html",
            tier_used="requests",
            url="https://example.com/not-a-pdf",
        )
        with patch.object(downloader, "_tiered_fetch", return_value=fetch_result):
            result = downloader.download_from_direct_url("https://example.com/not-a-pdf")
        assert result["success"] is False
        assert "not a PDF" in result["error"]

    def test_cache_hit(self, downloader, tmp_path):
        downloader.config.download_dir = str(tmp_path)
        # Pre-populate cache
        import hashlib
        url = "https://example.com/cached-direct.pdf"
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_path = downloader._cache_path(url_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"%PDF-1.4 cached")

        with patch.object(downloader, "_tiered_fetch") as mock_fetch:
            result = downloader.download_from_direct_url(url)
        assert result["success"] is True
        assert result["tier_used"] == "cache"
        mock_fetch.assert_not_called()


class TestExtractText:
    @patch("lib.downloader.subprocess.run")
    def test_extract_text_success(self, mock_run, downloader, tmp_path):
        """Successful extraction should return text."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 content")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="This is extracted text from the paper.",
            stderr="",
        )

        text = downloader.extract_text(str(pdf_path))
        assert "extracted text" in text

    @patch("lib.downloader.subprocess.run")
    def test_extract_text_truncation(self, mock_run, downloader, tmp_path):
        """Text exceeding max_chars should be truncated."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 content")

        long_text = "A" * 2000
        mock_run.return_value = MagicMock(
            returncode=0, stdout=long_text, stderr=""
        )

        text = downloader.extract_text(str(pdf_path))
        assert len(text) < 2000
        assert "truncated" in text.lower()

    def test_extract_text_missing_file(self, downloader):
        """Missing file should raise PDFTextExtractionError."""
        with pytest.raises(PDFTextExtractionError, match="not found"):
            downloader.extract_text("/nonexistent/file.pdf")

    @patch("lib.downloader.subprocess.run", side_effect=FileNotFoundError)
    def test_extract_text_pdftotext_missing(self, mock_run, downloader, tmp_path):
        """Missing pdftotext should raise clear error."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 content")

        with pytest.raises(PDFTextExtractionError, match="pdftotext not found"):
            downloader.extract_text(str(pdf_path))


class TestCopyToHost:
    def test_copy_to_host_readable_name(self, downloader, tmp_path):
        """Metadata should produce readable filename."""
        downloader.config.host_download_dir = str(tmp_path)
        pdf_path = tmp_path / "source.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 content")

        metadata = {
            "authors": ["Smith, John"],
            "date": "2023",
            "title": "Machine Learning Approaches",
        }
        result = downloader._copy_to_host(pdf_path, "rec_001", metadata)
        assert result is not None
        assert "Smith" in result
        assert "2023" in result
        assert "Machine Learning" in result

    def test_copy_to_host_no_metadata(self, downloader, tmp_path):
        """Without metadata, should use record_id as filename."""
        downloader.config.host_download_dir = str(tmp_path)
        pdf_path = tmp_path / "source.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 content")

        result = downloader._copy_to_host(pdf_path, "rec_fallback", None)
        assert result is not None
        assert "rec_fallback" in result

    def test_copy_to_host_dir_missing(self, downloader):
        """Missing host dir should return None."""
        downloader.config.host_download_dir = "/nonexistent/path"
        result = downloader._copy_to_host(
            Path("/tmp/fake.pdf"), "rec_001", None
        )
        assert result is None
