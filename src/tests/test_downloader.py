"""Unit tests for the downloader module."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import requests

from lib.config import ServerConfig
from lib.downloader import ArticleDownloader, DownloadError, PDFTextExtractionError


@pytest.fixture
def dl_config():
    return ServerConfig(
        download_dir="/tmp/test-dl-cache",
        host_download_dir="/tmp/test-host-downloads",
        download_timeout=30,
        max_pdf_text_chars=1000,
        ezproxy_prefix="https://proxy.lib.sfu.ca/login?url=",
    )


@pytest.fixture
def downloader(dl_config):
    return ArticleDownloader(dl_config, cookies={"session": "abc123"})


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


class TestDownloadPdf:
    @patch("lib.downloader.requests.Session")
    def test_download_pdf_success(self, mock_session_cls, downloader, tmp_path):
        """Successful download should write PDF and return success."""
        downloader.config.download_dir = str(tmp_path)
        downloader._session = None

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"%PDF-1.4 fake content"]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.history = []
        mock_resp.url = "https://example.com/test.pdf"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = ["session"]
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://example.com/test.pdf", "rec_001", copy_to_host=False
        )
        assert result["success"] is True
        assert result["container_path"] is not None
        assert result["size_bytes"] > 0

    @patch("lib.downloader.requests.Session")
    def test_download_pdf_not_a_pdf(self, mock_session_cls, downloader, tmp_path):
        """HTML response should be rejected with enriched error."""
        downloader.config.download_dir = str(tmp_path)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"<html>Login page</html>"]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.history = []
        mock_resp.url = "https://proxy.lib.sfu.ca/login"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = []
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://example.com/test.pdf", "rec_002", copy_to_host=False
        )
        assert result["success"] is False
        assert "not a PDF" in result["error"]
        assert "Content-Type" in result["error"]
        assert "text/html" in result["error"]

    def test_download_pdf_http_error(self, downloader):
        """HTTP errors should return failure with enriched diagnostics."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.url = "https://example.com/404"
        mock_resp.history = []
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )
        mock_session.get.return_value = mock_resp
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = []
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://example.com/nope.pdf", "rec_003", copy_to_host=False
        )
        assert result["success"] is False
        assert "404" in result["error"]
        assert "Content-Type" in result["error"]
        assert "final URL" in result["error"]

    def test_download_pdf_timeout(self, downloader):
        """Timeout should return failure."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.Timeout()
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = []
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://example.com/slow.pdf", "rec_004", copy_to_host=False
        )
        assert result["success"] is False
        assert "timed out" in result["error"]

    def test_download_pdf_cache_hit(self, downloader, tmp_path):
        """Cached PDF should return without HTTP call."""
        downloader.config.download_dir = str(tmp_path)
        cache_path = downloader._cache_path("rec_cached")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"%PDF-1.4 cached content")

        mock_session = MagicMock()
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://example.com/cached.pdf", "rec_cached", copy_to_host=False
        )
        assert result["success"] is True
        mock_session.get.assert_not_called()


class TestEZProxy403Fallback:
    """Verify 403 triggers EZProxy retry for non-proxied URLs."""

    def test_403_retries_via_ezproxy(self, downloader, tmp_path):
        """403 on a non-proxied URL should retry with EZProxy prefix."""
        downloader.config.download_dir = str(tmp_path)

        # First call: 403 error
        mock_403_resp = MagicMock()
        mock_403_resp.status_code = 403
        mock_403_resp.headers = {"Content-Type": "text/html"}
        mock_403_resp.url = "https://example.com/article.pdf"
        mock_403_resp.history = []
        mock_403_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_403_resp
        )

        # Second call (EZProxy retry): success
        mock_ok_resp = MagicMock()
        mock_ok_resp.status_code = 200
        mock_ok_resp.iter_content.return_value = [b"%PDF-1.4 retried content"]
        mock_ok_resp.raise_for_status.return_value = None
        mock_ok_resp.headers = {"Content-Type": "application/pdf"}
        mock_ok_resp.history = []
        mock_ok_resp.url = "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf"

        mock_session = MagicMock()
        mock_session.get.side_effect = [mock_403_resp, mock_ok_resp]
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = ["ezproxy"]
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://example.com/article.pdf", "rec_403_retry", copy_to_host=False
        )
        assert result["success"] is True
        assert result["size_bytes"] > 0
        # Should have made 2 calls: original + EZProxy retry
        assert mock_session.get.call_count == 2
        retry_url = mock_session.get.call_args_list[1][0][0]
        assert retry_url.startswith("https://proxy.lib.sfu.ca/login?url=")

    def test_403_no_retry_when_already_proxied(self, downloader):
        """403 on an already-proxied URL should NOT retry."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.url = "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf"
        mock_resp.history = []
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = ["ezproxy"]
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf",
            "rec_403_no_retry",
            copy_to_host=False,
        )
        assert result["success"] is False
        assert "403" in result["error"]
        # Should have made only 1 call (no retry)
        assert mock_session.get.call_count == 1

    def test_403_retry_also_fails(self, downloader):
        """403 retry via EZProxy that also fails should report both."""
        mock_403_resp = MagicMock()
        mock_403_resp.status_code = 403
        mock_403_resp.headers = {"Content-Type": "text/html"}
        mock_403_resp.url = "https://example.com/article.pdf"
        mock_403_resp.history = []
        mock_403_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_403_resp
        )

        mock_session = MagicMock()
        # Both calls fail
        mock_session.get.side_effect = [
            mock_403_resp,
            requests.exceptions.ConnectionError("EZProxy unreachable"),
        ]
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = []
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://example.com/article.pdf", "rec_403_both_fail", copy_to_host=False
        )
        assert result["success"] is False
        assert "retried via EZProxy" in result["error"]
        assert "also failed" in result["error"]


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
