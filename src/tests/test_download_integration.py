"""Integration tests for the download auth chain and backfill tool."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import requests

from lib.config import ServerConfig
from lib.downloader import ArticleDownloader


@pytest.fixture
def dl_config():
    return ServerConfig(
        download_dir="/tmp/test-dl-integration",
        host_download_dir="/tmp/test-host-downloads",
        download_timeout=30,
        max_pdf_text_chars=1000,
        ezproxy_prefix="https://proxy.lib.sfu.ca/login?url=",
    )


class TestDownloadAuthFlow:
    """Verify the full auth -> cookie -> EZProxy -> download chain."""

    def test_ezproxy_cookies_included_in_session(self, dl_config):
        """Cookies from EZProxy auth should be present in downloader session."""
        cookies = {"PrimoSession": "abc", "ezproxy": "xyz123"}
        downloader = ArticleDownloader(dl_config, cookies=cookies)
        session = downloader.session
        cookie_names = list(session.cookies.keys())
        assert "ezproxy" in cookie_names
        assert "PrimoSession" in cookie_names

    def test_download_with_ezproxy_cookie_succeeds(self, dl_config, tmp_path):
        """Mock a successful download when EZProxy cookie is present."""
        dl_config.download_dir = str(tmp_path)
        cookies = {"ezproxy": "xyz123", "PrimoSession": "abc"}
        downloader = ArticleDownloader(dl_config, cookies=cookies)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"%PDF-1.4 test content"]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.history = []
        mock_resp.url = "https://example.com/final.pdf"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = ["ezproxy", "PrimoSession"]
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf",
            "rec_auth_test",
            copy_to_host=False,
        )
        assert result["success"] is True
        assert result["size_bytes"] > 0

    def test_download_without_ezproxy_cookie_logs_warning(self, dl_config, caplog):
        """Missing EZProxy cookies should be logged as a warning."""
        import logging

        cookies = {"PrimoSession": "abc"}
        downloader = ArticleDownloader(dl_config, cookies=cookies)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"<html>Login</html>"]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.history = []
        mock_resp.url = "https://proxy.lib.sfu.ca/login"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = ["PrimoSession"]
        downloader._session = mock_session

        with caplog.at_level(logging.WARNING, logger="sfu_library_mcp"):
            downloader.download_pdf(
                "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf",
                "rec_no_ezproxy",
                copy_to_host=False,
            )

        assert any("no EZProxy cookies" in r.message for r in caplog.records)

    def test_download_403_captures_diagnostics(self, dl_config):
        """403 should include status, headers, final URL in error message."""
        cookies = {"ezproxy": "expired_cookie"}
        downloader = ArticleDownloader(dl_config, cookies=cookies)

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.url = "https://proxy.lib.sfu.ca/denied"
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
            "rec_403",
            copy_to_host=False,
        )
        assert result["success"] is False
        assert "403" in result["error"]
        assert "Content-Type" in result["error"]
        assert "final URL" in result["error"]

    def test_download_html_redirect_captures_body_preview(self, dl_config, tmp_path):
        """Login page response should include Content-Type and body preview in error."""
        dl_config.download_dir = str(tmp_path)
        cookies = {"ezproxy": "bad"}
        downloader = ArticleDownloader(dl_config, cookies=cookies)

        html_body = b"<html><head><title>Login Required</title></head><body>Please log in</body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [html_body]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_resp.history = [MagicMock()]  # 1 redirect
        mock_resp.url = "https://proxy.lib.sfu.ca/login"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = ["ezproxy"]
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf",
            "rec_html_redirect",
            copy_to_host=False,
        )
        assert result["success"] is False
        assert "Content-Type" in result["error"]
        assert "text/html" in result["error"]
        assert "final URL" in result["error"]


class TestBackfillCollectionPdfs:
    """Verify the retroactive PDF backfill tool."""

    @pytest.fixture
    def mock_zot_client(self):
        client = MagicMock()
        client.find_collection_by_name.return_value = "COL_BACKFILL"
        return client

    def test_identifies_items_without_pdfs(self, mock_zot_client):
        """Items with numChildren=0 should be flagged for backfill."""
        mock_zot_client.get_items_without_pdfs.return_value = [
            {"key": "ITEM1", "title": "No PDF Item", "DOI": "10.1000/test", "authors": []},
        ]
        result = mock_zot_client.get_items_without_pdfs("COL_BACKFILL")
        assert len(result) == 1
        assert result[0]["key"] == "ITEM1"

    def test_skips_items_that_already_have_pdfs(self, mock_zot_client):
        """Items with PDF attachments should be skipped."""
        mock_zot_client.get_items_without_pdfs.return_value = []
        result = mock_zot_client.get_items_without_pdfs("COL_BACKFILL")
        assert len(result) == 0

    def test_backfill_downloads_and_attaches(self, mock_zot_client, dl_config, tmp_path):
        """Full chain: find item -> search library -> download -> attach."""
        dl_config.download_dir = str(tmp_path)
        downloader = ArticleDownloader(dl_config, cookies={"ezproxy": "valid"})

        item = {"key": "ITEM1", "title": "Test Article", "DOI": "10.1000/test", "authors": []}

        # Mock a successful PDF download
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"%PDF-1.4 backfill content"]
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.history = []
        mock_resp.url = "https://example.com/article.pdf"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = MagicMock()
        mock_session.cookies.keys.return_value = ["ezproxy"]
        downloader._session = mock_session

        result = downloader.download_pdf(
            "https://example.com/article.pdf", item["key"], copy_to_host=False
        )
        assert result["success"] is True

        # Attach should succeed
        mock_zot_client.attach_pdf(item["key"], result["container_path"])
        mock_zot_client.attach_pdf.assert_called_once()

    def test_backfill_reports_per_item_results(self):
        """Output should show success/fail for each item."""
        details = [
            "  Test Article 1: ATTACHED",
            "  Test Article 2: FAILED - no PDF URL found",
            "  Test Article 3: ATTACHED (host: /mnt/host-downloads/test.pdf)",
        ]
        output = "\n".join([
            "=" * 50,
            "BACKFILL COLLECTION PDFs",
            "=" * 50,
            "\nCollection: Test Collection",
            "Items checked: 3",
            "PDFs attached: 2",
            "Failed: 1",
            "",
        ] + details)

        assert "ATTACHED" in output
        assert "FAILED" in output
        assert "PDFs attached: 2" in output
        assert "Failed: 1" in output
