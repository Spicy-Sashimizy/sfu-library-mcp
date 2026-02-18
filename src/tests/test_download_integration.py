"""Integration tests for the download auth chain and backfill tool."""

import os
import tempfile
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import requests

from lib.config import ServerConfig
from lib.downloader import ArticleDownloader, DownloadError, FetchResult


@pytest.fixture
def dl_config():
    return ServerConfig(
        download_dir="/tmp/test-dl-integration",
        host_download_dir="/tmp/test-host-downloads",
        download_timeout=30,
        max_pdf_text_chars=1000,
        ezproxy_prefix="https://proxy.lib.sfu.ca/login?url=",
        download_tiers=["curl_cffi", "playwright", "requests"],
    )


class TestDownloadAuthFlow:
    """Verify the full auth -> cookie -> EZProxy -> download chain."""

    def test_ezproxy_cookies_included_in_session(self, dl_config):
        """Both Primo and EZProxy cookies should be present in downloader session."""
        cookies = {
            "PrimoSession": "abc",
            "JSESSIONID": "primo_session_id",
            "ezproxy": "xyz123",
        }
        downloader = ArticleDownloader(dl_config, cookies=cookies)
        session = downloader.session
        cookie_names = list(session.cookies.keys())
        assert "ezproxy" in cookie_names
        assert "PrimoSession" in cookie_names
        assert "JSESSIONID" in cookie_names

    def test_download_with_ezproxy_cookie_succeeds(self, dl_config, tmp_path):
        """Mock a successful download when EZProxy cookie is present."""
        dl_config.download_dir = str(tmp_path)
        cookies = {"ezproxy": "xyz123", "PrimoSession": "abc"}
        downloader = ArticleDownloader(dl_config, cookies=cookies)

        fetch_result = FetchResult(
            content=b"%PDF-1.4 test content",
            status_code=200,
            content_type="application/pdf",
            tier_used="curl_cffi",
            url="https://example.com/final.pdf",
        )
        with patch.object(downloader, "_tiered_fetch", return_value=fetch_result):
            result = downloader.download_pdf(
                "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf",
                "rec_auth_test",
                copy_to_host=False,
            )
        assert result["success"] is True
        assert result["size_bytes"] > 0

    def test_download_without_proxy_cookie_logs_warning(self, dl_config, caplog):
        """Missing proxy cookies should be logged as a warning."""
        import logging

        cookies = {"PrimoSession": "abc"}
        downloader = ArticleDownloader(dl_config, cookies=cookies)

        fetch_result = FetchResult(
            content=b"<html><head><title>Authentication Required</title></head></html>",
            status_code=200,
            content_type="text/html",
            tier_used="curl_cffi",
            url="https://proxy.lib.sfu.ca/login",
        )
        with patch.object(downloader, "_tiered_fetch", return_value=fetch_result):
            with caplog.at_level(logging.WARNING, logger="sfu_library_mcp"):
                result = downloader.download_pdf(
                    "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf",
                    "rec_no_ezproxy",
                    copy_to_host=False,
                )

        assert any("no proxy cookies" in r.message for r in caplog.records)
        assert result["success"] is False
        assert "login" in result["error"].lower()

    def test_download_403_captures_diagnostics(self, dl_config):
        """403 should include status info in error message."""
        cookies = {"ezproxy": "expired_cookie"}
        downloader = ArticleDownloader(dl_config, cookies=cookies)

        with patch.object(
            downloader, "_tiered_fetch",
            side_effect=DownloadError("All download tiers failed for url. Last error: HTTP 403"),
        ):
            result = downloader.download_pdf(
                "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf",
                "rec_403",
                copy_to_host=False,
            )
        assert result["success"] is False
        assert "403" in result["error"]

    def test_download_html_redirect_captures_body_preview(self, dl_config, tmp_path):
        """Login page response should include Content-Type in error."""
        dl_config.download_dir = str(tmp_path)
        cookies = {"ezproxy": "bad"}
        downloader = ArticleDownloader(dl_config, cookies=cookies)

        fetch_result = FetchResult(
            content=b"<html><head><title>Login Required</title></head><body>Please log in</body></html>",
            status_code=200,
            content_type="text/html; charset=utf-8",
            tier_used="curl_cffi",
            url="https://proxy.lib.sfu.ca/login",
        )
        with patch.object(downloader, "_tiered_fetch", return_value=fetch_result):
            result = downloader.download_pdf(
                "https://proxy.lib.sfu.ca/login?url=https://example.com/article.pdf",
                "rec_html_redirect",
                copy_to_host=False,
            )
        assert result["success"] is False
        assert "Content-Type" in result["error"]
        assert "text/html" in result["error"]
        assert "URL" in result["error"]


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

        fetch_result = FetchResult(
            content=b"%PDF-1.4 backfill content",
            status_code=200,
            content_type="application/pdf",
            tier_used="curl_cffi",
            url="https://example.com/article.pdf",
        )
        with patch.object(downloader, "_tiered_fetch", return_value=fetch_result):
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


class TestClearTokenCacheInMemory:
    """Verify clear_token_cache resets in-memory auth state."""

    def test_clear_token_cache_resets_all_state(self):
        """clear_token_cache should clear file, jwt_token, cookies, user_info, and token_expiry."""
        from lib.client import SFULibraryClient

        fd, cache_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

        try:
            config = ServerConfig(
                token_cache_file=cache_path,
                sfu_username="test",
                sfu_password="test",
                mfa_secret="TESTSECRETBASE32A",
            )
            client = SFULibraryClient(config=config)
            # Simulate authenticated state
            client.jwt_token = "fake.jwt.token"
            client.cookies = {"session": "abc", "ezproxy": "xyz"}
            client.user_info = {"user": "testuser", "userName": "Test"}
            client.token_expiry = 9999999999

            # Write something to cache file
            with open(cache_path, "w") as f:
                f.write('{"jwt_token": "old"}')

            client.clear_token_cache()

            # File should be deleted
            assert not os.path.exists(cache_path)
            # In-memory state should be cleared
            assert client.jwt_token is None
            assert client.cookies == {}
            assert client.user_info == {}
            assert client.token_expiry is None
        finally:
            if os.path.exists(cache_path):
                os.remove(cache_path)

    def test_clear_token_cache_works_without_file(self):
        """clear_token_cache should work even if no cache file exists."""
        from lib.client import SFULibraryClient

        config = ServerConfig(
            token_cache_file="/tmp/nonexistent_cache_12345.json",
            sfu_username="test",
            sfu_password="test",
            mfa_secret="TESTSECRETBASE32A",
        )
        client = SFULibraryClient(config=config)
        client.jwt_token = "stale_token"
        client.cookies = {"old": "cookie"}

        client.clear_token_cache()

        assert client.jwt_token is None
        assert client.cookies == {}
