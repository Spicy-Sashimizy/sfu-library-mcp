"""Unit tests for the tools module."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from lib.tools import TOOL_DEFINITIONS, handle_tool_call, get_metrics, _reciprocal_rank_fusion, _get_downloader


class MockClient:
    """Mock SFULibraryClient for tool dispatch testing."""

    def __init__(self, authenticated=True, search_result=None, item_result=None):
        self._authenticated = authenticated
        self._search_result = search_result
        self._item_result = item_result
        self.token_cleared = False
        self.search_calls: list[dict] = []
        self.cookies: dict = {}

        # Provide a mock config for diagnostics
        from lib.config import ServerConfig
        self.config = ServerConfig(
            sfu_username="testuser",
            sfu_password="testpass",
            mfa_secret="TESTSECRET",
        )

    def ensure_authenticated(self, force=False):
        return self._authenticated

    def search(self, query="", limit=10, offset=0, field="any",
               precision="contains", sort="rank", tab="default_tab",
               scope="default_scope"):
        self.search_calls.append({
            "query": query, "limit": limit, "offset": offset,
            "field": field, "sort": sort, "tab": tab, "scope": scope,
        })
        return self._search_result

    def get_item_details(self, doc_id, context="L"):
        return self._item_result

    def get_token_status(self):
        if self._authenticated:
            return {
                "valid": True,
                "user": "Test User",
                "userId": "testid",
                "userGroup": "STUDENT",
                "expiresIn": "1h 30m",
                "expiresAt": "2026-01-15T12:00:00",
            }
        return {"valid": False, "message": "No token available"}

    def clear_token_cache(self):
        self.token_cleared = True


@pytest.fixture
def mock_client():
    return MockClient()


@pytest.fixture
def mock_client_with_results(sample_pnx_record, mock_search_response):
    return MockClient(
        search_result=mock_search_response,
        item_result=sample_pnx_record,
    )


class TestToolDefinitions:
    def test_exactly_26_tools(self):
        assert len(TOOL_DEFINITIONS) == 26

    def test_tool_names(self):
        names = [t.name for t in TOOL_DEFINITIONS]
        expected = [
            "search_library", "get_item_details", "get_token_status",
            "authenticate", "search_by_author", "search_by_subject",
            "search_by_isbn", "search_electronic_resources", "clear_cache",
            "get_full_text_links", "generate_citation",
            "batch_generate_citations", "export_search_results",
            "batch_isbn_lookup",
            "download_article", "read_article", "save_to_zotero",
            "list_zotero_collections", "batch_save_to_zotero",
            "search_zotero", "get_zotero_collection_items",
            "backfill_collection_pdfs", "download_from_url",
            "get_diagnostics", "get_zotero_status", "zotero_authenticate",
        ]
        assert names == expected

    def test_download_from_url_tool_exists(self):
        names = [t.name for t in TOOL_DEFINITIONS]
        assert "download_from_url" in names
        tool = next(t for t in TOOL_DEFINITIONS if t.name == "download_from_url")
        assert "url" in tool.inputSchema["required"]

    def test_all_tools_have_schemas(self):
        for tool in TOOL_DEFINITIONS:
            assert tool.inputSchema is not None
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"


class TestToolDispatch:
    @pytest.mark.asyncio
    async def test_search_library(self, mock_client_with_results):
        result = await handle_tool_call(
            "search_library",
            {"query": "test"},
            mock_client_with_results,
        )
        assert len(result) == 1
        assert "Found" in result[0].text

    @pytest.mark.asyncio
    async def test_get_item_details(self, mock_client_with_results):
        result = await handle_tool_call(
            "get_item_details",
            {"record_id": "alma123"},
            mock_client_with_results,
        )
        assert len(result) == 1
        assert "ITEM DETAILS" in result[0].text

    @pytest.mark.asyncio
    async def test_get_token_status(self, mock_client):
        result = await handle_tool_call("get_token_status", {}, mock_client)
        assert "VALID" in result[0].text

    @pytest.mark.asyncio
    async def test_authenticate(self, mock_client):
        result = await handle_tool_call("authenticate", {"force": False}, mock_client)
        assert "successful" in result[0].text

    @pytest.mark.asyncio
    async def test_search_by_author(self, mock_client_with_results):
        result = await handle_tool_call(
            "search_by_author",
            {"author": "Einstein"},
            mock_client_with_results,
        )
        assert "Found" in result[0].text

    @pytest.mark.asyncio
    async def test_search_by_subject(self, mock_client_with_results):
        result = await handle_tool_call(
            "search_by_subject",
            {"subject": "physics"},
            mock_client_with_results,
        )
        assert "Found" in result[0].text

    @pytest.mark.asyncio
    async def test_search_by_isbn(self, mock_client_with_results):
        result = await handle_tool_call(
            "search_by_isbn",
            {"isbn": "9780262018029"},
            mock_client_with_results,
        )
        assert "Found" in result[0].text

    @pytest.mark.asyncio
    async def test_search_electronic(self, mock_client_with_results):
        result = await handle_tool_call(
            "search_electronic_resources",
            {"query": "python"},
            mock_client_with_results,
        )
        assert "Found" in result[0].text

    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_client):
        result = await handle_tool_call("clear_cache", {}, mock_client)
        assert "cleared" in result[0].text.lower()
        assert mock_client.token_cleared is True

    @pytest.mark.asyncio
    async def test_get_full_text_links(self, mock_client_with_results):
        result = await handle_tool_call(
            "get_full_text_links",
            {"record_id": "alma123"},
            mock_client_with_results,
        )
        assert "FULL TEXT" in result[0].text

    @pytest.mark.asyncio
    async def test_generate_citation_apa(self, mock_client_with_results):
        result = await handle_tool_call(
            "generate_citation",
            {"record_id": "alma123", "format": "apa"},
            mock_client_with_results,
        )
        assert "APA" in result[0].text

    @pytest.mark.asyncio
    async def test_generate_citation_bibtex(self, mock_client_with_results):
        result = await handle_tool_call(
            "generate_citation",
            {"record_id": "alma123", "format": "bibtex"},
            mock_client_with_results,
        )
        assert "BibTeX" in result[0].text
        assert "@book{" in result[0].text

    @pytest.mark.asyncio
    async def test_batch_citations(self, mock_client_with_results):
        result = await handle_tool_call(
            "batch_generate_citations",
            {"record_ids": ["alma123", "alma456"], "format": "apa"},
            mock_client_with_results,
        )
        assert "Citations" in result[0].text

    @pytest.mark.asyncio
    async def test_export_bibtex(self, mock_client_with_results):
        result = await handle_tool_call(
            "export_search_results",
            {"query": "test", "format": "bibtex", "limit": 5},
            mock_client_with_results,
        )
        assert "BibTeX Export" in result[0].text

    @pytest.mark.asyncio
    async def test_export_csv(self, mock_client_with_results):
        result = await handle_tool_call(
            "export_search_results",
            {"query": "test", "format": "csv"},
            mock_client_with_results,
        )
        assert "CSV Export" in result[0].text

    @pytest.mark.asyncio
    async def test_export_json(self, mock_client_with_results):
        result = await handle_tool_call(
            "export_search_results",
            {"query": "test", "format": "json"},
            mock_client_with_results,
        )
        assert "JSON Export" in result[0].text

    @pytest.mark.asyncio
    async def test_export_ris(self, mock_client_with_results):
        result = await handle_tool_call(
            "export_search_results",
            {"query": "test", "format": "ris"},
            mock_client_with_results,
        )
        assert "RIS Export" in result[0].text

    @pytest.mark.asyncio
    async def test_batch_isbn_lookup(self, mock_client_with_results):
        result = await handle_tool_call(
            "batch_isbn_lookup",
            {"isbn_list": ["978-0-262-01802-9"]},
            mock_client_with_results,
        )
        assert "BATCH ISBN LOOKUP" in result[0].text

    @pytest.mark.asyncio
    async def test_unknown_tool(self, mock_client):
        result = await handle_tool_call("nonexistent_tool", {}, mock_client)
        assert "Unknown tool" in result[0].text

    @pytest.mark.asyncio
    async def test_auth_failure(self):
        unauth_client = MockClient(authenticated=False)
        result = await handle_tool_call(
            "search_library",
            {"query": "test"},
            unauth_client,
        )
        assert "Authentication failed" in result[0].text

    @pytest.mark.asyncio
    async def test_empty_batch_citations(self, mock_client):
        result = await handle_tool_call(
            "batch_generate_citations",
            {"record_ids": []},
            mock_client,
        )
        assert "No record IDs" in result[0].text

    @pytest.mark.asyncio
    async def test_empty_isbn_list(self, mock_client):
        result = await handle_tool_call(
            "batch_isbn_lookup",
            {"isbn_list": []},
            mock_client,
        )
        assert "No ISBN" in result[0].text

    @pytest.mark.asyncio
    async def test_search_with_expanded_terms(self, mock_client_with_results):
        result = await handle_tool_call(
            "search_library",
            {
                "query": "machine learning",
                "expanded_terms": "deep learning OR neural networks",
            },
            mock_client_with_results,
        )
        assert "Found" in result[0].text
        # Verify the combined query was sent to the client
        assert len(mock_client_with_results.search_calls) >= 1
        sent_query = mock_client_with_results.search_calls[0]["query"]
        assert "(machine learning) OR (deep learning OR neural networks)" == sent_query

    @pytest.mark.asyncio
    async def test_search_without_expanded_terms(self, mock_client_with_results):
        result = await handle_tool_call(
            "search_library",
            {"query": "machine learning"},
            mock_client_with_results,
        )
        assert "Found" in result[0].text
        sent_query = mock_client_with_results.search_calls[0]["query"]
        assert sent_query == "machine learning"

    @pytest.mark.asyncio
    async def test_comprehensive_search(self, mock_client_with_results):
        """Comprehensive search should make multiple parallel calls."""
        features = {
            "fusion_enabled": True,
            "rerank_enabled": False,
        }
        with patch("lib.tools._get_features", return_value=features):
            result = await handle_tool_call(
                "search_library",
                {"query": "climate change", "comprehensive": True},
                mock_client_with_results,
            )
        assert "Found" in result[0].text
        # Should have made 3 parallel searches (general, subject, electronic)
        assert len(mock_client_with_results.search_calls) == 3

    @pytest.mark.asyncio
    async def test_comprehensive_disabled_by_feature_flag(self, mock_client_with_results):
        """When fusion_enabled is False, comprehensive should fall back to single search."""
        features = {
            "fusion_enabled": False,
            "rerank_enabled": False,
        }
        with patch("lib.tools._get_features", return_value=features):
            result = await handle_tool_call(
                "search_library",
                {"query": "climate change", "comprehensive": True},
                mock_client_with_results,
            )
        assert "Found" in result[0].text
        # Only 1 search call when fusion is disabled
        assert len(mock_client_with_results.search_calls) == 1


class TestReciprocalRankFusion:
    def test_deduplication(self, sample_pnx_record):
        """Same record from different scopes should appear only once."""
        result_set_1 = {"docs": [sample_pnx_record], "info": {"total": 1}}
        result_set_2 = {"docs": [sample_pnx_record], "info": {"total": 1}}
        merged = _reciprocal_rank_fusion([result_set_1, result_set_2], limit=10)
        assert len(merged["docs"]) == 1

    def test_merges_different_records(self, sample_pnx_record, sample_article_record):
        result_set_1 = {"docs": [sample_pnx_record], "info": {"total": 1}}
        result_set_2 = {"docs": [sample_article_record], "info": {"total": 1}}
        merged = _reciprocal_rank_fusion([result_set_1, result_set_2], limit=10)
        assert len(merged["docs"]) == 2

    def test_handles_none_results(self, sample_pnx_record):
        result_set = {"docs": [sample_pnx_record], "info": {"total": 1}}
        merged = _reciprocal_rank_fusion([None, result_set, None], limit=10)
        assert len(merged["docs"]) == 1

    def test_limit_respected(self, sample_pnx_record, sample_article_record):
        result_set = {"docs": [sample_pnx_record, sample_article_record], "info": {"total": 2}}
        merged = _reciprocal_rank_fusion([result_set], limit=1)
        assert len(merged["docs"]) == 1


class TestGetDownloaderCookieStaleness:
    """Verify _get_downloader detects cookie changes via content hash."""

    def test_recreates_downloader_when_cookies_mutated(self):
        """Mutating the same dict object should trigger recreation."""
        import lib.tools as tools_module
        # Reset global state
        tools_module._downloader = None
        tools_module._downloader_cookie_id = None

        client = MockClient()
        client.cookies = {"session": "old_value"}

        dl1 = _get_downloader(client)
        assert dl1 is not None

        # Mutate the SAME dict (this is what re-auth does)
        client.cookies["ezproxy"] = "new_cookie"
        client.cookies["session"] = "new_value"

        dl2 = _get_downloader(client)
        # Should be a NEW downloader since cookies changed
        assert dl2 is not dl1

    def test_reuses_downloader_when_cookies_unchanged(self):
        """Same cookies should return the same downloader instance."""
        import lib.tools as tools_module
        tools_module._downloader = None
        tools_module._downloader_cookie_id = None

        client = MockClient()
        client.cookies = {"session": "abc"}

        dl1 = _get_downloader(client)
        dl2 = _get_downloader(client)
        assert dl2 is dl1


class TestDownloadFromUrlTool:
    """Tests for the download_from_url MCP tool."""

    @pytest.mark.asyncio
    async def test_download_from_url_success(self, mock_client):
        """Successful URL download should report tier and size."""
        mock_result = {
            "success": True,
            "container_path": "/tmp/test.pdf",
            "size_bytes": 12345,
            "tier_used": "curl_cffi",
            "error": None,
        }
        with patch("lib.tools._get_downloader") as mock_get_dl:
            mock_dl = MagicMock()
            mock_dl.download_from_direct_url.return_value = mock_result
            mock_get_dl.return_value = mock_dl
            result = await handle_tool_call(
                "download_from_url",
                {"url": "https://example.com/paper.pdf"},
                mock_client,
            )
        assert "PDF DOWNLOADED FROM URL" in result[0].text
        assert "12,345 bytes" in result[0].text
        assert "curl_cffi" in result[0].text

    @pytest.mark.asyncio
    async def test_download_from_url_failure(self, mock_client):
        """Failed download should return error message."""
        mock_result = {
            "success": False,
            "container_path": None,
            "size_bytes": 0,
            "tier_used": None,
            "error": "All tiers failed",
        }
        with patch("lib.tools._get_downloader") as mock_get_dl:
            mock_dl = MagicMock()
            mock_dl.download_from_direct_url.return_value = mock_result
            mock_get_dl.return_value = mock_dl
            result = await handle_tool_call(
                "download_from_url",
                {"url": "https://example.com/fail.pdf"},
                mock_client,
            )
        assert "Download failed" in result[0].text

    @pytest.mark.asyncio
    async def test_download_from_url_with_ezproxy(self, mock_client):
        """use_ezproxy=True should wrap the URL in hostname-based proxy format."""
        mock_result = {
            "success": True,
            "container_path": "/tmp/test.pdf",
            "size_bytes": 100,
            "tier_used": "requests",
            "error": None,
        }
        with patch("lib.tools._get_downloader") as mock_get_dl:
            mock_dl = MagicMock()
            mock_dl.download_from_direct_url.return_value = mock_result
            mock_get_dl.return_value = mock_dl
            result = await handle_tool_call(
                "download_from_url",
                {"url": "https://example.com/paper.pdf", "use_ezproxy": True},
                mock_client,
            )
        # Check that the URL was wrapped in hostname-based proxy format
        call_args = mock_dl.download_from_direct_url.call_args
        called_url = call_args[1].get("url", call_args[0][0] if call_args[0] else "")
        assert "example-com.proxy.lib.sfu.ca" in called_url

    @pytest.mark.asyncio
    async def test_download_from_url_empty_url(self, mock_client):
        """Empty URL should return error."""
        result = await handle_tool_call(
            "download_from_url",
            {"url": ""},
            mock_client,
        )
        assert "No URL provided" in result[0].text


class TestBackfillCap:
    """Verify backfill caps items at configured limit."""

    @pytest.mark.asyncio
    async def test_backfill_capped_at_config(self, mock_client):
        """Backfill should cap items_without_pdfs to download_backfill_cap."""
        # This tests the cap logic: create 20 items, expect only 10 processed
        mock_zot = MagicMock()
        mock_zot.find_collection_by_name.return_value = "col_key_123"
        # Return 20 items without PDFs
        items_without_pdfs = [
            {"title": f"Article {i}", "key": f"key_{i}", "DOI": f"10.1234/{i}"}
            for i in range(20)
        ]
        mock_zot.get_items_without_pdfs.return_value = items_without_pdfs

        mock_dl = MagicMock()
        mock_dl.resolve_pdf_url.return_value = None  # All fail to find URL

        with patch("lib.tools._get_zotero_client", return_value=mock_zot), \
             patch("lib.tools._get_downloader", return_value=mock_dl), \
             patch("lib.tools._get_rate_limiter") as mock_get_rl, \
             patch("lib.tools._get_config") as mock_get_config:
            from lib.config import ServerConfig
            mock_get_config.return_value = ServerConfig(download_backfill_cap=10)
            mock_rl = MagicMock()
            mock_rl.get_budget_status.return_value = {
                "session_remaining": 15, "session_budget": 15,
                "hourly_remaining": 20, "hourly_used": 0,
                "hourly_limit": 20, "per_domain": {},
            }
            mock_get_rl.return_value = mock_rl

            # The search calls will return None (no record found)
            mock_client._search_result = None
            result = await handle_tool_call(
                "backfill_collection_pdfs",
                {"collection_name": "Test Collection"},
                mock_client,
            )
        text = result[0].text
        # Should report 10 items checked (capped from 20)
        assert "Items checked: 10" in text


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_recorded(self, mock_client):
        await handle_tool_call("get_token_status", {}, mock_client)
        metrics = get_metrics()
        assert "get_token_status" in metrics
        assert metrics["get_token_status"]["count"] >= 1


class TestGetDiagnostics:
    """Tests for the get_diagnostics MCP tool."""

    @pytest.mark.asyncio
    async def test_diagnostics_returns_report(self, mock_client):
        """Diagnostics tool should return a comprehensive report."""
        result = await handle_tool_call("get_diagnostics", {}, mock_client)
        text = result[0].text
        assert "DIAGNOSTIC REPORT" in text
        assert "Token Status" in text
        assert "Cookie Inventory" in text
        assert "EZProxy Session" in text
        assert "Download Tier Availability" in text
        assert "Circuit Breaker" in text
        assert "Rate Limiter" in text
        assert "Log File" in text
        assert "Tool Metrics" in text

    @pytest.mark.asyncio
    async def test_diagnostics_without_errors(self, mock_client):
        """Diagnostics with include_recent_errors=False should skip error section."""
        result = await handle_tool_call(
            "get_diagnostics",
            {"include_recent_errors": False},
            mock_client,
        )
        text = result[0].text
        assert "DIAGNOSTIC REPORT" in text
        assert "Recent Errors" not in text

    @pytest.mark.asyncio
    async def test_diagnostics_shows_token_valid(self, mock_client):
        """Should show VALID token for authenticated client."""
        result = await handle_tool_call("get_diagnostics", {}, mock_client)
        assert "Status: VALID" in result[0].text

    @pytest.mark.asyncio
    async def test_diagnostics_shows_token_invalid(self):
        """Should show INVALID token for unauthenticated client."""
        unauth = MockClient(authenticated=False)
        result = await handle_tool_call("get_diagnostics", {}, unauth)
        assert "INVALID" in result[0].text


class TestSkipFlags:
    """Tests for download skip flags being passed through."""

    @pytest.mark.asyncio
    async def test_skip_flags_passed_to_downloader(self, mock_client):
        """Skip flags should be extracted and passed to downloader."""
        mock_result = {
            "success": True,
            "container_path": "/tmp/test.pdf",
            "size_bytes": 12345,
            "tier_used": "curl_cffi",
            "error": None,
        }
        with patch("lib.tools._get_downloader") as mock_get_dl:
            mock_dl = MagicMock()
            mock_dl.download_from_direct_url.return_value = mock_result
            mock_get_dl.return_value = mock_dl
            await handle_tool_call(
                "download_from_url",
                {
                    "url": "https://example.com/paper.pdf",
                    "skip_rate_limit": True,
                    "skip_tiers": ["playwright"],
                },
                mock_client,
            )
        call_kwargs = mock_dl.download_from_direct_url.call_args
        skip_flags = call_kwargs[1].get("skip_flags", call_kwargs[0][2] if len(call_kwargs[0]) > 2 else {})
        assert skip_flags.get("skip_rate_limit") is True
        assert skip_flags.get("skip_tiers") == ["playwright"]

    def test_skip_flag_properties_in_download_article_schema(self):
        """download_article schema should include skip flag properties."""
        tool = next(t for t in TOOL_DEFINITIONS if t.name == "download_article")
        props = tool.inputSchema["properties"]
        assert "skip_ezproxy" in props
        assert "skip_rate_limit" in props
        assert "skip_tiers" in props
        assert "skip_pdf_check" in props
        assert "skip_login_check" in props
        assert "skip_copy_to_host" in props

    def test_skip_flag_properties_in_read_article_schema(self):
        """read_article schema should include skip flag properties."""
        tool = next(t for t in TOOL_DEFINITIONS if t.name == "read_article")
        props = tool.inputSchema["properties"]
        assert "skip_ezproxy" in props

    def test_skip_flag_properties_in_download_from_url_schema(self):
        """download_from_url schema should include skip flag properties."""
        tool = next(t for t in TOOL_DEFINITIONS if t.name == "download_from_url")
        props = tool.inputSchema["properties"]
        assert "skip_rate_limit" in props
        assert "skip_tiers" in props
