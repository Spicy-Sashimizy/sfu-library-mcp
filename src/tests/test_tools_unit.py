"""Unit tests for the tools module."""

import asyncio

import pytest

from lib.tools import TOOL_DEFINITIONS, handle_tool_call, get_metrics


class MockClient:
    """Mock SFULibraryClient for tool dispatch testing."""

    def __init__(self, authenticated=True, search_result=None, item_result=None):
        self._authenticated = authenticated
        self._search_result = search_result
        self._item_result = item_result
        self.token_cleared = False

    def ensure_authenticated(self, force=False):
        return self._authenticated

    def search(self, query="", limit=10, offset=0, field="any",
               precision="contains", sort="rank", tab="default_tab",
               scope="default_scope"):
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
    def test_exactly_14_tools(self):
        assert len(TOOL_DEFINITIONS) == 14

    def test_tool_names(self):
        names = [t.name for t in TOOL_DEFINITIONS]
        expected = [
            "search_library", "get_item_details", "get_token_status",
            "authenticate", "search_by_author", "search_by_subject",
            "search_by_isbn", "search_electronic_resources", "clear_cache",
            "get_full_text_links", "generate_citation",
            "batch_generate_citations", "export_search_results",
            "batch_isbn_lookup",
        ]
        assert names == expected

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


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_recorded(self, mock_client):
        await handle_tool_call("get_token_status", {}, mock_client)
        metrics = get_metrics()
        assert "get_token_status" in metrics
        assert metrics["get_token_status"]["count"] >= 1
