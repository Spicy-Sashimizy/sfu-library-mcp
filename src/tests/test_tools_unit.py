"""Unit tests for the tools module."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from lib.tools import TOOL_DEFINITIONS, handle_tool_call, get_metrics, _reciprocal_rank_fusion


class MockClient:
    """Mock SFULibraryClient for tool dispatch testing."""

    def __init__(self, search_result=None, item_result=None):
        self._search_result = search_result
        self._item_result = item_result
        self.search_calls: list[dict] = []

        from lib.config import ServerConfig
        self.config = ServerConfig()

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
    def test_exactly_17_tools(self):
        assert len(TOOL_DEFINITIONS) == 17

    def test_tool_names(self):
        names = [t.name for t in TOOL_DEFINITIONS]
        expected = [
            "search_library", "get_item_details",
            "search_by_author", "search_by_subject",
            "search_by_isbn", "search_electronic_resources",
            "get_full_text_links", "generate_citation",
            "batch_generate_citations", "export_search_results",
            "batch_isbn_lookup", "save_to_zotero",
            "list_zotero_collections", "batch_save_to_zotero",
            "search_zotero", "get_zotero_collection_items",
            "get_zotero_status",
        ]
        assert names == expected

    def test_removed_tools_not_present(self):
        names = [t.name for t in TOOL_DEFINITIONS]
        for removed in ["authenticate", "get_token_status", "clear_cache",
                        "read_article", "get_diagnostics", "zotero_authenticate"]:
            assert removed not in names

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
        assert len(mock_client_with_results.search_calls) == 1


class TestReciprocalRankFusion:
    def test_deduplication(self, sample_pnx_record):
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


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_recorded(self, mock_client_with_results):
        await handle_tool_call(
            "search_library",
            {"query": "test"},
            mock_client_with_results,
        )
        metrics = get_metrics()
        assert "search_library" in metrics
        assert metrics["search_library"]["count"] >= 1
