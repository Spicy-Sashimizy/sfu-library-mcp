"""Unit tests for the tools module — new OpenAlex-based tool set."""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from lib.tools import TOOL_DEFINITIONS, handle_tool_call, get_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_WORK = {
    "title": "Machine Learning in Practice",
    "authors": ["LeCun, Yann", "Bengio, Yoshua"],
    "creators": ["LeCun, Yann", "Bengio, Yoshua"],
    "contributors": [],
    "date": "2024",
    "publisher": "MIT Press",
    "type": "article",
    "source": "Nature",
    "isbn": "",
    "issn": "0028-0836",
    "doi": "10.1038/test123",
    "volume": "12",
    "issue": "3",
    "spage": "100",
    "epage": "120",
    "pages": "100-120",
    "record_id": "https://openalex.org/W123",
    "is_cdi": False,
    "resource_type": "article",
    "openalex_id": "https://openalex.org/W123",
    "is_oa": True,
    "oa_url": "https://example.com/paper.pdf",
    "cited_by_count": 42,
    "abstract": "A foundational paper on machine learning.",
    "topics": ["Machine Learning", "Neural Networks"],
}

SAMPLE_OPENALEX_RESPONSE = {
    "results": [SAMPLE_WORK],
    "meta": {"count": 1, "page": 1, "per_page": 10},
}

SAMPLE_SFU_DB = {
    "id": "test123",
    "name": "PsycINFO",
    "description": "Psychology and behavioral science database.",
    "url": "https://search.ebscohost.com/",
    "provider": "EBSCOhost",
    "subjects": ["Psychology"],
    "contentTypes": ["Index"],
    "free": False,
    "proxy": True,
}


# ── Tool definition tests ─────────────────────────────────────────────────────

class TestToolDefinitions:
    def test_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 22

    def test_new_tool_names_present(self):
        names = {t.name for t in TOOL_DEFINITIONS}
        expected_new = {
            "search_academic", "search_by_author", "search_by_doi",
            "search_by_topic", "get_citations", "get_references",
            "get_paper_summary", "find_open_access", "get_full_text_link",
            "browse_sfu_databases", "check_sfu_access", "search_biomedical",
        }
        assert expected_new.issubset(names)

    def test_primo_tools_removed(self):
        names = {t.name for t in TOOL_DEFINITIONS}
        removed = {
            "search_library", "get_item_details", "search_by_subject",
            "search_by_isbn", "search_electronic_resources", "batch_isbn_lookup",
            "get_full_text_links",
        }
        assert removed.isdisjoint(names)

    def test_zotero_tools_kept(self):
        names = {t.name for t in TOOL_DEFINITIONS}
        assert {"save_to_zotero", "list_zotero_collections", "batch_save_to_zotero",
                "search_zotero", "get_zotero_collection_items", "get_zotero_status"}.issubset(names)

    def test_all_tools_have_object_schemas(self):
        for tool in TOOL_DEFINITIONS:
            assert tool.inputSchema is not None
            assert tool.inputSchema.get("type") == "object"

    def test_doi_based_tools_require_doi(self):
        doi_tools = {"search_by_doi", "get_citations", "get_references",
                     "get_paper_summary", "find_open_access", "get_full_text_link",
                     "generate_citation", "save_to_zotero"}
        for tool in TOOL_DEFINITIONS:
            if tool.name in doi_tools:
                required = tool.inputSchema.get("required", [])
                assert "doi" in required, f"{tool.name} should require doi"


# ── Dispatch / handler tests ──────────────────────────────────────────────────

class TestSearchAcademic:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.search_works.return_value = SAMPLE_OPENALEX_RESPONSE
            result = await handle_tool_call("search_academic", {"query": "machine learning"})
        assert len(result) == 1
        assert "Machine Learning in Practice" in result[0].text

    @pytest.mark.asyncio
    async def test_empty_query(self):
        result = await handle_tool_call("search_academic", {"query": ""})
        assert "Empty search query" in result[0].text

    @pytest.mark.asyncio
    async def test_with_year_filter(self):
        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.search_works.return_value = SAMPLE_OPENALEX_RESPONSE
            result = await handle_tool_call(
                "search_academic",
                {"query": "climate", "year_from": 2020, "year_to": 2024},
            )
        assert "Machine Learning" in result[0].text
        call_kwargs = mock_oa.return_value.search_works.call_args
        filters = call_kwargs[1].get("filters") or call_kwargs[0][1]
        assert "publication_year" in filters

    @pytest.mark.asyncio
    async def test_no_results(self):
        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.search_works.return_value = {"results": [], "meta": {"count": 0}}
            result = await handle_tool_call("search_academic", {"query": "xyznotreal"})
        assert "No results found" in result[0].text


class TestSearchByDoi:
    @pytest.mark.asyncio
    async def test_found_via_openalex(self):
        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.get_work_by_doi.return_value = SAMPLE_WORK
            result = await handle_tool_call("search_by_doi", {"doi": "10.1038/test123"})
        assert "Machine Learning in Practice" in result[0].text

    @pytest.mark.asyncio
    async def test_empty_doi(self):
        result = await handle_tool_call("search_by_doi", {"doi": ""})
        assert "No DOI" in result[0].text

    @pytest.mark.asyncio
    async def test_not_found(self):
        with patch("lib.tools._get_openalex") as mock_oa, \
             patch("lib.openalex.fetch_crossref_work", return_value=None):
            mock_oa.return_value.get_work_by_doi.return_value = None
            result = await handle_tool_call("search_by_doi", {"doi": "10.9999/notreal"})
        assert "No record found" in result[0].text


class TestBrowseSfuDatabases:
    @pytest.mark.asyncio
    async def test_basic_browse(self):
        with patch("lib.tools._get_registry") as mock_reg:
            mock_reg.return_value.search.return_value = [SAMPLE_SFU_DB]
            result = await handle_tool_call("browse_sfu_databases", {"query": "psychology"})
        assert "PsycINFO" in result[0].text

    @pytest.mark.asyncio
    async def test_no_results(self):
        with patch("lib.tools._get_registry") as mock_reg:
            mock_reg.return_value.search.return_value = []
            result = await handle_tool_call("browse_sfu_databases", {})
        assert "No databases found" in result[0].text


class TestCheckSfuAccess:
    @pytest.mark.asyncio
    async def test_found(self):
        with patch("lib.tools._get_registry") as mock_reg:
            mock_reg.return_value.search.return_value = [SAMPLE_SFU_DB]
            result = await handle_tool_call("check_sfu_access", {"name": "PsycINFO"})
        assert "FOUND" in result[0].text
        assert "PsycINFO" in result[0].text

    @pytest.mark.asyncio
    async def test_not_found(self):
        with patch("lib.tools._get_registry") as mock_reg:
            mock_reg.return_value.search.return_value = []
            result = await handle_tool_call("check_sfu_access", {"name": "FakeDatabase"})
        assert "NOT FOUND" in result[0].text

    @pytest.mark.asyncio
    async def test_no_name(self):
        result = await handle_tool_call("check_sfu_access", {"name": ""})
        assert "No database name" in result[0].text


class TestGenerateCitation:
    @pytest.mark.asyncio
    async def test_apa_citation(self):
        with patch("lib.tools._fetch_work_metadata", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_WORK
            result = await handle_tool_call(
                "generate_citation", {"doi": "10.1038/test123", "format": "apa"}
            )
        assert "APA" in result[0].text

    @pytest.mark.asyncio
    async def test_bibtex_citation(self):
        with patch("lib.tools._fetch_work_metadata", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_WORK
            result = await handle_tool_call(
                "generate_citation", {"doi": "10.1038/test123", "format": "bibtex"}
            )
        assert "BibTeX" in result[0].text

    @pytest.mark.asyncio
    async def test_no_doi(self):
        result = await handle_tool_call("generate_citation", {"doi": ""})
        assert "No DOI" in result[0].text

    @pytest.mark.asyncio
    async def test_not_found(self):
        with patch("lib.tools._fetch_work_metadata", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None
            result = await handle_tool_call(
                "generate_citation", {"doi": "10.9999/bad"}
            )
        assert "Could not retrieve" in result[0].text


class TestBatchCitations:
    @pytest.mark.asyncio
    async def test_batch(self):
        with patch("lib.tools._fetch_work_metadata", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_WORK
            result = await handle_tool_call(
                "batch_generate_citations",
                {"dois": ["10.1038/a", "10.1038/b"], "format": "apa"},
            )
        assert "Citations" in result[0].text

    @pytest.mark.asyncio
    async def test_empty_list(self):
        result = await handle_tool_call("batch_generate_citations", {"dois": []})
        assert "No DOIs" in result[0].text


class TestExportSearch:
    @pytest.mark.asyncio
    async def test_bibtex_export(self):
        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.search_works.return_value = SAMPLE_OPENALEX_RESPONSE
            result = await handle_tool_call(
                "export_search_results", {"query": "test", "format": "bibtex"}
            )
        assert "BibTeX Export" in result[0].text

    @pytest.mark.asyncio
    async def test_csv_export(self):
        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.search_works.return_value = SAMPLE_OPENALEX_RESPONSE
            result = await handle_tool_call(
                "export_search_results", {"query": "test", "format": "csv"}
            )
        assert "CSV Export" in result[0].text

    @pytest.mark.asyncio
    async def test_json_export(self):
        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.search_works.return_value = SAMPLE_OPENALEX_RESPONSE
            result = await handle_tool_call(
                "export_search_results", {"query": "test", "format": "json"}
            )
        assert "JSON Export" in result[0].text


class TestUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown(self):
        result = await handle_tool_call("nonexistent_tool", {})
        assert "Unknown tool" in result[0].text


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_recorded(self):
        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.search_works.return_value = SAMPLE_OPENALEX_RESPONSE
            await handle_tool_call("search_academic", {"query": "test"})
        metrics = get_metrics()
        assert "search_academic" in metrics
        assert metrics["search_academic"]["count"] >= 1
