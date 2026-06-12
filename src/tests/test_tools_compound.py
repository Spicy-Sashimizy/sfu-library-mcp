"""Tests for the Code-Mode-inspired optimizations: compound tools, fields param, tool gating.

Covers correctness, durability (error paths, cache bounds, concurrency) and
efficiency (token/char reduction of compound flows vs. the two-call equivalent).
"""

import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

import lib.tools as tools
from lib.tools import (
    TOOL_DEFINITIONS,
    handle_tool_call,
    _excluded_tool_names,
    _build_search_filters,
    _requested_fields,
)
from lib.formatters import format_openalex_results


def _make_work(i: int) -> dict:
    """A realistically sized OpenAlex work (long abstract, several authors)."""
    return {
        "title": f"Deep Learning Approaches to Protein Structure Prediction, Part {i}",
        "authors": [f"Author{j} Q. Lastname{i}" for j in range(5)],
        "creators": [f"Author{j} Q. Lastname{i}" for j in range(5)],
        "contributors": [],
        "date": f"{2015 + (i % 10)}",
        "publisher": "Springer Nature",
        "type": "article",
        "source": "Nature Methods",
        "isbn": "",
        "issn": "1548-7091",
        "doi": f"10.1038/test{i:04d}",
        "volume": str(10 + i),
        "issue": str(1 + (i % 4)),
        "spage": "100",
        "epage": "120",
        "pages": "100-120",
        "resource_type": "article",
        "openalex_id": f"https://openalex.org/W{i:06d}",
        "is_oa": i % 2 == 0,
        "oa_url": f"https://example.com/paper{i}.pdf",
        "cited_by_count": 40 + i,
        "abstract": (
            "Protein structure prediction has been transformed by deep learning. "
            "We benchmark several architectures across folding tasks and report "
            "substantial accuracy improvements over physics-based baselines. " * 3
        ),
        "topics": ["Machine Learning", "Structural Biology", "Bioinformatics", "Proteomics"],
    }


def _openalex_response(n: int) -> dict:
    return {"results": [_make_work(i) for i in range(1, n + 1)],
            "meta": {"count": n, "page": 1, "per_page": n}}


FEATURES_DIRECT = {
    "federated_search_enabled": False,
    "rerank_enabled": False,
    "zotero_enabled": True,
}


def _patch_direct_search(n_results: int = 10):
    """Patch the direct-OpenAlex search path with n mocked results."""
    mock_oa = MagicMock()
    mock_oa.search_works.return_value = _openalex_response(n_results)
    mock_oa.budget_status.return_value = {"exhausted": False}
    mock_oa.circuit_open = False
    return patch.multiple(
        "lib.tools",
        _get_features=MagicMock(return_value=dict(FEATURES_DIRECT)),
        _get_openalex=MagicMock(return_value=mock_oa),
        enrich_metadata_from_crossref=MagicMock(side_effect=lambda m: m),
    )


def _mock_zotero():
    zot = MagicMock()
    zot.ensure_authenticated.return_value = True
    zot.find_or_create_collection.return_value = "COLKEY"
    zot.check_duplicate.return_value = {"is_duplicate": False}
    zot.metadata_to_zotero_item.return_value = {}
    zot.create_item.return_value = "ITEMKEY"
    return zot


# ── Tool definitions ──────────────────────────────────────────────────────────

class TestDefinitions:
    def test_compound_tools_present(self):
        names = {t.name for t in TOOL_DEFINITIONS}
        assert {"search_and_cite", "search_and_save"}.issubset(names)

    def test_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 25

    def test_fields_param_on_search_tools(self):
        by_name = {t.name: t for t in TOOL_DEFINITIONS}
        for name in ("search_academic", "search_by_author", "search_by_doi", "search_by_topic"):
            assert "fields" in by_name[name].inputSchema["properties"], name

    def test_batch_citations_cap_raised(self):
        by_name = {t.name: t for t in TOOL_DEFINITIONS}
        desc = by_name["batch_generate_citations"].inputSchema["properties"]["dois"]["description"]
        assert "50" in desc


# ── Gating ────────────────────────────────────────────────────────────────────

class TestGating:
    def _config(self, *, zotero_enabled=True, api_key="k", user_id="u",
                engagement=False, pmc=False):
        cfg = MagicMock()
        cfg.features = {
            "zotero_enabled": zotero_enabled,
            "engagement_log_enabled": engagement,
            "europe_pmc_enabled": pmc,
        }
        cfg.zotero_api_key = api_key
        cfg.zotero_user_id = user_id
        return cfg

    def test_zotero_tools_gated_without_creds(self):
        with patch("lib.tools._get_config", return_value=self._config(api_key="")):
            excluded = _excluded_tool_names()
        assert "save_to_zotero" in excluded
        assert "search_and_save" in excluded
        assert "search_zotero" in excluded

    def test_zotero_tools_kept_with_creds(self):
        with patch("lib.tools._get_config", return_value=self._config()):
            excluded = _excluded_tool_names()
        assert "save_to_zotero" not in excluded
        assert "search_and_save" not in excluded

    def test_engagement_and_biomedical_gated_by_default(self):
        with patch("lib.tools._get_config", return_value=self._config()):
            excluded = _excluded_tool_names()
        assert "record_engagement" in excluded
        assert "search_biomedical" in excluded

    def test_engagement_and_biomedical_kept_when_enabled(self):
        cfg = self._config(engagement=True, pmc=True)
        with patch("lib.tools._get_config", return_value=cfg):
            excluded = _excluded_tool_names()
        assert "record_engagement" not in excluded
        assert "search_biomedical" not in excluded

    @pytest.mark.asyncio
    async def test_get_tool_definitions_filters(self):
        old_cache = tools._tool_definitions_cache
        tools._tool_definitions_cache = None
        try:
            with patch("lib.tools._get_config", return_value=self._config(api_key="")), \
                 patch("lib.tools.fetch_zotero_item_types", return_value=[]):
                defs = await tools.get_tool_definitions()
            names = {t.name for t in defs}
            assert "save_to_zotero" not in names
            assert "search_and_cite" in names
            assert "search_academic" in names
        finally:
            tools._tool_definitions_cache = old_cache


# ── fields parameter ──────────────────────────────────────────────────────────

class TestFieldsParam:
    def test_requested_fields_validation(self):
        assert _requested_fields({}) is None
        assert _requested_fields({"fields": "authors"}) is None
        assert _requested_fields({"fields": ["authors", 7, "date"]}) == ["authors", "date"]

    def test_formatter_full_output_unchanged_when_fields_none(self):
        data = _openalex_response(2)
        out = format_openalex_results(data, "q")
        assert "Abstract:" in out and "Topics:" in out and "Authors:" in out

    def test_formatter_limits_fields(self):
        data = _openalex_response(2)
        out = format_openalex_results(data, "q", fields=["authors", "date"])
        assert "Authors:" in out and "Date:" in out
        assert "Abstract:" not in out and "Topics:" not in out and "ID:" not in out
        assert "DOI:" in out  # always kept as the follow-up handle

    def test_formatter_empty_fields_minimal(self):
        data = _openalex_response(1)
        out = format_openalex_results(data, "q", fields=[])
        assert "DOI:" in out and "Abstract:" not in out and "Authors:" not in out

    @pytest.mark.asyncio
    async def test_search_academic_respects_fields(self):
        with _patch_direct_search(3):
            full = await handle_tool_call("search_academic", {"query": "proteins"})
            slim = await handle_tool_call(
                "search_academic", {"query": "proteins", "fields": ["authors"]}
            )
        assert "Abstract:" in full[0].text
        assert "Abstract:" not in slim[0].text
        assert "Authors:" in slim[0].text
        assert len(slim[0].text) < len(full[0].text)


# ── Compound: search_and_cite ─────────────────────────────────────────────────

class TestSearchAndCite:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        with _patch_direct_search(5):
            result = await handle_tool_call(
                "search_and_cite", {"query": "protein folding", "format": "apa", "limit": 5}
            )
        text = result[0].text
        assert "APA 7th Edition" in text
        assert "DOI: 10.1038/test0001" in text
        assert "Abstract:" not in text  # the whole point: no intermediate result list

    @pytest.mark.asyncio
    async def test_bibtex_format(self):
        with _patch_direct_search(2):
            result = await handle_tool_call(
                "search_and_cite", {"query": "protein folding", "format": "bibtex", "limit": 2}
            )
        assert "@" in result[0].text

    @pytest.mark.asyncio
    async def test_empty_query(self):
        result = await handle_tool_call("search_and_cite", {"query": "   "})
        assert "Empty search query" in result[0].text

    @pytest.mark.asyncio
    async def test_no_results(self):
        mock_oa = MagicMock()
        mock_oa.search_works.return_value = {"results": [], "meta": {"count": 0}}
        mock_oa.budget_status.return_value = {"exhausted": False}
        mock_oa.circuit_open = False
        with patch.multiple(
            "lib.tools",
            _get_features=MagicMock(return_value=dict(FEATURES_DIRECT)),
            _get_openalex=MagicMock(return_value=mock_oa),
        ):
            result = await handle_tool_call("search_and_cite", {"query": "zzzz"})
        assert "No results found" in result[0].text

    @pytest.mark.asyncio
    async def test_backend_exception_is_caught(self):
        mock_oa = MagicMock()
        mock_oa.search_works.side_effect = RuntimeError("boom")
        mock_oa.budget_status.return_value = {"exhausted": False}
        mock_oa.circuit_open = False
        with patch.multiple(
            "lib.tools",
            _get_features=MagicMock(return_value=dict(FEATURES_DIRECT)),
            _get_openalex=MagicMock(return_value=mock_oa),
        ):
            result = await handle_tool_call("search_and_cite", {"query": "x"})
        assert "Error in search_and_cite" in result[0].text

    @pytest.mark.asyncio
    async def test_federated_route_used_when_enabled(self):
        router = MagicMock()
        router.search.return_value = [_make_work(1)]
        router.last_degraded = False
        with patch.multiple(
            "lib.tools",
            _get_features=MagicMock(return_value={
                "federated_search_enabled": True, "rerank_enabled": False,
            }),
            _get_federated_router=MagicMock(return_value=router),
            enrich_metadata_from_crossref=MagicMock(side_effect=lambda m: m),
        ):
            result = await handle_tool_call("search_and_cite", {"query": "salmon", "limit": 1})
        assert router.search.called
        assert "DOI: 10.1038/test0001" in result[0].text

    @pytest.mark.asyncio
    async def test_degraded_local_index_noted(self):
        router = MagicMock()
        router.search.return_value = [_make_work(1)]
        router.last_degraded = True
        with patch.multiple(
            "lib.tools",
            _get_features=MagicMock(return_value={
                "federated_search_enabled": True, "rerank_enabled": False,
            }),
            _get_federated_router=MagicMock(return_value=router),
            enrich_metadata_from_crossref=MagicMock(side_effect=lambda m: m),
        ):
            result = await handle_tool_call("search_and_cite", {"query": "salmon", "limit": 1})
        assert "[Note:" in result[0].text

    @pytest.mark.asyncio
    async def test_s2_fallback_when_budget_exhausted(self):
        mock_oa = MagicMock()
        mock_oa.budget_status.return_value = {
            "exhausted": True, "calls_today": 900, "daily_limit": 900,
        }
        mock_oa.circuit_open = False
        mock_s2 = MagicMock()
        mock_s2.search_papers.return_value = [
            {"title": "S2 Paper", "authors": ["A. Author"], "year": 2021,
             "doi": "10.99/s2paper", "citation_count": 5},
        ]
        with patch.multiple(
            "lib.tools",
            _get_features=MagicMock(return_value=dict(FEATURES_DIRECT)),
            _get_openalex=MagicMock(return_value=mock_oa),
            _get_s2=MagicMock(return_value=mock_s2),
            enrich_metadata_from_crossref=MagicMock(side_effect=lambda m: m),
        ):
            result = await handle_tool_call("search_and_cite", {"query": "x", "limit": 5})
        text = result[0].text
        assert "Semantic Scholar" in text
        assert "10.99/s2paper" in text


# ── Compound: search_and_save ─────────────────────────────────────────────────

class TestSearchAndSave:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        zot = _mock_zotero()
        with _patch_direct_search(3), \
             patch("lib.tools._ensure_zotero_auth", return_value=None), \
             patch("lib.tools._get_zotero_client", return_value=zot):
            result = await handle_tool_call(
                "search_and_save",
                {"query": "protein folding", "collection_name": "Proteins", "limit": 3},
            )
        text = result[0].text
        assert "saving top 3" in text
        assert "3 saved" in text
        assert zot.create_item.call_count == 3
        assert "Abstract:" not in text

    @pytest.mark.asyncio
    async def test_missing_collection(self):
        with patch("lib.tools._ensure_zotero_auth", return_value=None):
            result = await handle_tool_call("search_and_save", {"query": "x"})
        assert "collection_name" in result[0].text

    @pytest.mark.asyncio
    async def test_auth_failure_short_circuits(self):
        from mcp.types import TextContent
        err = [TextContent(type="text", text="Zotero authentication failed.")]
        with patch("lib.tools._ensure_zotero_auth", return_value=err):
            result = await handle_tool_call(
                "search_and_save", {"query": "x", "collection_name": "C"}
            )
        assert "authentication failed" in result[0].text

    @pytest.mark.asyncio
    async def test_results_without_dois(self):
        mock_oa = MagicMock()
        works = [{**_make_work(1), "doi": "", "openalex_id": "https://openalex.org/W1"}]
        mock_oa.search_works.return_value = {"results": works, "meta": {"count": 1}}
        mock_oa.budget_status.return_value = {"exhausted": False}
        mock_oa.circuit_open = False
        with patch.multiple(
            "lib.tools",
            _get_features=MagicMock(return_value=dict(FEATURES_DIRECT)),
            _get_openalex=MagicMock(return_value=mock_oa),
        ), patch("lib.tools._ensure_zotero_auth", return_value=None):
            result = await handle_tool_call(
                "search_and_save", {"query": "x", "collection_name": "C"}
            )
        assert "none had a DOI" in result[0].text

    @pytest.mark.asyncio
    async def test_duplicates_skipped(self):
        zot = _mock_zotero()
        zot.check_duplicate.return_value = {"is_duplicate": True, "match_type": "doi"}
        with _patch_direct_search(2), \
             patch("lib.tools._ensure_zotero_auth", return_value=None), \
             patch("lib.tools._get_zotero_client", return_value=zot):
            result = await handle_tool_call(
                "search_and_save", {"query": "x", "collection_name": "C", "limit": 2}
            )
        assert "2 skipped" in result[0].text
        assert zot.create_item.call_count == 0


# ── Durability ────────────────────────────────────────────────────────────────

class TestDurability:
    @pytest.mark.asyncio
    async def test_work_cache_stays_bounded(self):
        """Repeated compound calls must not grow the work cache past its cap."""
        for batch in range(40):
            works = [_make_work(batch * 20 + i) for i in range(20)]
            tools._cache_works(works)
        assert len(tools._work_cache) <= 500

    @pytest.mark.asyncio
    async def test_twenty_concurrent_compound_calls(self):
        with _patch_direct_search(5):
            results = await asyncio.gather(*[
                handle_tool_call("search_and_cite", {"query": f"topic {i}", "limit": 3})
                for i in range(20)
            ])
        assert len(results) == 20
        for r in results:
            assert "Error" not in r[0].text

    @pytest.mark.asyncio
    async def test_repeated_sequential_calls_stable(self):
        with _patch_direct_search(5):
            for i in range(50):
                result = await handle_tool_call(
                    "search_and_cite", {"query": "stability", "limit": 2}
                )
                assert "APA" in result[0].text
        assert len(tools._work_cache) <= 500

    @pytest.mark.asyncio
    async def test_batch_citations_caps_at_50(self):
        meta = _make_work(1)
        with patch("lib.tools._fetch_work_metadata", new_callable=AsyncMock) as mock_fetch, \
             patch("lib.tools.enrich_metadata_from_crossref", side_effect=lambda m: m):
            mock_fetch.return_value = meta
            result = await handle_tool_call(
                "batch_generate_citations",
                {"dois": [f"10.1/{i}" for i in range(60)], "format": "apa"},
            )
        assert mock_fetch.await_count == 50
        assert "(50 items)" in result[0].text

    @pytest.mark.asyncio
    async def test_limit_clamping(self):
        with _patch_direct_search(10):
            result = await handle_tool_call(
                "search_and_cite", {"query": "x", "limit": 9999}
            )
        assert "Error" not in result[0].text

    @pytest.mark.asyncio
    async def test_unknown_fields_ignored(self):
        with _patch_direct_search(2):
            result = await handle_tool_call(
                "search_academic", {"query": "x", "fields": ["nonsense", "authors"]}
            )
        assert "Authors:" in result[0].text
        assert "Error" not in result[0].text


# ── Efficiency ────────────────────────────────────────────────────────────────

class TestEfficiency:
    @pytest.mark.asyncio
    async def test_compound_cite_flow_smaller_than_two_calls(self):
        """search_and_cite must return substantially less text than
        search_academic + batch_generate_citations for the same task."""
        with _patch_direct_search(10):
            search = await handle_tool_call("search_academic", {"query": "proteins", "limit": 10})
            compound = await handle_tool_call(
                "search_and_cite", {"query": "proteins", "format": "apa", "limit": 10}
            )
            dois = [f"10.1038/test{i:04d}" for i in range(1, 11)]
            batch = await handle_tool_call(
                "batch_generate_citations", {"dois": dois, "format": "apa"}
            )
        old_flow = len(search[0].text) + len(batch[0].text)
        new_flow = len(compound[0].text)
        reduction = 1 - new_flow / old_flow
        print(f"\n[efficiency] two-call flow: {old_flow} chars (~{old_flow // 4} tokens), "
              f"search_and_cite: {new_flow} chars (~{new_flow // 4} tokens), "
              f"reduction: {reduction:.0%}")
        assert reduction >= 0.40

    @pytest.mark.asyncio
    async def test_fields_param_reduction(self):
        with _patch_direct_search(10):
            full = await handle_tool_call("search_academic", {"query": "proteins", "limit": 10})
            slim = await handle_tool_call(
                "search_academic",
                {"query": "proteins", "limit": 10, "fields": ["authors", "date"]},
            )
        reduction = 1 - len(slim[0].text) / len(full[0].text)
        print(f"\n[efficiency] full search: {len(full[0].text)} chars, "
              f"fields=[authors,date]: {len(slim[0].text)} chars, reduction: {reduction:.0%}")
        assert reduction >= 0.40

    def test_gated_tool_list_smaller(self):
        """The advertised tool list (default config: no engagement, no Europe PMC)
        must serialize smaller than the full list."""
        full = sum(
            len(json.dumps({"name": t.name, "description": t.description,
                            "schema": t.inputSchema}))
            for t in TOOL_DEFINITIONS
        )
        excluded = {"record_engagement", "search_biomedical"}
        gated = sum(
            len(json.dumps({"name": t.name, "description": t.description,
                            "schema": t.inputSchema}))
            for t in TOOL_DEFINITIONS if t.name not in excluded
        )
        print(f"\n[efficiency] full tool list: {full} chars (~{full // 4} tokens), "
              f"gated: {gated} chars (~{gated // 4} tokens), "
              f"saved: {full - gated} chars (~{(full - gated) // 4} tokens)")
        assert gated < full


# ── Round 3: persistence, trimmed output, compact mode ───────────────────────

@pytest.fixture(autouse=True)
def _restore_work_cache_state():
    """Keep work-cache path/state mutations from leaking between tests."""
    yield
    try:
        from lib.config import load_config
        tools._get_config().work_cache_path = load_config().work_cache_path
    except Exception:
        pass
    tools._work_cache_loaded = True
    tools._work_cache_last_save = 0.0


class TestWorkCachePersistence:
    def _fresh(self, tmp_path):
        """Point the work cache at a temp file and reset its state."""
        tools._get_config().work_cache_path = str(tmp_path / "work_cache.json")
        tools._work_cache.clear()
        tools._work_cache_loaded = True  # don't restore other tests' state
        tools._work_cache_last_save = 0.0

    def test_persist_and_restore(self, tmp_path):
        self._fresh(tmp_path)
        tools._cache_works([_make_work(1), _make_work(2)])
        assert (tmp_path / "work_cache.json").is_file()
        # simulate restart
        tools._work_cache.clear()
        tools._work_cache_loaded = False
        work = tools._lookup_work("10.1038/test0001")
        assert work and work["title"].startswith("Deep Learning")

    def test_save_is_debounced(self, tmp_path):
        self._fresh(tmp_path)
        tools._cache_works([_make_work(1)])
        first_mtime = (tmp_path / "work_cache.json").stat().st_mtime_ns
        tools._cache_works([_make_work(2)])  # within debounce window
        assert (tmp_path / "work_cache.json").stat().st_mtime_ns == first_mtime

    def test_corrupt_cache_file_nonfatal(self, tmp_path):
        self._fresh(tmp_path)
        (tmp_path / "work_cache.json").write_text("{not json")
        tools._work_cache_loaded = False
        assert tools._lookup_work("10.1/none") is None  # no exception

    def test_unwritable_path_nonfatal(self):
        tools._get_config().work_cache_path = "/proc/definitely/not/writable.json"
        tools._work_cache_loaded = True
        tools._work_cache_last_save = 0.0
        tools._cache_works([_make_work(3)])  # must not raise
        assert tools._lookup_work("10.1038/test0003") is not None


class TestTrackerPersistence:
    def test_default_path_not_tmp(self):
        from lib.config import load_config
        import os
        old = os.environ.pop("OPENALEX_TRACKER_PATH", None)
        try:
            cfg = load_config()
            assert not cfg.openalex_tracker_path.startswith("/tmp")
            assert cfg.openalex_tracker_path.endswith("data/openalex_calls.json")
        finally:
            if old is not None:
                os.environ["OPENALEX_TRACKER_PATH"] = old

    def test_tracker_survives_reinit(self, tmp_path):
        from lib.openalex import DailyCallTracker
        path = str(tmp_path / "calls.json")
        t1 = DailyCallTracker(limit=900, path=path)
        for _ in range(5):
            t1.increment()
        # simulate restart
        t2 = DailyCallTracker(limit=900, path=path)
        assert t2.status()["calls_today"] == 5


class TestTrimmedOutput:
    @pytest.mark.asyncio
    async def test_no_ruler_lines_in_search(self):
        with _patch_direct_search(3):
            result = await handle_tool_call("search_academic", {"query": "x"})
        assert "======" not in result[0].text

    @pytest.mark.asyncio
    async def test_budget_no_bar_but_labels_kept(self):
        mock_oa = MagicMock()
        mock_oa.budget_status.return_value = {
            "exhausted": True, "calls_today": 900, "daily_limit": 900,
            "remaining": 0, "pct_used": 100.0,
        }
        with patch("lib.tools._get_openalex", return_value=mock_oa):
            result = await handle_tool_call("get_openalex_budget", {})
        text = result[0].text
        assert "EXHAUSTED" in text and "900" in text
        assert "█" not in text and "====" not in text

    @pytest.mark.asyncio
    async def test_zotero_save_no_banner(self):
        zot = _mock_zotero()
        with patch("lib.tools._ensure_zotero_auth", return_value=None), \
             patch("lib.tools._get_zotero_client", return_value=zot), \
             patch("lib.tools._fetch_work_metadata", new_callable=AsyncMock, return_value=_make_work(1)), \
             patch("lib.tools.enrich_metadata_from_crossref", side_effect=lambda m: m):
            result = await handle_tool_call("save_to_zotero", {"doi": "10.1038/test0001"})
        text = result[0].text
        assert "SAVED TO ZOTERO" in text
        assert "====" not in text


class TestCompactMode:
    @pytest.mark.asyncio
    async def test_generate_citation_compact(self):
        with patch("lib.tools._fetch_work_metadata", new_callable=AsyncMock, return_value=_make_work(1)), \
             patch("lib.tools.enrich_metadata_from_crossref", side_effect=lambda m: m):
            full = await handle_tool_call("generate_citation", {"doi": "10.1038/test0001"})
            compact = await handle_tool_call(
                "generate_citation", {"doi": "10.1038/test0001", "compact": True}
            )
        assert "--- APA" in full[0].text
        assert "---" not in compact[0].text
        assert len(compact[0].text) < len(full[0].text)

    @pytest.mark.asyncio
    async def test_batch_citations_compact(self):
        with patch("lib.tools._fetch_work_metadata", new_callable=AsyncMock, return_value=_make_work(1)), \
             patch("lib.tools.enrich_metadata_from_crossref", side_effect=lambda m: m):
            compact = await handle_tool_call(
                "batch_generate_citations",
                {"dois": ["10.1/a", "10.1/b"], "format": "apa", "compact": True},
            )
        text = compact[0].text
        assert "--- APA" not in text
        assert not text.lstrip().startswith("1.")

    @pytest.mark.asyncio
    async def test_search_and_cite_compact(self):
        with _patch_direct_search(5):
            full = await handle_tool_call(
                "search_and_cite", {"query": "x", "format": "apa", "limit": 5}
            )
            compact = await handle_tool_call(
                "search_and_cite", {"query": "x", "format": "apa", "limit": 5, "compact": True}
            )
        assert "DOI:" in full[0].text
        assert "DOI:" not in compact[0].text and "---" not in compact[0].text
        reduction = 1 - len(compact[0].text) / len(full[0].text)
        print(f"\n[efficiency] search_and_cite full: {len(full[0].text)} chars, "
              f"compact: {len(compact[0].text)} chars, reduction: {reduction:.0%}")
        # compact's primary value is structural (pasteable bibliography);
        # the size reduction is a secondary ~10-20% benefit
        assert reduction >= 0.10

    @pytest.mark.asyncio
    async def test_export_compact_drops_header(self):
        mock_oa = MagicMock()
        mock_oa.search_works.return_value = _openalex_response(2)
        with patch("lib.tools._get_openalex", return_value=mock_oa):
            out = await handle_tool_call(
                "export_search_results", {"query": "x", "format": "csv", "compact": True}
            )
        assert not out[0].text.startswith("#")
        assert out[0].text.startswith("doi,title")
