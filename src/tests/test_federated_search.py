"""Unit tests for FederatedSearchRouter (P.6)."""

import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock

from lib.federated_search import (
    FederatedSearchRouter,
    SearchSource,
    _normalize_doi,
    _rrf_merge,
)


# ── DOI normalization ─────────────────────────────────────────────────────────

def test_normalize_doi_strips_https():
    assert _normalize_doi("https://doi.org/10.1234/test") == "10.1234/test"


def test_normalize_doi_strips_http():
    assert _normalize_doi("http://doi.org/10.1234/test") == "10.1234/test"


def test_normalize_doi_bare():
    assert _normalize_doi("10.1234/test") == "10.1234/test"


def test_normalize_doi_empty():
    assert _normalize_doi("") == ""


# ── RRF merge ─────────────────────────────────────────────────────────────────

def _doc(doi, title):
    return {"doi": doi, "title": title}


def test_rrf_merge_deduplicates_by_doi():
    primary = [_doc("10.1/a", "A")]
    secondary = [_doc("10.1/a", "A-dup"), _doc("10.1/b", "B")]
    merged = _rrf_merge(primary, secondary, top_k=10)
    dois = [d["doi"] for d in merged]
    assert dois.count("10.1/a") == 1


def test_rrf_merge_respects_top_k():
    primary = [_doc(f"10.1/{i}", f"P{i}") for i in range(5)]
    secondary = [_doc(f"10.1/{i+5}", f"S{i}") for i in range(5)]
    merged = _rrf_merge(primary, secondary, top_k=3)
    assert len(merged) == 3


def test_rrf_merge_adds_rrf_score():
    primary = [_doc("10.1/a", "A")]
    secondary = [_doc("10.1/b", "B")]
    merged = _rrf_merge(primary, secondary, top_k=10)
    assert all("rrf_score" in d for d in merged)


def test_rrf_merge_higher_rank_wins():
    primary = [_doc("10.1/a", "A"), _doc("10.1/b", "B")]
    secondary = [_doc("10.1/b", "B")]
    merged = _rrf_merge(primary, secondary, top_k=10)
    # "B" appears in both lists so should score higher than "A"
    dois = [d["doi"] for d in merged]
    assert dois[0] == "10.1/b"


# ── Routing logic ─────────────────────────────────────────────────────────────

def _make_router(recency_days=30):
    oa = MagicMock()
    os_ = MagicMock()
    return FederatedSearchRouter(oa, os_, recency_days=recency_days)


def test_route_recent_date_filter_goes_to_live():
    router = _make_router()
    recent = (date.today() - timedelta(days=5)).isoformat()
    assert router.route("anything", {"from_publication_date": recent}) == SearchSource.LIVE_API


def test_route_old_date_filter_goes_to_local():
    router = _make_router()
    old = "2018-01-01"
    assert router.route("machine learning", {"from_publication_date": old}) == SearchSource.LOCAL_INDEX


def test_route_temporal_cue_recent_goes_to_live():
    router = _make_router()
    assert router.route("recent advances in CRISPR", {}) == SearchSource.LIVE_API


def test_route_temporal_cue_2025_goes_to_live():
    router = _make_router()
    assert router.route("AI papers 2025", {}) == SearchSource.LIVE_API


def test_route_historical_query_goes_to_local():
    router = _make_router()
    assert router.route("effects of climate change on salmon", {}) == SearchSource.LOCAL_INDEX


def test_route_no_filters_goes_to_local():
    router = _make_router()
    assert router.route("Indigenous land rights", {}) == SearchSource.LOCAL_INDEX


# ── Search dispatch ───────────────────────────────────────────────────────────

def _make_full_router():
    oa = MagicMock()
    os_ = MagicMock()
    oa.search_works.return_value = {"results": [_doc("10.1/a", "LiveA")]}
    os_.search.return_value = [_doc("10.1/b", "LocalB")]
    return FederatedSearchRouter(oa, os_, recency_days=30), oa, os_


def test_search_live_only_calls_only_openalex():
    router, oa, os_ = _make_full_router()
    results = router.search("query", {}, top_k=10, force_source=SearchSource.LIVE_API)
    oa.search_works.assert_called_once()
    os_.search.assert_not_called()
    assert results[0]["doi"] == "10.1/a"


def test_search_local_only_calls_only_opensearch():
    router, oa, os_ = _make_full_router()
    results = router.search("query", {}, top_k=10, force_source=SearchSource.LOCAL_INDEX)
    os_.search.assert_called_once()
    oa.search_works.assert_not_called()
    assert results[0]["doi"] == "10.1/b"


def test_search_both_merges_results():
    router, oa, os_ = _make_full_router()
    results = router.search("query", {}, top_k=10, force_source=SearchSource.BOTH)
    oa.search_works.assert_called_once()
    os_.search.assert_called_once()
    dois = {d["doi"] for d in results}
    assert "10.1/a" in dois
    assert "10.1/b" in dois


def test_search_live_exception_returns_empty():
    router, oa, os_ = _make_full_router()
    oa.search_works.side_effect = RuntimeError("down")
    results = router.search("query", {}, top_k=10, force_source=SearchSource.LIVE_API)
    assert results == []


def test_search_local_exception_returns_empty():
    router, oa, os_ = _make_full_router()
    os_.search.side_effect = RuntimeError("down")
    results = router.search("query", {}, top_k=10, force_source=SearchSource.LOCAL_INDEX)
    assert results == []


def test_search_doi_dedup_in_both_mode():
    oa = MagicMock()
    os_ = MagicMock()
    oa.search_works.return_value = {"results": [_doc("10.1/shared", "Live"), _doc("10.1/live-only", "L2")]}
    os_.search.return_value = [_doc("10.1/shared", "Dup"), _doc("10.1/local-only", "L3")]
    router = FederatedSearchRouter(oa, os_, recency_days=30)
    results = router.search("q", {}, top_k=10, force_source=SearchSource.BOTH)
    dois = [d["doi"] for d in results]
    assert dois.count("10.1/shared") == 1
    assert len(dois) == 3
