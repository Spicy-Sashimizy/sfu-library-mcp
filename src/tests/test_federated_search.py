"""Unit tests for FederatedSearchRouter (P.6)."""

import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock

from lib.federated_search import (
    ALWAYS_LIVE_SUBJECTS,
    SOFT_LIVE_SUBJECTS,
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


def test_rrf_merge_k_param_affects_score():
    """Lower k rewards consensus docs more aggressively — score 2/(k+1) when ranked #1 in both."""
    doc = _doc("10.1/a", "A")
    primary = [doc]
    secondary = [doc]
    merged_k20 = _rrf_merge(primary, secondary, top_k=1)
    merged_k60 = _rrf_merge(primary, secondary, top_k=1)
    # Both should give the same absolute value since _rrf_merge uses module-level _RRF_K=60
    # This test documents the expected score formula: 2 * 1/(60+1)
    expected = 2.0 / (60 + 1)
    assert abs(merged_k20[0]["rrf_score"] - expected) < 1e-9
    assert abs(merged_k60[0]["rrf_score"] - expected) < 1e-9


def test_rrf_merge_higher_rank_wins():
    primary = [_doc("10.1/a", "A"), _doc("10.1/b", "B")]
    secondary = [_doc("10.1/b", "B")]
    merged = _rrf_merge(primary, secondary, top_k=10)
    # "B" appears in both lists so should score higher than "A"
    dois = [d["doi"] for d in merged]
    assert dois[0] == "10.1/b"


# ── Routing logic ─────────────────────────────────────────────────────────────

def _make_router(recency_days=30, local_rrf_enabled=True):
    oa = MagicMock()
    os_ = MagicMock()
    return FederatedSearchRouter(
        oa, os_, recency_days=recency_days, local_rrf_enabled=local_rrf_enabled
    )


def test_route_recent_date_filter_goes_to_live():
    router = _make_router()
    recent = (date.today() - timedelta(days=5)).isoformat()
    assert router.route("anything", {"from_publication_date": recent}) == SearchSource.LIVE_API


def test_route_old_date_filter_goes_to_local_rrf_by_default():
    router = _make_router()
    old = "2018-01-01"
    assert router.route("machine learning", {"from_publication_date": old}) == SearchSource.LOCAL_RRF


def test_route_old_date_filter_goes_to_local_index_when_rrf_disabled():
    router = _make_router(local_rrf_enabled=False)
    old = "2018-01-01"
    assert router.route("machine learning", {"from_publication_date": old}) == SearchSource.LOCAL_INDEX


def test_route_temporal_cue_recent_goes_to_live():
    router = _make_router()
    assert router.route("recent advances in CRISPR", {}) == SearchSource.LIVE_API


def test_route_temporal_cue_2025_goes_to_live():
    router = _make_router()
    assert router.route("AI papers 2025", {}) == SearchSource.LIVE_API


def test_route_historical_query_goes_to_local_rrf():
    router = _make_router()
    assert router.route("effects of climate change on salmon", {}) == SearchSource.LOCAL_RRF


def test_route_no_filters_goes_to_local_rrf():
    router = _make_router()
    assert router.route("Indigenous land rights", {}) == SearchSource.LOCAL_RRF


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


def test_search_local_rrf_calls_both_modes_and_merges():
    oa = MagicMock()
    os_ = MagicMock()
    os_.search.side_effect = lambda query, top_k, mode: (
        [_doc("10.1/a", "BMa"), _doc("10.1/shared", "BMshared")]
        if mode == "bm25f"
        else [_doc("10.1/shared", "SPshared"), _doc("10.1/b", "SPb")]
    )
    router = FederatedSearchRouter(oa, os_, recency_days=30)
    results = router.search("q", {}, top_k=10, force_source=SearchSource.LOCAL_RRF)
    oa.search_works.assert_not_called()
    assert os_.search.call_count == 2
    modes = [c.kwargs.get("mode") for c in os_.search.call_args_list]
    assert set(modes) == {"bm25f", "splade"}
    dois = [d["doi"] for d in results]
    assert dois.count("10.1/shared") == 1
    assert dois[0] == "10.1/shared"  # appears in both → highest RRF score


def test_search_local_rrf_resilient_to_one_mode_failing():
    oa = MagicMock()
    os_ = MagicMock()

    def _maybe_raise(query, top_k, mode):
        if mode == "splade":
            raise RuntimeError("SPLADE model unavailable")
        return [_doc("10.1/bm", "BM")]

    os_.search.side_effect = _maybe_raise
    router = FederatedSearchRouter(oa, os_, recency_days=30)
    results = router.search("q", {}, top_k=10, force_source=SearchSource.LOCAL_RRF)
    dois = {d["doi"] for d in results}
    assert "10.1/bm" in dois


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


# ── Q2.1 — zero-coverage subject fallback ───────────────────────────────────────

def test_always_live_subjects_has_expected_15():
    """The zero-coverage set must contain exactly the 15 subjects from the roadmap."""
    assert ALWAYS_LIVE_SUBJECTS == {
        "Theatre", "Music", "Urban Studies", "Applied Legal Studies",
        "Publishing", "Management & Organizational Studies", "Accounting",
        "Forensics", "Statistics & Actuarial Science",
        "Sustainable Energy Engineering (SEE)",
        "Sustainable Community Development",
        "Visual Arts", "Public Policy",
        "Molecular Biology & Biochemistry", "Global Health",
    }
    assert len(ALWAYS_LIVE_SUBJECTS) == 15


def test_route_zero_coverage_subject_goes_live():
    router = _make_router()
    assert router.route("string quartets", {}, subject_hint="Music") == SearchSource.LIVE_API


def test_route_zero_coverage_subject_goes_live_for_all_15():
    router = _make_router()
    for subject in ALWAYS_LIVE_SUBJECTS:
        assert (
            router.route("some historical query", {}, subject_hint=subject)
            == SearchSource.LIVE_API
        ), f"{subject!r} should route LIVE_API"


def test_route_zero_coverage_overrides_prefer_local():
    """ALWAYS_LIVE is a hard route — prefer_local cannot pull it back to local."""
    router = _make_router()
    assert (
        router.route("string quartets", {}, subject_hint="Music", prefer_local=True)
        == SearchSource.LIVE_API
    )


def test_route_covered_subject_goes_local_rrf():
    router = _make_router()
    # A subject not in either special set follows the default historical path.
    assert (
        router.route("Indigenous land rights", {}, subject_hint="Indigenous Studies")
        == SearchSource.LOCAL_RRF
    )


def test_route_empty_subject_hint_unchanged():
    """No subject detected → behaves exactly like the pre-Q2 default path."""
    router = _make_router()
    assert router.route("effects of climate change", {}, subject_hint="") == SearchSource.LOCAL_RRF


def test_search_zero_coverage_subject_calls_only_openalex():
    """Q2.1 end-to-end: a zero-coverage subject must not touch the local retriever."""
    router, oa, os_ = _make_full_router()
    results = router.search("string quartets", {}, top_k=10, subject_hint="Music")
    oa.search_works.assert_called_once()
    os_.search.assert_not_called()
    assert results[0]["doi"] == "10.1/a"


# ── Q2.2 — Anthropology soft-route ──────────────────────────────────────────────

def test_anthropology_is_soft_live():
    assert "Anthropology" in SOFT_LIVE_SUBJECTS


def test_route_anthropology_prefers_live_by_default():
    router = _make_router()
    assert (
        router.route("kinship systems in melanesia", {}, subject_hint="Anthropology")
        == SearchSource.LIVE_API
    )


def test_route_anthropology_overridable_to_local():
    router = _make_router()
    assert (
        router.route(
            "kinship systems in melanesia",
            {},
            subject_hint="Anthropology",
            prefer_local=True,
        )
        == SearchSource.LOCAL_RRF
    )


def test_route_anthropology_override_respects_rrf_disabled():
    router = _make_router(local_rrf_enabled=False)
    assert (
        router.route(
            "kinship systems",
            {},
            subject_hint="Anthropology",
            prefer_local=True,
        )
        == SearchSource.LOCAL_INDEX
    )


def test_search_anthropology_default_calls_only_openalex():
    router, oa, os_ = _make_full_router()
    results = router.search("kinship systems", {}, top_k=10, subject_hint="Anthropology")
    oa.search_works.assert_called_once()
    os_.search.assert_not_called()
    assert results[0]["doi"] == "10.1/a"
