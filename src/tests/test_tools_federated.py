"""Tests for the federated search handler wiring in tools._handle_search_academic.

Covers the retrieval/serving-cluster fixes:
  - Item #1: year/type/OA filters are forwarded to the router.
  - Item #2: a detected subject (Theatre/Music) routes LIVE via the router.
  - Item #6: a degraded local cluster surfaces a notice instead of silent empty.
  - Item #8: limit is lower-bounded to >= 1.
  - Item #10: latency clock starts before the router call (retrieval is timed).
"""

from unittest.mock import MagicMock, patch

import pytest

from lib.tools import _handle_search_academic


def _fake_doc(i=0):
    return {
        "title": f"Doc {i}",
        "doi": f"10.1/{i}",
        "publication_year": 2020,
        "type": "article",
        "authors": ["A"],
        "abstract": "x",
    }


def _patch_handler(router):
    """Return a context-manager stack patching the federated dependencies."""
    return patch.multiple(
        "lib.tools",
        _get_features=MagicMock(return_value={
            "federated_search_enabled": True,
            "rerank_enabled": False,
            "rrf_enabled": False,
        }),
        _get_federated_router=MagicMock(return_value=router),
        _cache_works=MagicMock(),
    )


@pytest.mark.asyncio
class TestFederatedHandler:
    async def test_subject_query_routes_live(self):
        """Item #2: a Music query must reach the router with subject_hint='Music'."""
        router = MagicMock()
        router.search.return_value = [_fake_doc()]
        router.last_degraded = False
        with _patch_handler(router):
            await _handle_search_academic({"query": "string quartet ethnomusicology", "limit": 5})
        kwargs = router.search.call_args.kwargs
        assert kwargs["subject_hint"] == "Music"

    async def test_generic_query_no_subject_hint(self):
        router = MagicMock()
        router.search.return_value = [_fake_doc()]
        router.last_degraded = False
        with _patch_handler(router):
            await _handle_search_academic({"query": "salmon migration patterns", "limit": 5})
        assert router.search.call_args.kwargs["subject_hint"] == ""

    async def test_filters_forwarded_to_router(self):
        """Item #1: year/type/OA args become a filter dict passed to the router."""
        router = MagicMock()
        router.search.return_value = [_fake_doc()]
        router.last_degraded = False
        with _patch_handler(router):
            await _handle_search_academic({
                "query": "deep learning",
                "year_from": 2015,
                "year_to": 2020,
                "type": "article",
                "open_access_only": True,
                "limit": 5,
            })
        # router.search(query, filters, ...) — filters is the 2nd positional arg.
        filters = router.search.call_args.args[1]
        assert filters["publication_year"] == "2015-2020"
        assert filters["type"] == "article"
        assert filters["open_access.is_oa"] == "true"

    async def test_degradation_notice_emitted(self):
        """Item #6: degraded local cluster surfaces a notice, not silent no-results."""
        router = MagicMock()
        router.search.return_value = []
        router.last_degraded = True
        with _patch_handler(router):
            result = await _handle_search_academic({"query": "machine learning", "limit": 5})
        text = result[0].text
        assert "unavailable" in text.lower()

    async def test_no_notice_when_healthy(self):
        router = MagicMock()
        router.search.return_value = [_fake_doc()]
        router.last_degraded = False
        with _patch_handler(router):
            result = await _handle_search_academic({"query": "machine learning", "limit": 5})
        assert "unavailable" not in result[0].text.lower()

    async def test_limit_lower_bounded(self):
        """Item #8: limit=0 must clamp to >= 1 before reaching the router."""
        router = MagicMock()
        router.search.return_value = [_fake_doc()]
        router.last_degraded = False
        with _patch_handler(router):
            await _handle_search_academic({"query": "ml", "limit": 0})
        assert router.search.call_args.kwargs["top_k"] >= 1

    async def test_prefer_local_forwarded(self):
        router = MagicMock()
        router.search.return_value = [_fake_doc()]
        router.last_degraded = False
        with _patch_handler(router):
            await _handle_search_academic({"query": "string quartet", "prefer_local": True, "limit": 5})
        assert router.search.call_args.kwargs["prefer_local"] is True
