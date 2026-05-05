"""Integration tests for _maybe_rerank wiring in tool handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lib.tools import _maybe_rerank


def _make_oa_doc(title="Paper", year=2024):
    return {
        "title": title,
        "publication_year": year,
        "type": "article",
        "doi": f"https://doi.org/10.1/{title[:4]}",
        "authorships": [{"author": {"display_name": "Author"}}],
        "open_access": {"is_oa": False},
        "abstract": "",
    }


class TestMaybeRerank:
    def test_rerank_enabled_calls_rerank_results(self):
        docs = [_make_oa_doc(f"Paper {i}") for i in range(5)]
        with patch("lib.tools._get_features", return_value={"rerank_enabled": True, "rrf_enabled": False}):
            with patch("lib.tools.rerank_results", return_value=docs[:3]) as mock_rerank:
                result = _maybe_rerank(docs, "machine learning", 3)
        mock_rerank.assert_called_once()
        call_kwargs = mock_rerank.call_args
        assert call_kwargs[1]["use_rrf"] is False
        assert len(result) == 3

    def test_rerank_disabled_skips_rerank_results(self):
        docs = [_make_oa_doc(f"Paper {i}") for i in range(5)]
        with patch("lib.tools._get_features", return_value={"rerank_enabled": False, "rrf_enabled": False}):
            with patch("lib.tools.rerank_results") as mock_rerank:
                result = _maybe_rerank(docs, "query", 3)
        mock_rerank.assert_not_called()
        assert result == docs[:3]

    def test_rrf_flag_propagates(self):
        docs = [_make_oa_doc(f"Paper {i}") for i in range(3)]
        with patch("lib.tools._get_features", return_value={"rerank_enabled": True, "rrf_enabled": True}):
            with patch("lib.tools.rerank_results", return_value=docs) as mock_rerank:
                _maybe_rerank(docs, "query", 3)
        assert mock_rerank.call_args[1]["use_rrf"] is True

    def test_embedding_model_path_propagates(self):
        docs = [_make_oa_doc(f"Paper {i}") for i in range(3)]
        from types import SimpleNamespace
        fake_cfg = SimpleNamespace(embedding_model_path="models/sfu-academic-embed-v4-bge")
        with patch("lib.tools._get_features", return_value={"rerank_enabled": True, "rrf_enabled": True}):
            with patch("lib.tools._get_config", return_value=fake_cfg):
                with patch("lib.tools.rerank_results", return_value=docs) as mock_rerank:
                    _maybe_rerank(docs, "query", 3)
        assert mock_rerank.call_args[1]["embedding_model_path"] == "models/sfu-academic-embed-v4-bge"

    def test_empty_embedding_model_path_passes_none(self):
        docs = [_make_oa_doc(f"Paper {i}") for i in range(3)]
        from types import SimpleNamespace
        fake_cfg = SimpleNamespace(embedding_model_path="")
        with patch("lib.tools._get_features", return_value={"rerank_enabled": True, "rrf_enabled": False}):
            with patch("lib.tools._get_config", return_value=fake_cfg):
                with patch("lib.tools.rerank_results", return_value=docs) as mock_rerank:
                    _maybe_rerank(docs, "query", 3)
        assert mock_rerank.call_args[1]["embedding_model_path"] is None

    def test_reranker_exception_falls_back(self):
        docs = [_make_oa_doc(f"Paper {i}") for i in range(5)]
        with patch("lib.tools._get_features", return_value={"rerank_enabled": True, "rrf_enabled": False}):
            with patch("lib.tools.rerank_results", side_effect=RuntimeError("boom")):
                result = _maybe_rerank(docs, "query", 3)
        assert result == docs[:3]

    def test_limit_applied_when_rerank_disabled(self):
        docs = [_make_oa_doc(f"Paper {i}") for i in range(10)]
        with patch("lib.tools._get_features", return_value={"rerank_enabled": False}):
            result = _maybe_rerank(docs, "query", 4)
        assert len(result) == 4


@pytest.mark.asyncio
class TestHandlerRerank:
    async def test_search_academic_invokes_rerank_when_enabled(self):
        from lib.tools import _handle_search_academic

        fake_results = [_make_oa_doc(f"Work {i}") for i in range(5)]
        fake_data = {"results": fake_results, "meta": {"count": 5}}

        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.search_works.return_value = fake_data
            with patch("lib.tools._get_features", return_value={"rerank_enabled": True, "rrf_enabled": False}):
                with patch("lib.tools.rerank_results", return_value=fake_results[:3]) as mock_rerank:
                    with patch("lib.tools._cache_works"):
                        await _handle_search_academic({"query": "deep learning", "limit": 5})
        mock_rerank.assert_called_once()

    async def test_search_academic_skips_rerank_when_disabled(self):
        from lib.tools import _handle_search_academic

        fake_results = [_make_oa_doc(f"Work {i}") for i in range(3)]
        fake_data = {"results": fake_results, "meta": {"count": 3}}

        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.search_works.return_value = fake_data
            with patch("lib.tools._get_features", return_value={"rerank_enabled": False, "rrf_enabled": False}):
                with patch("lib.tools.rerank_results") as mock_rerank:
                    with patch("lib.tools._cache_works"):
                        await _handle_search_academic({"query": "test", "limit": 3})
        mock_rerank.assert_not_called()

    async def test_search_by_topic_invokes_rerank_when_enabled(self):
        from lib.tools import _handle_search_by_topic

        fake_results = [_make_oa_doc(f"Topic {i}") for i in range(4)]
        fake_data = {"results": fake_results, "meta": {"count": 4}}

        with patch("lib.tools._get_openalex") as mock_oa:
            mock_oa.return_value.search_works.return_value = fake_data
            with patch("lib.tools._get_features", return_value={"rerank_enabled": True, "rrf_enabled": True}):
                with patch("lib.tools.rerank_results", return_value=fake_results) as mock_rerank:
                    with patch("lib.tools._cache_works"):
                        await _handle_search_by_topic({"topic": "machine learning", "limit": 4})
        mock_rerank.assert_called_once()
        assert mock_rerank.call_args[1]["use_rrf"] is True
