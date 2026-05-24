"""Unit tests for OpenSearchRetriever (P.5)."""

from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock

from lib.opensearch_retriever import (
    OpenSearchRetriever,
    _resolve_onnx_path,
    _resolve_tokenizer_dir,
)


FAKE_HIT = {
    "_score": 1.5,
    "_source": {
        "doi": "10.1234/test",
        "title": "Quantum entanglement in cold atoms",
        "abstract": "We study ...",
        "publication_year": 2022,
        "type": "article",
        "is_oa": True,
    },
}

FAKE_SEARCH_RESPONSE = {
    "hits": {
        "total": {"value": 1},
        "hits": [FAKE_HIT],
    }
}


class TestOpenSearchRetrieverBM25F:
    def _retriever(self):
        return OpenSearchRetriever(
            url="http://localhost:9200",
            index="openalex_works",
            splade_enabled=False,
        )

    def test_search_returns_normalized_dicts(self):
        r = self._retriever()
        with patch.object(r, "_http", return_value=FAKE_SEARCH_RESPONSE):
            results = r.search("quantum entanglement", top_k=5)
        assert len(results) == 1
        doc = results[0]
        assert doc["doi"] == "10.1234/test"
        assert doc["title"] == "Quantum entanglement in cold atoms"
        assert doc["year"] == 2022
        assert doc["score"] == 1.5
        assert doc["source"] == "opensearch"
        assert doc["is_oa"] is True

    def test_search_emits_recency_keys(self):
        """Item #3: local docs must carry publication_year/date so the reranker
        and query logger keep the recency signal (they read those keys, not year)."""
        r = self._retriever()
        with patch.object(r, "_http", return_value=FAKE_SEARCH_RESPONSE):
            doc = r.search("quantum entanglement", top_k=5)[0]
        assert doc["publication_year"] == 2022
        assert doc["date"] == "2022"
        # back-compat key retained
        assert doc["year"] == 2022


class TestOpenSearchRetrieverFilters:
    """Item #1: year/type/OA filters must reach the local OpenSearch query."""

    def _retriever(self):
        return OpenSearchRetriever(url="http://localhost:9200", splade_enabled=False)

    def test_no_filters_no_filter_clause(self):
        r = self._retriever()
        body = r._build_bm25f_query("ml", 10, filters=None)
        assert "filter" not in body["query"]["bool"]

    def test_year_range_emits_range_filter(self):
        r = self._retriever()
        body = r._build_bm25f_query("ml", 10, filters={"publication_year": "2010-2020"})
        flt = body["query"]["bool"]["filter"]
        rng = next(c["range"]["publication_year"] for c in flt if "range" in c)
        assert rng == {"gte": 2010, "lte": 2020}

    def test_from_date_emits_gte(self):
        r = self._retriever()
        body = r._build_bm25f_query("ml", 10, filters={"from_publication_date": "2015-01-01"})
        rng = body["query"]["bool"]["filter"][0]["range"]["publication_year"]
        assert rng == {"gte": 2015}

    def test_type_emits_term_filter(self):
        r = self._retriever()
        body = r._build_bm25f_query("ml", 10, filters={"type": "article"})
        assert {"term": {"type": "article"}} in body["query"]["bool"]["filter"]

    def test_oa_openalex_key_emits_term_filter(self):
        r = self._retriever()
        body = r._build_bm25f_query("ml", 10, filters={"open_access.is_oa": "true"})
        assert {"term": {"is_oa": True}} in body["query"]["bool"]["filter"]

    def test_oa_plain_key_emits_term_filter(self):
        r = self._retriever()
        body = r._build_bm25f_query("ml", 10, filters={"is_oa": True})
        assert {"term": {"is_oa": True}} in body["query"]["bool"]["filter"]

    def test_splade_query_carries_filter(self):
        r = OpenSearchRetriever(url="http://localhost:9200", splade_enabled=True)
        with patch("lib.opensearch_retriever.encode_splade", return_value={"x": 1.0}):
            body = r._build_splade_query("ml", 10, filters={"type": "book"})
        assert {"term": {"type": "book"}} in body["query"]["bool"]["filter"]
        assert body["query"]["bool"]["minimum_should_match"] == 1

    def test_empty_response_returns_empty_list(self):
        r = self._retriever()
        with patch.object(r, "_http", return_value={"hits": {"hits": []}}):
            assert r.search("anything") == []

    def test_http_failure_returns_empty_list(self):
        r = self._retriever()
        with patch.object(r, "_http", return_value=None):
            assert r.search("anything") == []

    def test_bm25f_query_structure(self):
        r = self._retriever()
        body = r._build_bm25f_query("machine learning", 10)
        assert body["size"] == 10
        # multi_match is now wrapped in bool.must so filters can be attached.
        mm = body["query"]["bool"]["must"][0]["multi_match"]
        assert any("title" in f for f in mm["fields"])
        assert mm["type"] == "most_fields"
        assert mm["tie_breaker"] == 0.5

    def test_is_available_green(self):
        r = self._retriever()
        with patch.object(r, "_http", return_value={"status": "green"}):
            assert r.is_available() is True

    def test_is_available_yellow(self):
        r = self._retriever()
        with patch.object(r, "_http", return_value={"status": "yellow"}):
            assert r.is_available() is True

    def test_is_available_red(self):
        r = self._retriever()
        with patch.object(r, "_http", return_value={"status": "red"}):
            assert r.is_available() is False

    def test_is_available_unreachable(self):
        r = self._retriever()
        with patch.object(r, "_http", return_value=None):
            assert r.is_available() is False


class TestOpenSearchRetrieverSPLADE:
    def _retriever(self):
        return OpenSearchRetriever(
            url="http://localhost:9200",
            index="openalex_works",
            splade_enabled=True,
            splade_model_path="naver/splade-cocondenser-distil",
        )

    def test_splade_query_uses_bool_should(self):
        r = self._retriever()
        fake_sparse = {"quantum": 1.2, "entanglement": 0.8}
        with patch("lib.opensearch_retriever.encode_splade", return_value=fake_sparse):
            body = r._build_splade_query("quantum entanglement", 5)
        assert body["size"] == 5
        assert "bool" in body["query"]
        clauses = body["query"]["bool"]["should"]
        assert len(clauses) == 2
        clause = clauses[0]
        assert "rank_feature" in clause
        assert clause["rank_feature"]["log"]["scaling_factor"] == 4

    def test_splade_query_respects_max_terms(self):
        r = self._retriever()
        fake_sparse = {f"tok{i}": float(100 - i) for i in range(100)}
        with patch("lib.opensearch_retriever.encode_splade", return_value=fake_sparse):
            body = r._build_splade_query("query", 10)
        assert len(body["query"]["bool"]["should"]) == 64

    def test_splade_empty_vector_falls_back_to_bm25f(self):
        r = self._retriever()
        with patch("lib.opensearch_retriever.encode_splade", return_value={}):
            body = r._build_splade_query("anything", 5)
        # Fallback is the BM25F query (multi_match wrapped in bool.must).
        assert "multi_match" in body["query"]["bool"]["must"][0]

    def test_splade_search_returns_normalized_dicts(self):
        r = self._retriever()
        fake_sparse = {"quantum": 1.2}
        with patch("lib.opensearch_retriever.encode_splade", return_value=fake_sparse):
            with patch.object(r, "_http", return_value=FAKE_SEARCH_RESPONSE):
                results = r.search("quantum", top_k=5)
        assert len(results) == 1
        assert results[0]["source"] == "opensearch"


class TestSpladeOnnxLoading:
    """Item #5: SPLADE serving model must load from the local ONNX export, and
    raise a clear error for a bare HF hub id (which can't be fetched offline)."""

    def test_bare_hub_id_raises_clear_error(self):
        with pytest.raises(FileNotFoundError) as exc:
            _resolve_onnx_path("naver/splade-cocondenser-distil")
        assert "ONNX" in str(exc.value)

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            _resolve_onnx_path("/nonexistent/path/to/model")

    def test_resolves_onnx_dir_to_model_file(self, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"\x00")
        resolved = _resolve_onnx_path(str(tmp_path))
        assert resolved.name == "model.onnx"

    def test_resolves_direct_onnx_file(self, tmp_path):
        f = tmp_path / "model.onnx"
        f.write_bytes(b"\x00")
        assert _resolve_onnx_path(str(f)) == f

    def test_onnx_dir_without_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            _resolve_onnx_path(str(tmp_path))
        assert "model.onnx" in str(exc.value)

    def test_default_points_at_local_onnx_export(self):
        """The retriever no longer defaults to a bare hub id."""
        r = OpenSearchRetriever(url="http://localhost:9200", splade_enabled=True)
        assert r.splade_model_path.endswith("models/splade_onnx")
        assert "naver" not in r.splade_model_path

    def test_tokenizer_resolves_from_model_dir(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("{}")
        assert _resolve_tokenizer_dir(str(tmp_path)) == tmp_path

    def test_real_onnx_export_dir_resolves(self):
        """The shipped models/splade_onnx export is discoverable + has a tokenizer."""
        repo_root = Path(__file__).resolve().parents[2]
        onnx_dir = repo_root / "models" / "splade_onnx"
        if not (onnx_dir / "model.onnx").is_file():
            pytest.skip("local SPLADE ONNX export not present")
        assert _resolve_onnx_path(str(onnx_dir)).name == "model.onnx"
        # tokenizer falls back to an SFU embed model dir (bert-base-uncased vocab)
        tok = _resolve_tokenizer_dir(str(onnx_dir))
        assert (tok / "tokenizer.json").is_file() or (tok / "tokenizer_config.json").is_file()
