"""Unit tests for OpenSearchRetriever (P.5)."""

import pytest
from unittest.mock import patch, MagicMock

from lib.opensearch_retriever import OpenSearchRetriever


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
        mm = body["query"]["multi_match"]
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
        assert "multi_match" in body["query"]

    def test_splade_search_returns_normalized_dicts(self):
        r = self._retriever()
        fake_sparse = {"quantum": 1.2}
        with patch("lib.opensearch_retriever.encode_splade", return_value=fake_sparse):
            with patch.object(r, "_http", return_value=FAKE_SEARCH_RESPONSE):
                results = r.search("quantum", top_k=5)
        assert len(results) == 1
        assert results[0]["source"] == "opensearch"
