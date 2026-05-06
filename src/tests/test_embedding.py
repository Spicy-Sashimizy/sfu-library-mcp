"""Tests for the embedding module and embedding-based reranking."""

import pytest
from unittest.mock import patch, MagicMock

import numpy as np


class TestEmbeddingModule:
    """Unit tests for src/lib/embedding.py (mocked, no model download)."""

    def test_compute_similarity_empty_docs(self):
        from lib.embedding import compute_similarity
        result = compute_similarity("test query", [])
        assert result == []

    def test_compute_similarity_returns_scores(self):
        mock_model = MagicMock()
        fake_embeddings = np.array([
            [1.0, 0.0, 0.0],   # query
            [0.9, 0.1, 0.0],   # doc1 - similar
            [0.0, 0.0, 1.0],   # doc2 - dissimilar
        ])
        # Normalize
        norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
        fake_embeddings = fake_embeddings / norms
        mock_model.encode.return_value = fake_embeddings

        with patch("lib.embedding._load_model", return_value=mock_model):
            from lib.embedding import compute_similarity
            scores = compute_similarity("query", ["doc1", "doc2"])

        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_score_papers_semantic_empty(self):
        from lib.embedding import score_papers_semantic
        result = score_papers_semantic("test", [])
        assert result == []

    def test_score_papers_combines_title_abstract(self):
        mock_model = MagicMock()
        fake_embeddings = np.eye(3, dtype=np.float32)
        mock_model.encode.return_value = fake_embeddings

        with patch("lib.embedding._load_model", return_value=mock_model):
            from lib.embedding import score_papers_semantic
            papers = [
                {"title": "Paper A", "abstract": "Abstract A"},
                {"title": "Paper B", "abstract": ""},
            ]
            scores = score_papers_semantic("test query", papers)

        assert len(scores) == 2
        # Verify encode was called with combined title+abstract texts
        call_args = mock_model.encode.call_args
        texts = call_args[0][0]
        assert len(texts) == 3  # query + 2 papers
        assert "Paper A Abstract A" in texts[1]
        assert texts[2] == "Paper B"  # no abstract, title only

    def test_get_model_info_no_model(self):
        with patch("lib.embedding._load_model", return_value=None):
            from lib.embedding import get_model_info
            info = get_model_info()
        assert info["loaded"] is False

    def test_unload_model(self):
        import lib.embedding as emb
        emb._model = MagicMock()
        emb._model_name = "test"
        emb.unload_model()
        assert emb._model is None
        assert emb._model_name is None

    def test_encode_texts_failure_returns_none(self):
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("CUDA OOM")

        with patch("lib.embedding._load_model", return_value=mock_model):
            from lib.embedding import encode_texts
            result = encode_texts(["test"])
        assert result is None


class TestRerankerWithEmbedding:
    """Tests for reranker.py's embedding integration."""

    def _make_doc(self, title="Test", creator="Author", date="2024",
                  doc_type="article", doi="10.1234/test",
                  availability="available", description=""):
        return {
            "pnx": {
                "display": {
                    "title": [title],
                    "creator": [creator] if creator else [],
                    "creationdate": [date] if date else [],
                    "type": [doc_type],
                    "subject": [],
                    "description": [description] if description else [],
                },
                "addata": {
                    "doi": [doi] if doi else [],
                    "isbn": [],
                    "issn": [],
                },
                "control": {"recordid": [f"rec_{title[:10]}"]},
                "links": {"linktohtml": [], "linktorsrc": [], "linktopdf": []},
                "delivery": {"availability": [availability] if availability else []},
            }
        }

    def test_rerank_without_embedding(self):
        """Reranker works when embedding is disabled."""
        from lib.reranker import rerank_results
        docs = [self._make_doc(title=f"Doc {i}") for i in range(5)]
        result = rerank_results(docs, "test", limit=3, use_embedding=False)
        assert len(result) == 3

    def test_rerank_embedding_fallback(self):
        """Reranker falls back gracefully when embedding fails."""
        with patch("lib.reranker._compute_semantic_scores", return_value=None):
            from lib.reranker import rerank_results
            docs = [self._make_doc(title="Test Doc")]
            result = rerank_results(docs, "test", limit=1, use_embedding=True)
            assert len(result) == 1

    def test_rerank_with_embedding_scores(self):
        """Semantic scores influence ranking when available."""
        from lib.reranker import rerank_results

        doc_relevant = self._make_doc(
            title="History of Ancient Rome",  # title does NOT match query
            date="2020",
            description="A comprehensive study of deep learning neural networks",
        )
        doc_title_match = self._make_doc(
            title="Deep Learning Survey",  # title matches
            date="2020",
        )

        # Mock: doc_relevant gets high semantic score, doc_title_match gets low
        fake_scores = [0.95, 0.30]
        with patch("lib.reranker._compute_semantic_scores", return_value=fake_scores):
            result = rerank_results(
                [doc_relevant, doc_title_match],
                "deep learning",
                limit=2,
                use_embedding=True,
            )
        # With 35% semantic weight, high semantic score should boost doc_relevant
        assert result[0]["pnx"]["display"]["title"][0] == "History of Ancient Rome"

    def test_rerank_semantic_scores_length_match(self):
        """Semantic scores list must match docs length."""
        from lib.reranker import rerank_results

        docs = [self._make_doc(title=f"Doc {i}") for i in range(3)]
        fake_scores = [0.5, 0.8, 0.3]
        with patch("lib.reranker._compute_semantic_scores", return_value=fake_scores):
            result = rerank_results(docs, "test", limit=3, use_embedding=True)
        assert len(result) == 3

    def test_extract_doc_text(self):
        """_extract_doc_text takes a normalized dict (output of _normalize_for_rerank)."""
        from lib.reranker import _extract_doc_text, _normalize_for_rerank
        raw = self._make_doc(title="Test Title", description="Test Description")
        norm = _normalize_for_rerank(raw)
        text = _extract_doc_text(norm)
        assert "Test Title" in text
        assert "Test Description" in text

    def test_extract_doc_text_no_description(self):
        from lib.reranker import _extract_doc_text, _normalize_for_rerank
        raw = self._make_doc(title="Only Title")
        norm = _normalize_for_rerank(raw)
        text = _extract_doc_text(norm)
        assert "Only Title" in text
