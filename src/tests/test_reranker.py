"""Tests for the reranker module."""

from unittest.mock import patch

import pytest

from lib.reranker import _compute_rrf_scores, _RRF_K, rerank_results


def _make_doc(title="Test", creator="Author", date="2024", doc_type="article",
              doi="10.1234/test", availability="available",
              links_html=None, subjects=None):
    """Helper to build a minimal PNX doc for reranker tests."""
    return {
        "pnx": {
            "display": {
                "title": [title],
                "creator": [creator] if creator else [],
                "creationdate": [date] if date else [],
                "type": [doc_type],
                "subject": subjects or [],
            },
            "addata": {
                "doi": [doi] if doi else [],
                "isbn": [],
                "issn": [],
            },
            "control": {
                "recordid": [f"rec_{title[:10].replace(' ', '_')}"],
            },
            "links": {
                "linktohtml": links_html or [],
                "linktorsrc": [],
                "linktopdf": [],
            },
            "delivery": {
                "availability": [availability] if availability else [],
            },
        }
    }


class TestReranker:
    def test_recent_article_ranks_above_old_book(self):
        """A recent article with DOI should rank above an old book without."""
        recent_article = _make_doc(
            title="Machine Learning Advances",
            date="2025", doc_type="article", doi="10.1234/ml",
            availability="available",
        )
        old_book = _make_doc(
            title="Introduction to Computing",
            date="1995", doc_type="book", doi="",
            availability="",
        )
        result = rerank_results([old_book, recent_article], "machine learning", limit=2)
        assert result[0]["pnx"]["display"]["title"][0] == "Machine Learning Advances"

    def test_limit_respected(self):
        docs = [_make_doc(title=f"Doc {i}") for i in range(10)]
        result = rerank_results(docs, "test", limit=3)
        assert len(result) == 3

    def test_empty_input(self):
        result = rerank_results([], "test", limit=10)
        assert result == []

    def test_missing_fields_graceful(self):
        """Docs with missing fields should not crash, just score lower."""
        sparse_doc = _make_doc(
            title="Sparse", creator="", date="", doc_type="",
            doi="", availability="",
        )
        full_doc = _make_doc(
            title="Complete Test Document",
            creator="Author", date="2024", doc_type="article",
            doi="10.1234/test", availability="available",
        )
        result = rerank_results([sparse_doc, full_doc], "test", limit=2)
        # Full doc should rank higher
        assert result[0]["pnx"]["display"]["title"][0] == "Complete Test Document"

    def test_preserves_order_on_tied_scores(self):
        """When scores are equal, original input order should be preserved."""
        doc_a = _make_doc(title="Alpha Document", date="2024", doc_type="article",
                          doi="10.1/a", availability="available")
        doc_b = _make_doc(title="Beta Document", date="2024", doc_type="article",
                          doi="10.1/b", availability="available")
        # Both have identical scoring signals (neither title matches query)
        result = rerank_results([doc_a, doc_b], "unrelated query", limit=2)
        assert result[0]["pnx"]["display"]["title"][0] == "Alpha Document"
        assert result[1]["pnx"]["display"]["title"][0] == "Beta Document"

    def test_title_relevance_boosts_ranking(self):
        """Doc with query terms in title should rank higher."""
        relevant = _make_doc(title="Deep Learning Neural Networks", date="2020")
        irrelevant = _make_doc(title="History of Ancient Rome", date="2020")
        result = rerank_results([irrelevant, relevant], "deep learning", limit=2)
        assert result[0]["pnx"]["display"]["title"][0] == "Deep Learning Neural Networks"


class TestRRFScoring:
    def test_rrf_top_in_both_rankings_scores_one(self):
        """Doc at rank 0 in both lexical (input position) and embedding gets max=1.0."""
        # Position 0 = top lexical. semantic_scores[0]=highest = top embedding too.
        scores = [0.9, 0.5, 0.1]
        rrf = _compute_rrf_scores(scores, n_docs=3)
        assert rrf[0] == pytest.approx(1.0, abs=1e-6)
        # Doc 2 is bottom in both; should be the lowest
        assert rrf[2] < rrf[0]
        assert rrf[2] < rrf[1]

    def test_rrf_rescues_lexical_bottom_with_top_embedding(self):
        """A doc that lexical ranks last but embedding ranks first should still beat
        a middle-of-the-pack doc that neither ranker likes much."""
        # Lexical ranking: doc0 > doc1 > doc2 > doc3 (input order)
        # Embedding ranking: doc3 highest, doc0 second, then doc1, doc2
        semantic = [0.5, 0.3, 0.2, 0.9]
        rrf = _compute_rrf_scores(semantic, n_docs=4)
        # doc3 is lexical-last but embedding-first → should beat doc1, doc2
        assert rrf[3] > rrf[1]
        assert rrf[3] > rrf[2]
        # doc0 (lexical-first, embedding-second) still strongest overall
        assert rrf[0] >= rrf[3]

    def test_rrf_score_in_unit_interval(self):
        """Normalised RRF scores must be in [0, 1]."""
        rrf = _compute_rrf_scores([0.7, 0.3, 0.5, 0.1, 0.9], n_docs=5)
        assert all(0.0 <= s <= 1.0 for s in rrf)

    def test_rrf_constant_is_60(self):
        """k=60 is the literature default; document the choice."""
        assert _RRF_K == 60

    def test_rrf_disabled_uses_raw_cosine(self):
        """When use_rrf=False, semantic scoring uses raw cosine (existing behaviour)."""
        # Two docs identical on all signals except embedding cosine
        doc_a = _make_doc(title="Topic A", date="2024", doc_type="article",
                          doi="10.1/a", availability="available")
        doc_b = _make_doc(title="Topic B", date="2024", doc_type="article",
                          doi="10.1/b", availability="available")
        # doc_a (idx 0) has lower cosine; doc_b (idx 1) has higher cosine
        with patch("lib.reranker._compute_semantic_scores", return_value=[0.1, 0.9]):
            result = rerank_results([doc_a, doc_b], "topic", limit=2,
                                    use_embedding=True, use_rrf=False)
        # Without RRF, doc_b's higher cosine should win
        assert result[0]["pnx"]["display"]["title"][0] == "Topic B"

    def test_rrf_enabled_blends_lexical_and_embedding(self):
        """With RRF on, a doc top in lexical (position 0) AND top in embedding wins."""
        doc_a = _make_doc(title="Topic A", date="2024", doc_type="article",
                          doi="10.1/a", availability="available")
        doc_b = _make_doc(title="Topic B", date="2024", doc_type="article",
                          doi="10.1/b", availability="available")
        # doc_a is lexical-first (position 0). Give it the higher cosine too.
        with patch("lib.reranker._compute_semantic_scores", return_value=[0.9, 0.1]):
            result = rerank_results([doc_a, doc_b], "topic", limit=2,
                                    use_embedding=True, use_rrf=True)
        assert result[0]["pnx"]["display"]["title"][0] == "Topic A"

    def test_rrf_noop_when_no_embedding_available(self):
        """RRF requires embedding scores; should silently fall back when none."""
        doc_a = _make_doc(title="Alpha", date="2024", doc_type="article",
                          doi="10.1/a", availability="available")
        doc_b = _make_doc(title="Beta", date="2024", doc_type="article",
                          doi="10.1/b", availability="available")
        with patch("lib.reranker._compute_semantic_scores", return_value=None):
            result = rerank_results([doc_a, doc_b], "anything", limit=2,
                                    use_embedding=True, use_rrf=True)
        assert len(result) == 2  # No crash, returns ranked docs
