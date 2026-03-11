"""Tests for the reranker module."""

import pytest

from lib.reranker import rerank_results


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
        """When scores are equal, original Primo order should be preserved."""
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
