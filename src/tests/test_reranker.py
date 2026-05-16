"""Tests for the reranker module."""

from unittest.mock import patch

import pytest

from lib.reranker import _compute_rrf_scores, _normalize_for_rerank, _EMBEDDING_RRF_K, rerank_results


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
        assert _EMBEDDING_RRF_K == 60

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


def _make_openalex_doc(
    title="Test Paper",
    abstract="",
    year=2024,
    doc_type="article",
    doi="https://doi.org/10.1234/test",
    authors=None,
    is_oa=False,
):
    """Build a minimal OpenAlex flat doc for reranker tests."""
    return {
        "title": title,
        "abstract": abstract,
        "publication_year": year,
        "type": doc_type,
        "doi": doi,
        "authorships": [
            {"author": {"display_name": a}} for a in (authors or ["Test Author"])
        ],
        "open_access": {"is_oa": is_oa},
        "primary_location": {},
    }


class TestNormalizeForRerank:
    def test_pnx_shape_extracted_correctly(self):
        doc = _make_doc(title="PNX Title", date="2023", doc_type="article",
                        doi="10.1/pnx", availability="available", creator="Alice")
        norm = _normalize_for_rerank(doc)
        assert norm["title"] == "PNX Title"
        assert norm["year"] == 2023
        assert norm["type"] == "article"
        assert norm["doi"] == "10.1/pnx"
        assert norm["has_fulltext"] is True
        assert "Alice" in norm["authors"]

    def test_openalex_shape_extracted_correctly(self):
        doc = _make_openalex_doc(
            title="OA Title", year=2022, doc_type="article",
            doi="https://doi.org/10.99/oa", is_oa=True,
            authors=["Bob Smith"],
        )
        norm = _normalize_for_rerank(doc)
        assert norm["title"] == "OA Title"
        assert norm["year"] == 2022
        assert norm["type"] == "article"
        assert norm["doi"] == "https://doi.org/10.99/oa"
        assert norm["has_fulltext"] is True
        assert "Bob Smith" in norm["authors"]

    def test_openalex_no_fulltext(self):
        doc = _make_openalex_doc(is_oa=False)
        doc["primary_location"] = {}
        norm = _normalize_for_rerank(doc)
        assert norm["has_fulltext"] is False

    def test_openalex_missing_year_is_none(self):
        doc = _make_openalex_doc()
        doc["publication_year"] = None
        norm = _normalize_for_rerank(doc)
        assert norm["year"] is None

    def test_openalex_empty_authorships(self):
        doc = _make_openalex_doc()
        doc["authorships"] = []
        norm = _normalize_for_rerank(doc)
        assert norm["authors"] == []


def _make_normalized_doc(
    title="Test Paper",
    abstract="",
    date="2024",
    doc_type="article",
    doi="10.1234/test",
    authors=None,
    is_oa=False,
    oa_url="",
) -> dict:
    """Build a normalized OpenAlex doc (output of normalize_work()) for reranker tests.

    This is the shape that tools.py actually passes to _maybe_rerank in production.
    Keys differ from raw OpenAlex: 'date' not 'publication_year', 'authors' flat strings,
    'is_oa'/'oa_url' top-level not nested.
    """
    return {
        "title": title,
        "abstract": abstract,
        "date": date,
        "type": doc_type,
        "doi": doi,
        "authors": authors if authors is not None else ["Test Author"],
        "creators": authors if authors is not None else ["Test Author"],
        "is_oa": is_oa,
        "oa_url": oa_url,
        "publisher": "",
        "source": "",
        "cited_by_count": 0,
        "topics": [],
    }


class TestNormalizeForRerankNormalized:
    """Tests for the normalized OpenAlex shape (the production path from tools.py)."""

    def test_year_extracted_from_date_string(self):
        doc = _make_normalized_doc(date="2022-08-15")
        norm = _normalize_for_rerank(doc)
        assert norm["year"] == 2022

    def test_year_extracted_from_year_only_string(self):
        doc = _make_normalized_doc(date="2019")
        norm = _normalize_for_rerank(doc)
        assert norm["year"] == 2019

    def test_empty_date_gives_none_year(self):
        doc = _make_normalized_doc(date="")
        norm = _normalize_for_rerank(doc)
        assert norm["year"] is None

    def test_authors_flat_list_preserved(self):
        doc = _make_normalized_doc(authors=["Smith, Jane", "Doe, John"])
        norm = _normalize_for_rerank(doc)
        assert norm["authors"] == ["Smith, Jane", "Doe, John"]

    def test_is_oa_true_gives_fulltext(self):
        doc = _make_normalized_doc(is_oa=True, oa_url="")
        norm = _normalize_for_rerank(doc)
        assert norm["has_fulltext"] is True

    def test_oa_url_nonempty_gives_fulltext(self):
        doc = _make_normalized_doc(is_oa=False, oa_url="https://example.org/paper.pdf")
        norm = _normalize_for_rerank(doc)
        assert norm["has_fulltext"] is True

    def test_no_oa_no_url_gives_no_fulltext(self):
        doc = _make_normalized_doc(is_oa=False, oa_url="")
        norm = _normalize_for_rerank(doc)
        assert norm["has_fulltext"] is False

    def test_normalized_doc_does_not_use_raw_oa_keys(self):
        """Confirm normalized shape is routed to the correct branch (no authorships key)."""
        doc = _make_normalized_doc()
        assert "authorships" not in doc
        assert "publication_year" not in doc
        assert "open_access" not in doc
        norm = _normalize_for_rerank(doc)
        assert norm["year"] is not None or doc["date"] == ""


class TestRerankerNormalizedShape:
    """End-to-end reranker tests using normalized docs — the production path."""

    def test_recent_oa_ranks_above_old_non_oa(self):
        recent = _make_normalized_doc(
            title="Deep Learning Survey", date="2025",
            doc_type="article", is_oa=True,
        )
        old = _make_normalized_doc(
            title="History of Computing", date="1990",
            doc_type="book", is_oa=False,
        )
        result = rerank_results([old, recent], "deep learning", limit=2, use_embedding=False)
        assert result[0]["title"] == "Deep Learning Survey"

    def test_recency_signal_works_with_normalized_date(self):
        """Recency scoring must use 'date' key, not 'publication_year'."""
        new_doc = _make_normalized_doc(title="New Paper", date="2025")
        old_doc = _make_normalized_doc(title="Old Paper", date="1980")
        result = rerank_results([old_doc, new_doc], "research", limit=2, use_embedding=False)
        assert result[0]["title"] == "New Paper"

    def test_fulltext_signal_works_with_is_oa(self):
        """Fulltext scoring must use 'is_oa' key, not 'open_access.is_oa'."""
        oa_doc = _make_normalized_doc(title="OA Paper", date="2020", is_oa=True)
        closed_doc = _make_normalized_doc(title="Closed Paper", date="2020", is_oa=False)
        result = rerank_results([closed_doc, oa_doc], "study", limit=2, use_embedding=False)
        assert result[0]["title"] == "OA Paper"

    def test_author_completeness_signal_works_with_flat_authors(self):
        """Completeness scoring must use 'authors' flat list, not 'authorships'."""
        with_authors = _make_normalized_doc(
            title="Authored Paper", date="2023",
            authors=["Smith, Jane"], doi="10.1/a",
        )
        no_authors = _make_normalized_doc(
            title="No Author Paper", date="2023",
            authors=[], doi="10.1/b",
        )
        result = rerank_results([no_authors, with_authors], "paper", limit=2, use_embedding=False)
        assert result[0]["title"] == "Authored Paper"

    def test_title_relevance_works_with_normalized_docs(self):
        relevant = _make_normalized_doc(title="Machine Learning Methods", date="2023")
        irrelevant = _make_normalized_doc(title="Medieval History", date="2023")
        result = rerank_results([irrelevant, relevant], "machine learning", limit=2,
                                use_embedding=False)
        assert result[0]["title"] == "Machine Learning Methods"

    def test_limit_respected(self):
        docs = [_make_normalized_doc(title=f"Paper {i}", date="2023") for i in range(8)]
        assert len(rerank_results(docs, "test", limit=3, use_embedding=False)) == 3

    def test_empty_input_returns_empty(self):
        assert rerank_results([], "test", limit=5, use_embedding=False) == []

    def test_missing_fields_no_crash(self):
        sparse = {"title": None, "date": None, "type": None, "doi": ""}
        result = rerank_results([sparse], "test", limit=1, use_embedding=False)
        assert len(result) == 1


class TestRerankerOpenAlexShape:
    def test_recent_oa_article_ranks_above_old_non_oa(self):
        recent = _make_openalex_doc(
            title="Deep Learning Survey", year=2025,
            doc_type="article", is_oa=True,
        )
        old = _make_openalex_doc(
            title="History of Computing", year=1990,
            doc_type="book", is_oa=False,
        )
        result = rerank_results([old, recent], "deep learning", limit=2,
                                use_embedding=False)
        assert result[0]["title"] == "Deep Learning Survey"

    def test_openalex_docs_ranked_by_title_match(self):
        relevant = _make_openalex_doc(title="Machine Learning Methods", year=2023)
        irrelevant = _make_openalex_doc(title="Medieval History", year=2023)
        result = rerank_results([irrelevant, relevant], "machine learning", limit=2,
                                use_embedding=False)
        assert result[0]["title"] == "Machine Learning Methods"

    def test_openalex_limit_respected(self):
        docs = [_make_openalex_doc(title=f"Paper {i}") for i in range(8)]
        result = rerank_results(docs, "test", limit=3, use_embedding=False)
        assert len(result) == 3

    def test_openalex_empty_input(self):
        assert rerank_results([], "test", limit=5, use_embedding=False) == []

    def test_openalex_missing_fields_no_crash(self):
        sparse = {"title": None, "publication_year": None, "type": None}
        result = rerank_results([sparse], "test", limit=1, use_embedding=False)
        assert len(result) == 1

    def test_pnx_and_openalex_mixed_no_crash(self):
        pnx_doc = _make_doc(title="PNX Paper", date="2023")
        oa_doc = _make_openalex_doc(title="OpenAlex Paper", year=2023)
        result = rerank_results([pnx_doc, oa_doc], "paper", limit=2,
                                use_embedding=False)
        assert len(result) == 2
