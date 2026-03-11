"""Tests for citations module."""

import pytest

from lib.citations import (
    extract_metadata,
    format_author_apa,
    format_author_mla,
    format_apa_citation,
    format_mla_citation,
    format_chicago_citation,
    format_bibtex_entry,
    format_ris_entry,
    extract_full_text_links,
)


class TestExtractMetadata:
    def test_extracts_book_metadata(self, sample_pnx_record):
        metadata = extract_metadata(sample_pnx_record)
        assert metadata is not None
        assert metadata["title"] == "Machine Learning: A Probabilistic Perspective"
        assert metadata["resource_type"] == "book"
        assert metadata["isbn"] == "9780262018029"
        assert len(metadata["authors"]) == 1

    def test_extracts_article_metadata(self, sample_article_record):
        metadata = extract_metadata(sample_article_record)
        assert metadata is not None
        assert metadata["resource_type"] == "article"
        assert metadata["doi"] == "10.1613/jair.1.12345"
        assert metadata["volume"] == "76"
        assert metadata["issue"] == "3"
        assert len(metadata["authors"]) == 2

    def test_none_input(self):
        assert extract_metadata(None) is None

    def test_empty_dict(self):
        # extract_metadata returns None for items without a 'pnx' key
        metadata = extract_metadata({})
        assert metadata is None

    def test_missing_authors_uses_contributors(self):
        item = {
            "pnx": {
                "display": {
                    "title": ["Test"],
                    "creator": [],
                    "contributor": ["Editor, Joe"],
                    "creationdate": ["2020"],
                    "publisher": [""],
                    "type": ["other"],
                    "source": [""],
                },
                "addata": {},
                "control": {},
            }
        }
        metadata = extract_metadata(item)
        assert metadata["authors"] == ["Editor, Joe"]


class TestFormatAuthorAPA:
    def test_last_first_format(self):
        result = format_author_apa("Murphy, Kevin P.")
        assert result == "Murphy, K. P."

    def test_first_last_format(self):
        result = format_author_apa("Kevin Murphy")
        assert result == "Murphy, K."

    def test_with_dollar_suffix(self):
        result = format_author_apa("Murphy, Kevin P.$$QMurphy, Kevin P.")
        assert result == "Murphy, K. P."

    def test_empty_string(self):
        assert format_author_apa("") == ""

    def test_single_name(self):
        result = format_author_apa("Madonna")
        assert result == "Madonna"


class TestFormatAuthorMLA:
    def test_preserves_full_name(self):
        result = format_author_mla("Murphy, Kevin P.")
        assert result == "Murphy, Kevin P."

    def test_strips_dollar_suffix(self):
        result = format_author_mla("Smith, John$$QSmith, John")
        assert result == "Smith, John"

    def test_empty_string(self):
        assert format_author_mla("") == ""


class TestFormatAPACitation:
    def test_book_citation(self, sample_pnx_record):
        metadata = extract_metadata(sample_pnx_record)
        citation = format_apa_citation(metadata)
        assert "Murphy" in citation
        assert "(2012)" in citation
        assert "Machine Learning" in citation
        assert "MIT Press" in citation

    def test_article_citation(self, sample_article_record):
        metadata = extract_metadata(sample_article_record)
        citation = format_apa_citation(metadata)
        assert "Smith" in citation
        assert "Doe" in citation
        assert "(2023)" in citation
        assert "doi.org" in citation

    def test_none_metadata(self):
        citation = format_apa_citation(None)
        assert "no metadata" in citation.lower()

    def test_missing_authors(self):
        metadata = {
            "authors": [],
            "date": "2020",
            "title": "Test Title",
            "resource_type": "book",
        }
        citation = format_apa_citation(metadata)
        assert "(2020)" in citation
        assert "Test Title" in citation

    def test_no_date(self):
        metadata = {
            "authors": ["Smith, John"],
            "date": "",
            "title": "Test",
            "resource_type": "book",
        }
        citation = format_apa_citation(metadata)
        # Empty date string produces "()" — the original monolith behavior
        assert "Smith" in citation
        assert "Test" in citation


class TestFormatMLACitation:
    def test_book_citation(self, sample_pnx_record):
        metadata = extract_metadata(sample_pnx_record)
        citation = format_mla_citation(metadata)
        assert "Murphy" in citation
        assert "Machine Learning" in citation
        assert "MIT Press" in citation

    def test_article_citation(self, sample_article_record):
        metadata = extract_metadata(sample_article_record)
        citation = format_mla_citation(metadata)
        assert "Smith" in citation
        assert "vol." in citation
        assert "no." in citation

    def test_none_metadata(self):
        assert "no metadata" in format_mla_citation(None).lower()


class TestFormatChicagoCitation:
    def test_book_citation(self, sample_pnx_record):
        metadata = extract_metadata(sample_pnx_record)
        citation = format_chicago_citation(metadata)
        assert "Murphy" in citation
        assert "Machine Learning" in citation

    def test_article_citation(self, sample_article_record):
        metadata = extract_metadata(sample_article_record)
        citation = format_chicago_citation(metadata)
        assert "Smith" in citation
        assert "(2023)" in citation

    def test_none_metadata(self):
        assert "no metadata" in format_chicago_citation(None).lower()


class TestFormatBibTeX:
    def test_book_entry(self, sample_pnx_record):
        metadata = extract_metadata(sample_pnx_record)
        entry = format_bibtex_entry(metadata)
        assert entry.startswith("@book{")
        assert "author = {" in entry
        assert "title = {" in entry
        assert "publisher = {MIT Press}" in entry

    def test_article_entry(self, sample_article_record):
        metadata = extract_metadata(sample_article_record)
        entry = format_bibtex_entry(metadata)
        assert entry.startswith("@article{")
        assert "journal = {" in entry
        assert "doi = {" in entry

    def test_none_metadata(self):
        entry = format_bibtex_entry(None)
        assert entry.startswith("%")

    def test_no_authors(self):
        metadata = {
            "authors": [],
            "date": "2020",
            "title": "Test",
            "resource_type": "book",
        }
        entry = format_bibtex_entry(metadata)
        assert "author" not in entry.lower() or "unknown" in entry.lower()


class TestFormatRIS:
    def test_book_entry(self, sample_pnx_record):
        metadata = extract_metadata(sample_pnx_record)
        entry = format_ris_entry(metadata)
        assert "TY  - BOOK" in entry
        assert "AU  - " in entry
        assert "TI  - " in entry
        assert "ER  -" in entry

    def test_article_entry(self, sample_article_record):
        metadata = extract_metadata(sample_article_record)
        entry = format_ris_entry(metadata)
        assert "TY  - JOUR" in entry
        assert "DO  - " in entry
        assert "VL  - 76" in entry

    def test_none_metadata(self):
        entry = format_ris_entry(None)
        assert "TY  - GEN" in entry
        assert "ER  -" in entry


class TestExtractFullTextLinks:
    def test_extracts_source_links(self, sample_pnx_record):
        links = extract_full_text_links(sample_pnx_record)
        assert links is not None
        assert len(links["source_links"]) == 1
        assert links["open_access"] is False

    def test_extracts_article_links(self, sample_article_record):
        links = extract_full_text_links(sample_article_record)
        assert links is not None
        assert len(links["html_links"]) == 1
        assert len(links["pdf_links"]) == 1
        assert links["doi_url"] == "https://doi.org/10.1613/jair.1.12345"
        assert links["open_access"] is True

    def test_none_input(self):
        assert extract_full_text_links(None) is None

    def test_empty_item(self):
        # Empty dict has no 'pnx'; function returns None for falsy items
        # but {} is falsy, so it returns None
        links = extract_full_text_links({})
        assert links is None
