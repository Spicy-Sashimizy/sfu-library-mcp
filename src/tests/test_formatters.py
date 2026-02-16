"""Tests for formatters module."""

import pytest

from lib.formatters import format_search_results, format_item_details


class TestFormatSearchResults:
    def test_formats_results(self, mock_search_response):
        result = format_search_results(mock_search_response)
        assert "Found 2 total results" in result
        assert "Machine Learning" in result

    def test_none_results(self):
        result = format_search_results(None)
        assert "No results found" in result

    def test_empty_docs(self):
        result = format_search_results({"docs": [], "info": {"total": 0}})
        assert "Found 0 total results" in result

    def test_single_result(self, sample_pnx_record):
        results = {
            "docs": [sample_pnx_record],
            "info": {"total": 1},
        }
        result = format_search_results(results)
        assert "Found 1 total results" in result
        assert "Machine Learning" in result

    def test_subject_headings_in_output(self, sample_pnx_record):
        results = {
            "docs": [sample_pnx_record],
            "info": {"total": 1},
        }
        result = format_search_results(results)
        assert "Subjects: Machine learning, Probabilities, Artificial intelligence" in result

    def test_search_metadata_footer(self, mock_search_response):
        metadata = {"query": "machine learning", "field": "any", "sort": "rank"}
        result = format_search_results(mock_search_response, metadata=metadata)
        assert "--- Search Metadata ---" in result
        assert "Query: machine learning | Field: any | Sort: rank" in result
        assert "Top subjects across results:" in result
        assert "Electronic resources available:" in result

    def test_backward_compatible_no_metadata(self, mock_search_response):
        # Calling without metadata kwarg still works
        result = format_search_results(mock_search_response)
        assert "Found 2 total results" in result
        # Metadata footer still appears but without query line
        assert "--- Search Metadata ---" in result
        assert "Top subjects across results:" in result

    def test_non_ascii_titles(self):
        doc = {
            "pnx": {
                "display": {
                    "title": ["Recherche en fran\u00e7ais"],
                    "creator": ["Auteur, Un"],
                    "creationdate": ["2020"],
                    "type": ["book"],
                },
                "control": {},
                "addata": {},
                "links": {},
                "delivery": {},
            }
        }
        results = {"docs": [doc], "info": {"total": 1}}
        result = format_search_results(results)
        assert "fran" in result

    def test_missing_fields(self):
        doc = {
            "pnx": {
                "display": {"title": ["Minimal"]},
                "control": {},
                "addata": {},
                "links": {},
                "delivery": {},
            }
        }
        results = {"docs": [doc], "info": {"total": 1}}
        result = format_search_results(results)
        assert "Minimal" in result


class TestFormatItemDetails:
    def test_formats_book(self, sample_pnx_record):
        result = format_item_details(sample_pnx_record)
        assert "ITEM DETAILS" in result
        assert "Machine Learning" in result
        assert "Murphy" in result
        assert "9780262018029" in result
        assert "SFUL" in result

    def test_formats_article(self, sample_article_record):
        result = format_item_details(sample_article_record)
        assert "Deep Learning" in result
        assert "Smith" in result
        assert "1076-9757" in result

    def test_none_item(self):
        result = format_item_details(None)
        assert "Could not retrieve" in result

    def test_empty_item(self):
        # Empty dict is falsy, so format_item_details returns error message
        result = format_item_details({})
        assert "Could not retrieve" in result
