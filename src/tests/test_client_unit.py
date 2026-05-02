"""Unit tests for the new API client modules (OpenAlex, SFU databases, access resolver)."""

import pytest

from lib.config import ServerConfig
from lib.openalex import normalize_doi, reconstruct_abstract, normalize_work
from lib.sfu_databases import extract_domain, normalize_provider
from lib.access_resolver import _is_proxy


# ── openalex helpers ──────────────────────────────────────────────────────────

class TestNormalizeDoi:
    def test_bare_doi(self):
        assert normalize_doi("10.1000/xyz123") == "10.1000/xyz123"

    def test_https_prefix(self):
        assert normalize_doi("https://doi.org/10.1000/xyz123") == "10.1000/xyz123"

    def test_http_prefix(self):
        assert normalize_doi("http://doi.org/10.1000/xyz123") == "10.1000/xyz123"

    def test_empty(self):
        assert normalize_doi("") == ""

    def test_none(self):
        assert normalize_doi(None) == ""


class TestReconstructAbstract:
    def test_basic(self):
        inverted = {"hello": [0], "world": [1]}
        assert reconstruct_abstract(inverted) == "hello world"

    def test_out_of_order(self):
        inverted = {"b": [1], "a": [0]}
        assert reconstruct_abstract(inverted) == "a b"

    def test_empty(self):
        assert reconstruct_abstract(None) == ""
        assert reconstruct_abstract({}) == ""


class TestNormalizeWork:
    def test_minimal_work(self):
        work = {
            "id": "https://openalex.org/W123",
            "title": "Test Paper",
            "doi": "https://doi.org/10.1000/test",
            "type": "article",
            "authorships": [{"author": {"display_name": "Smith, John"}}],
            "publication_year": 2024,
            "open_access": {"is_oa": True, "oa_url": "https://example.com/paper.pdf"},
            "biblio": {},
            "primary_location": {"source": {"display_name": "Nature", "issn_l": "0028-0836"}},
        }
        result = normalize_work(work)
        assert result["title"] == "Test Paper"
        assert result["doi"] == "10.1000/test"
        assert result["authors"] == ["Smith, John"]
        assert result["is_oa"] is True
        assert result["issn"] == "0028-0836"
        assert result["resource_type"] == "article"
        assert result["record_id"] == "https://openalex.org/W123"

    def test_missing_fields_are_empty_strings(self):
        result = normalize_work({"id": "W1", "title": "Minimal"})
        assert result["doi"] == ""
        assert result["authors"] == []
        assert result["is_oa"] is False


# ── sfu_databases helpers ─────────────────────────────────────────────────────

class TestExtractDomain:
    def test_full_url(self):
        assert extract_domain("https://www.jstor.org/stable/123") == "jstor.org"

    def test_no_www(self):
        assert extract_domain("https://search.proquest.com/") == "search.proquest.com"

    def test_bare_domain(self):
        assert extract_domain("jstor.org") == "jstor.org"

    def test_empty(self):
        assert extract_domain("") is None

    def test_none_url(self):
        assert extract_domain(None) is None


class TestNormalizeProvider:
    def test_strips_inc(self):
        assert "inc" not in normalize_provider("Springer Inc.")

    def test_lowercases(self):
        assert normalize_provider("EBSCOhost") == "ebscohost"

    def test_empty(self):
        assert normalize_provider("") == ""

    def test_collapses_whitespace(self):
        result = normalize_provider("Taylor   &   Francis")
        assert "  " not in result


# ── access_resolver helpers ───────────────────────────────────────────────────

class TestIsProxy:
    def test_true_bool(self):
        assert _is_proxy({"proxy": True}) is True

    def test_false_bool(self):
        assert _is_proxy({"proxy": False}) is False

    def test_true_string(self):
        assert _is_proxy({"proxy": "true"}) is True

    def test_missing(self):
        assert _is_proxy({}) is False
