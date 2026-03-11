"""Tests for validators module."""

import pytest

from lib.validators import (
    validate_isbn,
    validate_issn,
    sanitize_search_query,
    sanitize_search_query_advanced,
    validate_api_response,
    normalize_encoding,
)


class TestValidateISBN:
    def test_valid_isbn13(self):
        # "The Art of Computer Programming" Vol 1
        valid, cleaned = validate_isbn("978-0-201-89683-1")
        assert valid is True
        assert cleaned == "9780201896831"

    def test_valid_isbn13_no_hyphens(self):
        valid, _ = validate_isbn("9780262018029")
        assert valid is True

    def test_valid_isbn10(self):
        valid, cleaned = validate_isbn("0-201-89683-4")
        assert valid is True
        assert cleaned == "0201896834"

    def test_isbn10_with_x_check_digit(self):
        valid, _ = validate_isbn("080442957X")
        assert valid is True

    def test_invalid_isbn_wrong_check_digit(self):
        valid, _ = validate_isbn("978-0-201-89683-9")
        assert valid is False

    def test_invalid_isbn_too_short(self):
        valid, _ = validate_isbn("12345")
        assert valid is False

    def test_invalid_isbn_too_long(self):
        valid, _ = validate_isbn("12345678901234")
        assert valid is False

    def test_empty_isbn(self):
        valid, cleaned = validate_isbn("")
        assert valid is False
        assert cleaned == ""

    def test_isbn_with_spaces(self):
        valid, cleaned = validate_isbn("978 0 262 01802 9")
        assert valid is True
        assert cleaned == "9780262018029"

    def test_isbn10_all_digits(self):
        valid, _ = validate_isbn("0201896834")
        assert valid is True

    def test_isbn_non_numeric(self):
        valid, _ = validate_isbn("abcdefghij")
        assert valid is False


class TestValidateISSN:
    def test_valid_issn_with_hyphen(self):
        valid, cleaned = validate_issn("0378-5955")
        assert valid is True
        assert cleaned == "03785955"

    def test_valid_issn_no_hyphen(self):
        valid, _ = validate_issn("03785955")
        assert valid is True

    def test_issn_with_x_check(self):
        valid, _ = validate_issn("0317-8471")
        assert valid is True

    def test_invalid_issn_wrong_check(self):
        valid, _ = validate_issn("0378-5959")
        assert valid is False

    def test_invalid_issn_too_short(self):
        valid, _ = validate_issn("1234")
        assert valid is False

    def test_empty_issn(self):
        valid, cleaned = validate_issn("")
        assert valid is False
        assert cleaned == ""


class TestSanitizeSearchQuery:
    def test_normal_query_unchanged(self):
        result = sanitize_search_query("machine learning")
        assert result == "machine learning"

    def test_strips_html_tags(self):
        result = sanitize_search_query("<script>alert('xss')</script>machine learning")
        assert "<script>" not in result
        assert "alert" in result  # text content preserved

    def test_removes_null_bytes(self):
        result = sanitize_search_query("machine\x00learning")
        assert "\x00" not in result
        assert "machine" in result

    def test_truncates_oversized_input(self):
        long_query = "a" * 2000
        result = sanitize_search_query(long_query)
        assert len(result) <= 1000

    def test_empty_query(self):
        result = sanitize_search_query("")
        assert result == ""

    def test_collapses_whitespace(self):
        result = sanitize_search_query("  machine   learning  ")
        assert result == "machine learning"

    def test_preserves_quotes_and_ampersands(self):
        # Quotes and ampersands pass through for Primo API syntax
        result = sanitize_search_query('query & "test"')
        assert '&' in result
        assert '"test"' in result

    def test_preserves_phrase_quotes(self):
        result = sanitize_search_query('"machine learning"')
        assert result == '"machine learning"'

    def test_preserves_boolean_operators(self):
        result = sanitize_search_query('"CRISPR" AND "sickle cell"')
        assert result == '"CRISPR" AND "sickle cell"'

    def test_preserves_wildcards(self):
        result = sanitize_search_query("cultur*")
        assert result == "cultur*"
        result2 = sanitize_search_query("wom?n")
        assert result2 == "wom?n"

    def test_strips_dangerous_chars(self):
        # HTML tags still stripped
        result = sanitize_search_query("<script>alert('xss')</script>test")
        assert "<script>" not in result
        assert "test" in result
        # Null bytes stripped
        result2 = sanitize_search_query("test\x00query")
        assert "\x00" not in result2
        # Primo subfield delimiters stripped
        result3 = sanitize_search_query("Murphy$$QMurphy")
        assert "$$Q" not in result3

    def test_max_boolean_operators(self):
        # Build query with 35 boolean operators (exceeds 30 limit)
        terms = " AND ".join([f"term{i}" for i in range(36)])
        result = sanitize_search_query_advanced(terms)
        # Should have at most 30 AND operators
        assert result.count(" AND ") <= 30

    def test_non_ascii_preserved(self):
        result = sanitize_search_query("recherche en français")
        assert "français" in result


class TestValidateApiResponse:
    def test_valid_response(self):
        valid, issues = validate_api_response({"docs": [], "info": {}}, ["docs", "info"])
        assert valid is True
        assert issues == []

    def test_missing_required_key(self):
        valid, issues = validate_api_response({"docs": []}, ["docs", "info"])
        assert valid is False
        assert len(issues) == 1

    def test_non_dict_response(self):
        valid, issues = validate_api_response("not a dict")  # type: ignore
        assert valid is False

    def test_no_required_keys(self):
        valid, issues = validate_api_response({"anything": True})
        assert valid is True


class TestNormalizeEncoding:
    def test_smart_quotes_replaced(self):
        result = normalize_encoding("\u201cHello\u201d")
        assert result == '"Hello"'

    def test_em_dash_replaced(self):
        result = normalize_encoding("word\u2014word")
        assert result == "word--word"

    def test_empty_string(self):
        result = normalize_encoding("")
        assert result == ""

    def test_normal_ascii_unchanged(self):
        result = normalize_encoding("normal text")
        assert result == "normal text"
