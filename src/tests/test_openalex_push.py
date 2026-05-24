"""Tests for OpenAlexClient._push_to_opensearch (Item 2).

Covers:
- SPLADE-served index → live push is skipped (no _bulk HTTP call), because
  live docs carry no `sparse_field` and would be invisible to the SPLADE leg.
- Non-SPLADE index → push happens and the bulk body carries an integer/None
  `publication_year` (never a raw string) plus a `sparse_field`-free body
  that is valid for a non-sparse index.
- _coerce_year: integer/None coercion for empty / malformed / valid dates.

All HTTP is mocked; no model loading, no network.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from lib.openalex import OpenAlexClient, _coerce_year


def _make_client(**kwargs) -> OpenAlexClient:
    defaults = dict(
        tracker_path="/tmp/test_openalex_push_tracker.json",
        opensearch_enabled=True,
        opensearch_url="http://os:9200",
        opensearch_index="openalex_works",
    )
    defaults.update(kwargs)
    return OpenAlexClient(**defaults)


def _sample_results():
    return [
        {
            "doi": "10.1/abc",
            "openalex_id": "https://openalex.org/W1",
            "title": "T",
            "abstract": "A",
            "topics": ["ml", "ir"],
            "date": "2021-05-03",
            "type": "article",
            "is_oa": True,
        }
    ]


def _join_push_threads():
    """Wait for the fire-and-forget daemon push thread(s) to finish."""
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(timeout=2.0)


class TestCoerceYear:
    def test_valid_date_string(self):
        assert _coerce_year("2021-05-03") == 2021

    def test_year_only_string(self):
        assert _coerce_year("1999") == 1999

    def test_empty_string_returns_none(self):
        assert _coerce_year("") is None

    def test_none_returns_none(self):
        assert _coerce_year(None) is None

    def test_malformed_returns_none(self):
        assert _coerce_year("n/a") is None
        assert _coerce_year("20XX-01") is None

    def test_int_passthrough(self):
        assert _coerce_year(2015) == 2015


class TestPushSkippedForSpladeIndex:
    def test_splade_served_index_skips_push(self):
        client = _make_client(opensearch_splade_served=True)
        with patch("requests.post") as mock_post:
            client._push_to_opensearch(_sample_results())
            _join_push_threads()
        mock_post.assert_not_called()

    def test_default_is_splade_served(self):
        # Default must be the safe behavior: skip (index is openalex_works).
        client = _make_client()
        assert client._opensearch_splade_served is True
        with patch("requests.post") as mock_post:
            client._push_to_opensearch(_sample_results())
            _join_push_threads()
        mock_post.assert_not_called()

    def test_disabled_opensearch_no_push(self):
        client = _make_client(opensearch_enabled=False, opensearch_splade_served=False)
        with patch("requests.post") as mock_post:
            client._push_to_opensearch(_sample_results())
            _join_push_threads()
        mock_post.assert_not_called()


class TestPushToNonSpladeIndex:
    def _capture_bulk_body(self, results):
        client = _make_client(
            opensearch_splade_served=False, opensearch_index="openalex_dense"
        )
        captured = {}

        def _fake_post(url, data=None, headers=None, timeout=None):
            captured["url"] = url
            captured["payload"] = data
            return MagicMock(status_code=200)

        with patch("requests.post", side_effect=_fake_post):
            client._push_to_opensearch(results)
            _join_push_threads()
        return captured

    def _parse_bodies(self, payload):
        """Bulk payload is alternating meta/body NDJSON lines."""
        lines = [ln for ln in payload.split("\n") if ln.strip()]
        bodies = [json.loads(ln) for ln in lines[1::2]]
        return bodies

    def test_push_happens_for_non_splade_index(self):
        captured = self._capture_bulk_body(_sample_results())
        assert "payload" in captured, "expected a bulk POST for non-SPLADE index"
        assert captured["url"].endswith("/_bulk")

    def test_publication_year_is_int(self):
        captured = self._capture_bulk_body(_sample_results())
        bodies = self._parse_bodies(captured["payload"])
        assert bodies[0]["publication_year"] == 2021
        assert isinstance(bodies[0]["publication_year"], int)

    def test_publication_year_none_when_empty(self):
        results = _sample_results()
        results[0]["date"] = ""
        captured = self._capture_bulk_body(results)
        bodies = self._parse_bodies(captured["payload"])
        assert bodies[0]["publication_year"] is None

    def test_publication_year_none_when_missing(self):
        results = _sample_results()
        del results[0]["date"]
        captured = self._capture_bulk_body(results)
        bodies = self._parse_bodies(captured["payload"])
        assert bodies[0]["publication_year"] is None
