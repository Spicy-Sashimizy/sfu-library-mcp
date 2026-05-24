"""Tests for low-risk tool fixes: CSV escaping (#9), limit clamping (#8),
and Europe PMC single-parse (#11)."""

from unittest.mock import MagicMock, patch

import pytest

from lib.tools import _handle_export_search, _fetch_europe_pmc


def _patch_oa(works):
    data = {"results": works, "meta": {"count": len(works)}}
    m = MagicMock()
    m.return_value.search_works.return_value = data
    return m


@pytest.mark.asyncio
class TestCsvExport:
    async def test_embedded_newline_stays_one_row(self):
        """Item #9: a title with an embedded newline must be quoted so the CSV
        body has exactly one data row (header + 1)."""
        works = [{
            "doi": "10.1/x",
            "title": "A title\nwith a newline",
            "authors": ["Smith, J"],
            "date": "2024",
            "source": "Nature",
            "type": "article",
            "is_oa": True,
            "cited_by_count": 3,
        }]
        with patch("lib.tools._get_openalex", _patch_oa(works)):
            with patch("lib.tools._cache_works"):
                result = await _handle_export_search({"query": "q", "format": "csv"})
        text = result[0].text
        # The newline inside the title must be wrapped in quotes.
        assert '"A title\nwith a newline"' in text

        # Validate with the stdlib csv parser: body must be a single record.
        import csv, io
        body = text.split("\n", 1)[1]  # drop the "# CSV Export: ..." comment line
        rows = list(csv.reader(io.StringIO(body)))
        # row 0 = header, row 1 = the single data record (newline kept inside field)
        assert len(rows) == 2
        assert rows[1][1] == "A title\nwith a newline"

    async def test_carriage_return_quoted(self):
        works = [{"doi": "", "title": "x\ry", "authors": [], "date": "",
                  "source": "", "type": "", "is_oa": False, "cited_by_count": 0}]
        with patch("lib.tools._get_openalex", _patch_oa(works)):
            with patch("lib.tools._cache_works"):
                result = await _handle_export_search({"query": "q", "format": "csv"})
        assert '"x\ry"' in result[0].text


@pytest.mark.asyncio
class TestExportLimitClamp:
    async def test_limit_zero_clamps_to_one(self):
        """Item #8: limit=0 must not reach the client as 0."""
        oa = _patch_oa([{"title": "t", "authors": [], "date": "2024"}])
        with patch("lib.tools._get_openalex", oa):
            with patch("lib.tools._cache_works"):
                await _handle_export_search({"query": "q", "format": "json", "limit": 0})
        per_page = oa.return_value.search_works.call_args.kwargs["per_page"]
        assert per_page >= 1

    async def test_negative_limit_clamps_to_one(self):
        oa = _patch_oa([{"title": "t", "authors": [], "date": "2024"}])
        with patch("lib.tools._get_openalex", oa):
            with patch("lib.tools._cache_works"):
                await _handle_export_search({"query": "q", "format": "json", "limit": -5})
        per_page = oa.return_value.search_works.call_args.kwargs["per_page"]
        assert per_page >= 1


class TestEuropePmcSingleParse:
    def test_resp_json_called_once(self):
        """Item #11: the response body is parsed exactly once."""
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "hitCount": 2,
            "resultList": {"result": [
                {"title": "T1", "pubYear": "2020"},
                {"title": "T2", "pubYear": "2021"},
            ]},
        }
        with patch("lib.tools.requests.get", return_value=fake_resp):
            out = _fetch_europe_pmc("cancer", 10)
        assert fake_resp.json.call_count == 1
        assert "2 results" in out
