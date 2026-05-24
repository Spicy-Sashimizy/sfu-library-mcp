"""Unit tests for opensearch_sync delta logic and model-skew guard.

Item 3: compute_delta must diff by stable part identity (the updated_date
partition + filename), NOT the raw URL — so OpenAlex's monthly full-snapshot
republish (new URLs, unchanged data) does not look like an all-new delta and
re-index the entire corpus.

Item 6: the sync's SPLADE indexing model must match splade_indexer.DEFAULT_MODEL
(the model the index was built with) to avoid train/serve skew.

All pure-function tests — no network, no model loading, no OpenSearch.
"""

import sys
from pathlib import Path

import pytest

# opensearch_sync top-level imports are stdlib + requests only; the heavy
# splade_indexer import is lazy inside run_sync, so this is safe.
sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root (for `scripts` pkg)

import opensearch_sync as oss


def _entry(url):
    return {"url": url, "meta": {"record_count": 100}}


BASE = "https://openalex.s3.amazonaws.com/data/works"


class TestPartIdentity:
    def test_extracts_updated_date_partition(self):
        url = f"{BASE}/updated_date=2024-01-15/part_000.gz"
        assert oss.part_identity(url) == "updated_date=2024-01-15/part_000.gz"

    def test_strips_query_string(self):
        url = f"{BASE}/updated_date=2024-01-15/part_000.gz?X-Amz-Signature=deadbeef"
        assert oss.part_identity(url) == "updated_date=2024-01-15/part_000.gz"

    def test_ignores_host_changes(self):
        a = f"https://openalex.s3.amazonaws.com/data/works/updated_date=2024-01-15/part_000.gz"
        b = f"https://openalex-mirror.example.com/x/y/updated_date=2024-01-15/part_000.gz"
        assert oss.part_identity(a) == oss.part_identity(b)

    def test_filename_fallback_when_no_partition(self):
        url = "https://openalex.s3.amazonaws.com/data/works/part_007.gz"
        assert oss.part_identity(url) == "part_007.gz"

    def test_empty_url(self):
        assert oss.part_identity("") == ""


class TestComputeDelta:
    def test_no_last_sync_returns_all(self):
        entries = [_entry(f"{BASE}/updated_date=2024-01-15/part_000.gz")]
        assert oss.compute_delta(entries, None) == entries

    def test_same_content_new_url_returns_empty(self):
        """Item 3 core fix: monthly republish with NEW urls but SAME partition
        identity must produce an EMPTY delta — not the whole corpus."""
        prev_urls = [
            f"{BASE}/updated_date=2024-01-15/part_000.gz",
            f"{BASE}/updated_date=2024-01-15/part_001.gz",
        ]
        last_sync = {"synced_urls": prev_urls}

        # OpenAlex republishes the full snapshot: brand-new URLs (signed,
        # different host) but the same updated_date partitions + filenames.
        current = [
            _entry(f"https://new-bucket.s3.amazonaws.com/feb/updated_date=2024-01-15/part_000.gz?sig=1"),
            _entry(f"https://new-bucket.s3.amazonaws.com/feb/updated_date=2024-01-15/part_001.gz?sig=2"),
        ]
        delta = oss.compute_delta(current, last_sync)
        assert delta == [], "republished-but-unchanged parts must not be re-indexed"

    def test_genuinely_new_partition_in_delta(self):
        last_sync = {"synced_urls": [f"{BASE}/updated_date=2024-01-15/part_000.gz"]}
        current = [
            _entry(f"{BASE}/updated_date=2024-01-15/part_000.gz"),     # already synced
            _entry(f"{BASE}/updated_date=2024-02-19/part_000.gz"),     # NEW month
        ]
        delta = oss.compute_delta(current, last_sync)
        assert len(delta) == 1
        assert oss.part_identity(delta[0]["url"]) == "updated_date=2024-02-19/part_000.gz"

    def test_new_part_in_existing_partition_in_delta(self):
        last_sync = {"synced_urls": [f"{BASE}/updated_date=2024-01-15/part_000.gz"]}
        current = [
            _entry(f"{BASE}/updated_date=2024-01-15/part_000.gz"),
            _entry(f"{BASE}/updated_date=2024-01-15/part_001.gz"),  # new part file
        ]
        delta = oss.compute_delta(current, last_sync)
        assert len(delta) == 1
        assert "part_001.gz" in delta[0]["url"]

    def test_backward_compatible_with_synced_ids(self):
        """A last_sync that already stored synced_ids works directly."""
        last_sync = {"synced_ids": ["updated_date=2024-01-15/part_000.gz"]}
        current = [_entry(f"https://x/y/updated_date=2024-01-15/part_000.gz?sig=z")]
        assert oss.compute_delta(current, last_sync) == []


class TestModelSkewGuard:
    def test_sync_default_matches_indexer_default(self):
        """Item 6: the sync's indexing model must equal the model the index
        was built with (splade_indexer.DEFAULT_MODEL)."""
        from scripts.splade_indexer import DEFAULT_MODEL as INDEXER_MODEL

        assert oss.DEFAULT_MODEL == INDEXER_MODEL
