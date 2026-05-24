"""Unit tests for snapshot_downloader resilience fixes.

Item 4: partial-chunk buffer durability — a hard kill between full-chunk
flushes must NOT silently drop buffered records. The writer persists its
partial buffer alongside every checkpoint and recovers it on resume.

Item 5: the prefetch fallback must use the same processing fn + schema as the
prefetch path (_part_fn with legacy_schema), not a hardcoded 3-arg call.

All tests are fully mocked — no network, no model loading, no real OpenSearch.
"""

import gzip
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))  # scripts/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

import snapshot_downloader as sd


def _read_all_chunk_records(output_dir: Path) -> list[dict]:
    """Read every record written across all numbered chunk files."""
    records = []
    for path in sorted(output_dir.glob("works_part_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ── Item 4: ChunkWriter partial-buffer durability ────────────────────────────


class TestChunkWriterBufferDurability:
    def test_persist_and_load_roundtrip(self, tmp_path):
        w = sd.ChunkWriter(tmp_path, chunk_size=1000)
        w.current_chunk = [{"id": "W1"}, {"id": "W2"}, {"id": "W3"}]
        w.persist_buffer()
        assert (tmp_path / sd.PENDING_BUFFER_FILE).exists()

        # New writer (simulating a fresh process after a crash) recovers them.
        w2 = sd.ChunkWriter(tmp_path, chunk_size=1000)
        recovered = w2.load_buffer()
        assert recovered == 3
        assert w2.current_chunk == [{"id": "W1"}, {"id": "W2"}, {"id": "W3"}]

    def test_persist_empty_buffer_removes_file(self, tmp_path):
        w = sd.ChunkWriter(tmp_path, chunk_size=1000)
        w.current_chunk = [{"id": "W1"}]
        w.persist_buffer()
        assert (tmp_path / sd.PENDING_BUFFER_FILE).exists()
        # After a full flush the buffer is empty; persisting must drop the file.
        w.current_chunk = []
        w.persist_buffer()
        assert not (tmp_path / sd.PENDING_BUFFER_FILE).exists()

    def test_load_buffer_absent_file_returns_zero(self, tmp_path):
        w = sd.ChunkWriter(tmp_path, chunk_size=1000)
        assert w.load_buffer() == 0
        assert w.current_chunk == []

    def test_clear_buffer_file(self, tmp_path):
        w = sd.ChunkWriter(tmp_path, chunk_size=1000)
        w.current_chunk = [{"id": "W1"}]
        w.persist_buffer()
        w.clear_buffer_file()
        assert not (tmp_path / sd.PENDING_BUFFER_FILE).exists()

    def test_hard_kill_mid_buffer_then_resume_no_record_loss(self, tmp_path):
        """Simulate a hard crash with records buffered below chunk_size, then a
        resume. Buffered records must survive (recovered + later flushed),
        i.e. record-count continuity with no silent drops."""
        chunk_size = 10

        # ── Phase 1: process some parts; buffer sits below chunk_size ──
        w = sd.ChunkWriter(tmp_path, chunk_size=chunk_size)
        # 7 records buffered — NOT enough to trigger a full-chunk flush.
        part_a = [{"id": f"A{i}"} for i in range(7)]
        flushed = w.add_records(part_a)
        assert flushed == []                      # nothing flushed yet
        assert len(w.current_chunk) == 7

        # Checkpoint sequence (Item 4): persist buffer BEFORE advancing.
        w.persist_buffer()
        # ...then a HARD KILL happens here (no graceful flush). The 7 buffered
        # records are ONLY in the pending buffer file, not in a numbered chunk.
        assert _read_all_chunk_records(tmp_path) == []      # no chunk on disk
        assert (tmp_path / sd.PENDING_BUFFER_FILE).exists()

        # ── Phase 2: resume — fresh writer recovers the buffer ──
        w2 = sd.ChunkWriter(tmp_path, chunk_size=chunk_size)
        recovered = w2.load_buffer()
        assert recovered == 7

        # Continue processing the next part; eventually flush everything.
        part_b = [{"id": f"B{i}"} for i in range(5)]
        w2.add_records(part_b)                    # 7 + 5 = 12 → one chunk of 10
        w2.flush()                                # remaining 2
        w2.clear_buffer_file()

        all_ids = {r["id"] for r in _read_all_chunk_records(tmp_path)}
        expected = {f"A{i}" for i in range(7)} | {f"B{i}" for i in range(5)}
        assert all_ids == expected                # zero records dropped
        assert len(all_ids) == 12


# ── Item 5: prefetch fallback schema parity ──────────────────────────────────


class TestFallbackSchemaParity:
    def test_fallback_source_uses_part_fn_with_legacy_schema(self):
        """The fallback (else branch when a part wasn't prefetched) must route
        through _part_fn(session, part_url, min_year, legacy_schema) — NOT the
        old hardcoded download_and_process_part(session, part_url, min_year),
        which dropped the GPU path and legacy_schema flag."""
        src = inspect.getsource(sd.run_download)
        # The buggy 3-arg hardcoded fallback must be gone.
        assert "download_and_process_part(session, part_url, min_year)" not in src
        # The corrected fallback must pass through _part_fn with legacy_schema.
        assert "_part_fn(" in src
        assert "legacy_schema" in src

    def test_part_fn_selection_respects_use_gpu(self, tmp_path):
        """_part_fn is chosen by use_gpu; both candidates accept legacy_schema,
        so prefetch and fallback share the same schema-aware signature."""
        cpu_sig = inspect.signature(sd.download_and_process_part)
        gpu_sig = inspect.signature(sd.download_and_process_part_gpu)
        assert "legacy_schema" in cpu_sig.parameters
        assert "legacy_schema" in gpu_sig.parameters

    def test_prefetch_path_passes_legacy_schema(self, tmp_path):
        """End-to-end (mocked): run_download must submit _part_fn with the
        legacy_schema flag for prefetched parts."""
        manifest = [{"url": "https://x/data/works/updated_date=2024-01-15/part_000.gz", "meta": {}}]

        captured = {"calls": []}

        def _fake_part_fn(session, url, min_year, legacy_schema):
            captured["calls"].append((url, min_year, legacy_schema))
            return ([{"id": "W1", "title": "t"}], {
                "total": 1, "kept": 1, "no_abstract": 0,
                "too_old": 0, "parse_error": 0, "retracted": 0,
            })

        with patch.object(sd, "fetch_manifest", return_value=manifest), \
             patch.object(sd, "download_and_process_part", side_effect=_fake_part_fn), \
             patch("requests.Session", return_value=MagicMock()):
            result = sd.run_download(
                output_dir=tmp_path,
                min_year=2015,
                chunk_size=1000,
                resume=False,
                dry_run=False,
                workers=1,
                legacy_schema=True,
                use_gpu=False,
            )

        assert result["state"] == "completed"
        assert captured["calls"], "expected _part_fn to be invoked"
        # legacy_schema=True must be threaded through to the processing fn.
        assert all(c[2] is True for c in captured["calls"])
        # Records were written and the pending buffer cleaned up.
        assert {r["id"] for r in _read_all_chunk_records(tmp_path)} == {"W1"}
        assert not (tmp_path / sd.PENDING_BUFFER_FILE).exists()
