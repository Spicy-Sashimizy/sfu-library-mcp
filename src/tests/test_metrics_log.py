"""Tests for persistent JSONL metrics logging in tools.py."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lib.tools import _record_metric, _persist_metric, get_metrics


class TestInMemoryMetrics:
    def test_count_increments(self):
        before = get_metrics().get("test_tool_count", {}).get("count", 0)
        _record_metric("test_tool_count", 0.1, True)
        after = get_metrics()["test_tool_count"]["count"]
        assert after == before + 1

    def test_error_count_increments_on_failure(self):
        before = get_metrics().get("test_tool_err", {}).get("errors", 0)
        _record_metric("test_tool_err", 0.1, False)
        after = get_metrics()["test_tool_err"]["errors"]
        assert after == before + 1

    def test_success_does_not_increment_errors(self):
        before = get_metrics().get("test_tool_ok", {}).get("errors", 0)
        _record_metric("test_tool_ok", 0.1, True)
        after = get_metrics()["test_tool_ok"].get("errors", 0)
        assert after == before

    def test_latency_accumulated(self):
        before = get_metrics().get("test_tool_lat", {}).get("total_latency", 0.0)
        _record_metric("test_tool_lat", 0.25, True)
        after = get_metrics()["test_tool_lat"]["total_latency"]
        assert after == pytest.approx(before + 0.25, abs=1e-6)


class TestPersistMetric:
    def test_no_op_when_path_not_configured(self, tmp_path):
        """When SFU_METRICS_LOG_PATH is empty, no file is written."""
        from lib.config import ServerConfig
        cfg = ServerConfig(metrics_log_path="")
        with patch("lib.tools._get_config", return_value=cfg):
            _persist_metric("some_tool", 0.05, True)
        assert not any(tmp_path.iterdir())

    def test_writes_jsonl_entry_when_path_configured(self, tmp_path):
        log_file = tmp_path / "metrics.jsonl"
        from lib.config import ServerConfig
        cfg = ServerConfig(metrics_log_path=str(log_file))
        with patch("lib.tools._get_config", return_value=cfg):
            _persist_metric("search_academic", 0.123, True)
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tool"] == "search_academic"
        assert entry["success"] is True
        assert entry["latency_ms"] == pytest.approx(123.0, abs=1.0)

    def test_failure_entry_written_correctly(self, tmp_path):
        log_file = tmp_path / "metrics.jsonl"
        from lib.config import ServerConfig
        cfg = ServerConfig(metrics_log_path=str(log_file))
        with patch("lib.tools._get_config", return_value=cfg):
            _persist_metric("search_academic", 0.5, False)
        entry = json.loads(log_file.read_text().strip())
        assert entry["success"] is False

    def test_multiple_entries_appended(self, tmp_path):
        log_file = tmp_path / "metrics.jsonl"
        from lib.config import ServerConfig
        cfg = ServerConfig(metrics_log_path=str(log_file))
        with patch("lib.tools._get_config", return_value=cfg):
            _persist_metric("tool_a", 0.1, True)
            _persist_metric("tool_b", 0.2, False)
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["tool"] == "tool_a"
        assert json.loads(lines[1])["tool"] == "tool_b"

    def test_parent_dir_created_automatically(self, tmp_path):
        nested = tmp_path / "nested" / "dir" / "metrics.jsonl"
        from lib.config import ServerConfig
        cfg = ServerConfig(metrics_log_path=str(nested))
        with patch("lib.tools._get_config", return_value=cfg):
            _persist_metric("tool_x", 0.01, True)
        assert nested.exists()

    def test_bad_path_does_not_raise(self):
        """A path that cannot be created must not propagate an exception."""
        from lib.config import ServerConfig
        cfg = ServerConfig(metrics_log_path="/proc/cannot-write-here/metrics.jsonl")
        with patch("lib.tools._get_config", return_value=cfg):
            _persist_metric("tool_y", 0.01, True)  # must not raise

    def test_entry_contains_required_fields(self, tmp_path):
        log_file = tmp_path / "metrics.jsonl"
        from lib.config import ServerConfig
        cfg = ServerConfig(metrics_log_path=str(log_file))
        with patch("lib.tools._get_config", return_value=cfg):
            _persist_metric("generate_citation", 0.05, True)
        entry = json.loads(log_file.read_text().strip())
        for field in ("ts", "tool", "latency_ms", "success", "server_version"):
            assert field in entry, f"Missing field: {field}"
        assert entry["ts"].endswith("Z")

    def test_record_metric_calls_persist(self, tmp_path):
        """_record_metric must also trigger _persist_metric."""
        log_file = tmp_path / "metrics.jsonl"
        from lib.config import ServerConfig
        cfg = ServerConfig(metrics_log_path=str(log_file))
        with patch("lib.tools._get_config", return_value=cfg):
            _record_metric("search_by_doi", 0.08, True)
        assert log_file.exists()
