"""Tests for DailyCallTracker load/save paths and logging."""

import json
import logging
from pathlib import Path

import pytest

from lib.openalex import DailyCallTracker


def _tracker(tmp_path, limit: int = 900) -> DailyCallTracker:
    return DailyCallTracker(limit=limit, path=str(tmp_path / "tracker.json"))


class TestTrackerLoadMissingFile:
    def test_missing_file_resets_count_to_zero(self, tmp_path):
        t = _tracker(tmp_path)
        assert t._count == 0

    def test_missing_file_logs_warning(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="sfu_library_mcp"):
            _tracker(tmp_path)
        assert any("not found" in r.message.lower() for r in caplog.records)

    def test_missing_file_sets_today_date(self, tmp_path):
        t = _tracker(tmp_path)
        assert t._date == t._today()


class TestTrackerLoadCorruptFile:
    def test_corrupt_json_resets_count(self, tmp_path):
        p = tmp_path / "tracker.json"
        p.write_text("NOT VALID JSON {{{{")
        t = _tracker(tmp_path)
        assert t._count == 0

    def test_corrupt_json_logs_warning(self, tmp_path, caplog):
        p = tmp_path / "tracker.json"
        p.write_text("NOT VALID JSON {{{{")
        with caplog.at_level(logging.WARNING, logger="sfu_library_mcp"):
            _tracker(tmp_path)
        assert any("corrupt" in r.message.lower() for r in caplog.records)

    def test_wrong_type_in_count_resets(self, tmp_path):
        p = tmp_path / "tracker.json"
        p.write_text(json.dumps({"date": "2099-01-01", "count": "not-an-int"}))
        t = _tracker(tmp_path)
        # ValueError from int("not-an-int") → treated as corrupt
        assert t._count == 0


class TestTrackerLoadValidFile:
    def test_todays_count_restored(self, tmp_path):
        p = tmp_path / "tracker.json"
        t = DailyCallTracker(limit=900, path=str(p))
        today = t._today()
        p.write_text(json.dumps({"date": today, "count": 42}))
        # Re-load by constructing a new instance pointing at same file
        t2 = DailyCallTracker(limit=900, path=str(p))
        assert t2._count == 42

    def test_previous_day_file_resets_count(self, tmp_path):
        p = tmp_path / "tracker.json"
        p.write_text(json.dumps({"date": "2000-01-01", "count": 500}))
        t = _tracker(tmp_path)
        assert t._count == 0

    def test_previous_day_does_not_log_warning(self, tmp_path, caplog):
        p = tmp_path / "tracker.json"
        p.write_text(json.dumps({"date": "2000-01-01", "count": 500}))
        with caplog.at_level(logging.WARNING, logger="sfu_library_mcp"):
            _tracker(tmp_path)
        # Previous-day reset is expected — should not emit a warning
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("corrupt" in m.lower() for m in warning_msgs)
        assert not any("not found" in m.lower() for m in warning_msgs)


class TestTrackerSaveFailure:
    def test_save_failure_logs_warning(self, tmp_path, caplog):
        t = _tracker(tmp_path)
        # Make the path a directory so write fails
        p = tmp_path / "tracker.json"
        p.mkdir()
        with caplog.at_level(logging.WARNING, logger="sfu_library_mcp"):
            t._save()
        assert any("persist" in r.message.lower() or "failed" in r.message.lower()
                   for r in caplog.records)


class TestTrackerBudgetLogic:
    def test_not_exhausted_below_limit(self, tmp_path):
        t = _tracker(tmp_path, limit=100)
        t._count = 50
        assert not t.status()["exhausted"]

    def test_exhausted_at_limit(self, tmp_path):
        t = _tracker(tmp_path, limit=100)
        t._count = 100
        assert t.status()["exhausted"]

    def test_increment_persists_and_returns_status(self, tmp_path):
        p = tmp_path / "tracker.json"
        t = DailyCallTracker(limit=900, path=str(p))
        status = t.increment()
        assert status["calls_today"] == 1
        assert p.is_file()
        data = json.loads(p.read_text())
        assert data["count"] == 1


class TestTrackerValidateConfigWarning:
    def test_tmp_path_triggers_config_warning(self):
        from lib.config import ServerConfig, validate_config
        cfg = ServerConfig(openalex_tracker_path="/tmp/openalex_calls.json")
        warnings = validate_config(cfg)
        assert any("OPENALEX_TRACKER_PATH" in w for w in warnings)

    def test_persistent_path_no_config_warning(self):
        from lib.config import ServerConfig, validate_config
        cfg = ServerConfig(openalex_tracker_path="/data/openalex_calls.json")
        warnings = validate_config(cfg)
        assert not any("OPENALEX_TRACKER_PATH" in w for w in warnings)
