"""Tests for durable state: runtime accounting (incl. the midnight-clamp fix) and
the single-flight wake lock. Pure stdlib.

    python3 scripts/demo_broker/test_state.py
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from state import StateStore

TZ = ZoneInfo("America/Vancouver")


class TestRuntimeAccounting(unittest.TestCase):
    def setUp(self):
        self.store = StateStore(os.path.join(tempfile.mkdtemp(), "s.db"))

    def test_midnight_clamp_does_not_overcount(self):
        # Droplet booted 30h ago and still up; runtime_day is stale (yesterday).
        # Without the clamp, runtime_hours_today would report ~30h and trip the
        # killswitch wrongly. With it, only time since today's midnight counts.
        self.store.set("status", "up")
        self.store.set("booted_at", time.time() - 30 * 3600)
        self.store.set("runtime_day", "1999-01-01")  # stale day -> accrued resets to 0
        self.store.set("runtime_seconds", 999999)
        h = self.store.runtime_hours_today(datetime.now(TZ))
        self.assertLess(h, 24.0)   # clamped to "since midnight", not 30h
        self.assertGreaterEqual(h, 0.0)

    def test_fresh_boot_today_counts_live(self):
        self.store.mark_booting("d1")
        self.store.mark_up("d1")
        self.store.set("booted_at", time.time() - 3600)  # 1h ago, today
        self.store.set("runtime_day", datetime.now(TZ).date().isoformat())
        h = self.store.runtime_hours_today(datetime.now(TZ))
        self.assertGreater(h, 0.9)
        self.assertLess(h, 1.2)


class TestWakeLock(unittest.TestCase):
    def setUp(self):
        self.store = StateStore(os.path.join(tempfile.mkdtemp(), "s.db"))

    def test_single_flight(self):
        self.assertTrue(self.store.acquire_wake_lock())
        self.assertFalse(self.store.acquire_wake_lock())  # held
        self.store.release_wake_lock()
        self.assertTrue(self.store.acquire_wake_lock())


if __name__ == "__main__":
    unittest.main(verbosity=2)
