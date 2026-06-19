"""Tests for per-person tenancy, rate caps, events, and notify payloads.

    python3 scripts/demo_broker/test_tenants.py
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest

from notify import build_payload
from tenants import TenantStore


class TestInvitesSessions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TenantStore(os.path.join(self.tmp, "t.db"))

    def test_mint_and_open_session(self):
        tok = self.store.mint_invite("Prof X")
        s = self.store.open_session(tok)
        self.assertIsNotNone(s)
        self.assertEqual(s.name, "Prof X")
        self.assertTrue(s.bearer.startswith("demo_"))

    def test_session_is_stable_per_invite(self):
        tok = self.store.mint_invite("Prof X")
        s1 = self.store.open_session(tok)
        s2 = self.store.open_session(tok)  # second click reuses the session
        self.assertEqual(s1.session_id, s2.session_id)
        self.assertEqual(s1.bearer, s2.bearer)

    def test_distinct_invites_are_isolated(self):
        a = self.store.open_session(self.store.mint_invite("A"))
        b = self.store.open_session(self.store.mint_invite("B"))
        self.assertNotEqual(a.session_id, b.session_id)
        self.assertNotEqual(a.bearer, b.bearer)

    def test_bad_or_revoked_token_rejected(self):
        self.assertIsNone(self.store.open_session("inv_nonexistent"))
        tok = self.store.mint_invite("X")
        self.store.revoke_invite(tok)
        self.assertIsNone(self.store.open_session(tok))

    def test_bearer_lookup(self):
        s = self.store.open_session(self.store.mint_invite("X"))
        got = self.store.session_by_bearer(s.bearer)
        self.assertEqual(got.session_id, s.session_id)
        self.assertIsNone(self.store.session_by_bearer("demo_wrong"))

    def test_max_sessions_cap(self):
        tok = self.store.mint_invite("seat", max_sessions=1)
        s1 = self.store.open_session(tok)
        self.assertIsNotNone(s1)
        # same invite reuses, so still fine; a cap matters only if a new session is forced.
        self.assertEqual(self.store.open_session(tok).session_id, s1.session_id)


class TestRateAndEvents(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TenantStore(os.path.join(self.tmp, "t.db"))
        self.s = self.store.open_session(self.store.mint_invite("X"))

    def test_rate_window_cap(self):
        for _ in range(3):
            self.store.record(self.s.session_id, "search", "query")
        ok, _ = self.store.rate_ok(self.s.session_id, window_s=60, max_in_window=3, daily_max=1000)
        self.assertFalse(ok)  # 3 >= 3
        ok2, _ = self.store.rate_ok(self.s.session_id, window_s=60, max_in_window=5, daily_max=1000)
        self.assertTrue(ok2)

    def test_daily_cap(self):
        for _ in range(4):
            self.store.record(self.s.session_id, "search", "query")
        ok, msg = self.store.rate_ok(self.s.session_id, window_s=60, max_in_window=100, daily_max=4)
        self.assertFalse(ok)
        self.assertIn("daily", msg)

    def test_report_counts(self):
        self.store.record(self.s.session_id, "search", "query", status=200, latency_ms=42)
        self.store.bump(self.s.session_id, query=1)
        self.store.record(self.s.session_id, "save_to_zotero", "blocked", status=403)
        self.store.bump(self.s.session_id, blocked=1)
        self.store.record(self.s.session_id, "search", "error", status=502, detail="upstream down")
        self.store.bump(self.s.session_id, error=1)
        rep = self.store.report()
        self.assertEqual(rep["events_by_kind"].get("query"), 1)
        self.assertEqual(rep["events_by_kind"].get("blocked"), 1)
        self.assertEqual(rep["events_by_kind"].get("error"), 1)
        self.assertEqual(rep["sessions"], 1)
        self.assertEqual(rep["per_person"][0]["queries"], 1)
        self.assertEqual(rep["per_person"][0]["blocked"], 1)
        self.assertTrue(rep["recent_errors"])


class TestNotifyPayload(unittest.TestCase):
    def test_ntfy(self):
        body, headers = build_payload("ntfy", "SFU demo started", "Prof X @ 14:32")
        self.assertEqual(body, b"Prof X @ 14:32")
        self.assertEqual(headers["Title"], "SFU demo started")

    def test_slack(self):
        body, headers = build_payload("slack", "t", "m")
        self.assertIn(b"*t*", body)
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_discord(self):
        body, _ = build_payload("discord", "t", "m")
        self.assertIn(b"**t**", body)

    def test_generic_json(self):
        import json
        body, _ = build_payload("json", "t", "m")
        self.assertEqual(json.loads(body), {"title": "t", "message": "m"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
