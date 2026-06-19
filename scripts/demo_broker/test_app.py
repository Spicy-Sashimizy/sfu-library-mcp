"""End-to-end broker wiring test (dry-run, no DO spend) via FastAPI TestClient.

Requires fastapi + httpx (present in the project venv):
    .venv/bin/python3 -m pytest scripts/demo_broker/test_app.py
Skips cleanly if fastapi/httpx are absent (e.g. system python).
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest

_HAVE = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx"))


@unittest.skipUnless(_HAVE, "needs fastapi+httpx (project venv)")
class TestBrokerWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DEMO_STATE_PATH"] = os.path.join(tempfile.mkdtemp(), "state.db")
        os.environ["DEMO_LOG_PATH"] = os.path.join(tempfile.mkdtemp(), "b.jsonl")
        os.environ["DEMO_ADMIN_TOKEN"] = "adm"
        from fastapi.testclient import TestClient
        import app  # imported after env is set so CONFIG picks it up
        cls.app = app
        cls.c = TestClient(app.app)

    def test_full_flow(self):
        app, c = self.app, self.c
        tok = app.tenants.mint_invite("Prof Test")

        self.assertEqual(c.get("/start", params={"t": "inv_bad"}).status_code, 403)
        r = c.get("/start", params={"t": tok})
        self.assertEqual(r.status_code, 200)
        self.assertIn("/mcp", r.text)

        bearer = app.tenants.open_session(tok).bearer
        self.assertEqual(c.post("/mcp", content=b"{}").status_code, 401)

        blk = c.post("/mcp", headers={"Authorization": f"Bearer {bearer}"},
                     content=b'{"method":"tools/call","params":{"name":"save_to_zotero"}}')
        self.assertEqual(blk.status_code, 403)

        ok = c.post("/mcp", headers={"Authorization": f"Bearer {bearer}"},
                    content=b'{"method":"tools/call","params":{"name":"search"}}')
        self.assertEqual(ok.status_code, 503)  # gating passed; dry-run compute not serving

        self.assertEqual(c.get("/admin/report").status_code, 403)
        rep = c.get("/admin/report", headers={"x-admin-token": "adm"}).json()
        self.assertEqual(rep["sessions"], 1)
        self.assertEqual(rep["events_by_kind"].get("blocked"), 1)
        self.assertEqual(rep["events_by_kind"].get("session_start"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
