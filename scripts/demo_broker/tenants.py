"""Per-person, account-free tenancy + durable usage/debug events.

Each invitee gets an unguessable invite token (baked into their email link). A
click mints/reuses a *session* with its own bearer token, so people are isolated
by identity and tracked separately WITHOUT making accounts — and without anyone
needing an OpenAlex key (OpenAlex is keyless; the server uses one shared
polite-pool identity, and per-session rate caps stop any one visitor exhausting it).

Durable: everything is sqlite (survives a NAS reboot), so usage history and the
killswitch-relevant counters can't be lost by bouncing the process. Raw debug rows
live in the `events` table; a JSONL mirror is written by the app's logger.
"""
from __future__ import annotations

import secrets
import sqlite3
import time
from dataclasses import dataclass

_SCHEMA = """
CREATE TABLE IF NOT EXISTS invites (
  token TEXT PRIMARY KEY, name TEXT, created_at REAL, max_sessions INTEGER DEFAULT 0,
  revoked INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, invite_token TEXT, bearer TEXT UNIQUE, name TEXT,
  created_at REAL, last_seen REAL, query_count INTEGER DEFAULT 0,
  error_count INTEGER DEFAULT 0, blocked_count INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, session_id TEXT, name TEXT,
  kind TEXT, status INTEGER, latency_ms REAL, detail TEXT);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS ix_sessions_invite ON sessions(invite_token);
"""


@dataclass
class Session:
    session_id: str
    invite_token: str
    bearer: str
    name: str
    created_at: float
    last_seen: float
    query_count: int = 0
    error_count: int = 0
    blocked_count: int = 0


class TenantStore:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    # --- invites ---
    def mint_invite(self, name: str, max_sessions: int = 0) -> str:
        token = "inv_" + secrets.token_urlsafe(18)
        self._db.execute("INSERT INTO invites(token,name,created_at,max_sessions,revoked) VALUES(?,?,?,?,0)",
                         (token, name, time.time(), max_sessions))
        return token

    def get_invite(self, token: str) -> sqlite3.Row | None:
        return self._db.execute("SELECT * FROM invites WHERE token=? AND revoked=0", (token,)).fetchone()

    def revoke_invite(self, token: str) -> None:
        self._db.execute("UPDATE invites SET revoked=1 WHERE token=?", (token,))

    def list_invites(self) -> list[sqlite3.Row]:
        return self._db.execute("SELECT * FROM invites ORDER BY created_at").fetchall()

    # --- sessions ---
    def _sessions_for(self, token: str) -> int:
        return self._db.execute("SELECT COUNT(*) FROM sessions WHERE invite_token=?", (token,)).fetchone()[0]

    def open_session(self, token: str) -> Session | None:
        """Validate the invite and return its session, creating one on first click.

        One active session per invite (reused on subsequent clicks) so a person's
        connector URL is stable; honours invite.max_sessions if > 0.
        """
        inv = self.get_invite(token)
        if inv is None:
            return None
        row = self._db.execute(
            "SELECT * FROM sessions WHERE invite_token=? ORDER BY created_at DESC LIMIT 1", (token,)).fetchone()
        if row is not None:
            self._db.execute("UPDATE sessions SET last_seen=? WHERE session_id=?", (time.time(), row["session_id"]))
            return self._row_to_session(row)
        if inv["max_sessions"] and self._sessions_for(token) >= inv["max_sessions"]:
            return None
        now = time.time()
        sid = "s_" + secrets.token_urlsafe(9)
        bearer = "demo_" + secrets.token_urlsafe(24)
        self._db.execute(
            "INSERT INTO sessions(session_id,invite_token,bearer,name,created_at,last_seen) VALUES(?,?,?,?,?,?)",
            (sid, token, bearer, inv["name"], now, now))
        return Session(sid, token, bearer, inv["name"], now, now)

    def session_by_bearer(self, bearer: str) -> Session | None:
        row = self._db.execute("SELECT * FROM sessions WHERE bearer=?", (bearer,)).fetchone()
        return self._row_to_session(row) if row else None

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        return Session(row["session_id"], row["invite_token"], row["bearer"], row["name"],
                       row["created_at"], row["last_seen"], row["query_count"],
                       row["error_count"], row["blocked_count"])

    def bump(self, session_id: str, *, query: int = 0, error: int = 0, blocked: int = 0) -> None:
        self._db.execute(
            "UPDATE sessions SET query_count=query_count+?, error_count=error_count+?, "
            "blocked_count=blocked_count+?, last_seen=? WHERE session_id=?",
            (query, error, blocked, time.time(), session_id))

    # --- rate caps (sliding window + daily, from the durable events log) ---
    def rate_ok(self, session_id: str, window_s: int, max_in_window: int, daily_max: int) -> tuple[bool, str]:
        now = time.time()
        win = self._db.execute(
            "SELECT COUNT(*) FROM events WHERE session_id=? AND kind='query' AND ts>=?",
            (session_id, now - window_s)).fetchone()[0]
        if win >= max_in_window:
            return False, f"rate: {win} reqs in {window_s}s >= {max_in_window}"
        day = self._db.execute(
            "SELECT COUNT(*) FROM events WHERE session_id=? AND kind='query' AND ts>=?",
            (session_id, now - 86400)).fetchone()[0]
        if day >= daily_max:
            return False, f"daily: {day} reqs/24h >= {daily_max}"
        return True, ""

    # --- events ---
    def record(self, session_id: str | None, name: str, kind: str,
               status: int = 0, latency_ms: float = 0.0, detail: str = "") -> None:
        self._db.execute(
            "INSERT INTO events(ts,session_id,name,kind,status,latency_ms,detail) VALUES(?,?,?,?,?,?,?)",
            (time.time(), session_id, name, kind, status, latency_ms, detail[:2000]))

    # --- report ---
    def report(self, since_s: float | None = None) -> dict:
        where, args = "", []
        if since_s:
            where, args = "WHERE ts>=?", [time.time() - since_s]
        ev = self._db.execute(f"SELECT kind, COUNT(*) c FROM events {where} GROUP BY kind", args).fetchall()
        per_person = self._db.execute(
            "SELECT name, COUNT(*) sessions, SUM(query_count) queries, SUM(error_count) errors, "
            "SUM(blocked_count) blocked, MAX(last_seen) last_seen FROM sessions GROUP BY invite_token "
            "ORDER BY queries DESC").fetchall()
        recent_errors = self._db.execute(
            "SELECT ts, session_id, name, detail FROM events WHERE kind='error' ORDER BY ts DESC LIMIT 20").fetchall()
        return {
            "events_by_kind": {r["kind"]: r["c"] for r in ev},
            "invites": len(self.list_invites()),
            "sessions": self._db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "per_person": [dict(r) for r in per_person],
            "recent_errors": [dict(r) for r in recent_errors],
        }
