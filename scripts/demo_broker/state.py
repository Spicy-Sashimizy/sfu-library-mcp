"""Durable broker state — sqlite KV with single-flight wake lock + daily runtime
accounting. Survives a NAS reboot so the killswitch budget cap can't be reset by
bouncing the process.

The runtime accumulator is keyed by local-tz calendar day; `runtime_hours_today`
= persisted accrual for today + live delta since the current boot. That feeds the
killswitch in scheduler.decide().
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime

from scheduler import BrokerState, DropletStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


class StateStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    # --- low-level KV ---
    def get(self, k: str, default=None):
        row = self._db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, k: str, v) -> None:
        self._db.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                         (k, json.dumps(v)))

    # --- single-flight wake lock: 15 simultaneous clicks => one boot ---
    def acquire_wake_lock(self, ttl_s: int = 600) -> bool:
        now = time.time()
        held = self.get("wake_lock")
        if held and now - held < ttl_s:
            return False
        self.set("wake_lock", now)
        return True

    def release_wake_lock(self) -> None:
        self.set("wake_lock", 0)

    # --- droplet lifecycle bookkeeping ---
    def mark_booting(self, droplet_id: str | None = None) -> None:
        self.set("status", DropletStatus.BOOTING.value)
        self.set("booted_at", time.time())
        self.set("droplet_id", droplet_id)
        self.touch_activity()

    def mark_up(self, droplet_id: str | None = None) -> None:
        self.set("status", DropletStatus.UP.value)
        if droplet_id is not None:
            self.set("droplet_id", droplet_id)
        if not self.get("booted_at"):
            self.set("booted_at", time.time())

    def mark_down(self, now: datetime) -> None:
        # fold the just-ended run into today's accumulator before clearing.
        self._accrue_runtime(now)
        self.set("status", DropletStatus.DOWN.value)
        self.set("booted_at", 0)
        self.set("droplet_id", None)
        self.set("on_demand_session", False)

    def touch_activity(self) -> None:
        self.set("last_activity", time.time())

    def set_on_demand(self, flag: bool) -> None:
        self.set("on_demand_session", bool(flag))

    # --- runtime accounting (local-tz day) ---
    def _accrue_runtime(self, now: datetime) -> None:
        booted_at = self.get("booted_at") or 0
        if not booted_at:
            return
        day = now.date().isoformat()
        if self.get("runtime_day") != day:
            self.set("runtime_day", day)
            self.set("runtime_seconds", 0.0)
        elapsed = max(0.0, time.time() - booted_at)
        self.set("runtime_seconds", (self.get("runtime_seconds") or 0.0) + elapsed)
        self.set("booted_at", time.time())  # reset baseline so we don't double-count

    def runtime_hours_today(self, now: datetime) -> float:
        day = now.date().isoformat()
        accrued = (self.get("runtime_seconds") or 0.0) if self.get("runtime_day") == day else 0.0
        booted_at = self.get("booted_at") or 0
        if booted_at:
            # clamp the live portion to today's local midnight so a run that crosses
            # midnight doesn't charge yesterday's hours against today's killswitch cap
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            live = max(0.0, time.time() - max(booted_at, midnight))
        else:
            live = 0.0
        return (accrued + live) / 3600.0

    # --- snapshot for the scheduler ---
    def snapshot(self, now: datetime) -> BrokerState:
        status = DropletStatus(self.get("status", DropletStatus.DOWN.value))
        last = self.get("last_activity")
        idle = (time.time() - last) / 60.0 if (last and status == DropletStatus.UP) else None
        return BrokerState(
            status=status,
            idle_minutes=idle,
            runtime_hours_today=self.runtime_hours_today(now),
            on_demand_session=bool(self.get("on_demand_session", False)),
        )

    def droplet_id(self) -> str | None:
        return self.get("droplet_id")
