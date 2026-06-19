"""Demo broker config — all knobs are env-driven so the NAS deploy needs no code edits.

Loaded once at startup. Safe defaults are conservative for the $200 credit:
dry-run is ON by default (no DO API spend until DEMO_BROKER_LIVE=1), the hard
daily killswitch is the budget backstop, and the business-hours window keeps the
droplet hot 09:00-17:00 weekdays only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _days(name: str, default: str) -> frozenset[int]:
    # 0=Mon .. 6=Sun. Default Mon-Fri.
    raw = os.environ.get(name, default)
    out: set[int] = set()
    for tok in raw.replace(" ", "").split(","):
        if not tok:
            continue
        out.add(int(tok))
    return frozenset(out)


@dataclass(frozen=True)
class Config:
    # --- identity / DO ---
    do_token: str = field(default_factory=lambda: os.environ.get("DIGITALOCEAN_ACCESS_TOKEN", ""))
    do_region: str = field(default_factory=lambda: os.environ.get("DEMO_DO_REGION", "tor1"))
    do_size: str = field(default_factory=lambda: os.environ.get("DEMO_DO_SIZE", "g-16vcpu-64gb"))
    do_snapshot_id: str = field(default_factory=lambda: os.environ.get("DEMO_DO_SNAPSHOT_ID", ""))
    do_volume_id: str = field(default_factory=lambda: os.environ.get("DEMO_DO_VOLUME_ID", ""))
    do_droplet_name: str = field(default_factory=lambda: os.environ.get("DEMO_DO_DROPLET_NAME", "sfu-demo"))
    do_firewall_id: str = field(default_factory=lambda: os.environ.get("DEMO_DO_FIREWALL_ID", ""))

    # --- safety gates ---
    live: bool = field(default_factory=lambda: _b("DEMO_BROKER_LIVE", False))  # OFF => dry-run, no spend
    link_token: str = field(default_factory=lambda: os.environ.get("DEMO_LINK_TOKEN", ""))
    bearer_token: str = field(default_factory=lambda: os.environ.get("DEMO_BEARER_TOKEN", ""))

    # --- business-hours hot/cold schedule ---
    tz: str = field(default_factory=lambda: os.environ.get("DEMO_TZ", "America/Vancouver"))
    business_start_hour: int = field(default_factory=lambda: _i("DEMO_BUSINESS_START_HOUR", 9))
    business_end_hour: int = field(default_factory=lambda: _i("DEMO_BUSINESS_END_HOUR", 17))
    business_days: frozenset[int] = field(default_factory=lambda: _days("DEMO_BUSINESS_DAYS", "0,1,2,3,4"))
    # pre-warm: boot this many minutes before the window opens so 9:00 is already hot
    prewarm_minutes: int = field(default_factory=lambda: _i("DEMO_PREWARM_MINUTES", 5))

    # --- reaper / killswitch ---
    idle_minutes: int = field(default_factory=lambda: _i("DEMO_IDLE_MINUTES", 20))  # after-hours only
    max_runtime_hours_per_day: float = field(default_factory=lambda: _f("DEMO_MAX_RUNTIME_HOURS_PER_DAY", 10.0))
    poll_seconds: int = field(default_factory=lambda: _i("DEMO_POLL_SECONDS", 60))

    # --- proxy ---
    droplet_port: int = field(default_factory=lambda: _i("DEMO_DROPLET_PORT", 8080))
    health_timeout_s: float = field(default_factory=lambda: _f("DEMO_HEALTH_TIMEOUT_S", 3.0))
    state_path: str = field(default_factory=lambda: os.environ.get("DEMO_STATE_PATH", "data/demo_broker/state.db"))

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty == ready for LIVE."""
        problems: list[str] = []
        if self.live:
            if not self.do_token:
                problems.append("DIGITALOCEAN_ACCESS_TOKEN unset (required in LIVE mode)")
            if not self.do_snapshot_id:
                problems.append("DEMO_DO_SNAPSHOT_ID unset (no image to boot)")
            if not self.do_volume_id:
                problems.append("DEMO_DO_VOLUME_ID unset (no index volume to attach)")
        if not self.link_token:
            problems.append("DEMO_LINK_TOKEN unset (anyone could wake the droplet)")
        if self.business_end_hour <= self.business_start_hour:
            problems.append("DEMO_BUSINESS_END_HOUR must be > DEMO_BUSINESS_START_HOUR")
        return problems


CONFIG = Config()
