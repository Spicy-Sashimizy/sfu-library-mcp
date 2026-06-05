"""Click-through / engagement tracking for the analytics GUI (Phase N).

Captures what users actually do with search results — which results they click, how
deep they engage (abstract -> tldr -> pdf -> citation -> zotero), and how they refine
queries. Two consumers:

  1. Position-bias / propensity panel — P(click | rank) needs, per rank, both an
     IMPRESSION count (denominator) and a CLICK count (numerator). The search path
     emits one impression row per ranked result via tools._log_query; the client posts
     `result_click` / `action` events as the user interacts.
  2. Session-replay tree — query -> result -> action chain, reconstructed per session.

These signals are also the future implicit-relevance labels for LambdaMART training
(clicked@1 -> grade 3, etc. — see scripts/train_lambdamart.py).

Storage is a single append-only JSONL file (SFU_ENGAGEMENT_LOG_PATH, default
logs/engagement_log.jsonl). Writes are best-effort and never raise into the caller.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("sfu_library_mcp")

# Event taxonomy — mirrors the design's ACTION_STYLE / session-replay kinds.
KINDS = {"query", "impression", "result_click", "action"}
ACTION_TYPES = {"abstract", "tldr", "pdf", "citation", "zotero"}

_DEFAULT_LOG = Path(__file__).resolve().parents[2] / "logs" / "engagement_log.jsonl"
_write_lock = threading.Lock()


def _log_path() -> Path:
    return Path(os.environ.get("SFU_ENGAGEMENT_LOG_PATH", str(_DEFAULT_LOG)))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_event(event: dict) -> dict:
    """Coerce/validate one event. Raises ValueError on an unusable event.

    Required: session_id (str), kind (one of KINDS). Optional: query, rank (int),
    doc_id, action_type (one of ACTION_TYPES for kind=='action'), refined_from, ts.
    """
    if not isinstance(event, dict):
        raise ValueError("event must be an object")
    session_id = str(event.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("session_id is required")
    kind = str(event.get("kind") or "").strip()
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")

    out = {
        "ts": str(event.get("ts") or _now_iso()),
        "session_id": session_id,
        "kind": kind,
    }
    if event.get("query") is not None:
        out["query"] = str(event["query"])[:300]
    if event.get("doc_id") is not None:
        out["doc_id"] = str(event["doc_id"])
    if event.get("rank") is not None:
        try:
            out["rank"] = int(event["rank"])
        except (TypeError, ValueError):
            raise ValueError("rank must be an integer")
    if kind == "action":
        action_type = str(event.get("action_type") or "").strip()
        if action_type not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {sorted(ACTION_TYPES)}")
        out["action_type"] = action_type
    if event.get("refined_from") is not None:
        out["refined_from"] = str(event["refined_from"])[:300]
    return out


def log_engagement(session_id: str, kind: str, **fields) -> bool:
    """Append a single engagement event. Best-effort; returns success bool."""
    try:
        event = validate_event({"session_id": session_id, "kind": kind, **fields})
    except ValueError as e:
        logger.debug("Dropping invalid engagement event: %s", e)
        return False
    return _append([event]) == 1


def record_engagement(payload) -> int:
    """Record one event dict or a list/{"events":[...]} batch. Returns count written.

    This is the single validated write path shared by the POST /engagement HTTP route
    and the record_engagement MCP tool. Invalid events are skipped, not fatal.
    """
    if isinstance(payload, dict) and "events" in payload:
        payload = payload["events"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return 0
    valid = []
    for raw in payload:
        try:
            valid.append(validate_event(raw))
        except ValueError as e:
            logger.debug("Skipping invalid engagement event: %s", e)
    return _append(valid)


def log_impressions(session_id: str, query: str, results: list[dict]) -> None:
    """Emit one impression row per ranked result — the propensity denominators.

    Called from the search path (tools._log_query). doc_id prefers DOI, falls back to
    openalex_id. Best-effort and silent.
    """
    events = []
    for i, r in enumerate(results[:20]):
        doc_id = r.get("doi") or r.get("openalex_id") or r.get("record_id") or ""
        events.append({
            "session_id": session_id, "kind": "impression",
            "query": query, "rank": i + 1, "doc_id": str(doc_id),
        })
    if events:
        try:
            valid = [validate_event(e) for e in events]
            _append(valid)
        except Exception:
            logger.debug("Impression logging failed (non-fatal)")


def _append(events: list[dict]) -> int:
    if not events:
        return 0
    path = _log_path()
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                for e in events:
                    f.write(json.dumps(e) + "\n")
        return len(events)
    except Exception:
        logger.debug("Engagement log write failed (non-fatal)")
        return 0


def load_events(path: Path | None = None) -> list[dict]:
    """Read all engagement events (oldest first). Empty list when absent."""
    p = path or _log_path()
    if not p.exists():
        return []
    events = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events
