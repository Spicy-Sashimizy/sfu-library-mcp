"""Best-effort push notifications to the owner (ntfy / Slack / Discord / generic JSON).

One DEMO_NOTIFY_WEBHOOK URL covers ntfy.sh, Pushover-compatible, Slack/Discord
incoming webhooks, or any JSON sink. Fire-and-forget on a daemon thread so a slow
or dead webhook never adds latency to a visitor's request, and never raises.

`build_payload` is PURE (no network) so it is unit-tested without a webhook.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.request

log = logging.getLogger("demo_broker.notify")


def build_payload(kind: str, title: str, message: str) -> tuple[bytes, dict]:
    """Return (body_bytes, headers) for the given webhook flavour."""
    kind = (kind or "json").lower()
    if kind == "ntfy":
        # ntfy: body is the message, metadata via headers.
        return message.encode(), {"Title": title, "Priority": "default", "Tags": "books"}
    if kind == "slack":
        return json.dumps({"text": f"*{title}*\n{message}"}).encode(), {"Content-Type": "application/json"}
    if kind == "discord":
        return json.dumps({"content": f"**{title}**\n{message}"}).encode(), {"Content-Type": "application/json"}
    # generic JSON (Pushover-ish / custom)
    return json.dumps({"title": title, "message": message}).encode(), {"Content-Type": "application/json"}


def _post(url: str, body: bytes, headers: dict) -> None:
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            if r.status >= 300:
                log.warning("notify webhook returned %s", r.status)
    except Exception as e:  # never propagate into the request path
        log.warning("notify webhook failed: %s", e)


def notify(cfg, title: str, message: str) -> None:
    """Fire a notification in the background. No-op if no webhook configured."""
    if not cfg.notify_webhook:
        log.info("[no webhook] %s — %s", title, message)
        return
    body, headers = build_payload(cfg.notify_kind, title, message)
    threading.Thread(target=_post, args=(cfg.notify_webhook, body, headers),
                     daemon=True, name="notify").start()
