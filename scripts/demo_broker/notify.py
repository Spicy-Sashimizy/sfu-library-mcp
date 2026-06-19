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


def build_payload(kind: str, title: str, message: str, token: str = "") -> tuple[bytes, dict]:
    """Return (body_bytes, headers) for the given webhook flavour.

    `token` adds `Authorization: Bearer` for auth-locked ntfy / generic JSON sinks
    (Slack/Discord carry auth in the webhook URL itself, so it's ignored there).
    """
    kind = (kind or "json").lower()
    if kind == "ntfy":
        # ntfy: body is the message, metadata via headers.
        h = {"Title": title, "Priority": "default", "Tags": "books"}
        if token:
            h["Authorization"] = f"Bearer {token}"
        return message.encode(), h
    if kind == "slack":
        return json.dumps({"text": f"*{title}*\n{message}"}).encode(), {"Content-Type": "application/json"}
    if kind == "discord":
        return json.dumps({"content": f"**{title}**\n{message}"}).encode(), {"Content-Type": "application/json"}
    # generic JSON (Pushover-ish / custom)
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return json.dumps({"title": title, "message": message}).encode(), h


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
    body, headers = build_payload(cfg.notify_kind, title, message, getattr(cfg, "notify_token", ""))
    threading.Thread(target=_post, args=(cfg.notify_webhook, body, headers),
                     daemon=True, name="notify").start()


if __name__ == "__main__":
    # Manual test push (blocking, so you see the result):
    #   DEMO_NOTIFY_WEBHOOK=... DEMO_NOTIFY_KIND=ntfy DEMO_NOTIFY_TOKEN=... \
    #     python3 notify.py "hello from the demo broker"
    import sys
    from config import CONFIG
    logging.basicConfig(level=logging.INFO)
    msg = " ".join(sys.argv[1:]) or "demo broker test notification"
    if not CONFIG.notify_webhook:
        print("set DEMO_NOTIFY_WEBHOOK (+DEMO_NOTIFY_TOKEN for auth-locked ntfy)")
        raise SystemExit(2)
    b, h = build_payload(CONFIG.notify_kind, "SFU demo broker — test", msg, CONFIG.notify_token)
    _post(CONFIG.notify_webhook, b, h)
    print(f"posted to {CONFIG.notify_webhook} (kind={CONFIG.notify_kind}, auth={'yes' if CONFIG.notify_token else 'no'})")
