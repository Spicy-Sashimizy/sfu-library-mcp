"""NAS demo broker — FastAPI gateway (always-on) in front of the on-demand DO droplet.

Responsibilities (docs/infrastructure/HYBRID_DEMO_DEPLOYMENT.md §2A + multi-tenant ext):
  * /start?t=<invite>  per-person link -> isolated session + access info, notify owner
  * /status            JSON readiness for the waiting room to poll
  * /mcp[/...]         per-session bearer-gated reverse-proxy with rate caps,
                       visitor tool-blocking (no Zotero writes), durable event log
  * /admin/report      owner-token-gated usage + debug report
  * /health            broker self-health
  * background loop    hot/cold schedule + idle reaper + killswitch (scheduler.decide)

Account-free: each invitee is identified by an unguessable token in their link; no
sign-ups, and no OpenAlex key needed (OpenAlex is keyless — one shared server-side
polite-pool identity, protected by per-session rate caps). Engine-agnostic; runs in
DRY-RUN until DEMO_BROKER_LIVE=1 (no DO spend).

Run:  uvicorn app:app --host 0.0.0.0 --port 8088
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
import time
import urllib.request
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from config import CONFIG
from do_client import DOClient
from notify import notify
from scheduler import Action, DropletStatus, decide
from state import StateStore
from tenants import TenantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("demo_broker.app")

# durable JSONL mirror of structured events (rotating, survives restarts)
os.makedirs(os.path.dirname(CONFIG.log_path) or ".", exist_ok=True)
_jsonl = logging.getLogger("demo_broker.events")
_jsonl.setLevel(logging.INFO)
_jsonl.addHandler(logging.handlers.RotatingFileHandler(
    CONFIG.log_path, maxBytes=20_000_000, backupCount=10))


def jlog(**fields) -> None:
    fields["ts"] = time.time()
    _jsonl.info(json.dumps(fields))


app = FastAPI(title="SFU library demo broker")
store = StateStore(CONFIG.state_path)
tenants = TenantStore(CONFIG.state_path)
do = DOClient(CONFIG)
_stop = threading.Event()


def _now() -> datetime:
    from datetime import datetime as _dt
    return _dt.now(CONFIG.zone)


def _droplet_health_ready() -> bool:
    did = store.droplet_id()
    if not did:
        return False
    ip = do.droplet_ip(did)
    if not ip:
        return False
    try:
        with urllib.request.urlopen(f"http://{ip}:{CONFIG.droplet_port}/health",
                                    timeout=CONFIG.health_timeout_s) as r:
            return r.status == 200
    except Exception:
        return False


def _boot(reason: str, on_demand: bool) -> None:
    if not store.acquire_wake_lock():
        log.info("wake suppressed (single-flight lock held): %s", reason)
        return
    try:
        log.info("BOOT (%s, on_demand=%s)", reason, on_demand)
        did = do.create_droplet()
        store.mark_booting(did)
        store.set_on_demand(on_demand)
        tenants.record(None, "droplet", "wake", detail=reason)
        jlog(event="wake", reason=reason, on_demand=on_demand)
    finally:
        store.release_wake_lock()


def _destroy(reason: str) -> None:
    did = store.droplet_id()
    log.info("DESTROY (%s) droplet=%s", reason, did)
    do.destroy_droplet(did or "")
    store.mark_down(_now())
    tenants.record(None, "droplet", "reap", detail=reason)
    jlog(event="reap", reason=reason)


def _scheduler_loop() -> None:
    while not _stop.is_set():
        try:
            now = _now()
            if store.snapshot(now).status == DropletStatus.BOOTING and _droplet_health_ready():
                store.mark_up()
                log.info("droplet healthy -> UP")
            st = store.snapshot(now)
            d = decide(now, st, CONFIG)
            if d.action == Action.BOOT:
                _boot(d.reason, on_demand=False)
            elif d.action == Action.DESTROY:
                _destroy(d.reason)
        except Exception:
            log.exception("scheduler tick failed")
        _stop.wait(CONFIG.poll_seconds)


@app.on_event("startup")
def _startup() -> None:
    mode = "LIVE" if CONFIG.live else "DRY-RUN"
    log.info("broker starting in %s mode; tier=%s region=%s window=%02d-%02d days=%s notify=%s",
             mode, CONFIG.do_size, CONFIG.do_region, CONFIG.business_start_hour,
             CONFIG.business_end_hour, sorted(CONFIG.business_days),
             CONFIG.notify_kind if CONFIG.notify_webhook else "off")
    for p in CONFIG.validate():
        log.warning("config: %s", p)
    threading.Thread(target=_scheduler_loop, daemon=True, name="scheduler").start()


@app.on_event("shutdown")
def _shutdown() -> None:
    _stop.set()


@app.get("/health")
def health():
    st = store.snapshot(_now())
    return {"status": "ok", "mode": "live" if CONFIG.live else "dry-run",
            "droplet": st.status.value, "runtime_hours_today": round(st.runtime_hours_today, 3),
            "idle_minutes": st.idle_minutes}


@app.get("/status")
def status():
    st = store.snapshot(_now())
    return {"ready": st.status == DropletStatus.UP, "droplet": st.status.value}


@app.get("/start", response_class=HTMLResponse)
def start(request: Request, t: str = ""):
    sess = tenants.open_session(t) if t else None
    if sess is None:
        tenants.record(None, "start", "denied", status=403, detail=f"bad invite token from {request.client.host}")
        jlog(event="start_denied", ip=request.client.host)
        raise HTTPException(status_code=403, detail="invalid, revoked, or missing invite link")

    st = store.snapshot(_now())
    if st.status == DropletStatus.DOWN:
        _boot(f"on-demand /start ({sess.name})", on_demand=True)

    tenants.record(sess.session_id, "start", "session_start", detail=sess.name)
    jlog(event="session_start", session=sess.session_id, name=sess.name, ip=request.client.host)
    notify(CONFIG, "SFU demo started",
           f"{sess.name} connected ({_now():%Y-%m-%d %H:%M %Z}) — session {sess.session_id}")

    mcp_url = f"{CONFIG.public_base.rstrip('/')}/mcp"
    return HTMLResponse(_waiting_room(mcp_url, sess.bearer))


def _rpc_info(body: bytes) -> tuple[bool, str | None]:
    """Parse the JSON-RPC body → (is_tool_call, blocked_tool_or_None).

    Only `tools/call` is a billable query. Protocol messages (initialize,
    tools/list, notifications/*) must NOT count toward the rate caps, or the MCP
    handshake every session performs would exhaust its own quota.
    """
    if not body:
        return False, None
    try:
        msg = json.loads(body)
    except Exception:
        return False, None
    is_call = False
    blocked = None
    for m in (msg if isinstance(msg, list) else [msg]):
        if isinstance(m, dict) and m.get("method") == "tools/call":
            is_call = True
            name = (m.get("params") or {}).get("name")
            if name in CONFIG.blocked_tools:
                blocked = name
    return is_call, blocked


def _proxy(request: Request, subpath: str, body: bytes) -> Response:
    # 1) per-session bearer
    auth = request.headers.get("authorization", "")
    bearer = auth[7:] if auth.startswith("Bearer ") else ""
    sess = tenants.session_by_bearer(bearer) if bearer else None
    if sess is None:
        raise HTTPException(status_code=401, detail="missing/invalid session bearer; open your /start link")

    # 2) classify: only tools/call is billable; protocol msgs pass freely
    is_call, blocked = _rpc_info(body)
    if blocked:  # visitor tool-blocking (e.g. writes to the owner's Zotero)
        tenants.bump(sess.session_id, blocked=1)
        tenants.record(sess.session_id, blocked, "blocked", status=403)
        jlog(event="blocked", session=sess.session_id, tool=blocked, name=sess.name)
        raise HTTPException(status_code=403, detail=f"tool '{blocked}' is disabled in the demo")

    # 3) per-session rate caps — count ONLY tool calls (searches), so the MCP
    #    handshake (initialize/tools/list/notifications) never exhausts quota
    if is_call:
        ok, why = tenants.rate_ok(sess.session_id, CONFIG.session_rate_window_s,
                                  CONFIG.session_rate_max, CONFIG.session_daily_max)
        if not ok:
            tenants.record(sess.session_id, "ratelimit", "blocked", status=429, detail=why)
            jlog(event="ratelimited", session=sess.session_id, why=why)
            raise HTTPException(status_code=429, detail=f"rate limit: {why}")

    # 4) compute must be up
    did = store.droplet_id()
    if store.snapshot(_now()).status != DropletStatus.UP or not did:
        raise HTTPException(status_code=503, detail="compute warming; retry shortly (see /start)")

    ip = do.droplet_ip(did)
    url = f"http://{ip}:{CONFIG.droplet_port}/{subpath}"
    upreq = urllib.request.Request(url, data=body or None, method=request.method)
    for h in ("content-type", "accept", "mcp-session-id"):
        if h in request.headers:
            upreq.add_header(h, request.headers[h])
    upreq.add_header("X-Demo-Session", sess.session_id)  # tag upstream for attribution
    t0 = time.perf_counter()
    try:
        # NOT a context manager: the body is streamed (and closed) by the generator
        # below so MCP SSE (text/event-stream) responses pass through — a buffered
        # .read() would hang on a long-lived event stream.
        resp = urllib.request.urlopen(upreq, timeout=CONFIG.upstream_timeout_s)
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        tenants.bump(sess.session_id, error=1)
        tenants.record(sess.session_id, subpath, "error", status=502, latency_ms=dt, detail=str(e)[:500])
        jlog(event="error", session=sess.session_id, detail=str(e)[:500])
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")

    code = resp.status
    ctype = resp.headers.get("content-type", "application/json")
    # forward Mcp-Session-Id (REQUIRED: client must echo it on later calls) + caching
    fwd = {h: resp.headers[h] for h in ("mcp-session-id", "cache-control") if h in resp.headers}

    def _stream():
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                yield chunk
        finally:
            resp.close()
            dt = (time.perf_counter() - t0) * 1000
            store.touch_activity()
            if is_call:
                tenants.bump(sess.session_id, query=1)
            tenants.record(sess.session_id, subpath, "query" if is_call else "mcp",
                           status=code, latency_ms=dt)
            jlog(event="query" if is_call else "mcp", session=sess.session_id,
                 name=sess.name, status=code, latency_ms=round(dt, 1))
            if is_call and CONFIG.notify_on_query:
                notify(CONFIG, "SFU demo query", f"{sess.name}: {subpath} ({code}, {dt:.0f}ms)")

    return StreamingResponse(_stream(), status_code=code, media_type=ctype, headers=fwd)


@app.api_route("/mcp", methods=["GET", "POST"])
async def mcp_root(request: Request):
    return _proxy(request, "mcp", await request.body())


@app.api_route("/mcp/{subpath:path}", methods=["GET", "POST"])
async def mcp_sub(request: Request, subpath: str):
    return _proxy(request, f"mcp/{subpath}", await request.body())


@app.get("/admin/report")
def admin_report(request: Request, since_hours: float = 24.0):
    tok = request.headers.get("x-admin-token", "") or request.query_params.get("admin", "")
    if not CONFIG.admin_token or tok != CONFIG.admin_token:
        raise HTTPException(status_code=403, detail="admin token required")
    st = store.snapshot(_now())
    rep = tenants.report(since_s=since_hours * 3600)
    rep["droplet"] = {"status": st.status.value, "runtime_hours_today": round(st.runtime_hours_today, 3),
                      "est_cost_today_usd": round(st.runtime_hours_today * 0.50, 2)}  # g-16vcpu-64gb est.
    return rep


def _waiting_room(mcp_url: str, bearer: str) -> str:
    return _WAITING_ROOM.replace("__MCP_URL__", mcp_url).replace("__BEARER__", bearer)


_WAITING_ROOM = """<!doctype html><html><head><meta charset=utf-8>
<title>SFU Library demo — your access</title>
<style>body{font-family:system-ui;max-width:42rem;margin:3rem auto;padding:0 1rem;color:#222}
.spin{display:inline-block;width:1rem;height:1rem;border:2px solid #ccc;border-top-color:#c00;
border-radius:50%;animation:s 1s linear infinite;vertical-align:middle}
@keyframes s{to{transform:rotate(360deg)}}code{background:#f4f4f4;padding:.15rem .35rem;
border-radius:4px;word-break:break-all}.box{background:#f7f7f7;border:1px solid #e0e0e0;
border-radius:8px;padding:1rem;margin:1rem 0}</style></head>
<body><h1>Your SFU Library search demo</h1>
<p><span class=spin id=sp></span> <span id=msg>Starting compute…</span></p>
<div id=ready style=display:none>
<h2>Ready ✓ — your private access</h2>
<div class=box>
<p><b>Connector URL</b> (Claude → Settings → Connectors → Add custom connector):</p>
<p><code>__MCP_URL__</code></p>
<p><b>Authorization header</b> (set as a bearer/secret header if your client supports it):</p>
<p><code>Bearer __BEARER__</code></p>
</div>
<p style=color:#666>This link is personal to you. Usage is rate-limited and logged for the demo.</p>
</div>
<script>
async function poll(){
 try{const r=await fetch('/status');const j=await r.json();
  if(j.ready){document.getElementById('sp').style.display='none';
   document.getElementById('msg').textContent='Compute ready.';
   document.getElementById('ready').style.display='block';return;}
  document.getElementById('msg').textContent='Status: '+j.droplet+' — warming (~2-4 min after hours)…';
 }catch(e){document.getElementById('msg').textContent='Waiting…';}
 setTimeout(poll,4000);}
poll();
</script></body></html>"""
