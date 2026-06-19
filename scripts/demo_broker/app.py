"""NAS demo broker — FastAPI gateway (always-on) in front of the on-demand DO droplet.

Responsibilities (see docs/infrastructure/HYBRID_DEMO_DEPLOYMENT.md §2A):
  * /start?t=<token>  landing + waiting room; after-hours valid click wakes the droplet
  * /status           JSON readiness for the waiting room to poll
  * /mcp[/...]        reverse-proxy to the current droplet :8080, bearer-gated
  * /health           broker self-health
  * background loop   hot/cold schedule + idle reaper + killswitch (scheduler.decide)

Engine-agnostic: it brokers a droplet, not a search engine, so it is unaffected by
the BMP->Qdrant migration. Runs in DRY-RUN until DEMO_BROKER_LIVE=1 (no DO spend).

Run:  uvicorn app:app --host 0.0.0.0 --port 8088
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.request
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from config import CONFIG
from do_client import DOClient
from scheduler import Action, DropletStatus, decide
from state import StateStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("demo_broker.app")

app = FastAPI(title="SFU library demo broker")
store = StateStore(CONFIG.state_path)
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
    finally:
        store.release_wake_lock()


def _destroy(reason: str) -> None:
    did = store.droplet_id()
    log.info("DESTROY (%s) droplet=%s", reason, did)
    do.destroy_droplet(did or "")
    store.mark_down(_now())


def _scheduler_loop() -> None:
    """Background tick: apply the hot/cold + reaper + killswitch policy."""
    while not _stop.is_set():
        try:
            now = _now()
            # promote BOOTING -> UP once the droplet app answers /health
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
    problems = CONFIG.validate()
    mode = "LIVE" if CONFIG.live else "DRY-RUN"
    log.info("broker starting in %s mode; tier=%s region=%s window=%02d-%02d days=%s",
             mode, CONFIG.do_size, CONFIG.do_region, CONFIG.business_start_hour,
             CONFIG.business_end_hour, sorted(CONFIG.business_days))
    for p in problems:
        log.warning("config: %s", p)
    threading.Thread(target=_scheduler_loop, daemon=True, name="scheduler").start()


@app.on_event("shutdown")
def _shutdown() -> None:
    _stop.set()


@app.get("/health")
def health():
    now = _now()
    st = store.snapshot(now)
    return {"status": "ok", "mode": "live" if CONFIG.live else "dry-run",
            "droplet": st.status.value, "runtime_hours_today": round(st.runtime_hours_today, 3),
            "idle_minutes": st.idle_minutes}


@app.get("/status")
def status():
    now = _now()
    st = store.snapshot(now)
    ready = st.status == DropletStatus.UP
    return {"ready": ready, "droplet": st.status.value}


@app.get("/start", response_class=HTMLResponse)
def start(t: str = ""):
    if not CONFIG.link_token or t != CONFIG.link_token:
        raise HTTPException(status_code=403, detail="invalid or missing link token")
    now = _now()
    st = store.snapshot(now)
    if st.status == DropletStatus.DOWN:
        # after-hours on-demand wake (during business hours the scheduler already booted it)
        _boot("on-demand /start click", on_demand=True)
    return HTMLResponse(_WAITING_ROOM)


def _proxy(request: Request, subpath: str) -> Response:
    # bearer gate
    if CONFIG.bearer_token:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {CONFIG.bearer_token}":
            raise HTTPException(status_code=401, detail="missing/invalid bearer token")
    did = store.droplet_id()
    st = store.snapshot(_now())
    if st.status != DropletStatus.UP or not did:
        raise HTTPException(status_code=503, detail="compute not ready; visit /start")
    ip = do.droplet_ip(did)
    url = f"http://{ip}:{CONFIG.droplet_port}/{subpath}"
    body = request._body if hasattr(request, "_body") else None
    req = urllib.request.Request(url, data=body, method=request.method)
    for h in ("content-type", "accept", "mcp-session-id"):
        if h in request.headers:
            req.add_header(h, request.headers[h])
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            store.touch_activity()
            return Response(content=r.read(), status_code=r.status,
                            media_type=r.headers.get("content-type", "application/json"))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")


@app.api_route("/mcp", methods=["GET", "POST"])
async def mcp_root(request: Request):
    request._body = await request.body()
    return _proxy(request, "mcp")


@app.api_route("/mcp/{subpath:path}", methods=["GET", "POST"])
async def mcp_sub(request: Request, subpath: str):
    request._body = await request.body()
    return _proxy(request, f"mcp/{subpath}")


_WAITING_ROOM = """<!doctype html><html><head><meta charset=utf-8>
<title>SFU Library demo — starting</title>
<style>body{font-family:system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem;color:#222}
.spin{display:inline-block;width:1rem;height:1rem;border:2px solid #ccc;border-top-color:#c00;
border-radius:50%;animation:s 1s linear infinite;vertical-align:middle}
@keyframes s{to{transform:rotate(360deg)}}code{background:#f4f4f4;padding:.1rem .3rem}</style></head>
<body><h1>Starting the SFU Library search demo…</h1>
<p><span class=spin></span> Booting compute. This takes ~2-4 minutes after hours; during
business hours it is usually already warm.</p>
<p id=msg>Checking status…</p>
<div id=ready style=display:none>
<h2>Ready ✓</h2><p>Add this custom connector in Claude:</p>
<p><code id=url></code></p></div>
<script>
async function poll(){
 try{const r=await fetch('/status');const j=await r.json();
  if(j.ready){document.getElementById('msg').style.display='none';
   const d=document.getElementById('ready');d.style.display='block';
   document.getElementById('url').textContent=location.origin+'/mcp';return;}
  document.getElementById('msg').textContent='Status: '+j.droplet+' — still warming…';
 }catch(e){document.getElementById('msg').textContent='Waiting…';}
 setTimeout(poll,4000);}
poll();
</script></body></html>"""
