"""Capture server for browser-assisted PDF capture.

Receives PDF URLs + cookies from the Chrome extension and downloads PDFs
using ArticleDownloader's tiered strategy.

Usage:
    python -m capture_server
    # or
    uvicorn capture_server:app --port 8787
"""

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lib.config import load_config
from lib.downloader import ArticleDownloader

logger = logging.getLogger("sfu_library_mcp.capture_server")

# ─── State ─────────────────────────────────────────────────────

_downloader: ArticleDownloader | None = None
_captures: list[dict] = []
MAX_CAPTURE_HISTORY = 100


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize downloader on startup."""
    global _downloader
    config = load_config()
    _downloader = ArticleDownloader(config)
    logger.info("Capture server started on port %d", config.capture_server_port)
    yield
    logger.info("Capture server shutting down")


# ─── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="SFU Library PDF Capture Server",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from Chrome extensions
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^chrome-extension://.*$",
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response models ──────────────────────────────────

class CaptureRequest(BaseModel):
    url: str
    cookies: dict[str, str] = {}
    filename: str | None = None


class CaptureResponse(BaseModel):
    success: bool
    container_path: str | None = None
    size_bytes: int = 0
    tier_used: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_captures: int


class CaptureHistoryItem(BaseModel):
    url: str
    timestamp: str
    success: bool
    size_bytes: int
    tier_used: str | None
    error: str | None


# ─── Track uptime ─────────────────────────────────────────────

_start_time = time.time()


# ─── Background download task ─────────────────────────────────

def _do_capture(url: str, cookies: dict, filename: str | None) -> dict:
    """Synchronous capture task run in background."""
    global _downloader
    if _downloader is None:
        return {"success": False, "error": "Downloader not initialized"}

    result = _downloader.download_from_direct_url(url, cookies=cookies, filename=filename)
    return result


# ─── Endpoints ─────────────────────────────────────────────────

@app.post("/capture", response_model=CaptureResponse)
async def capture(req: CaptureRequest):
    """Receive a PDF URL + cookies from the Chrome extension and download it."""
    logger.info("Capture request: %s (cookies: %d)", req.url[:80], len(req.cookies))

    result = _do_capture(req.url, req.cookies, req.filename)

    # Record in history
    entry = {
        "url": req.url[:200],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "success": result.get("success", False),
        "size_bytes": result.get("size_bytes", 0),
        "tier_used": result.get("tier_used"),
        "error": result.get("error"),
    }
    _captures.insert(0, entry)
    if len(_captures) > MAX_CAPTURE_HISTORY:
        _captures[:] = _captures[:MAX_CAPTURE_HISTORY]

    return CaptureResponse(
        success=result.get("success", False),
        container_path=result.get("container_path"),
        size_bytes=result.get("size_bytes", 0),
        tier_used=result.get("tier_used"),
        error=result.get("error"),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for extension connectivity."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
        total_captures=len(_captures),
    )


@app.get("/captures", response_model=list[CaptureHistoryItem])
async def captures():
    """List recent captures with status."""
    return [CaptureHistoryItem(**entry) for entry in _captures[:50]]


# ─── CLI entrypoint ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    config = load_config()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "capture_server:app",
        host="0.0.0.0",
        port=config.capture_server_port,
        log_level="info",
    )
