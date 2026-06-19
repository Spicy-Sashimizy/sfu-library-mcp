"""
SFU Library MCP Server — Streamable HTTP Transport.

Production entry point serving the same MCP server over HTTP
instead of stdio. Used for remote access from Claude Desktop via mcp-remote.

Endpoints:
  POST /mcp        — MCP JSON-RPC (Streamable HTTP transport)
  GET  /mcp        — SSE stream for server-initiated messages
  GET  /health     — Health check (returns {"status": "ok", "tools": <count>})
  GET  /analytics  — Research-analytics dashboard data bundle (Phase N GUI tracking).
                     Optional ?panel=<name> for a single panel.
  POST /engagement — Record click-through / engagement events (one or a batch).
"""

import contextlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Tool, TextContent

from lib.logging_setup import setup_logging
from lib.config import load_config, validate_config
from lib.tools import (
    TOOL_DEFINITIONS,
    get_tool_definitions,
    handle_tool_call,
    search_academic_structured,
)

# Static web GUI (the SFU Library Suite). Served read-only at /app.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

config = load_config()
logger = setup_logging(level=config.log_level, log_file=config.log_file)

for warning in validate_config(config):
    logger.warning("Config: %s", warning)

mcp_server = Server("sfu-library")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return await get_tool_definitions()


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    return await handle_tool_call(name, arguments)


async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "tools": len(TOOL_DEFINITIONS)})


async def analytics(request: Request) -> JSONResponse:
    """Read-only analytics bundle for the research-analytics GUI."""
    from lib.analytics import build_analytics_bundle
    panel = request.query_params.get("panel")
    try:
        return JSONResponse(build_analytics_bundle(panel))
    except Exception as e:
        logger.exception("Analytics bundle failed")
        return JSONResponse({"error": str(e)}, status_code=500)


async def engagement(request: Request) -> JSONResponse:
    """Capture click-through / engagement events (single object or batch)."""
    from lib.engagement import record_engagement
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    n = record_engagement(payload)
    return JSONResponse({"recorded": n})


# ── GUI REST endpoints (consumed by web/lib/api.js) ──────────────────────────
# Thin JSON wrappers so the browser never has to drive the MCP JSON-RPC
# handshake. Each reuses the existing tool logic. The GUI degrades to mock data
# on any non-2xx, so these stay simple and surface real errors as 5xx.

def _tool_json(result) -> dict:
    """Parse the first TextContent of a tool result as JSON (tools that emit
    JSON), else wrap the raw text under {'text': ...}."""
    if not result:
        return {}
    text = getattr(result[0], "text", "") or ""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return {"text": text}


async def api_search(request: Request) -> JSONResponse:
    """Structured academic search for the GUI result cards."""
    try:
        args = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(args, dict) or not args.get("query"):
        return JSONResponse({"error": "missing 'query'"}, status_code=400)
    try:
        data = await search_academic_structured(args)
        return JSONResponse(data)
    except Exception as e:
        logger.exception("GUI search failed")
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_index_status(request: Request) -> JSONResponse:
    """Index metrics + pack state + unpack jobs (for the Index Manager app)."""
    try:
        result = await handle_tool_call("get_index_status", {})
        return JSONResponse(_tool_json(result))
    except Exception as e:
        logger.exception("GUI index_status failed")
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_personas(request: Request) -> JSONResponse:
    """Persona registry + live/cold section lists (for the Index Manager app)."""
    try:
        result = await handle_tool_call("list_personas", {})
        return JSONResponse(_tool_json(result))
    except Exception as e:
        logger.exception("GUI personas failed")
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_unpack(request: Request) -> JSONResponse:
    """Kick off a background section unpack (for the Index Manager app)."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    section = (payload or {}).get("section", "")
    if not section:
        return JSONResponse({"error": "missing 'section'"}, status_code=400)
    try:
        result = await handle_tool_call("request_section_unpack", {"section": section})
        return JSONResponse({"ok": True, "message": getattr(result[0], "text", "") if result else ""})
    except Exception as e:
        logger.exception("GUI unpack failed")
        return JSONResponse({"error": str(e)}, status_code=500)


async def app_root(request: Request) -> RedirectResponse:
    """/app -> the suite shell."""
    return RedirectResponse(url="/app/suite.html")


session_manager = StreamableHTTPSessionManager(
    app=mcp_server,
    json_response=True,
    stateless=True,
)


@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with session_manager.run():
        yield


_routes = [
    Route("/health", health_check, methods=["GET"]),
    Route("/analytics", analytics, methods=["GET"]),
    Route("/engagement", engagement, methods=["POST"]),
    Route("/api/search", api_search, methods=["POST"]),
    Route("/api/index_status", api_index_status, methods=["GET"]),
    Route("/api/personas", api_personas, methods=["GET"]),
    Route("/api/unpack", api_unpack, methods=["POST"]),
    Route("/app", app_root, methods=["GET"]),
]

# Serve the static web GUI only if the bundle is present (keeps the server
# usable in headless/index-only deployments that don't ship the GUI).
if WEB_DIR.is_dir():
    _routes.append(Mount("/app", app=StaticFiles(directory=str(WEB_DIR), html=True)))
    logger.info("Serving web GUI from %s at /app", WEB_DIR)
else:
    logger.info("Web GUI dir %s not found; /app disabled", WEB_DIR)

_starlette = Starlette(lifespan=lifespan, routes=_routes)


async def app(scope, receive, send):
    if scope["type"] == "http" and scope.get("path", "").rstrip("/") == "/mcp":
        await session_manager.handle_request(scope, receive, send)
    else:
        await _starlette(scope, receive, send)


def main():
    host = os.environ.get("MCP_HTTP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_HTTP_PORT", "8080"))
    logger.info("Starting SFU Library MCP HTTP server on %s:%d", host, port)
    logger.info("Tools available: %d", len(TOOL_DEFINITIONS))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
