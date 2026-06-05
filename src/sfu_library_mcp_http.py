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
import os
from collections.abc import AsyncIterator

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Tool, TextContent

from lib.logging_setup import setup_logging
from lib.config import load_config, validate_config
from lib.tools import TOOL_DEFINITIONS, get_tool_definitions, handle_tool_call

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


session_manager = StreamableHTTPSessionManager(
    app=mcp_server,
    json_response=True,
    stateless=True,
)


@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with session_manager.run():
        yield


_starlette = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/health", health_check, methods=["GET"]),
        Route("/analytics", analytics, methods=["GET"]),
        Route("/engagement", engagement, methods=["POST"]),
    ],
)


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
