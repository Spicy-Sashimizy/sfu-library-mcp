"""
SFU Library MCP Server — Streamable HTTP Transport

Production entry point that serves the same MCP server over HTTP
instead of stdio. Used for remote access from Claude Desktop via mcp-remote.

Endpoints:
  POST /mcp    — MCP JSON-RPC (Streamable HTTP transport)
  GET  /mcp    — SSE stream for server-initiated messages
  GET  /health — Health check (returns {"status": "ok", "tools": <count>})
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp.server import Server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.types import Tool, TextContent

from lib.logging_setup import setup_logging
from lib.config import load_config, validate_config
from lib.client import SFULibraryClient
from lib.tools import TOOL_DEFINITIONS, handle_tool_call

# Configure logging
config = load_config()
logger = setup_logging(level=config.log_level, log_file=config.log_file)

for warning in validate_config(config):
    logger.warning("Config: %s", warning)

# Initialize MCP server — same name as stdio version
mcp_server = Server("sfu-library")

# Global client instance
_client = None


def get_client() -> SFULibraryClient:
    """Get or create the library client."""
    global _client
    if _client is None:
        _client = SFULibraryClient(headless=True, config=config)
    return _client


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return TOOL_DEFINITIONS


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    return await handle_tool_call(name, arguments, get_client())


# ─── Health endpoint ──────────────────────────────────────────────

async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "ok",
        "tools": len(TOOL_DEFINITIONS),
    })


# ─── MCP Streamable HTTP handler ─────────────────────────────────

async def handle_mcp(request: Request):
    """Handle MCP requests using StreamableHTTPServerTransport.

    Each request gets its own transport instance (stateless mode).
    """
    # Create a transport for this request (stateless — no session persistence)
    transport = StreamableHTTPServerTransport(
        mcp_session_id=None,  # Stateless mode
        is_json_response_enabled=True,
    )

    async with transport.connect() as (read_stream, write_stream):
        # Start the MCP server processing in the background
        server_task = asyncio.create_task(
            mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())
        )

        try:
            # Let the transport handle the HTTP request
            await transport.handle_request(
                request.scope, request.receive, request._send
            )
        finally:
            # Clean up: terminate transport and wait for server task
            if not transport.is_terminated:
                await transport.terminate()
            # Give the server task a moment to finish
            try:
                await asyncio.wait_for(server_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass


# ─── Starlette app ────────────────────────────────────────────────

app = Starlette(
    routes=[
        Route("/health", health_check, methods=["GET"]),
        Route("/mcp", handle_mcp, methods=["GET", "POST", "DELETE"]),
    ],
)


def main():
    """Run the HTTP server."""
    host = os.environ.get("MCP_HTTP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_HTTP_PORT", "8080"))

    logger.info("Starting SFU Library MCP HTTP server on %s:%d", host, port)
    logger.info("Health endpoint: GET /health")
    logger.info("MCP endpoint: POST /mcp")
    logger.info("Tools available: %d", len(TOOL_DEFINITIONS))

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
