"""
SFU Library MCP Server
Provides Claude Desktop access to the SFU Library database through the Primo API.
Supports JWT token caching, authenticated searches, and detailed item retrieval.
"""

import asyncio
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from lib.logging_setup import setup_logging
from lib.config import load_config, validate_config
from lib.client import SFULibraryClient
from lib.tools import TOOL_DEFINITIONS, handle_tool_call

# Configure logging to stderr (never stdout — MCP uses stdio JSON-RPC)
config = load_config()
logger = setup_logging(level=config.log_level, log_file=config.log_file)

# Log config warnings at startup
for warning in validate_config(config):
    logger.warning("Config: %s", warning)

# Initialize the MCP server — name MUST stay "sfu-library"
server = Server("sfu-library")

# Global client instance
client = None


def get_client() -> SFULibraryClient:
    """Get or create the library client."""
    global client
    if client is None:
        client = SFULibraryClient(headless=True, config=config)
    return client


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return TOOL_DEFINITIONS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    return await handle_tool_call(name, arguments, get_client())


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
