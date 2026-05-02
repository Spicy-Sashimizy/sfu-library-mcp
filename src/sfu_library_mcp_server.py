"""
SFU Library MCP Server — stdio transport.

Provides Claude Desktop access to open academic APIs (OpenAlex, Semantic Scholar,
SFU Database Registry) with no proprietary API dependencies.
"""

import asyncio
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from lib.logging_setup import setup_logging
from lib.config import load_config, validate_config
from lib.tools import get_tool_definitions, handle_tool_call

# Configure logging to stderr (never stdout — MCP uses stdio JSON-RPC)
config = load_config()
logger = setup_logging(level=config.log_level, log_file=config.log_file)

for warning in validate_config(config):
    logger.warning("Config: %s", warning)

# Initialize the MCP server — name MUST stay "sfu-library"
server = Server("sfu-library")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return await get_tool_definitions()


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    return await handle_tool_call(name, arguments)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
