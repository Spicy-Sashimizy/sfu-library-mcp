#!/usr/bin/env python3
"""
Pommel MCP Server for Claude Code (Container-side)
Queries Pommel daemons running in ClaudeBox containers via shared Docker network.
Supports both local project search and cross-project discovery.
"""

import json
import os
import urllib.request
import urllib.error
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Initialize MCP server
app = Server("pommel-search")

# Current project name (set via environment variable)
CURRENT_PROJECT = os.environ.get("PROJECT_NAME", "unknown")

# All Pommel services accessible via Docker shared network
# Container names follow pattern: claudebox-{project}-pommel
# All listen on port 7420 internally
def get_pommel_projects():
    """Dynamically get known projects - can be extended via env vars."""
    projects = {
        "dashboard": {"container": "claudebox-dashboard-pommel", "description": "ClaudeBox Dashboard"},
        "business-plan-cataloger": {"container": "claudebox-business-plan-cataloger-pommel", "description": "Business plan cataloger"},
        "ccpsandyass-cybermap": {"container": "claudebox-ccpsandyass-cybermap-pommel", "description": "CCPS cybermap"},
        "ccpsandyass-mcp-research": {"container": "claudebox-ccpsandyass-mcp-research-pommel", "description": "CCPS MCP research"},
        "ccpsandyass-scripts": {"container": "claudebox-ccpsandyass-scripts-pommel", "description": "CCPS scripts"},
        "ccpsandyass-separate": {"container": "claudebox-ccpsandyass-separate-pommel", "description": "CCPS separate"},
        "essay-script": {"container": "claudebox-essay-script-pommel", "description": "Essay script"},
        "geo-property-data": {"container": "claudebox-geo-property-data-pommel", "description": "Geo property data"},
        "graph-visualization": {"container": "claudebox-graph-visualization-pommel", "description": "Graph visualization"},
        "llm-api": {"container": "claudebox-llm-api-pommel", "description": "LLM API"},
        "misc-scripts": {"container": "claudebox-misc-scripts-pommel", "description": "Miscellaneous scripts"},
        "sfu-auto-researcher": {"container": "claudebox-sfu-auto-researcher-pommel", "description": "SFU auto researcher"},
        "sfu-library-api": {"container": "claudebox-sfu-library-api-pommel", "description": "SFU library API"},
        "sfu-library-mcp": {"container": "claudebox-sfu-library-mcp-pommel", "description": "SFU library MCP"},
        "website-cataloger": {"container": "claudebox-website-cataloger-pommel", "description": "Website cataloger"},
        "auto-ass-googlecal-populator": {"container": "claudebox-auto-ass-googlecal-populator-pommel", "description": "Auto assignment Google Calendar populator"},
        "auto-ahh-google-drive-ahh": {"container": "claudebox-auto-ahh-google-drive-ahh-pommel", "description": "Auto AHH Google Drive"},
    }

    # Add current project if not in list
    if CURRENT_PROJECT and CURRENT_PROJECT not in projects:
        projects[CURRENT_PROJECT] = {
            "container": f"claudebox-{CURRENT_PROJECT}-pommel",
            "description": f"{CURRENT_PROJECT} project"
        }

    return projects

POMMEL_PROJECTS = get_pommel_projects()


def query_pommel_http(host: str, port: int, query: str, limit: int = 5, level: str = "method") -> dict:
    """Query Pommel API via HTTP."""
    try:
        url = f"http://{host}:{port}/search"
        data = json.dumps({
            "query": query,
            "limit": limit,
            "level": level
        }).encode('utf-8')

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))

    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response: {e}"}
    except Exception as e:
        return {"error": str(e)}


def get_pommel_status_http(host: str, port: int) -> dict:
    """Get Pommel status via HTTP."""
    try:
        url = f"http://{host}:{port}/status"
        req = urllib.request.Request(url, method="GET")

        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode('utf-8'))

    except Exception as e:
        return {"error": str(e)}


def trigger_reindex_http(host: str, port: int) -> dict:
    """Trigger reindex via HTTP API."""
    try:
        url = f"http://{host}:{port}/reindex"
        req = urllib.request.Request(url, method="POST")

        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))

    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available Pommel search tools."""
    return [
        Tool(
            name="pommel_search_project",
            description=(
                "Search code in a specific ClaudeBox project using semantic search. "
                "Returns relevant code snippets with file paths and line numbers. "
                "Use this to find code patterns, implementations, or examples."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name to search in (use pommel_list_projects to see available)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'database connection', 'auth handler')"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "level": {
                        "type": "string",
                        "enum": ["file", "class", "method"],
                        "description": "Granularity level to search at (default: method)",
                        "default": "method"
                    }
                },
                "required": ["project", "query"]
            }
        ),
        Tool(
            name="pommel_search_all",
            description=(
                "Search code across ALL ClaudeBox projects using semantic search. "
                "Returns results from multiple projects. "
                "Use this to find implementations across projects or discover reusable code."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max results per project (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="pommel_search_local",
            description=(
                f"Search code in the CURRENT project ({CURRENT_PROJECT}) using semantic search. "
                "Fastest option when you only need to search the project you're working in."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "level": {
                        "type": "string",
                        "enum": ["file", "class", "method"],
                        "description": "Granularity level (default: method)",
                        "default": "method"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="pommel_list_projects",
            description=(
                "List all ClaudeBox projects that have Pommel indexing enabled. "
                "Shows which projects are available for semantic code search."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="pommel_reindex",
            description=(
                "Trigger a full reindex of a project's codebase. "
                "Use after major code changes or if search results seem outdated."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name to reindex (omit for current project)"
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Execute Pommel search tools."""

    if name == "pommel_search_project":
        return await search_project(arguments)
    elif name == "pommel_search_all":
        return await search_all_projects(arguments)
    elif name == "pommel_search_local":
        return await search_local(arguments)
    elif name == "pommel_list_projects":
        return await list_projects()
    elif name == "pommel_reindex":
        return await reindex_project(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def search_local(args: dict) -> Sequence[TextContent]:
    """Search the current project's local Pommel instance."""
    query = args.get("query")
    limit = args.get("limit", 5)
    level = args.get("level", "method")

    result = query_pommel_http("localhost", 7420, query, limit, level)

    if "error" in result:
        return [TextContent(type="text", text=f"Local search error: {result['error']}")]

    return [TextContent(type="text", text=format_search_results(result, CURRENT_PROJECT, query))]


async def search_project(args: dict) -> Sequence[TextContent]:
    """Search a specific project."""
    project = args.get("project")
    query = args.get("query")
    limit = args.get("limit", 5)
    level = args.get("level", "method")

    # Check if searching current project
    if project == CURRENT_PROJECT or project == "local":
        result = query_pommel_http("localhost", 7420, query, limit, level)
    elif project in POMMEL_PROJECTS:
        # Query via container name on shared network
        container = POMMEL_PROJECTS[project]["container"]
        result = query_pommel_http(container, 7420, query, limit, level)
    else:
        # Try dynamic container name
        container = f"claudebox-{project}-pommel"
        result = query_pommel_http(container, 7420, query, limit, level)

    if "error" in result:
        return [TextContent(type="text", text=f"Search error: {result['error']}")]

    return [TextContent(type="text", text=format_search_results(result, project, query))]


async def search_all_projects(args: dict) -> Sequence[TextContent]:
    """Search across all projects."""
    query = args.get("query")
    limit = args.get("limit", 3)

    all_results = []

    # Search local project first
    local_result = query_pommel_http("localhost", 7420, query, limit)
    if "error" not in local_result:
        all_results.append({
            'project': f"{CURRENT_PROJECT} (local)",
            'results': local_result.get('results', [])
        })

    # Search all other projects via shared network
    for project, info in POMMEL_PROJECTS.items():
        if project == CURRENT_PROJECT:
            continue  # Already searched locally

        result = query_pommel_http(info["container"], 7420, query, limit)
        if "error" not in result:
            all_results.append({
                'project': project,
                'results': result.get('results', [])
            })

    if not all_results:
        return [TextContent(
            type="text",
            text="No Pommel projects are currently available."
        )]

    return [TextContent(type="text", text=format_cross_project_results(all_results, query))]


async def list_projects() -> Sequence[TextContent]:
    """List available projects."""
    output = ["# Available Projects with Pommel Search\n"]
    output.append(f"Current project: **{CURRENT_PROJECT}**\n")

    # Check local project first
    output.append("## Local Project\n")
    local_status = get_pommel_status_http("localhost", 7420)
    if isinstance(local_status, dict) and "error" not in local_status:
        files = local_status.get('files', local_status.get('index', {}).get('files', 'N/A'))
        chunks = local_status.get('chunks', local_status.get('index', {}).get('chunks', 'N/A'))
        output.append(f"- **{CURRENT_PROJECT}** (local): {files} files, {chunks} chunks indexed")
    else:
        error_msg = local_status.get('error', 'unavailable') if isinstance(local_status, dict) else 'unavailable'
        output.append(f"- **{CURRENT_PROJECT}** (local): Not running - {error_msg}")

    # List remote projects
    output.append("\n## Remote Projects (via shared network)\n")
    for project, info in POMMEL_PROJECTS.items():
        if project == CURRENT_PROJECT:
            continue

        status = get_pommel_status_http(info["container"], 7420)

        if isinstance(status, dict) and "error" not in status:
            files = status.get('files', status.get('index', {}).get('files', 'N/A'))
            chunks = status.get('chunks', status.get('index', {}).get('chunks', 'N/A'))
            output.append(f"- **{project}**: {files} files, {chunks} chunks ({info['description']})")
        else:
            output.append(f"- **{project}**: Not running ({info['description']})")

    return [TextContent(type="text", text="\n".join(output))]


async def reindex_project(args: dict) -> Sequence[TextContent]:
    """Trigger reindex for a project."""
    project = args.get("project", CURRENT_PROJECT)

    # Check if reindexing local or remote
    if project == CURRENT_PROJECT or project == "local" or not project:
        result = trigger_reindex_http("localhost", 7420)
        target = f"{CURRENT_PROJECT} (local)"
    elif project in POMMEL_PROJECTS:
        container = POMMEL_PROJECTS[project]["container"]
        result = trigger_reindex_http(container, 7420)
        target = project
    else:
        container = f"claudebox-{project}-pommel"
        result = trigger_reindex_http(container, 7420)
        target = project

    if "error" in result:
        return [TextContent(type="text", text=f"Reindex failed: {result['error']}")]

    return [TextContent(type="text", text=f"Reindex triggered for {target}. This runs in the background.")]


def format_search_results(data: dict, project: str, query: str) -> str:
    """Format search results for display."""
    results = data.get('results', [])

    if not results:
        return f"No results found in {project} for: {query}"

    output = [f"# Search Results from {project}"]
    output.append(f"Query: {query}")
    output.append(f"Found {len(results)} matches\n")

    for i, result in enumerate(results, 1):
        file_path = result.get('file', 'unknown')
        start_line = result.get('start_line', '?')
        end_line = result.get('end_line', '?')
        score = result.get('score', 0)
        name = result.get('name', '')
        level = result.get('level', 'unknown')

        output.append(f"## {i}. {file_path}:{start_line}-{end_line}")
        output.append(f"**Score:** {score:.3f} | **Level:** {level} | **Name:** {name}")

        content = result.get('content', '')
        lang = result.get('language', '')
        if content:
            if len(content) > 500:
                content = content[:500] + "\n... (truncated)"
            output.append(f"```{lang}")
            output.append(content)
            output.append("```\n")

    return "\n".join(output)


def format_cross_project_results(all_results: list[dict], query: str) -> str:
    """Format cross-project search results."""
    if not all_results or all(len(r.get('results', [])) == 0 for r in all_results):
        return f"No results found across projects for: {query}"

    output = ["# Cross-Project Search Results"]
    output.append(f"Query: {query}")
    output.append(f"Searched {len(all_results)} projects\n")

    for project_result in all_results:
        project = project_result['project']
        results = project_result.get('results', [])

        if not results:
            continue

        output.append(f"## From: {project}")
        output.append(f"Found {len(results)} matches\n")

        for i, result in enumerate(results[:5], 1):
            file_path = result.get('file', 'unknown')
            start_line = result.get('start_line', '?')
            score = result.get('score', 0)

            output.append(f"### {i}. {file_path}:{start_line} (score: {score:.3f})")

            content = result.get('content', '')[:300]
            lang = result.get('language', '')
            if content:
                output.append(f"```{lang}")
                output.append(content + "...")
                output.append("```\n")

    return "\n".join(output)


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
