"""MCP tool definitions and handler dispatch."""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from mcp.types import Tool, TextContent

from lib.citations import (
    extract_metadata,
    extract_full_text_links,
    enrich_metadata_from_crossref,
    format_apa_citation,
    format_mla_citation,
    format_chicago_citation,
    format_bibtex_entry,
    format_ris_entry,
)
from lib.config import load_config
from lib.formatters import format_search_results, format_item_details
from lib.reranker import rerank_results
from lib.validators import sanitize_search_query, validate_isbn
from lib.zotero import ZoteroClient, ZoteroError

logger = logging.getLogger("sfu_library_mcp")


# Semaphore to limit concurrent API requests
_request_semaphore = asyncio.Semaphore(8)

# Lazy-loaded config for feature flags
_config = None


def _get_config() -> "ServerConfig":
    """Get config (lazy-loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_features() -> dict[str, bool]:
    """Get feature flags from config (lazy-loaded)."""
    return _get_config().features


# Lazy-loaded zotero client
_zotero_client: ZoteroClient | None = None


def _get_zotero_client() -> ZoteroClient:
    """Get or create ZoteroClient."""
    global _zotero_client
    if _zotero_client is None:
        _zotero_client = ZoteroClient(_get_config())
    return _zotero_client


def _ensure_zotero_auth() -> list[TextContent] | None:
    """Zotero-only auth pre-flight check. Returns error response or None if OK."""
    features = _get_features()
    if not features.get("zotero_enabled", True):
        return [TextContent(type="text", text="Zotero integration is disabled.")]
    try:
        zot = _get_zotero_client()
        if not zot.ensure_authenticated():
            return [TextContent(type="text", text=(
                "Zotero authentication failed. Check SFU_ZOTERO_API_KEY and "
                "SFU_ZOTERO_USER_ID."
            ))]
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero auth error: {e}")]
    return None


# Simple metrics counters
_metrics: dict[str, dict[str, Any]] = {}

# Search result cache indexed by record ID.
# Populated by every search call so that generate_citation / get_full_text_links
# can fall back to cached PNX data when get_item_details fails for CDI records.
_record_cache: dict[str, dict] = {}


def _record_metric(tool_name: str, latency: float, success: bool) -> None:
    """Record a metric for a tool call."""
    if tool_name not in _metrics:
        _metrics[tool_name] = {"count": 0, "errors": 0, "total_latency": 0.0}
    _metrics[tool_name]["count"] += 1
    _metrics[tool_name]["total_latency"] += latency
    if not success:
        _metrics[tool_name]["errors"] += 1
    logger.debug(
        "Metric: %s count=%d latency=%.3fs",
        tool_name,
        _metrics[tool_name]["count"],
        latency,
    )


def get_metrics() -> dict:
    """Return current metrics snapshot."""
    return dict(_metrics)


def _cache_search_docs(docs: list[dict]) -> None:
    """Cache docs from search results by record ID."""
    for doc in docs:
        pnx = doc.get("pnx", {})
        control = pnx.get("control", {})
        record_id = control.get("recordid", [""])[0] if control.get("recordid") else ""
        if record_id:
            _record_cache[record_id] = doc
    # Cap cache to prevent unbounded growth
    if len(_record_cache) > 500:
        keys = list(_record_cache.keys())
        for k in keys[:len(keys) - 500]:
            del _record_cache[k]


def _lookup_cached_record(record_id: str) -> dict | None:
    """Look up a previously searched record from cache."""
    return _record_cache.get(record_id)


# ─── Tool definitions ───────────────────────────────────────────

TOOL_DEFINITIONS: list[Tool] = [
    Tool(
        name="search_library",
        description=(
            "Search the SFU Library database for books, articles, journals, and other academic resources. "
            "Returns titles, authors, dates, subjects, availability, and record IDs.\n\n"
            "QUERY SYNTAX:\n"
            "- Boolean operators: AND, OR, NOT (MUST be uppercase). Example: '\"CRISPR\" AND \"sickle cell\"'\n"
            "- Phrase search: wrap exact phrases in double quotes. Example: '\"machine learning\"'\n"
            "- Wildcards: ? (single char), * (multiple chars). Example: 'cultur*' matches culture, cultures, cultural\n\n"
            "SEARCH STRATEGY:\n"
            "- For comprehensive results on complex topics, make multiple calls with different field/scope combinations\n"
            "- Use field='sub' for controlled subject vocabulary (most precise for topic searches)\n"
            "- Use field='title' for known work titles\n"
            "- Use field='any' for broad discovery when unsure\n"
            "- Use resource_type='electronic' when user needs immediate online access\n"
            "- Use sort='date' for recent publications, sort='rank' for best relevance\n"
            "- Results include subject headings from the library's controlled vocabulary — use these for follow-up searches"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Supports boolean operators (AND, OR, NOT uppercase), phrase search (\"quoted\"), and wildcards (?, *)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10, max: 50)",
                    "default": 10
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting offset for pagination (default: 0)",
                    "default": 0
                },
                "field": {
                    "type": "string",
                    "description": "Field to search in: 'any' (all fields, broad), 'title' (known works), 'creator' (author), 'sub' (subject headings, most precise), 'isbn', 'issn'",
                    "enum": ["any", "title", "creator", "sub", "isbn", "issn"],
                    "default": "any"
                },
                "sort": {
                    "type": "string",
                    "description": "Sort order: 'rank' (relevance), 'date' (newest first), 'author', 'title'",
                    "enum": ["rank", "date", "author", "title"],
                    "default": "rank"
                },
                "resource_type": {
                    "type": "string",
                    "description": "Type of resources: 'all' (everything), 'electronic' (online only — use when user needs immediate access), 'courses' (course reserves)",
                    "enum": ["all", "electronic", "courses"],
                    "default": "all"
                },
                "expanded_terms": {
                    "type": "string",
                    "description": (
                        "Optional: additional search terms to OR with the main query. "
                        "Generate academic synonyms/related terms. "
                        "Example: for query 'machine learning', expanded_terms might be "
                        "'deep learning OR neural networks OR artificial intelligence'. "
                        "Server will construct: (query) OR (expanded_terms)"
                    )
                },
                "comprehensive": {
                    "type": "boolean",
                    "description": (
                        "Enable comprehensive search: runs parallel searches across "
                        "multiple scopes (general, electronic, subject-focused) and "
                        "merges results using rank fusion. Use for broad research questions. "
                        "Costs 3-4x API calls but provides much broader coverage."
                    ),
                    "default": False
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="get_item_details",
        description="Get detailed information about a specific library item using its record ID. Returns full metadata including description, subjects, availability, and access links.",
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "The record ID of the item (obtained from search results)"
                }
            },
            "required": ["record_id"]
        }
    ),
    Tool(
        name="search_by_author",
        description=(
            "Search for works by a specific author in the SFU Library. "
            "For best results use 'LastName, FirstName' format. "
            "Combine with search_library (field='sub') or search_by_subject to find an author's works on a specific topic."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "author": {
                    "type": "string",
                    "description": "Author name to search for (best: 'LastName, FirstName')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10
                }
            },
            "required": ["author"]
        }
    ),
    Tool(
        name="search_by_subject",
        description=(
            "Search for resources by subject heading in the SFU Library. "
            "Uses the library's controlled vocabulary (LCSH). "
            "Check subject headings returned in search results for the exact vocabulary to use. "
            "For broader discovery, combine with search_library (field='any') or search_electronic_resources."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Subject heading to search for (use exact terms from result subjects when possible)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10
                }
            },
            "required": ["subject"]
        }
    ),
    Tool(
        name="search_by_isbn",
        description="Look up a specific book by its ISBN number.",
        inputSchema={
            "type": "object",
            "properties": {
                "isbn": {
                    "type": "string",
                    "description": "ISBN number (10 or 13 digits)"
                }
            },
            "required": ["isbn"]
        }
    ),
    Tool(
        name="search_electronic_resources",
        description="Search specifically for electronic/online resources available through SFU Library (e-books, online journals, databases).",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="get_full_text_links",
        description="Extract all full-text access URLs (HTML, PDF, DOI) for a specific library item. Use this to get direct links to articles and documents.",
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "The record ID of the item (obtained from search results)"
                }
            },
            "required": ["record_id"]
        }
    ),
    Tool(
        name="generate_citation",
        description="Generate a citation for a library item in various formats (APA, MLA, Chicago, BibTeX).",
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "The record ID of the item to cite"
                },
                "format": {
                    "type": "string",
                    "description": "Citation format: 'apa' (APA 7th), 'mla' (MLA 9th), 'chicago' (Chicago 17th), 'bibtex'",
                    "enum": ["apa", "mla", "chicago", "bibtex"],
                    "default": "apa"
                }
            },
            "required": ["record_id"]
        }
    ),
    Tool(
        name="batch_generate_citations",
        description="Generate citations for multiple library items at once. Returns all citations in the specified format.",
        inputSchema={
            "type": "object",
            "properties": {
                "record_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of record IDs to generate citations for (max 20)"
                },
                "format": {
                    "type": "string",
                    "description": "Citation format: 'apa', 'mla', 'chicago', 'bibtex'",
                    "enum": ["apa", "mla", "chicago", "bibtex"],
                    "default": "apa"
                }
            },
            "required": ["record_ids"]
        }
    ),
    Tool(
        name="export_search_results",
        description="Search and export results in various formats (JSON, CSV, BibTeX, RIS). Useful for importing into reference managers or spreadsheets.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "format": {
                    "type": "string",
                    "description": "Export format: 'json', 'csv', 'bibtex', 'ris' (EndNote/Zotero compatible)",
                    "enum": ["json", "csv", "bibtex", "ris"],
                    "default": "bibtex"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to export (default: 10, max: 100)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="batch_isbn_lookup",
        description="Look up multiple books by their ISBN numbers in a single request. Returns basic info and availability for each.",
        inputSchema={
            "type": "object",
            "properties": {
                "isbn_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ISBN numbers to look up (max 20)"
                }
            },
            "required": ["isbn_list"]
        }
    ),
    Tool(
        name="save_to_zotero",
        description=(
            "Save a library item's metadata to the user's Zotero library. "
            "Automatically checks for duplicates before saving. "
            "Optionally specify a collection name to organize the item."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "The record ID of the item (obtained from search results)"
                },
                "collection_name": {
                    "type": "string",
                    "description": "Optional: Zotero collection name to add the item to (created if it doesn't exist)"
                },
                "parent_collection": {
                    "type": "string",
                    "description": "Optional: parent collection name. When specified, the item's collection becomes a subcollection under this parent."
                },
            },
            "required": ["record_id"]
        }
    ),
    Tool(
        name="list_zotero_collections",
        description="List all collections in the user's Zotero library with item counts.",
        inputSchema={
            "type": "object",
            "properties": {}
        }
    ),
    Tool(
        name="batch_save_to_zotero",
        description=(
            "Save multiple library items to a Zotero collection at once. "
            "Checks each item for duplicates and skips items already in the library. "
            "Returns a summary showing how many were saved, skipped, or failed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "record_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of record IDs to save (max 20)"
                },
                "collection_name": {
                    "type": "string",
                    "description": "Zotero collection name to add items to (created if it doesn't exist)"
                },
                "parent_collection": {
                    "type": "string",
                    "description": "Optional: parent collection name. When specified, the collection becomes a subcollection under this parent."
                },
            },
            "required": ["record_ids", "collection_name"]
        }
    ),
    Tool(
        name="search_zotero",
        description=(
            "Search the user's existing Zotero library. Use this BEFORE saving to check for "
            "duplicates, and to answer questions about what the user already has saved. "
            "Returns titles, authors, dates, types, DOIs, collections, and tags."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find items in the Zotero library"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 20, max: 100)",
                    "default": 20
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="get_zotero_collection_items",
        description=(
            "List all items in a specific Zotero collection. Use to browse what's already "
            "saved in a collection or to help the user review their saved research."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Name of the Zotero collection to list items from"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of items to return (default: 50)",
                    "default": 50
                }
            },
            "required": ["collection_name"]
        }
    ),
    Tool(
        name="get_zotero_status",
        description=(
            "Check Zotero API connection, credentials, and permissions."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# ─── Shared record resolution ─────────────────────────────────

async def _resolve_record(record_id: str, client) -> dict | None:
    """Resolve a record by ID using 3-tier fallback: API → cache → search."""
    async with _request_semaphore:
        item = client.get_item_details(record_id)
    if item:
        return item

    item = _lookup_cached_record(record_id)
    if item:
        return item

    async with _request_semaphore:
        results = client.search(query=record_id, limit=1)
    if results and results.get("docs"):
        return results["docs"][0]

    return None


# ─── Tool handler dispatch ──────────────────────────────────────

async def handle_tool_call(
    name: str,
    arguments: dict[str, Any],
    lib_client,
) -> list[TextContent]:
    """Handle a tool call by dispatching to the appropriate handler."""
    start_time = time.time()
    success = True

    try:
        result = await _dispatch_tool(name, arguments, lib_client)
        return result
    except Exception as e:
        success = False
        logger.error("Tool %s failed: %s", name, e)
        return [TextContent(type="text", text=f"Error in {name}: {str(e)}")]
    finally:
        latency = time.time() - start_time
        _record_metric(name, latency, success)


async def _dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    lib_client,
) -> list[TextContent]:
    """Route tool call to the correct handler."""

    if name == "search_library":
        return await _handle_search_library(arguments, lib_client)
    elif name == "get_item_details":
        return await _handle_get_item_details(arguments, lib_client)
    elif name == "search_by_author":
        return await _handle_search_by_author(arguments, lib_client)
    elif name == "search_by_subject":
        return await _handle_search_by_subject(arguments, lib_client)
    elif name == "search_by_isbn":
        return await _handle_search_by_isbn(arguments, lib_client)
    elif name == "search_electronic_resources":
        return await _handle_search_electronic(arguments, lib_client)
    elif name == "get_full_text_links":
        return await _handle_get_full_text_links(arguments, lib_client)
    elif name == "generate_citation":
        return await _handle_generate_citation(arguments, lib_client)
    elif name == "batch_generate_citations":
        return await _handle_batch_citations(arguments, lib_client)
    elif name == "export_search_results":
        return await _handle_export_search(arguments, lib_client)
    elif name == "batch_isbn_lookup":
        return await _handle_batch_isbn(arguments, lib_client)
    elif name == "save_to_zotero":
        return await _handle_save_to_zotero(arguments, lib_client)
    elif name == "list_zotero_collections":
        return _handle_list_zotero_collections()
    elif name == "batch_save_to_zotero":
        return await _handle_batch_save_to_zotero(arguments, lib_client)
    elif name == "search_zotero":
        return _handle_search_zotero(arguments)
    elif name == "get_zotero_collection_items":
        return _handle_get_zotero_collection_items(arguments)
    elif name == "get_zotero_status":
        return _handle_get_zotero_status()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ─── Individual tool handlers ───────────────────────────────────

async def _single_search(client, query: str, limit: int, offset: int,
                         field: str, sort: str, tab: str, scope: str) -> dict | None:
    """Execute a single search against the Primo API."""
    async with _request_semaphore:
        return client.search(
            query=query, limit=limit, offset=offset,
            field=field, sort=sort, tab=tab, scope=scope,
        )


def _reciprocal_rank_fusion(results_sets: list[dict | None], limit: int, k: int = 60) -> dict:
    """Merge multiple result sets using Reciprocal Rank Fusion.

    RRF formula: score(d) = sum(1 / (k + rank_i)) for each result set.
    Deduplicates by record ID.

    Args:
        results_sets: List of Primo API response dicts (may contain None).
        limit: Maximum number of merged results to return.
        k: RRF constant (default 60, standard value).

    Returns:
        Merged results dict with docs and info.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    total_results = 0

    for result_set in results_sets:
        if not result_set or not result_set.get("docs"):
            continue
        total_results = max(total_results, result_set.get("info", {}).get("total", 0))
        for rank, doc in enumerate(result_set["docs"]):
            pnx = doc.get("pnx", {})
            record_id = pnx.get("control", {}).get("recordid", [""])[0] if pnx.get("control", {}).get("recordid") else ""
            if not record_id:
                record_id = f"_fallback_{pnx.get('display', {}).get('title', [''])[0][:50]}"
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (k + rank)
            if record_id not in doc_map:
                doc_map[record_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda rid: scores[rid], reverse=True)
    merged_docs = [doc_map[rid] for rid in sorted_ids[:limit]]

    return {
        "docs": merged_docs,
        "info": {"total": total_results, "first": 0, "last": len(merged_docs) - 1},
    }


async def _handle_search_library(args: dict, client) -> list[TextContent]:
    query = args.get("query", "")
    limit = min(args.get("limit", 10), 50)
    offset = args.get("offset", 0)
    field = args.get("field", "any")
    sort = args.get("sort", "rank")
    resource_type = args.get("resource_type", "all")
    expanded_terms = args.get("expanded_terms", "")
    comprehensive = args.get("comprehensive", False)

    # Construct combined boolean query if expanded_terms provided
    if expanded_terms and expanded_terms.strip():
        search_query = f"({query}) OR ({expanded_terms.strip()})"
    else:
        search_query = query

    tab = "default_tab"
    scope = "default_scope"
    if resource_type == "electronic":
        tab = "online_only_tab"
        scope = "ElectronicOnly_scope"
    elif resource_type == "courses":
        tab = "course_tab"
        scope = "course_scope"

    features = _get_features()

    # Determine how many results to fetch (over-fetch for re-ranking)
    rerank_enabled = features.get("rerank_enabled", False)
    fetch_limit = min(limit * 3, 50) if rerank_enabled else limit

    # Fusion retrieval: parallel searches across multiple scopes
    fusion_enabled = features.get("fusion_enabled", False)
    if comprehensive and fusion_enabled:
        search_tasks = [
            _single_search(client, search_query, fetch_limit, offset, field, sort, tab, scope),
            _single_search(client, search_query, fetch_limit, offset, "sub", sort, "default_tab", "default_scope"),
            _single_search(client, search_query, fetch_limit, offset, field, sort, "online_only_tab", "ElectronicOnly_scope"),
        ]
        results_sets = await asyncio.gather(*search_tasks, return_exceptions=True)
        valid_results = [r if not isinstance(r, Exception) else None for r in results_sets]
        results = _reciprocal_rank_fusion(valid_results, fetch_limit)
    else:
        results = await _single_search(client, search_query, fetch_limit, offset, field, sort, tab, scope)

    # Cache all returned docs by record ID
    if results and results.get("docs"):
        _cache_search_docs(results["docs"])

    # Re-rank results if enabled
    if rerank_enabled and results and results.get("docs"):
        results["docs"] = rerank_results(results["docs"], query, limit)
    elif results and results.get("docs") and len(results["docs"]) > limit:
        results["docs"] = results["docs"][:limit]

    search_metadata = {"query": query, "field": field, "sort": sort, "resource_type": resource_type}
    if comprehensive and fusion_enabled:
        search_metadata["comprehensive"] = True
    formatted = format_search_results(results, metadata=search_metadata)
    return [TextContent(type="text", text=formatted)]


async def _handle_get_item_details(args: dict, client) -> list[TextContent]:
    record_id = args.get("record_id", "")
    async with _request_semaphore:
        item = client.get_item_details(record_id)
    formatted = format_item_details(item)
    return [TextContent(type="text", text=formatted)]


async def _handle_search_by_author(args: dict, client) -> list[TextContent]:
    author = args.get("author", "")
    limit = args.get("limit", 10)
    async with _request_semaphore:
        results = client.search(query=author, limit=limit, field="creator")
    if results and results.get("docs"):
        _cache_search_docs(results["docs"])
    search_metadata = {"query": author, "field": "creator", "sort": "rank"}
    formatted = format_search_results(results, metadata=search_metadata)
    return [TextContent(type="text", text=formatted)]


async def _handle_search_by_subject(args: dict, client) -> list[TextContent]:
    subject = args.get("subject", "")
    limit = args.get("limit", 10)
    async with _request_semaphore:
        results = client.search(query=subject, limit=limit, field="sub")
    if results and results.get("docs"):
        _cache_search_docs(results["docs"])
    search_metadata = {"query": subject, "field": "sub", "sort": "rank"}
    formatted = format_search_results(results, metadata=search_metadata)
    return [TextContent(type="text", text=formatted)]


async def _handle_search_by_isbn(args: dict, client) -> list[TextContent]:
    isbn = args.get("isbn", "")
    async with _request_semaphore:
        results = client.search(query=isbn, limit=5, field="isbn")
    if results and results.get("docs"):
        _cache_search_docs(results["docs"])
    search_metadata = {"query": isbn, "field": "isbn", "sort": "rank"}
    formatted = format_search_results(results, metadata=search_metadata)
    return [TextContent(type="text", text=formatted)]


async def _handle_search_electronic(args: dict, client) -> list[TextContent]:
    query = args.get("query", "")
    limit = args.get("limit", 10)
    async with _request_semaphore:
        results = client.search(
            query=query, limit=limit,
            tab="online_only_tab", scope="ElectronicOnly_scope",
        )
    if results and results.get("docs"):
        _cache_search_docs(results["docs"])
    search_metadata = {"query": query, "field": "any", "sort": "rank", "resource_type": "electronic"}
    formatted = format_search_results(results, metadata=search_metadata)
    return [TextContent(type="text", text=formatted)]


async def _handle_get_full_text_links(args: dict, client) -> list[TextContent]:
    record_id = args.get("record_id", "")

    async with _request_semaphore:
        item = client.get_item_details(record_id)
    if not item:
        item = _lookup_cached_record(record_id)
    if not item:
        async with _request_semaphore:
            results = client.search(query=record_id, limit=1)
        if results and results.get("docs"):
            item = results["docs"][0]

    if not item:
        return [TextContent(type="text", text=f"Could not find item with record ID: {record_id}")]

    links = extract_full_text_links(item)
    if not links:
        return [TextContent(type="text", text="No access links found for this item.")]

    output = ["=" * 50, "FULL TEXT ACCESS LINKS", "=" * 50]
    if links["doi_url"]:
        output.append(f"\nDOI: {links['doi_url']}")
    if links["open_access"]:
        output.append("\nOpen Access: Yes")
    if links["html_links"]:
        output.append(f"\nHTML Links ({len(links['html_links'])}):")
        for link in links["html_links"][:5]:
            output.append(f"  - {link}")
    if links["pdf_links"]:
        output.append(f"\nPDF Links ({len(links['pdf_links'])}):")
        for link in links["pdf_links"][:5]:
            output.append(f"  - {link}")
    if links["source_links"]:
        output.append(f"\nSource Links ({len(links['source_links'])}):")
        for link in links["source_links"][:5]:
            output.append(f"  - {link}")
    if not any([links["html_links"], links["pdf_links"], links["source_links"], links["doi_url"]]):
        output.append("\nNo direct access links available for this item.")
        output.append("Try searching for it on the library website.")

    return [TextContent(type="text", text="\n".join(output))]


def _format_single_citation(metadata: dict | None, fmt: str) -> str:
    """Format a single citation in the given format."""
    formatters = {
        "apa": format_apa_citation,
        "mla": format_mla_citation,
        "chicago": format_chicago_citation,
        "bibtex": format_bibtex_entry,
    }
    formatter = formatters.get(fmt, format_apa_citation)
    return formatter(metadata)


async def _handle_generate_citation(args: dict, client) -> list[TextContent]:
    record_id = args.get("record_id", "")
    citation_format = args.get("format", "apa").lower()

    async with _request_semaphore:
        item = client.get_item_details(record_id)
    if not item:
        item = _lookup_cached_record(record_id)
    if not item:
        async with _request_semaphore:
            results = client.search(query=record_id, limit=1)
        if results and results.get("docs"):
            item = results["docs"][0]

    if not item:
        return [TextContent(type="text", text=f"Could not find item with record ID: {record_id}")]

    metadata = extract_metadata(item)
    metadata = enrich_metadata_from_crossref(metadata)

    format_names = {
        "apa": "APA 7th Edition",
        "mla": "MLA 9th Edition",
        "chicago": "Chicago 17th Edition",
        "bibtex": "BibTeX",
    }
    format_name = format_names.get(citation_format, "APA 7th Edition (default)")
    citation = _format_single_citation(metadata, citation_format)

    return [TextContent(type="text", text=f"--- {format_name} ---\n\n{citation}")]


async def _handle_batch_citations(args: dict, client) -> list[TextContent]:
    record_ids = args.get("record_ids", [])[:20]
    citation_format = args.get("format", "apa").lower()

    if not record_ids:
        return [TextContent(type="text", text="No record IDs provided.")]

    format_names = {
        "apa": "APA 7th Edition",
        "mla": "MLA 9th Edition",
        "chicago": "Chicago 17th Edition",
        "bibtex": "BibTeX",
    }
    format_name = format_names.get(citation_format, "APA 7th Edition")

    # Fetch items concurrently
    async def fetch_item(rid: str) -> tuple[str, dict | None]:
        async with _request_semaphore:
            item = client.get_item_details(rid)
        if not item:
            item = _lookup_cached_record(rid)
        if not item:
            async with _request_semaphore:
                results = client.search(query=rid, limit=1)
            if results and results.get("docs"):
                item = results["docs"][0]
        return rid, item

    tasks = [fetch_item(rid) for rid in record_ids]
    fetched = await asyncio.gather(*tasks, return_exceptions=True)

    output = [f"--- {format_name} Citations ({len(record_ids)} items) ---\n"]
    for i, result in enumerate(fetched, 1):
        if isinstance(result, Exception):
            output.append(f"{i}. [Error: {result}]\n")
            continue
        rid, item = result
        if not item:
            output.append(f"{i}. [Error: Could not find record {rid}]\n")
            continue
        metadata = extract_metadata(item)
        metadata = enrich_metadata_from_crossref(metadata)
        citation = _format_single_citation(metadata, citation_format)
        if citation_format == "bibtex":
            output.append(f"{citation}\n")
        else:
            output.append(f"{i}. {citation}\n")

    return [TextContent(type="text", text="\n".join(output))]


async def _handle_export_search(args: dict, client) -> list[TextContent]:
    query = args.get("query", "")
    export_format = args.get("format", "bibtex").lower()
    limit = min(args.get("limit", 10), 100)

    async with _request_semaphore:
        results = client.search(query=query, limit=limit)
    if not results or not results.get("docs"):
        return [TextContent(type="text", text="No results found to export.")]

    docs = results.get("docs", [])
    total = results.get("info", {}).get("total", 0)
    _cache_search_docs(docs)

    if export_format == "json":
        export_data = []
        for doc in docs:
            pnx = doc.get("pnx", {})
            display = pnx.get("display", {})
            addata = pnx.get("addata", {})
            control = pnx.get("control", {})
            export_data.append({
                "record_id": control.get("recordid", [""])[0] if control.get("recordid") else "",
                "title": display.get("title", [""])[0],
                "authors": display.get("creator", []),
                "date": display.get("creationdate", [""])[0],
                "type": display.get("type", [""])[0],
                "publisher": display.get("publisher", [""])[0],
                "isbn": addata.get("isbn", [""])[0] if addata.get("isbn") else "",
                "issn": addata.get("issn", [""])[0] if addata.get("issn") else "",
                "doi": addata.get("doi", [""])[0] if addata.get("doi") else "",
            })
        output = json.dumps(export_data, indent=2)
        header = f"// JSON Export: {len(docs)} of {total:,} results for '{query}'\n\n"
        return [TextContent(type="text", text=header + output)]

    elif export_format == "csv":
        lines = ["record_id,title,authors,date,type,publisher,isbn,issn,doi"]
        for doc in docs:
            pnx = doc.get("pnx", {})
            display = pnx.get("display", {})
            addata = pnx.get("addata", {})
            control = pnx.get("control", {})

            def escape_csv(val):
                if not val:
                    return ""
                val = str(val).replace('"', '""')
                if "," in val or '"' in val or "\n" in val:
                    return f'"{val}"'
                return val

            row = [
                escape_csv(control.get("recordid", [""])[0] if control.get("recordid") else ""),
                escape_csv(display.get("title", [""])[0]),
                escape_csv("; ".join(display.get("creator", []))),
                escape_csv(display.get("creationdate", [""])[0]),
                escape_csv(display.get("type", [""])[0]),
                escape_csv(display.get("publisher", [""])[0]),
                escape_csv(addata.get("isbn", [""])[0] if addata.get("isbn") else ""),
                escape_csv(addata.get("issn", [""])[0] if addata.get("issn") else ""),
                escape_csv(addata.get("doi", [""])[0] if addata.get("doi") else ""),
            ]
            lines.append(",".join(row))

        header = f"# CSV Export: {len(docs)} of {total:,} results for '{query}'\n"
        return [TextContent(type="text", text=header + "\n".join(lines))]

    elif export_format == "bibtex":
        entries = [format_bibtex_entry(enrich_metadata_from_crossref(extract_metadata(doc))) for doc in docs]
        header = f"% BibTeX Export: {len(docs)} of {total:,} results for '{query}'\n\n"
        return [TextContent(type="text", text=header + "\n\n".join(entries))]

    elif export_format == "ris":
        entries = [format_ris_entry(enrich_metadata_from_crossref(extract_metadata(doc))) for doc in docs]
        header = f"# RIS Export: {len(docs)} of {total:,} results for '{query}'\n\n"
        return [TextContent(type="text", text=header + "\n\n".join(entries))]

    else:
        return [TextContent(type="text", text=f"Unknown export format: {export_format}")]


async def _handle_batch_isbn(args: dict, client) -> list[TextContent]:
    isbn_list = args.get("isbn_list", [])[:20]

    if not isbn_list:
        return [TextContent(type="text", text="No ISBN numbers provided.")]

    output = ["=" * 50, f"BATCH ISBN LOOKUP ({len(isbn_list)} items)", "=" * 50 + "\n"]

    # Concurrent ISBN lookups
    async def lookup_isbn(isbn: str) -> tuple[str, dict | None]:
        isbn_clean = isbn.replace("-", "").replace(" ", "")
        async with _request_semaphore:
            results = client.search(query=isbn_clean, limit=1, field="isbn")
        if results and results.get("docs"):
            _cache_search_docs(results["docs"])
            return isbn, results["docs"][0]
        return isbn, None

    tasks = [lookup_isbn(isbn) for isbn in isbn_list]
    fetched = await asyncio.gather(*tasks, return_exceptions=True)

    found_count = 0
    not_found_count = 0

    for result in fetched:
        if isinstance(result, Exception):
            output.append(f"ISBN: [Error: {result}]\n")
            not_found_count += 1
            continue

        isbn, doc = result
        if doc:
            pnx = doc.get("pnx", {})
            display = pnx.get("display", {})
            delivery = pnx.get("delivery", {})

            title = display.get("title", ["No title"])[0][:60]
            authors = display.get("creator", ["Unknown"])
            author = authors[0].split("$$")[0] if authors else "Unknown"
            date = display.get("creationdate", ["N/A"])[0]
            availability = delivery.get("availability", ["Unknown"])[0] if delivery.get("availability") else "Unknown"

            output.append(f"ISBN: {isbn}")
            output.append(f"  Status: FOUND")
            output.append(f"  Title: {title}")
            output.append(f"  Author: {author}")
            output.append(f"  Date: {date}")
            output.append(f"  Availability: {availability}")
            output.append("")
            found_count += 1
        else:
            output.append(f"ISBN: {isbn}")
            output.append(f"  Status: NOT FOUND")
            output.append("")
            not_found_count += 1

    output.append("-" * 50)
    output.append(f"Summary: {found_count} found, {not_found_count} not found")

    return [TextContent(type="text", text="\n".join(output))]


# ─── Zotero handlers ──────────────────────────────────────────

async def _handle_save_to_zotero(args: dict, client) -> list[TextContent]:
    zotero_auth_err = _ensure_zotero_auth()
    if zotero_auth_err:
        return zotero_auth_err

    record_id = args.get("record_id", "")
    collection_name = args.get("collection_name", "")
    parent_collection = args.get("parent_collection", "")

    item = await _resolve_record(record_id, client)
    if not item:
        return [TextContent(type="text", text=f"Could not find item with record ID: {record_id}")]

    metadata = extract_metadata(item)
    if not metadata:
        return [TextContent(type="text", text=f"Could not extract metadata for record {record_id}")]

    metadata = enrich_metadata_from_crossref(metadata)

    zot_client = _get_zotero_client()

    # Duplicate check
    try:
        dup = zot_client.check_duplicate(metadata)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error during duplicate check: {e}")]

    if dup["is_duplicate"]:
        output = [
            "=" * 50,
            "ALREADY IN ZOTERO",
            "=" * 50,
            f"\nThis item is already in your Zotero library ({dup['match_type']} match).",
            f"\nExisting item:",
            dup["existing_item_summary"] or "N/A",
        ]
        return [TextContent(type="text", text="\n".join(output))]

    # Map metadata → Zotero item
    zotero_item = zot_client.metadata_to_zotero_item(metadata)

    # Resolve parent collection first, then child collection
    collection_key = None
    if collection_name:
        try:
            parent_key = None
            if parent_collection:
                parent_key = zot_client.find_or_create_collection(parent_collection)
            collection_key = zot_client.find_or_create_collection(collection_name, parent_key=parent_key)
        except ZoteroError as e:
            return [TextContent(type="text", text=f"Zotero collection error: {e}")]

    # Create item
    try:
        item_key = zot_client.create_item(zotero_item, collection_key)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Failed to create Zotero item: {e}")]

    output = [
        "=" * 50,
        "SAVED TO ZOTERO",
        "=" * 50,
        f"\nTitle: {metadata.get('title', 'Unknown')}",
        f"Zotero Key: {item_key}",
    ]
    if collection_name:
        output.append(f"Collection: {collection_name}")

    return [TextContent(type="text", text="\n".join(output))]


def _handle_list_zotero_collections() -> list[TextContent]:
    auth_err = _ensure_zotero_auth()
    if auth_err:
        return auth_err

    zot_client = _get_zotero_client()

    try:
        collections = zot_client.list_collections()
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error: {e}")]

    if not collections:
        return [TextContent(type="text", text="No collections found in your Zotero library.")]

    output = ["=" * 50, f"ZOTERO COLLECTIONS ({len(collections)})", "=" * 50, ""]
    for c in collections:
        parent = f" (in: {c['parent_key']})" if c.get("parent_key") else ""
        output.append(f"  {c['name']} — {c['num_items']} items{parent}")
        output.append(f"    Key: {c['key']}")

    return [TextContent(type="text", text="\n".join(output))]


async def _handle_batch_save_to_zotero(args: dict, client) -> list[TextContent]:
    zotero_auth_err = _ensure_zotero_auth()
    if zotero_auth_err:
        return zotero_auth_err

    record_ids = args.get("record_ids", [])[:20]
    collection_name = args.get("collection_name", "")
    parent_collection = args.get("parent_collection", "")

    if not record_ids:
        return [TextContent(type="text", text="No record IDs provided.")]

    zot_client = _get_zotero_client()

    # Resolve parent collection first, then child collection
    collection_key = None
    if collection_name:
        try:
            parent_key = None
            if parent_collection:
                parent_key = zot_client.find_or_create_collection(parent_collection)
            collection_key = zot_client.find_or_create_collection(collection_name, parent_key=parent_key)
        except ZoteroError as e:
            return [TextContent(type="text", text=f"Zotero collection error: {e}")]

    saved = 0
    skipped = 0
    failed = 0
    details = []

    for rid in record_ids:
        item = await _resolve_record(rid, client)
        if not item:
            details.append(f"  {rid}: FAILED — record not found")
            failed += 1
            continue

        metadata = extract_metadata(item)
        if not metadata:
            details.append(f"  {rid}: FAILED — no metadata")
            failed += 1
            continue

        metadata = enrich_metadata_from_crossref(metadata)
        title_short = metadata.get("title", "Unknown")[:60]

        # Duplicate check
        try:
            dup = zot_client.check_duplicate(metadata)
        except ZoteroError:
            dup = {"is_duplicate": False}

        if dup["is_duplicate"]:
            details.append(f"  {title_short}: SKIPPED — already in Zotero ({dup['match_type']} match)")
            skipped += 1
            continue

        # Create item
        zotero_item = zot_client.metadata_to_zotero_item(metadata)
        try:
            item_key = zot_client.create_item(zotero_item, collection_key)
        except ZoteroError as e:
            details.append(f"  {title_short}: FAILED — {e}")
            failed += 1
            continue

        details.append(f"  {title_short}: SAVED")
        saved += 1

    output = [
        "=" * 50,
        "BATCH SAVE TO ZOTERO",
        "=" * 50,
        f"\nCollection: {collection_name or '(none)'}",
        f"Results: {saved} saved, {skipped} skipped (already in library), {failed} failed",
        "",
    ] + details

    return [TextContent(type="text", text="\n".join(output))]


def _handle_search_zotero(args: dict) -> list[TextContent]:
    auth_err = _ensure_zotero_auth()
    if auth_err:
        return auth_err

    query = args.get("query", "")
    limit = min(args.get("limit", 20), 100)

    if not query:
        return [TextContent(type="text", text="No search query provided.")]

    zot_client = _get_zotero_client()

    try:
        items = zot_client.search_items(query, limit)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero search error: {e}")]

    if not items:
        return [TextContent(type="text", text=f"No items found in Zotero for '{query}'.")]

    output = ["=" * 50, f"ZOTERO SEARCH: '{query}' ({len(items)} results)", "=" * 50, ""]
    for i, item in enumerate(items, 1):
        output.append(f"--- Item {i} ---")
        output.append(zot_client.format_item_summary(item))
        output.append("")

    return [TextContent(type="text", text="\n".join(output))]


def _handle_get_zotero_collection_items(args: dict) -> list[TextContent]:
    auth_err = _ensure_zotero_auth()
    if auth_err:
        return auth_err

    collection_name = args.get("collection_name", "")
    limit = min(args.get("limit", 50), 100)

    if not collection_name:
        return [TextContent(type="text", text="No collection name provided.")]

    zot_client = _get_zotero_client()

    try:
        collection_key = zot_client.find_collection_by_name(collection_name)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error: {e}")]

    if not collection_key:
        return [TextContent(type="text", text=f"Collection '{collection_name}' not found in Zotero.")]

    try:
        items = zot_client.get_collection_items(collection_key, limit)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error: {e}")]

    if not items:
        return [TextContent(type="text", text=f"No items in collection '{collection_name}'.")]

    output = [
        "=" * 50,
        f"ZOTERO COLLECTION: '{collection_name}' ({len(items)} items)",
        "=" * 50,
        "",
    ]
    for i, item in enumerate(items, 1):
        output.append(f"--- Item {i} ---")
        output.append(zot_client.format_item_summary(item))
        output.append("")

    return [TextContent(type="text", text="\n".join(output))]


def _handle_get_zotero_status() -> list[TextContent]:
    """Handle get_zotero_status tool — verify credentials and report status."""
    features = _get_features()
    if not features.get("zotero_enabled", True):
        return [TextContent(type="text", text="Zotero integration is disabled.")]

    try:
        zot = _get_zotero_client()
        result = zot.verify_credentials()
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error: {e}")]

    output = ["=" * 50, "ZOTERO STATUS", "=" * 50]
    if result["valid"]:
        output.append(f"\nCredentials: VALID")
        output.append(f"Username: {result.get('username', 'N/A')}")
        output.append(f"User ID: {result.get('userID', 'N/A')}")
        access = result.get("access", {})
        output.append(f"\nPermissions:")
        output.append(f"  Library access: {access.get('library', False)}")
        output.append(f"  File access: {access.get('files', False)}")
        output.append(f"  Notes access: {access.get('notes', False)}")
        output.append(f"  Write access: {access.get('write', False)}")
    else:
        output.append(f"\nCredentials: INVALID")
        output.append(f"Reason: {result.get('message', 'Unknown')}")

    return [TextContent(type="text", text="\n".join(output))]
