"""MCP tool definitions and handler dispatch.

Extracted from the monolith with:
- PERF-002: asyncio.gather() for batch operations
- PERF-003: Semaphore-based request queuing
- PERF-004: Metrics logging (request count, latency) per tool call
- Input validation on all tools
"""

import asyncio
import json
import logging
import random
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
from lib.downloader import ArticleDownloader, DownloadError, PDFTextExtractionError
from lib.proxy_utils import make_proxied_url
from lib.publisher_router import PublisherRouter
from lib.rate_limiter import DownloadRateLimiter, RateLimitExceeded
from lib.formatters import format_search_results, format_item_details
from lib.reranker import rerank_results
from lib.validators import sanitize_search_query, validate_isbn
from lib.zotero import ZoteroClient, ZoteroError

logger = logging.getLogger("sfu_library_mcp")

# PERF-003: Semaphore to limit concurrent API requests (bumped from 5 for fusion)
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


# Lazy-loaded downloader, rate limiter, publisher router, and zotero client
_downloader: ArticleDownloader | None = None
_downloader_cookie_id: int | None = None  # Track cookie changes
_rate_limiter: DownloadRateLimiter | None = None
_publisher_router: PublisherRouter | None = None
_zotero_client: ZoteroClient | None = None


def _get_rate_limiter() -> DownloadRateLimiter:
    """Get or create the singleton DownloadRateLimiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = DownloadRateLimiter(_get_config())
    return _rate_limiter


def _get_publisher_router() -> PublisherRouter:
    """Get or create the singleton PublisherRouter.

    Persists across downloader recreations (cookie changes) so learned
    domain preferences survive re-authentication.
    """
    global _publisher_router
    if _publisher_router is None:
        _publisher_router = PublisherRouter()
    return _publisher_router


def _get_downloader(lib_client) -> ArticleDownloader:
    """Get or create ArticleDownloader, recreating if cookies changed."""
    global _downloader, _downloader_cookie_id
    cookies = getattr(lib_client, "cookies", {})
    # Content-based fingerprint so we detect when cookies are updated in place
    cookie_id = hash(frozenset(cookies.items())) if cookies else 0

    if _downloader is None or _downloader_cookie_id != cookie_id:
        _downloader = ArticleDownloader(
            _get_config(), cookies,
            rate_limiter=_get_rate_limiter(),
            publisher_router=_get_publisher_router(),
        )
        _downloader_cookie_id = cookie_id
    return _downloader


def _get_zotero_client() -> ZoteroClient:
    """Get or create ZoteroClient."""
    global _zotero_client
    if _zotero_client is None:
        _zotero_client = ZoteroClient(_get_config())
    return _zotero_client


def _ensure_zotero_auth() -> list[TextContent] | None:
    """Zotero-only auth pre-flight check. Returns error response or None if OK.

    Independent of SFU Library auth — error messages explicitly note that
    the other auth path is still available.
    """
    features = _get_features()
    if not features.get("zotero_enabled", True):
        return [TextContent(type="text", text="Zotero integration is disabled.")]
    try:
        zot = _get_zotero_client()
        if not zot.ensure_authenticated():
            return [TextContent(type="text", text=(
                "Zotero authentication failed. Check SFU_ZOTERO_API_KEY and "
                "SFU_ZOTERO_USER_ID. SFU Library search/download still available."
            ))]
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero auth error: {e}. SFU Library tools still available.")]
    return None


# PERF-004: Simple metrics counters
_metrics: dict[str, dict[str, Any]] = {}

# Strategy B: Search result cache indexed by record ID.
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
    """Strategy B: Cache docs from search results by record ID."""
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
    """Strategy B: Look up a previously searched record from cache."""
    return _record_cache.get(record_id)


# ─── Skip flag schema properties (shared across download tools) ──

_SKIP_FLAG_PROPERTIES = {
    "skip_ezproxy": {
        "type": "boolean",
        "description": "Skip EZProxy fallback entirely (default: false)",
        "default": False,
    },
    "skip_rate_limit": {
        "type": "boolean",
        "description": "Skip rate limiter delay/budget checks (default: false)",
        "default": False,
    },
    "skip_tiers": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Tiers to skip, e.g. [\"playwright\", \"requests\"]",
    },
    "skip_pdf_check": {
        "type": "boolean",
        "description": "Skip PDF magic-byte validation — accept any content (default: false)",
        "default": False,
    },
    "skip_login_check": {
        "type": "boolean",
        "description": "Skip login/auth page detection (default: false)",
        "default": False,
    },
    "skip_copy_to_host": {
        "type": "boolean",
        "description": "Skip copying PDF to host Downloads folder (default: false)",
        "default": False,
    },
}


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
        name="get_token_status",
        description="Check the current authentication token status. Shows if authenticated, time remaining, and user information.",
        inputSchema={
            "type": "object",
            "properties": {}
        }
    ),
    Tool(
        name="authenticate",
        description="Authenticate with the SFU Library system. Usually done automatically, but can be called manually to refresh authentication or force re-authentication.",
        inputSchema={
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Force re-authentication even if a valid token exists",
                    "default": False
                }
            }
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
        name="clear_cache",
        description="Clear the cached authentication token. Useful if experiencing authentication issues.",
        inputSchema={
            "type": "object",
            "properties": {}
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
        name="download_article",
        description=(
            "Download the PDF of a library article to the container cache and optionally "
            "to the host Downloads folder. Use after searching to save articles locally."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "The record ID of the item (obtained from search results)"
                },
                "save_to_host": {
                    "type": "boolean",
                    "description": "Also copy PDF to host Downloads folder (default: true)",
                    "default": True
                },
                **_SKIP_FLAG_PROPERTIES,
            },
            "required": ["record_id"]
        }
    ),
    Tool(
        name="read_article",
        description=(
            "Download a library article's PDF and extract its full text for analysis. "
            "Returns the article text content directly so you can read, summarize, or answer questions about it."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "The record ID of the item (obtained from search results)"
                },
                **_SKIP_FLAG_PROPERTIES,
            },
            "required": ["record_id"]
        }
    ),
    Tool(
        name="save_to_zotero",
        description=(
            "Save a library item's metadata and PDF to the user's Zotero library. "
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
                "attach_pdf": {
                    "type": "boolean",
                    "description": "Download and attach the PDF to the Zotero item (default: true)",
                    "default": True
                }
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
                "attach_pdfs": {
                    "type": "boolean",
                    "description": "Download and attach PDFs to each Zotero item (default: true)",
                    "default": True
                }
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
        name="backfill_collection_pdfs",
        description=(
            "Find items in a Zotero collection that only have citation metadata "
            "(no PDF attached) and attempt to download and attach PDFs for each. "
            "Use after batch saves where downloads failed, or to enrich an "
            "existing collection with full-text PDFs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Name of the Zotero collection to backfill PDFs for"
                },
                "save_to_host": {
                    "type": "boolean",
                    "description": "Also copy PDFs to host Downloads folder (default: true)",
                    "default": True,
                },
            },
            "required": ["collection_name"],
        },
    ),
    Tool(
        name="download_from_url",
        description=(
            "Download a PDF from a direct URL. Use when you have a known PDF URL "
            "from any source (publisher page, DOI link, direct PDF link). "
            "Supports tiered download strategies to bypass TLS fingerprint detection."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Direct URL to the PDF"
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename for the downloaded PDF"
                },
                "use_ezproxy": {
                    "type": "boolean",
                    "description": "Wrap URL with EZProxy prefix for authenticated access",
                    "default": False,
                },
                **_SKIP_FLAG_PROPERTIES,
            },
            "required": ["url"],
        },
    ),
    Tool(
        name="get_diagnostics",
        description=(
            "Get a comprehensive diagnostic report for the SFU Library MCP server. "
            "Shows token status, cookie inventory, EZProxy session state, download tier "
            "availability, circuit breaker and rate limiter state, log file info, "
            "recent errors, and tool metrics."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "include_recent_errors": {
                    "type": "boolean",
                    "description": "Include last 10 ERROR/WARNING lines from log file (default: true)",
                    "default": True,
                }
            },
        },
    ),
    Tool(
        name="get_zotero_status",
        description=(
            "Check Zotero API connection, credentials, and permissions. "
            "Independent of SFU Library authentication."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="zotero_authenticate",
        description=(
            "Verify or re-verify Zotero API credentials. "
            "Independent of SFU Library authentication."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Force re-verification even if already validated",
                    "default": False,
                }
            },
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
    elif name == "get_token_status":
        return _handle_get_token_status(lib_client)
    elif name == "authenticate":
        return _handle_authenticate(arguments, lib_client)
    elif name == "search_by_author":
        return await _handle_search_by_author(arguments, lib_client)
    elif name == "search_by_subject":
        return await _handle_search_by_subject(arguments, lib_client)
    elif name == "search_by_isbn":
        return await _handle_search_by_isbn(arguments, lib_client)
    elif name == "search_electronic_resources":
        return await _handle_search_electronic(arguments, lib_client)
    elif name == "clear_cache":
        return _handle_clear_cache(lib_client)
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
    elif name == "download_article":
        return await _handle_download_article(arguments, lib_client)
    elif name == "read_article":
        return await _handle_read_article(arguments, lib_client)
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
    elif name == "backfill_collection_pdfs":
        return await _handle_backfill_collection_pdfs(arguments, lib_client)
    elif name == "download_from_url":
        return _handle_download_from_url(arguments, lib_client)
    elif name == "get_diagnostics":
        return _handle_get_diagnostics(arguments, lib_client)
    elif name == "get_zotero_status":
        return _handle_get_zotero_status()
    elif name == "zotero_authenticate":
        return _handle_zotero_authenticate(arguments)
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
                # Use a fallback key based on title
                record_id = f"_fallback_{pnx.get('display', {}).get('title', [''])[0][:50]}"
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (k + rank)
            if record_id not in doc_map:
                doc_map[record_id] = doc

    # Sort by RRF score descending
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

    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed. Please try again or check your credentials.")]

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
        # Filter out exceptions, treat them as None
        valid_results = [r if not isinstance(r, Exception) else None for r in results_sets]
        results = _reciprocal_rank_fusion(valid_results, fetch_limit)
    else:
        results = await _single_search(client, search_query, fetch_limit, offset, field, sort, tab, scope)

        if results is None:
            if client.ensure_authenticated(force=True):
                results = await _single_search(client, search_query, fetch_limit, offset, field, sort, tab, scope)

    # Strategy B: Cache all returned docs by record ID
    if results and results.get("docs"):
        _cache_search_docs(results["docs"])

    # Re-rank results if enabled
    if rerank_enabled and results and results.get("docs"):
        results["docs"] = rerank_results(results["docs"], query, limit)
    elif results and results.get("docs") and len(results["docs"]) > limit:
        # Trim to requested limit if we over-fetched but reranking is off
        results["docs"] = results["docs"][:limit]

    search_metadata = {"query": query, "field": field, "sort": sort, "resource_type": resource_type}
    if comprehensive and fusion_enabled:
        search_metadata["comprehensive"] = True
    formatted = format_search_results(results, metadata=search_metadata)
    return [TextContent(type="text", text=formatted)]


async def _handle_get_item_details(args: dict, client) -> list[TextContent]:
    record_id = args.get("record_id", "")
    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]
    async with _request_semaphore:
        item = client.get_item_details(record_id)
    formatted = format_item_details(item)
    return [TextContent(type="text", text=formatted)]


def _handle_get_token_status(client) -> list[TextContent]:
    status = client.get_token_status()
    if status.get("valid"):
        text = f"""Token Status: VALID
User: {status.get('user', 'Unknown')} ({status.get('userId', '')})
User Group: {status.get('userGroup', '')}
Expires In: {status.get('expiresIn', '')}
Expires At: {status.get('expiresAt', '')}"""
    else:
        text = f"Token Status: INVALID\nReason: {status.get('message', 'Unknown')}"
    return [TextContent(type="text", text=text)]


def _handle_authenticate(args: dict, client) -> list[TextContent]:
    force = args.get("force", False)
    success = client.ensure_authenticated(force=force)
    if success:
        status = client.get_token_status()
        text = f"""Authentication successful!
User: {status.get('user', 'Unknown')} ({status.get('userId', '')})
User Group: {status.get('userGroup', '')}
Token valid for: {status.get('expiresIn', '')}"""
    else:
        text = "Authentication failed. Please check your credentials and network connection."
    return [TextContent(type="text", text=text)]


async def _handle_search_by_author(args: dict, client) -> list[TextContent]:
    author = args.get("author", "")
    limit = args.get("limit", 10)
    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]
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
    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]
    async with _request_semaphore:
        results = client.search(query=subject, limit=limit, field="sub")
    if results and results.get("docs"):
        _cache_search_docs(results["docs"])
    search_metadata = {"query": subject, "field": "sub", "sort": "rank"}
    formatted = format_search_results(results, metadata=search_metadata)
    return [TextContent(type="text", text=formatted)]


async def _handle_search_by_isbn(args: dict, client) -> list[TextContent]:
    isbn = args.get("isbn", "")
    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]
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
    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]
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


def _handle_clear_cache(client) -> list[TextContent]:
    client.clear_token_cache()
    return [TextContent(type="text", text="Token cache cleared successfully.")]


async def _handle_get_full_text_links(args: dict, client) -> list[TextContent]:
    record_id = args.get("record_id", "")
    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]

    async with _request_semaphore:
        item = client.get_item_details(record_id)
    if not item:
        # Strategy B: Try cached search result before expensive API fallback
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

    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]

    async with _request_semaphore:
        item = client.get_item_details(record_id)
    if not item:
        # Strategy B: Try cached search result before expensive API fallback
        item = _lookup_cached_record(record_id)
    if not item:
        async with _request_semaphore:
            results = client.search(query=record_id, limit=1)
        if results and results.get("docs"):
            item = results["docs"][0]

    if not item:
        return [TextContent(type="text", text=f"Could not find item with record ID: {record_id}")]

    metadata = extract_metadata(item)
    # Strategy D: Enrich with CrossRef if key fields are missing
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
    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]

    format_names = {
        "apa": "APA 7th Edition",
        "mla": "MLA 9th Edition",
        "chicago": "Chicago 17th Edition",
        "bibtex": "BibTeX",
    }
    format_name = format_names.get(citation_format, "APA 7th Edition")

    # PERF-002: Fetch items concurrently
    async def fetch_item(rid: str) -> tuple[str, dict | None]:
        async with _request_semaphore:
            item = client.get_item_details(rid)
        if not item:
            # Strategy B: Try cached search result first
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

    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]

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
    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]

    output = ["=" * 50, f"BATCH ISBN LOOKUP ({len(isbn_list)} items)", "=" * 50 + "\n"]

    # PERF-002: Concurrent ISBN lookups
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


# ─── PDF Download + Zotero handlers ───────────────────────────

def _extract_skip_flags(args: dict) -> dict:
    """Extract skip flag arguments into a dict for downloader."""
    flags = {}
    for key in ("skip_ezproxy", "skip_rate_limit", "skip_pdf_check",
                "skip_login_check", "skip_copy_to_host"):
        if args.get(key):
            flags[key] = True
    if args.get("skip_tiers"):
        flags["skip_tiers"] = args["skip_tiers"]
    return flags


async def _handle_download_article(args: dict, client) -> list[TextContent]:
    features = _get_features()
    if not features.get("pdf_download_enabled", True):
        return [TextContent(type="text", text="PDF download is disabled. Set SFU_FEATURE_PDF_DOWNLOAD_ENABLED=true to enable.")]

    record_id = args.get("record_id", "")
    save_to_host = args.get("save_to_host", True)
    skip_flags = _extract_skip_flags(args)

    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]

    item = await _resolve_record(record_id, client)
    if not item:
        return [TextContent(type="text", text=f"Could not find item with record ID: {record_id}")]

    downloader = _get_downloader(client)
    urls = downloader.resolve_all_pdf_urls(item)
    if not urls:
        return [TextContent(type="text", text=f"No PDF URL found for record {record_id}. The item may not have an accessible PDF.")]

    metadata = extract_metadata(item)
    copy_to_host = save_to_host and features.get("host_download_enabled", True)
    if skip_flags.get("skip_copy_to_host"):
        copy_to_host = False

    # Try each available URL (direct + EZProxy fallback per URL) until one succeeds
    config = _get_config()
    budget = config.download_budget_seconds
    start_time = time.time()
    errors = []
    result = None
    for url in urls:
        elapsed = time.time() - start_time
        if elapsed > budget:
            errors.append(f"  (skipped remaining URLs — {budget:.0f}s time budget exceeded)")
            break
        result = downloader.download_pdf(url, record_id, metadata, copy_to_host=copy_to_host, skip_flags=skip_flags)
        if result["success"]:
            break
        errors.append(f"  {url}: {result['error']}")

    if not result or not result["success"]:
        error_detail = "\n".join(errors)
        return [TextContent(type="text", text=(
            f"Download failed — tried {len(urls)} URL(s) with direct + EZProxy strategies:\n{error_detail}"
        ))]

    output = ["=" * 50, "ARTICLE DOWNLOADED", "=" * 50]
    if metadata:
        output.append(f"\nTitle: {metadata.get('title', 'Unknown')}")
    output.append(f"Size: {result['size_bytes']:,} bytes")
    output.append(f"Container path: {result['container_path']}")
    if result.get("host_path"):
        output.append(f"Host Downloads: {result['host_path']}")
    elif save_to_host:
        output.append("Note: Could not copy to host Downloads folder.")

    return [TextContent(type="text", text="\n".join(output))]


async def _handle_read_article(args: dict, client) -> list[TextContent]:
    features = _get_features()
    if not features.get("pdf_download_enabled", True):
        return [TextContent(type="text", text="PDF download is disabled. Set SFU_FEATURE_PDF_DOWNLOAD_ENABLED=true to enable.")]

    record_id = args.get("record_id", "")
    skip_flags = _extract_skip_flags(args)

    if not client.ensure_authenticated():
        return [TextContent(type="text", text="Authentication failed.")]

    item = await _resolve_record(record_id, client)
    if not item:
        return [TextContent(type="text", text=f"Could not find item with record ID: {record_id}")]

    downloader = _get_downloader(client)
    metadata = extract_metadata(item)

    # Check cache or download — try all available URLs
    urls = downloader.resolve_all_pdf_urls(item)
    if not urls:
        return [TextContent(type="text", text=f"No PDF URL found for record {record_id}. The item may not have an accessible PDF.")]

    config = _get_config()
    budget = config.download_budget_seconds
    start_time = time.time()
    errors = []
    result = None
    for url in urls:
        elapsed = time.time() - start_time
        if elapsed > budget:
            errors.append(f"  (skipped remaining URLs — {budget:.0f}s time budget exceeded)")
            break
        result = downloader.download_pdf(url, record_id, metadata, copy_to_host=False, skip_flags=skip_flags)
        if result["success"]:
            break
        errors.append(f"  {url}: {result['error']}")

    if not result or not result["success"]:
        error_detail = "\n".join(errors)
        return [TextContent(type="text", text=(
            f"Download failed — tried {len(urls)} URL(s) with direct + EZProxy strategies:\n{error_detail}"
        ))]

    try:
        text = downloader.extract_text(result["container_path"])
    except PDFTextExtractionError as e:
        return [TextContent(type="text", text=f"Text extraction failed: {e}")]

    # Build header with metadata
    header_parts = ["=" * 50, "ARTICLE TEXT", "=" * 50]
    if metadata:
        header_parts.append(f"Title: {metadata.get('title', 'Unknown')}")
        authors = metadata.get("authors") or metadata.get("creators", [])
        if authors:
            author_strs = [a.split("$$")[0] for a in authors[:5]]
            header_parts.append(f"Authors: {'; '.join(author_strs)}")
        if metadata.get("date"):
            header_parts.append(f"Date: {metadata['date']}")
        if metadata.get("source"):
            header_parts.append(f"Source: {metadata['source']}")
        if metadata.get("doi"):
            header_parts.append(f"DOI: {metadata['doi']}")
    header_parts.append("=" * 50)
    header_parts.append("")

    return [TextContent(type="text", text="\n".join(header_parts) + text)]


async def _handle_save_to_zotero(args: dict, client) -> list[TextContent]:
    # Independent auth checks — Zotero and SFU Library are separate paths
    zotero_auth_err = _ensure_zotero_auth()
    if zotero_auth_err:
        # Zotero is down — tell user SFU Library still works
        return zotero_auth_err

    record_id = args.get("record_id", "")
    collection_name = args.get("collection_name", "")
    parent_collection = args.get("parent_collection", "")
    attach_pdf = args.get("attach_pdf", True)

    sfu_authenticated = client.ensure_authenticated()
    if not sfu_authenticated:
        return [TextContent(type="text", text=(
            "SFU Library authentication failed — cannot resolve record. "
            "Zotero is connected. Use Zotero-only tools (search_zotero, "
            "list_zotero_collections) which work independently."
        ))]

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

    # Optionally attach PDF
    if attach_pdf and features.get("pdf_download_enabled", True):
        downloader = _get_downloader(client)
        urls = downloader.resolve_all_pdf_urls(item)
        if urls:
            dl_result = None
            for url in urls:
                dl_result = downloader.download_pdf(url, record_id, metadata, copy_to_host=False)
                if dl_result["success"]:
                    break
            if dl_result and dl_result["success"]:
                try:
                    zot_client.attach_pdf(item_key, dl_result["container_path"])
                    output.append("PDF: Attached successfully")
                except ZoteroError as e:
                    output.append(f"PDF: Attachment failed ({e})")
            else:
                output.append(f"PDF: Download failed ({dl_result['error'] if dl_result else 'no result'})")
        else:
            output.append("PDF: No accessible PDF URL found")

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
    # Independent auth checks — Zotero and SFU Library are separate paths
    zotero_auth_err = _ensure_zotero_auth()
    if zotero_auth_err:
        return zotero_auth_err

    record_ids = args.get("record_ids", [])[:20]
    collection_name = args.get("collection_name", "")
    parent_collection = args.get("parent_collection", "")
    attach_pdfs = args.get("attach_pdfs", True)

    if not record_ids:
        return [TextContent(type="text", text="No record IDs provided.")]

    sfu_authenticated = client.ensure_authenticated()
    if not sfu_authenticated:
        return [TextContent(type="text", text=(
            "SFU Library authentication failed — cannot resolve records. "
            "Zotero is connected. Use Zotero-only tools (search_zotero, "
            "list_zotero_collections) which work independently."
        ))]

    zot_client = _get_zotero_client()
    downloader = _get_downloader(client) if attach_pdfs and features.get("pdf_download_enabled", True) else None

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

        # Attach PDF
        pdf_status = ""
        if downloader:
            urls = downloader.resolve_all_pdf_urls(item)
            dl_result = None
            for url in urls:
                dl_result = downloader.download_pdf(url, rid, metadata, copy_to_host=False)
                if dl_result["success"]:
                    break
            if dl_result and dl_result["success"]:
                try:
                    zot_client.attach_pdf(item_key, dl_result["container_path"])
                    pdf_status = " + PDF"
                except ZoteroError:
                    pdf_status = " (PDF attach failed)"
            else:
                pdf_status = " (PDF download failed)"
            # Humanized delay between batch PDF downloads
            await asyncio.sleep(random.uniform(2.0, 5.0))

        details.append(f"  {title_short}: SAVED{pdf_status}")
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


def _handle_download_from_url(args: dict, client) -> list[TextContent]:
    features = _get_features()
    if not features.get("pdf_download_enabled", True):
        return [TextContent(type="text", text="PDF download is disabled. Set SFU_FEATURE_PDF_DOWNLOAD_ENABLED=true to enable.")]

    url = args.get("url", "")
    filename = args.get("filename")
    use_ezproxy = args.get("use_ezproxy", False)
    skip_flags = _extract_skip_flags(args)

    if not url:
        return [TextContent(type="text", text="No URL provided.")]

    config = _get_config()
    if use_ezproxy and config.ezproxy_proxy_base not in url:
        url = make_proxied_url(url, config.ezproxy_proxy_base)

    downloader = _get_downloader(client)
    result = downloader.download_from_direct_url(url, filename=filename, skip_flags=skip_flags)

    if not result["success"]:
        return [TextContent(type="text", text=f"Download failed: {result['error']}")]

    output = [
        "=" * 50,
        "PDF DOWNLOADED FROM URL",
        "=" * 50,
        f"\nURL: {url}",
        f"Size: {result['size_bytes']:,} bytes",
        f"Tier: {result['tier_used']}",
        f"Container path: {result['container_path']}",
    ]
    if filename:
        output.append(f"Filename: {filename}")

    return [TextContent(type="text", text="\n".join(output))]


async def _handle_backfill_collection_pdfs(args: dict, client) -> list[TextContent]:
    # Independent auth checks — Zotero and SFU Library are separate paths
    zotero_auth_err = _ensure_zotero_auth()
    if zotero_auth_err:
        return zotero_auth_err

    features = _get_features()
    if not features.get("pdf_download_enabled", True):
        return [TextContent(type="text", text="PDF download is disabled.")]

    collection_name = args.get("collection_name", "")
    save_to_host = args.get("save_to_host", True)

    if not collection_name:
        return [TextContent(type="text", text="No collection name provided.")]

    sfu_authenticated = client.ensure_authenticated()
    if not sfu_authenticated:
        return [TextContent(type="text", text=(
            "SFU Library authentication failed — cannot search for PDFs. "
            "Zotero is connected. Use Zotero-only tools (search_zotero, "
            "list_zotero_collections) which work independently."
        ))]

    zot_client = _get_zotero_client()
    downloader = _get_downloader(client)

    # Find collection
    try:
        collection_key = zot_client.find_collection_by_name(collection_name)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error: {e}")]

    if not collection_key:
        return [TextContent(type="text", text=f"Collection '{collection_name}' not found in Zotero.")]

    # Get items without PDFs
    try:
        items_without_pdfs = zot_client.get_items_without_pdfs(collection_key)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error: {e}")]

    if not items_without_pdfs:
        return [TextContent(type="text", text=f"All items in '{collection_name}' already have PDFs attached.")]

    # Cap backfill to configured limit
    config = _get_config()
    backfill_cap = config.download_backfill_cap
    if len(items_without_pdfs) > backfill_cap:
        logger.info(
            "Backfill capped: %d items without PDFs, limiting to %d",
            len(items_without_pdfs), backfill_cap,
        )
        items_without_pdfs = items_without_pdfs[:backfill_cap]

    # Check budget before starting
    rate_limiter = _get_rate_limiter()
    budget = rate_limiter.get_budget_status()
    if budget["session_remaining"] == 0:
        return [TextContent(type="text", text="Session download budget exhausted. Restart server to reset.")]

    attached = 0
    failed = 0
    details = []
    copy_to_host = save_to_host and features.get("host_download_enabled", True)

    for item in items_without_pdfs:
        title_short = item.get("title", "Unknown")[:60]
        item_key = item.get("key", "")
        doi = item.get("DOI", "")

        # Try to find the article via DOI or title search
        search_query = doi if doi else item.get("title", "")
        if not search_query:
            details.append(f"  {title_short}: FAILED - no DOI or title for search")
            failed += 1
            continue

        # Search library for full record with links
        record = None
        try:
            async with _request_semaphore:
                results = client.search(query=search_query, limit=1)
            if results and results.get("docs"):
                record = results["docs"][0]
        except Exception as e:
            logger.warning("Backfill search failed for '%s': %s", search_query[:50], e)

        if not record:
            details.append(f"  {title_short}: FAILED - not found in library search")
            failed += 1
            continue

        # Resolve PDF URLs and try each one
        urls = downloader.resolve_all_pdf_urls(record)
        if not urls:
            details.append(f"  {title_short}: FAILED - no PDF URL found")
            failed += 1
            continue

        # Download PDF — try all available URLs
        metadata = extract_metadata(record)
        dl_result = None
        for url in urls:
            dl_result = downloader.download_pdf(url, item_key, metadata, copy_to_host=copy_to_host)
            if dl_result["success"]:
                break
        if not dl_result or not dl_result["success"]:
            details.append(f"  {title_short}: FAILED - {dl_result['error'] if dl_result else 'no URLs'}")
            failed += 1
            continue

        # Attach to Zotero item
        try:
            zot_client.attach_pdf(item_key, dl_result["container_path"])
            host_info = f" (host: {dl_result['host_path']})" if dl_result.get("host_path") else ""
            details.append(f"  {title_short}: ATTACHED{host_info}")
            attached += 1
        except ZoteroError as e:
            details.append(f"  {title_short}: FAILED - attach error: {e}")
            failed += 1

        # Humanized delay between backfill downloads
        await asyncio.sleep(random.uniform(3.0, 6.0))

    # Include budget status in output
    budget = rate_limiter.get_budget_status()
    output = [
        "=" * 50,
        "BACKFILL COLLECTION PDFs",
        "=" * 50,
        f"\nCollection: {collection_name}",
        f"Items checked: {len(items_without_pdfs)}",
        f"PDFs attached: {attached}",
        f"Failed: {failed}",
        f"Downloads remaining: {budget['session_remaining']}/{budget['session_budget']}",
        "",
    ] + details

    return [TextContent(type="text", text="\n".join(output))]


# ─── Zotero status handlers ──────────────────────────────────────

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

    output.append(f"\nNote: Independent of SFU Library authentication.")
    return [TextContent(type="text", text="\n".join(output))]


def _handle_zotero_authenticate(args: dict) -> list[TextContent]:
    """Handle zotero_authenticate tool — verify or re-verify credentials."""
    features = _get_features()
    if not features.get("zotero_enabled", True):
        return [TextContent(type="text", text="Zotero integration is disabled.")]

    force = args.get("force", False)

    try:
        zot = _get_zotero_client()
        success = zot.ensure_authenticated(force=force)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero auth error: {e}")]

    if success:
        status = zot.get_auth_status()
        access = status.get("access", {})
        output = [
            "Zotero authentication successful!",
            f"Username: {status.get('username', 'N/A')}",
            f"User ID: {status.get('userID', 'N/A')}",
            f"Write access: {access.get('write', False)}",
            "",
            "Note: Independent of SFU Library authentication.",
        ]
        return [TextContent(type="text", text="\n".join(output))]
    else:
        return [TextContent(type="text", text=(
            "Zotero authentication failed. Check SFU_ZOTERO_API_KEY and "
            "SFU_ZOTERO_USER_ID. SFU Library search/download still available."
        ))]


# ─── Diagnostics handler ────────────────────────────────────────

def _handle_get_diagnostics(args: dict, client) -> list[TextContent]:
    """Generate a comprehensive diagnostic report."""
    import os
    import datetime

    include_errors = args.get("include_recent_errors", True)
    output = ["=" * 60, "SFU LIBRARY MCP — DIAGNOSTIC REPORT", "=" * 60]

    # 1. Token status
    output.append("\n--- Token Status ---")
    try:
        status = client.get_token_status()
        if status.get("valid"):
            output.append("  Status: VALID")
            output.append(f"  User: {status.get('user', 'Unknown')} ({status.get('userId', '')})")
            output.append(f"  Group: {status.get('userGroup', '')}")
            output.append(f"  Expires: {status.get('expiresIn', '')} ({status.get('expiresAt', '')})")
        else:
            output.append(f"  Status: INVALID — {status.get('message', 'Unknown')}")
    except Exception as e:
        output.append(f"  Error: {e}")

    # 2. Zotero status (independent of SFU Library token)
    output.append("\n--- Zotero Status ---")
    try:
        zot = _get_zotero_client()
        zot_status = zot.get_auth_status()
        output.append(f"  Credentials configured: {zot_status.get('credentials_configured', False)}")
        output.append(f"  Validated: {zot_status.get('validated', False)}")
        if zot_status.get("validated"):
            output.append(f"  Username: {zot_status.get('username', 'N/A')}")
            output.append(f"  User ID: {zot_status.get('userID', 'N/A')}")
            access = zot_status.get("access", {})
            output.append(f"  Write access: {access.get('write', False)}")
        elif zot_status.get("message"):
            output.append(f"  Status: {zot_status['message']}")
    except Exception as e:
        output.append(f"  Error: {e}")

    # 3. Cookie inventory
    output.append("\n--- Cookie Inventory ---")
    cookies = getattr(client, "cookies", {})
    output.append(f"  Total cookies: {len(cookies)}")
    proxy_cookies = [n for n in cookies if "proxy" in n.lower() or "ezproxy" in n.lower()]
    secure_cookies = [n for n in cookies if n.startswith("__Secure-") or n.startswith("__Host-")]
    output.append(f"  Proxy cookies: {proxy_cookies if proxy_cookies else '(none)'}")
    output.append(f"  __Secure-/__Host- cookies: {secure_cookies if secure_cookies else '(none)'}")
    if cookies:
        output.append(f"  All cookie names: {list(cookies.keys())}")

    # 3. EZProxy session status
    output.append("\n--- EZProxy Session ---")
    if proxy_cookies:
        output.append("  Status: Cookies present (session likely active)")
    else:
        output.append("  Status: NO proxy cookies — session not established")
        output.append("  Action: Re-authenticate to establish EZProxy session")

    # 4. Download tier availability
    output.append("\n--- Download Tier Availability ---")
    tier_status = {}
    try:
        import curl_cffi  # noqa: F401
        tier_status["curl_cffi"] = "AVAILABLE"
    except ImportError:
        tier_status["curl_cffi"] = "NOT INSTALLED"
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        tier_status["playwright"] = "AVAILABLE"
    except ImportError:
        tier_status["playwright"] = "NOT INSTALLED"
    tier_status["requests"] = "AVAILABLE (built-in)"
    for tier, status_str in tier_status.items():
        output.append(f"  {tier}: {status_str}")

    # 5. Circuit breaker state
    output.append("\n--- Circuit Breaker ---")
    try:
        downloader = _get_downloader(client)
        cb = downloader._circuit_breaker
        can_proceed = cb.can_proceed()
        output.append(f"  Can proceed: {can_proceed}")
        output.append(f"  Failure count: {cb._failure_count}")
        output.append(f"  Threshold: {cb._threshold}")
        output.append(f"  Timeout: {cb._timeout}s")
        if hasattr(cb, '_last_failure_time') and cb._last_failure_time:
            last_fail = datetime.datetime.fromtimestamp(cb._last_failure_time).isoformat()
            output.append(f"  Last failure: {last_fail}")
    except Exception as e:
        output.append(f"  Error: {e}")

    # 6. Rate limiter state
    output.append("\n--- Rate Limiter ---")
    try:
        rate_limiter = _get_rate_limiter()
        budget_status = rate_limiter.get_budget_status()
        output.append(f"  Session remaining: {budget_status['session_remaining']}/{budget_status['session_budget']}")
        output.append(f"  Hourly: {budget_status['hourly_used']}/{budget_status['hourly_limit']} used")
        if budget_status.get("per_domain"):
            output.append(f"  Per-domain stats: {budget_status['per_domain']}")
    except Exception as e:
        output.append(f"  Error: {e}")

    # 7. Log file info
    output.append("\n--- Log File ---")
    log_path = getattr(client.config, "log_file", "") or "/tmp/sfu-library-mcp.log"
    try:
        if os.path.exists(log_path):
            stat = os.stat(log_path)
            size_kb = stat.st_size / 1024
            modified = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
            output.append(f"  Path: {log_path}")
            output.append(f"  Size: {size_kb:.1f} KB")
            output.append(f"  Last modified: {modified}")
        else:
            output.append(f"  Path: {log_path} (not found)")
    except Exception as e:
        output.append(f"  Error: {e}")

    # 8. Recent errors from log
    if include_errors:
        output.append("\n--- Recent Errors/Warnings (last 10) ---")
        try:
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    lines = f.readlines()
                error_lines = [
                    line.rstrip() for line in lines
                    if " ERROR " in line or " WARNING " in line
                ]
                for line in error_lines[-10:]:
                    output.append(f"  {line[:200]}")
                if not error_lines:
                    output.append("  (no errors or warnings found)")
            else:
                output.append("  (log file not found)")
        except Exception as e:
            output.append(f"  Error reading log: {e}")

    # 9. Tool metrics
    output.append("\n--- Tool Metrics ---")
    metrics = get_metrics()
    if metrics:
        for tool_name, m in sorted(metrics.items()):
            avg_latency = m["total_latency"] / m["count"] if m["count"] > 0 else 0
            output.append(f"  {tool_name}: {m['count']} calls, {m['errors']} errors, avg {avg_latency:.2f}s")
    else:
        output.append("  (no metrics recorded yet)")

    output.append("\n" + "=" * 60)
    return [TextContent(type="text", text="\n".join(output))]
