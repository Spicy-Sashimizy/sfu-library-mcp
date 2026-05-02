"""MCP tool definitions and handler dispatch."""

import asyncio
import json
import logging
import time
from typing import Any

import requests
from mcp.types import Tool, TextContent

from lib.citations import (
    enrich_metadata_from_crossref,
    format_apa_citation,
    format_mla_citation,
    format_chicago_citation,
    format_bibtex_entry,
    format_ris_entry,
)
from lib.config import load_config, ServerConfig
from lib.formatters import (
    format_openalex_results,
    format_sfu_databases_list,
    format_sfu_database,
    format_semantic_scholar_papers,
)
from lib.reranker import rerank_results
from lib.validators import sanitize_search_query
from lib.zotero import ZoteroClient, ZoteroError

logger = logging.getLogger("sfu_library_mcp")

# Semaphore to limit concurrent outbound API requests
_request_semaphore = asyncio.Semaphore(8)

# Lazy-loaded config
_config: ServerConfig | None = None


def _get_config() -> ServerConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_features() -> dict[str, bool]:
    return _get_config().features


# ── Lazy-loaded API clients ───────────────────────────────────────────────────

_zotero_client: ZoteroClient | None = None
_openalex_client = None
_sfu_registry = None
_access_resolver = None
_s2_client = None


def _get_zotero_client() -> ZoteroClient:
    global _zotero_client
    if _zotero_client is None:
        _zotero_client = ZoteroClient(_get_config())
    return _zotero_client


def _get_openalex() -> "OpenAlexClient":
    global _openalex_client
    if _openalex_client is None:
        from lib.openalex import OpenAlexClient
        cfg = _get_config()
        _openalex_client = OpenAlexClient(
            mailto=cfg.openalex_mailto,
            api_key=cfg.openalex_api_key,
        )
    return _openalex_client


def _get_registry() -> "SFUDatabaseRegistry":
    global _sfu_registry
    if _sfu_registry is None:
        from lib.sfu_databases import SFUDatabaseRegistry
        cfg = _get_config()
        _sfu_registry = SFUDatabaseRegistry(
            cache_ttl=cfg.sfu_db_registry_cache_ttl,
            cache_file=cfg.sfu_db_registry_cache_file,
        )
    return _sfu_registry


def _get_resolver() -> "AccessResolver":
    global _access_resolver
    if _access_resolver is None:
        from lib.access_resolver import AccessResolver
        _access_resolver = AccessResolver(
            registry=_get_registry(),
            unpaywall_email=_get_config().unpaywall_email,
        )
    return _access_resolver


def _get_s2() -> "SemanticScholarClient":
    global _s2_client
    if _s2_client is None:
        from lib.semantic_scholar import SemanticScholarClient
        _s2_client = SemanticScholarClient(api_key=_get_config().semantic_scholar_api_key)
    return _s2_client


# ── Work cache (keyed by DOI) ─────────────────────────────────────────────────
# Populated by search results so citation/Zotero tools can avoid a second API call.

_work_cache: dict[str, dict] = {}


def _cache_works(works: list[dict]) -> None:
    for w in works:
        doi = w.get("doi", "")
        openalex_id = w.get("openalex_id", "")
        if doi:
            _work_cache[doi] = w
        if openalex_id:
            _work_cache[openalex_id] = w
    # Cap at 500 entries
    if len(_work_cache) > 500:
        for k in list(_work_cache)[:len(_work_cache) - 500]:
            del _work_cache[k]


def _lookup_work(doi_or_id: str) -> dict | None:
    return _work_cache.get(doi_or_id)


# ── Metrics ───────────────────────────────────────────────────────────────────

_metrics: dict[str, dict[str, Any]] = {}


def _record_metric(tool_name: str, latency: float, success: bool) -> None:
    if tool_name not in _metrics:
        _metrics[tool_name] = {"count": 0, "errors": 0, "total_latency": 0.0}
    _metrics[tool_name]["count"] += 1
    _metrics[tool_name]["total_latency"] += latency
    if not success:
        _metrics[tool_name]["errors"] += 1


def get_metrics() -> dict:
    return dict(_metrics)


# ── Zotero auth helper ────────────────────────────────────────────────────────

def _ensure_zotero_auth() -> list[TextContent] | None:
    if not _get_features().get("zotero_enabled", True):
        return [TextContent(type="text", text="Zotero integration is disabled.")]
    try:
        zot = _get_zotero_client()
        if not zot.ensure_authenticated():
            return [TextContent(type="text", text=(
                "Zotero authentication failed. Check SFU_ZOTERO_API_KEY and SFU_ZOTERO_USER_ID."
            ))]
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero auth error: {e}")]
    return None


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOL_DEFINITIONS: list[Tool] = [
    # ── Search ────────────────────────────────────────────────────────────────
    Tool(
        name="search_academic",
        description=(
            "Search 250M+ scholarly works via OpenAlex (articles, books, datasets, theses). "
            "Returns titles, authors, DOIs, open-access status, citation counts, and abstracts.\n\n"
            "FILTERS (all optional):\n"
            "- year_from / year_to: publication year range (e.g. year_from=2020)\n"
            "- open_access_only: true to limit to freely available papers\n"
            "- type: 'article', 'book', 'dataset', 'dissertation', 'preprint'\n\n"
            "TIPS:\n"
            "- Use the returned DOI with generate_citation, save_to_zotero, or get_full_text_link\n"
            "- For citations/references of a specific paper use get_citations / get_references\n"
            "- For biomedical literature add search_biomedical for PubMed coverage"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Results to return (default 10, max 50)", "default": 10},
                "page": {"type": "integer", "description": "Page number (default 1)", "default": 1},
                "year_from": {"type": "integer", "description": "Earliest publication year"},
                "year_to": {"type": "integer", "description": "Latest publication year"},
                "open_access_only": {"type": "boolean", "description": "Limit to open-access works", "default": False},
                "type": {
                    "type": "string",
                    "description": "Work type filter",
                    "enum": ["article", "book", "dataset", "dissertation", "preprint"],
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_by_author",
        description=(
            "Find scholarly works by a specific author via OpenAlex. "
            "Use 'LastName, FirstName' or just the last name. "
            "Returns works with DOIs you can use for citations and Zotero."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "author": {"type": "string", "description": "Author name (best: 'LastName, FirstName')"},
                "limit": {"type": "integer", "description": "Results to return (default 10)", "default": 10},
            },
            "required": ["author"],
        },
    ),
    Tool(
        name="search_by_doi",
        description=(
            "Look up a specific work by its DOI. Returns full metadata including "
            "title, authors, abstract, open-access status, and citations. "
            "Use this when you have a DOI and want full details."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI (with or without https://doi.org/ prefix)"},
            },
            "required": ["doi"],
        },
    ),
    Tool(
        name="search_by_topic",
        description=(
            "Search for works by academic topic or concept via OpenAlex. "
            "Good for broad discipline-level discovery. "
            "For narrower keyword searches use search_academic instead."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Academic topic or concept (e.g. 'machine learning', 'climate change')"},
                "limit": {"type": "integer", "description": "Results to return (default 10)", "default": 10},
                "open_access_only": {"type": "boolean", "description": "Limit to open-access works", "default": False},
            },
            "required": ["topic"],
        },
    ),
    Tool(
        name="get_citations",
        description=(
            "Get papers that cite a given work, via Semantic Scholar. "
            "Useful for forward citation tracking. "
            "Provide the DOI (preferred) or a Semantic Scholar paper ID."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper to find citations for"},
                "limit": {"type": "integer", "description": "Max citations to return (default 20)", "default": 20},
            },
            "required": ["doi"],
        },
    ),
    Tool(
        name="get_references",
        description=(
            "Get the reference list of a paper (papers it cites), via Semantic Scholar. "
            "Useful for backward citation tracking. "
            "Provide the DOI."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper whose references to fetch"},
                "limit": {"type": "integer", "description": "Max references to return (default 20)", "default": 20},
            },
            "required": ["doi"],
        },
    ),
    Tool(
        name="get_paper_summary",
        description=(
            "Get an AI-generated one-sentence summary (TLDR) of a paper from Semantic Scholar. "
            "Provide the DOI."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper"},
            },
            "required": ["doi"],
        },
    ),
    Tool(
        name="find_open_access",
        description=(
            "Check whether a free full-text copy of a paper is available anywhere, "
            "using Unpaywall. Searches PubMed Central, institutional repositories, "
            "author websites, and legal preprint servers. Provide the DOI."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI to check for open-access availability"},
            },
            "required": ["doi"],
        },
    ),
    Tool(
        name="get_full_text_link",
        description=(
            "Get the best available access URL for a paper. "
            "Tries in order: open-access copy, Unpaywall, SFU subscription (with EZProxy if needed), "
            "then falls back to the DOI link. "
            "SFU students/staff can authenticate via EZProxy in their browser."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper"},
            },
            "required": ["doi"],
        },
    ),
    Tool(
        name="browse_sfu_databases",
        description=(
            "Browse or search SFU Library's subscribed databases and electronic resources. "
            "Returns database names, descriptions, subjects, content types, and access URLs. "
            "Use this to discover which specialized databases SFU subscribes to for a given subject."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term (database name or keyword)"},
                "subject": {"type": "string", "description": "Filter by subject area (e.g. 'Psychology', 'Chemistry')"},
                "content_type": {
                    "type": "string",
                    "description": "Filter by content type (e.g. 'Full-text database', 'Datasets', 'Ebook collection')",
                },
                "free_only": {"type": "boolean", "description": "Show only freely accessible databases", "default": False},
                "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
            },
            "required": [],
        },
    ),
    Tool(
        name="check_sfu_access",
        description=(
            "Check whether SFU Library subscribes to a specific database or resource. "
            "Provide the database name, publisher, or URL. "
            "Returns subscription status and EZProxy information."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Database or resource name to check (e.g. 'JSTOR', 'Nature', 'Web of Science')"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="search_biomedical",
        description=(
            "Search biomedical and life sciences literature via Europe PMC "
            "(43M+ records from PubMed, PMC, and preprints). "
            "Complements search_academic for health sciences, medicine, and biology."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Biomedical search query"},
                "limit": {"type": "integer", "description": "Results to return (default 10)", "default": 10},
            },
            "required": ["query"],
        },
    ),
    # ── Citations ──────────────────────────────────────────────────────────────
    Tool(
        name="generate_citation",
        description=(
            "Generate a formatted citation for a paper using its DOI. "
            "Supports APA 7th, MLA 9th, Chicago 17th, and BibTeX formats. "
            "Fetches metadata from OpenAlex and enriches with CrossRef if needed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper to cite"},
                "format": {
                    "type": "string",
                    "description": "Citation format: 'apa', 'mla', 'chicago', 'bibtex'",
                    "enum": ["apa", "mla", "chicago", "bibtex"],
                    "default": "apa",
                },
            },
            "required": ["doi"],
        },
    ),
    Tool(
        name="batch_generate_citations",
        description="Generate citations for multiple papers at once using their DOIs.",
        inputSchema={
            "type": "object",
            "properties": {
                "dois": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of DOIs (max 20)",
                },
                "format": {
                    "type": "string",
                    "enum": ["apa", "mla", "chicago", "bibtex"],
                    "default": "apa",
                },
            },
            "required": ["dois"],
        },
    ),
    Tool(
        name="export_search_results",
        description=(
            "Search and export results in various formats (JSON, CSV, BibTeX, RIS). "
            "Useful for importing into reference managers or spreadsheets."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "format": {
                    "type": "string",
                    "enum": ["json", "csv", "bibtex", "ris"],
                    "default": "bibtex",
                },
                "limit": {"type": "integer", "description": "Max results to export (default 10, max 50)", "default": 10},
            },
            "required": ["query"],
        },
    ),
    # ── Zotero ────────────────────────────────────────────────────────────────
    Tool(
        name="save_to_zotero",
        description=(
            "Save a paper to the user's Zotero library using its DOI. "
            "Automatically checks for duplicates. "
            "Optionally specify a collection name."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper to save"},
                "collection_name": {"type": "string", "description": "Zotero collection name (created if it doesn't exist)"},
                "parent_collection": {"type": "string", "description": "Parent collection name for nested collections"},
            },
            "required": ["doi"],
        },
    ),
    Tool(
        name="list_zotero_collections",
        description="List all collections in the user's Zotero library with item counts.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="batch_save_to_zotero",
        description=(
            "Save multiple papers to Zotero at once using their DOIs. "
            "Checks each for duplicates and skips items already in the library."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "dois": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of DOIs to save (max 20)",
                },
                "collection_name": {"type": "string", "description": "Zotero collection name"},
                "parent_collection": {"type": "string", "description": "Parent collection name"},
            },
            "required": ["dois", "collection_name"],
        },
    ),
    Tool(
        name="search_zotero",
        description=(
            "Search the user's existing Zotero library. "
            "Use this BEFORE saving to check for duplicates, "
            "or to answer questions about what the user already has saved."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_zotero_collection_items",
        description="List all items in a specific Zotero collection.",
        inputSchema={
            "type": "object",
            "properties": {
                "collection_name": {"type": "string", "description": "Collection name"},
                "limit": {"type": "integer", "description": "Max items (default 50)", "default": 50},
            },
            "required": ["collection_name"],
        },
    ),
    Tool(
        name="get_zotero_status",
        description="Check Zotero API connection, credentials, and permissions.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


# ── Tool handler dispatch ─────────────────────────────────────────────────────

async def handle_tool_call(
    name: str,
    arguments: dict[str, Any],
    lib_client=None,  # kept for signature compatibility, unused
) -> list[TextContent]:
    """Handle a tool call by dispatching to the appropriate handler."""
    start_time = time.time()
    success = True
    try:
        result = await _dispatch_tool(name, arguments)
        return result
    except Exception as e:
        success = False
        logger.error("Tool %s failed: %s", name, e, exc_info=True)
        return [TextContent(type="text", text=f"Error in {name}: {str(e)}")]
    finally:
        _record_metric(name, time.time() - start_time, success)


async def _dispatch_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    dispatch = {
        "search_academic": _handle_search_academic,
        "search_by_author": _handle_search_by_author,
        "search_by_doi": _handle_search_by_doi,
        "search_by_topic": _handle_search_by_topic,
        "get_citations": _handle_get_citations,
        "get_references": _handle_get_references,
        "get_paper_summary": _handle_get_paper_summary,
        "find_open_access": _handle_find_open_access,
        "get_full_text_link": _handle_get_full_text_link,
        "browse_sfu_databases": _handle_browse_sfu_databases,
        "check_sfu_access": _handle_check_sfu_access,
        "search_biomedical": _handle_search_biomedical,
        "generate_citation": _handle_generate_citation,
        "batch_generate_citations": _handle_batch_citations,
        "export_search_results": _handle_export_search,
        "save_to_zotero": _handle_save_to_zotero,
        "list_zotero_collections": _handle_list_zotero_collections,
        "batch_save_to_zotero": _handle_batch_save_to_zotero,
        "search_zotero": _handle_search_zotero,
        "get_zotero_collection_items": _handle_get_zotero_collection_items,
        "get_zotero_status": _handle_get_zotero_status,
    }
    handler = dispatch.get(name)
    if handler is None:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    return await handler(arguments)


# ── Search handlers ───────────────────────────────────────────────────────────

async def _handle_search_academic(args: dict) -> list[TextContent]:
    query = sanitize_search_query(args.get("query", ""))
    if not query:
        return [TextContent(type="text", text="Empty search query.")]
    limit = min(args.get("limit", 10), 50)
    page = max(args.get("page", 1), 1)

    filters: dict[str, str] = {}
    year_from = args.get("year_from")
    year_to = args.get("year_to")
    if year_from and year_to:
        filters["publication_year"] = f"{year_from}-{year_to}"
    elif year_from:
        filters["from_publication_date"] = f"{year_from}-01-01"
    elif year_to:
        filters["to_publication_date"] = f"{year_to}-12-31"
    if args.get("open_access_only"):
        filters["open_access.is_oa"] = "true"
    work_type = args.get("type", "")
    if work_type:
        filters["type"] = work_type

    async with _request_semaphore:
        data = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_openalex().search_works(query, filters=filters, per_page=limit, page=page),
        )

    if data.get("results"):
        _cache_works(data["results"])
    return [TextContent(type="text", text=format_openalex_results(data, query))]


async def _handle_search_by_author(args: dict) -> list[TextContent]:
    author = args.get("author", "").strip()
    if not author:
        return [TextContent(type="text", text="No author name provided.")]
    limit = min(args.get("limit", 10), 50)

    async with _request_semaphore:
        data = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_openalex().search_works(
                query=author,
                filters={"authorships.author.display_name.search": author},
                per_page=limit,
            ),
        )

    if data.get("results"):
        _cache_works(data["results"])
    return [TextContent(type="text", text=format_openalex_results(data, f"author:{author}"))]


async def _handle_search_by_doi(args: dict) -> list[TextContent]:
    doi = args.get("doi", "").strip()
    if not doi:
        return [TextContent(type="text", text="No DOI provided.")]

    # Check work cache first
    work = _lookup_work(doi)
    if not work:
        async with _request_semaphore:
            work = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _get_openalex().get_work_by_doi(doi),
            )

    if not work:
        # Fallback to CrossRef
        from lib.openalex import fetch_crossref_work
        async with _request_semaphore:
            work = await asyncio.get_event_loop().run_in_executor(
                None, lambda: fetch_crossref_work(doi)
            )

    if not work:
        return [TextContent(type="text", text=f"No record found for DOI: {doi}")]

    _cache_works([work])
    data = {"results": [work], "meta": {"count": 1}}
    return [TextContent(type="text", text=format_openalex_results(data, doi))]


async def _handle_search_by_topic(args: dict) -> list[TextContent]:
    topic = args.get("topic", "").strip()
    if not topic:
        return [TextContent(type="text", text="No topic provided.")]
    limit = min(args.get("limit", 10), 50)

    filters: dict[str, str] = {}
    if args.get("open_access_only"):
        filters["open_access.is_oa"] = "true"

    async with _request_semaphore:
        data = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_openalex().search_works(
                query=topic,
                filters=filters or None,
                sort="cited_by_count:desc",
                per_page=limit,
            ),
        )

    if data.get("results"):
        _cache_works(data["results"])
    return [TextContent(type="text", text=format_openalex_results(data, topic))]


# ── Semantic Scholar handlers ─────────────────────────────────────────────────

async def _handle_get_citations(args: dict) -> list[TextContent]:
    doi = args.get("doi", "").strip()
    if not doi:
        return [TextContent(type="text", text="No DOI provided.")]
    limit = min(args.get("limit", 20), 100)

    async with _request_semaphore:
        papers = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_s2().get_citations(f"DOI:{doi}", limit=limit),
        )

    if not papers:
        return [TextContent(type="text", text=f"No citing papers found for DOI: {doi}")]
    return [TextContent(type="text", text=format_semantic_scholar_papers(papers, f"Papers citing {doi}"))]


async def _handle_get_references(args: dict) -> list[TextContent]:
    doi = args.get("doi", "").strip()
    if not doi:
        return [TextContent(type="text", text="No DOI provided.")]
    limit = min(args.get("limit", 20), 100)

    async with _request_semaphore:
        papers = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_s2().get_references(f"DOI:{doi}", limit=limit),
        )

    if not papers:
        return [TextContent(type="text", text=f"No references found for DOI: {doi}")]
    return [TextContent(type="text", text=format_semantic_scholar_papers(papers, f"References of {doi}"))]


async def _handle_get_paper_summary(args: dict) -> list[TextContent]:
    doi = args.get("doi", "").strip()
    if not doi:
        return [TextContent(type="text", text="No DOI provided.")]

    async with _request_semaphore:
        tldr = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_s2().get_tldr(f"DOI:{doi}"),
        )

    if not tldr:
        return [TextContent(type="text", text=f"No TLDR available for DOI: {doi}")]
    return [TextContent(type="text", text=f"TLDR for {doi}:\n\n{tldr}")]


# ── Access resolution handlers ────────────────────────────────────────────────

async def _handle_find_open_access(args: dict) -> list[TextContent]:
    doi = args.get("doi", "").strip()
    if not doi:
        return [TextContent(type="text", text="No DOI provided.")]

    from lib.openalex import normalize_doi
    from lib.access_resolver import AccessResolver
    doi_norm = normalize_doi(doi)

    resolver = _get_resolver()

    async with _request_semaphore:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: resolver._check_unpaywall(doi_norm),
        )

    doi_url = f"https://doi.org/{doi_norm}"
    if result:
        output = [
            "Open Access: AVAILABLE",
            f"URL: {result['url']}",
        ]
        if result.get("version"):
            output.append(f"Version: {result['version']}")
        if result.get("host_type"):
            output.append(f"Host: {result['host_type']}")
        output.append(f"DOI: {doi_url}")
    else:
        # Check if OpenAlex cached work is OA
        work = _lookup_work(doi_norm)
        if work and work.get("is_oa") and work.get("oa_url"):
            output = [
                "Open Access: AVAILABLE (via OpenAlex)",
                f"URL: {work['oa_url']}",
                f"DOI: {doi_url}",
            ]
        else:
            output = [
                "Open Access: NOT FOUND",
                f"No freely available copy found via Unpaywall for {doi}.",
                f"DOI: {doi_url}",
            ]

    return [TextContent(type="text", text="\n".join(output))]


async def _handle_get_full_text_link(args: dict) -> list[TextContent]:
    doi = args.get("doi", "").strip()
    if not doi:
        return [TextContent(type="text", text="No DOI provided.")]

    from lib.openalex import normalize_doi
    doi_norm = normalize_doi(doi)

    # Pull cached work metadata for OA info and source URL
    work = _lookup_work(doi_norm)
    if not work:
        work = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_openalex().get_work_by_doi(doi_norm),
        )
        if work:
            _cache_works([work])

    is_oa = (work or {}).get("is_oa", False)
    oa_url = (work or {}).get("oa_url", "")
    publisher = (work or {}).get("publisher", "")
    source_url = (work or {}).get("oa_url", "") or f"https://doi.org/{doi_norm}"

    async with _request_semaphore:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_resolver().resolve(
                doi=doi_norm,
                source_url=source_url,
                publisher=publisher,
                is_oa=is_oa,
                oa_url=oa_url,
            ),
        )

    access_labels = {
        "oa": "Open Access",
        "unpaywall": "Open Access (Unpaywall)",
        "ezproxy": "SFU Subscription (EZProxy login required)",
        "direct": "SFU Subscription (direct access)",
        "doi_fallback": "DOI link (may require personal/institutional access)",
    }

    output = [
        "=" * 50,
        "FULL TEXT ACCESS",
        "=" * 50,
        f"Access type: {access_labels.get(result['access_type'], result['access_type'])}",
        f"URL: {result['access_url']}",
    ]
    if result.get("db_name"):
        output.append(f"Database: {result['db_name']}")
    if result.get("proxy_needed"):
        output.append("Note: Click the URL above and log in with your SFU credentials.")
    if result.get("doi_url") and result["doi_url"] != result["access_url"]:
        output.append(f"DOI fallback: {result['doi_url']}")

    return [TextContent(type="text", text="\n".join(output))]


# ── SFU Database Registry handlers ───────────────────────────────────────────

async def _handle_browse_sfu_databases(args: dict) -> list[TextContent]:
    query = args.get("query", "")
    subject = args.get("subject", "")
    content_type = args.get("content_type", "")
    free_only = args.get("free_only", False)
    limit = min(args.get("limit", 20), 100)

    registry = _get_registry()
    async with _request_semaphore:
        docs = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: registry.search(
                query=query,
                subject=subject,
                content_type=content_type,
                free_only=free_only,
                limit=limit,
            ),
        )

    return [TextContent(type="text", text=format_sfu_databases_list(docs, query))]


async def _handle_check_sfu_access(args: dict) -> list[TextContent]:
    name = args.get("name", "").strip()
    if not name:
        return [TextContent(type="text", text="No database name provided.")]

    registry = _get_registry()
    async with _request_semaphore:
        docs = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: registry.search(query=name, limit=5),
        )

    if not docs:
        output = [
            f"SFU Access Check: '{name}'",
            "Result: NOT FOUND in SFU database registry.",
            "SFU may not subscribe to this resource, or it may be listed under a different name.",
        ]
    else:
        output = [
            f"SFU Access Check: '{name}'",
            f"Result: FOUND — {len(docs)} matching database(s)\n",
        ]
        for doc in docs:
            output.append(format_sfu_database(doc))
            output.append("")

    return [TextContent(type="text", text="\n".join(output))]


# ── Biomedical handler ────────────────────────────────────────────────────────

async def _handle_search_biomedical(args: dict) -> list[TextContent]:
    if not _get_features().get("europe_pmc_enabled", False):
        # Fall back to OpenAlex with biomedical filter
        args_copy = dict(args)
        args_copy["query"] = args.get("query", "")
        return await _handle_search_academic(args_copy)

    query = args.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="No query provided.")]
    limit = min(args.get("limit", 10), 25)

    async with _request_semaphore:
        data = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _fetch_europe_pmc(query, limit),
        )

    return [TextContent(type="text", text=data)]


def _fetch_europe_pmc(query: str, limit: int) -> str:
    try:
        resp = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "resultType": "core", "format": "json", "pageSize": limit},
            headers={"User-Agent": "SFULibraryMCP/1.0"},
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])
        if not results:
            return "No results found in Europe PMC."
        total = resp.json().get("hitCount", len(results))
        lines = [f"Europe PMC: {total:,} results for '{query}'\n", "=" * 60 + "\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            authors = r.get("authorString", "")
            year = r.get("pubYear", "")
            doi = r.get("doi", "")
            pmid = r.get("pmid", "")
            abstract = (r.get("abstractText", "") or "")[:200]
            lines.append(f"{i}. {title}\n")
            if authors:
                lines.append(f"   Authors: {authors[:100]}\n")
            if year:
                lines.append(f"   Year: {year}\n")
            if doi:
                lines.append(f"   DOI: {doi}\n")
            if pmid:
                lines.append(f"   PMID: {pmid}\n")
            if abstract:
                lines.append(f"   Abstract: {abstract}...\n")
            lines.append("\n")
        return "".join(lines)
    except Exception as e:
        logger.error("Europe PMC request failed: %s", e)
        return f"Europe PMC search failed: {e}"


# ── Citation handlers ─────────────────────────────────────────────────────────

async def _fetch_work_metadata(doi: str) -> dict | None:
    """Fetch work metadata by DOI using cache → OpenAlex → CrossRef."""
    from lib.openalex import normalize_doi
    doi_norm = normalize_doi(doi)
    work = _lookup_work(doi_norm) or _lookup_work(doi)
    if work:
        return work

    work = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _get_openalex().get_work_by_doi(doi_norm),
    )
    if work:
        _cache_works([work])
        return work

    from lib.openalex import fetch_crossref_work
    work = await asyncio.get_event_loop().run_in_executor(
        None, lambda: fetch_crossref_work(doi_norm)
    )
    if work:
        _cache_works([work])
    return work


def _format_single_citation(metadata: dict | None, fmt: str) -> str:
    formatters = {
        "apa": format_apa_citation,
        "mla": format_mla_citation,
        "chicago": format_chicago_citation,
        "bibtex": format_bibtex_entry,
    }
    return formatters.get(fmt, format_apa_citation)(metadata)


async def _handle_generate_citation(args: dict) -> list[TextContent]:
    doi = args.get("doi", "").strip()
    if not doi:
        return [TextContent(type="text", text="No DOI provided.")]
    fmt = args.get("format", "apa").lower()

    async with _request_semaphore:
        metadata = await _fetch_work_metadata(doi)

    if not metadata:
        return [TextContent(type="text", text=f"Could not retrieve metadata for DOI: {doi}")]

    metadata = enrich_metadata_from_crossref(metadata)
    format_names = {
        "apa": "APA 7th Edition",
        "mla": "MLA 9th Edition",
        "chicago": "Chicago 17th Edition",
        "bibtex": "BibTeX",
    }
    citation = _format_single_citation(metadata, fmt)
    return [TextContent(type="text", text=f"--- {format_names.get(fmt, fmt)} ---\n\n{citation}")]


async def _handle_batch_citations(args: dict) -> list[TextContent]:
    dois = args.get("dois", [])[:20]
    fmt = args.get("format", "apa").lower()

    if not dois:
        return [TextContent(type="text", text="No DOIs provided.")]

    format_names = {
        "apa": "APA 7th Edition", "mla": "MLA 9th Edition",
        "chicago": "Chicago 17th Edition", "bibtex": "BibTeX",
    }

    async def fetch_one(doi: str) -> tuple[str, dict | None]:
        async with _request_semaphore:
            return doi, await _fetch_work_metadata(doi)

    fetched = await asyncio.gather(*[fetch_one(doi) for doi in dois], return_exceptions=True)

    output = [f"--- {format_names.get(fmt, fmt)} Citations ({len(dois)} items) ---\n"]
    for i, result in enumerate(fetched, 1):
        if isinstance(result, Exception):
            output.append(f"{i}. [Error: {result}]\n")
            continue
        doi, metadata = result
        if not metadata:
            output.append(f"{i}. [Error: Could not retrieve metadata for {doi}]\n")
            continue
        metadata = enrich_metadata_from_crossref(metadata)
        citation = _format_single_citation(metadata, fmt)
        output.append(f"{citation}\n" if fmt == "bibtex" else f"{i}. {citation}\n")

    return [TextContent(type="text", text="\n".join(output))]


async def _handle_export_search(args: dict) -> list[TextContent]:
    query = sanitize_search_query(args.get("query", ""))
    export_format = args.get("format", "bibtex").lower()
    limit = min(args.get("limit", 10), 50)

    async with _request_semaphore:
        data = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_openalex().search_works(query, per_page=limit),
        )

    works = data.get("results", [])
    if not works:
        return [TextContent(type="text", text="No results found.")]

    _cache_works(works)
    total = data.get("meta", {}).get("count", len(works))

    if export_format == "json":
        export_data = [
            {
                "title": w.get("title", ""),
                "authors": w.get("authors", []),
                "date": w.get("date", ""),
                "type": w.get("type", ""),
                "publisher": w.get("publisher", ""),
                "source": w.get("source", ""),
                "doi": w.get("doi", ""),
                "issn": w.get("issn", ""),
                "is_oa": w.get("is_oa", False),
                "cited_by_count": w.get("cited_by_count", 0),
                "openalex_id": w.get("openalex_id", ""),
            }
            for w in works
        ]
        header = f"// JSON Export: {len(works)} of {total:,} results for '{query}'\n\n"
        return [TextContent(type="text", text=header + json.dumps(export_data, indent=2))]

    if export_format == "csv":
        lines = ["doi,title,authors,date,source,type,is_oa,cited_by_count"]
        for w in works:
            def esc(v):
                v = str(v or "").replace('"', '""')
                return f'"{v}"' if "," in v or '"' in v else v
            lines.append(",".join([
                esc(w.get("doi", "")), esc(w.get("title", "")),
                esc("; ".join(w.get("authors", []))),
                esc(w.get("date", "")), esc(w.get("source", "")),
                esc(w.get("type", "")), esc(w.get("is_oa", False)),
                esc(w.get("cited_by_count", 0)),
            ]))
        return [TextContent(type="text", text=f"# CSV Export: {len(works)} of {total:,} results\n" + "\n".join(lines))]

    if export_format == "bibtex":
        entries = [format_bibtex_entry(enrich_metadata_from_crossref(w)) for w in works]
        header = f"% BibTeX Export: {len(works)} of {total:,} results for '{query}'\n\n"
        return [TextContent(type="text", text=header + "\n\n".join(entries))]

    if export_format == "ris":
        entries = [format_ris_entry(enrich_metadata_from_crossref(w)) for w in works]
        header = f"# RIS Export: {len(works)} of {total:,} results for '{query}'\n\n"
        return [TextContent(type="text", text=header + "\n\n".join(entries))]

    return [TextContent(type="text", text=f"Unknown export format: {export_format}")]


# ── Zotero handlers ───────────────────────────────────────────────────────────

async def _handle_save_to_zotero(args: dict) -> list[TextContent]:
    auth_err = _ensure_zotero_auth()
    if auth_err:
        return auth_err

    doi = args.get("doi", "").strip()
    if not doi:
        return [TextContent(type="text", text="No DOI provided.")]
    collection_name = args.get("collection_name", "")
    parent_collection = args.get("parent_collection", "")

    async with _request_semaphore:
        metadata = await _fetch_work_metadata(doi)

    if not metadata:
        return [TextContent(type="text", text=f"Could not retrieve metadata for DOI: {doi}")]

    metadata = enrich_metadata_from_crossref(metadata)
    zot = _get_zotero_client()

    try:
        dup = zot.check_duplicate(metadata)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error during duplicate check: {e}")]

    if dup["is_duplicate"]:
        return [TextContent(type="text", text=(
            "=" * 50 + "\nALREADY IN ZOTERO\n" + "=" * 50 +
            f"\nThis item is already in your Zotero library ({dup['match_type']} match).\n"
            f"\nExisting item:\n{dup.get('existing_item_summary', 'N/A')}"
        ))]

    zotero_item = zot.metadata_to_zotero_item(metadata)
    collection_key = None
    if collection_name:
        try:
            parent_key = zot.find_or_create_collection(parent_collection) if parent_collection else None
            collection_key = zot.find_or_create_collection(collection_name, parent_key=parent_key)
        except ZoteroError as e:
            return [TextContent(type="text", text=f"Zotero collection error: {e}")]

    try:
        item_key = zot.create_item(zotero_item, collection_key)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Failed to create Zotero item: {e}")]

    output = ["=" * 50, "SAVED TO ZOTERO", "=" * 50,
              f"\nTitle: {metadata.get('title', 'Unknown')}", f"Zotero Key: {item_key}"]
    if collection_name:
        output.append(f"Collection: {collection_name}")
    return [TextContent(type="text", text="\n".join(output))]


async def _handle_list_zotero_collections(args: dict) -> list[TextContent]:
    auth_err = _ensure_zotero_auth()
    if auth_err:
        return auth_err
    try:
        collections = _get_zotero_client().list_collections()
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error: {e}")]
    if not collections:
        return [TextContent(type="text", text="No collections found in your Zotero library.")]
    output = ["=" * 50, f"ZOTERO COLLECTIONS ({len(collections)})", "=" * 50, ""]
    for c in collections:
        parent = f" (in: {c['parent_key']})" if c.get("parent_key") else ""
        output.append(f"  {c['name']} — {c['num_items']} items{parent}\n    Key: {c['key']}")
    return [TextContent(type="text", text="\n".join(output))]


async def _handle_batch_save_to_zotero(args: dict) -> list[TextContent]:
    auth_err = _ensure_zotero_auth()
    if auth_err:
        return auth_err

    dois = args.get("dois", [])[:20]
    collection_name = args.get("collection_name", "")
    parent_collection = args.get("parent_collection", "")

    if not dois:
        return [TextContent(type="text", text="No DOIs provided.")]

    zot = _get_zotero_client()
    collection_key = None
    if collection_name:
        try:
            parent_key = zot.find_or_create_collection(parent_collection) if parent_collection else None
            collection_key = zot.find_or_create_collection(collection_name, parent_key=parent_key)
        except ZoteroError as e:
            return [TextContent(type="text", text=f"Zotero collection error: {e}")]

    saved, skipped, failed = 0, 0, 0
    details = []

    for doi in dois:
        async with _request_semaphore:
            metadata = await _fetch_work_metadata(doi)
        if not metadata:
            details.append(f"  {doi}: FAILED — could not retrieve metadata")
            failed += 1
            continue

        metadata = enrich_metadata_from_crossref(metadata)
        title_short = metadata.get("title", "Unknown")[:60]

        try:
            dup = zot.check_duplicate(metadata)
        except ZoteroError:
            dup = {"is_duplicate": False}

        if dup["is_duplicate"]:
            details.append(f"  {title_short}: SKIPPED — already in Zotero ({dup['match_type']} match)")
            skipped += 1
            continue

        try:
            zot.create_item(zot.metadata_to_zotero_item(metadata), collection_key)
            details.append(f"  {title_short}: SAVED")
            saved += 1
        except ZoteroError as e:
            details.append(f"  {title_short}: FAILED — {e}")
            failed += 1

    output = [
        "=" * 50, "BATCH SAVE TO ZOTERO", "=" * 50,
        f"\nCollection: {collection_name or '(none)'}",
        f"Results: {saved} saved, {skipped} skipped, {failed} failed\n",
    ] + details
    return [TextContent(type="text", text="\n".join(output))]


async def _handle_search_zotero(args: dict) -> list[TextContent]:
    auth_err = _ensure_zotero_auth()
    if auth_err:
        return auth_err
    query = args.get("query", "")
    limit = min(args.get("limit", 20), 100)
    if not query:
        return [TextContent(type="text", text="No search query provided.")]
    try:
        items = _get_zotero_client().search_items(query, limit)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero search error: {e}")]
    if not items:
        return [TextContent(type="text", text=f"No items found in Zotero for '{query}'.")]
    zot = _get_zotero_client()
    output = ["=" * 50, f"ZOTERO SEARCH: '{query}' ({len(items)} results)", "=" * 50, ""]
    for i, item in enumerate(items, 1):
        output.append(f"--- Item {i} ---\n{zot.format_item_summary(item)}\n")
    return [TextContent(type="text", text="\n".join(output))]


async def _handle_get_zotero_collection_items(args: dict) -> list[TextContent]:
    auth_err = _ensure_zotero_auth()
    if auth_err:
        return auth_err
    collection_name = args.get("collection_name", "")
    limit = min(args.get("limit", 50), 100)
    if not collection_name:
        return [TextContent(type="text", text="No collection name provided.")]
    zot = _get_zotero_client()
    try:
        collection_key = zot.find_collection_by_name(collection_name)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error: {e}")]
    if not collection_key:
        return [TextContent(type="text", text=f"Collection '{collection_name}' not found.")]
    try:
        items = zot.get_collection_items(collection_key, limit)
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error: {e}")]
    if not items:
        return [TextContent(type="text", text=f"No items in collection '{collection_name}'.")]
    output = ["=" * 50, f"ZOTERO COLLECTION: '{collection_name}' ({len(items)} items)", "=" * 50, ""]
    for i, item in enumerate(items, 1):
        output.append(f"--- Item {i} ---\n{zot.format_item_summary(item)}\n")
    return [TextContent(type="text", text="\n".join(output))]


async def _handle_get_zotero_status(args: dict) -> list[TextContent]:
    if not _get_features().get("zotero_enabled", True):
        return [TextContent(type="text", text="Zotero integration is disabled.")]
    try:
        result = _get_zotero_client().verify_credentials()
    except ZoteroError as e:
        return [TextContent(type="text", text=f"Zotero error: {e}")]
    output = ["=" * 50, "ZOTERO STATUS", "=" * 50]
    if result["valid"]:
        access = result.get("access", {})
        output += [
            "\nCredentials: VALID",
            f"Username: {result.get('username', 'N/A')}",
            f"User ID: {result.get('userID', 'N/A')}",
            "\nPermissions:",
            f"  Library: {access.get('library', False)}",
            f"  Files: {access.get('files', False)}",
            f"  Write: {access.get('write', False)}",
        ]
    else:
        output += ["\nCredentials: INVALID", f"Reason: {result.get('message', 'Unknown')}"]
    return [TextContent(type="text", text="\n".join(output))]
