# sfu-library-mcp

An MCP (Model Context Protocol) server that provides authenticated access to the Simon Fraser University Library database, enabling Claude Desktop to search, retrieve, cite, and download academic articles directly within the AI assistant interface.

## Tech Stack

| Technology | Usage |
|------------|-------|
| Python | 95% - Core application logic, MCP server implementation, API clients |
| MCP SDK | 2% - Model Context Protocol server framework |
| Selenium | 1% - Browser automation for CAS/MFA authentication |
| Playwright | 0.5% - Headless browser for PDF downloads (Tier 2 fallback) |
| curl_cffi | 0.5% - TLS fingerprint impersonation for anti-detection |
| FastAPI/Uvicorn | 0.5% - Production HTTP server |
| PyZotero | 0.5% - Zotero API integration |

## Architecture

The project follows a modular architecture with clear separation between transport layers, business logic, and integration points.

### Transport Layers

Two entry points support different deployment scenarios:

- **stdio transport** (`sfu_library_mcp_server.py`) - Standard MCP transport for local Claude Desktop
- **HTTP transport** (`sfu_library_mcp_http.py`) - StreamableHTTPSessionManager-based server for remote/mcp-remote access with health check endpoint

### Core Modules (`src/lib/`)

| Module | Responsibility |
|--------|----------------|
| `client.py` | SFULibraryClient - Primo API authentication, JWT token lifecycle, search operations, session persistence |
| `tools.py` | MCP tool definitions (14 tools) with input validation, metrics logging, batch operations |
| `zotero.py` | Zotero API wrapper - saves items, manages collections, multi-signal duplicate detection |
| `downloader.py` | ArticleDownloader - tiered PDF download strategy, caching, text extraction |
| `citations.py` | Citation formatter - APA, MLA, Chicago, BibTeX, RIS with CrossRef enrichment |
| `formatters.py` | Result formatting for search responses with character encoding handling |
| `config.py` | ServerConfig - environment variable and Docker secret loading, feature flags |
| `reranker.py` | Two-stage result re-ranking by relevance, recency, availability, completeness |
| `rate_limiter.py` | DownloadRateLimiter - anti-detection with session budgets, hourly caps, per-domain limits |
| `publisher_router.py` | PublisherRouter - runtime learning of per-domain download preferences |
| `stealth.py` | Browser fingerprint spoofing and anti-detection hardening |
| `retry.py` | Exponential backoff decorator and CircuitBreaker for fault tolerance |
| `cache.py` | ResponseCache - TTL-based LRU cache with memory bounds |

### External Services

- **SFU Primo API** - Library database search and retrieval
- **SFU CAS + Duo** - Central authentication with multi-factor authentication
- **CrossRef API** - Metadata enrichment for incomplete citation records
- **Zotero API** - Reference management integration
- **Publisher platforms** - Direct PDF downloads (Wiley, Springer, SAGE, etc.)

## Key Features

### Search & Discovery
- General search with field filtering (title, author, subject, ISBN)
- Author-specific and subject/topic searches
- Batch ISBN lookup (up to 20 items)
- Electronic resource-only filtering
- Detailed item retrieval with full metadata

### Authentication
- Automatic CAS + Duo MFA via Selenium automation
- JWT token caching with 24-hour expiry and auto-refresh
- Token encryption at rest (Fernet, optional)
- Session persistence across server restarts
- MFA method fallback and retry logic

### Citation Management
- Generate citations in APA, MLA, Chicago, and BibTeX formats
- Batch citation generation for multiple items
- Export results to JSON, CSV, BibTeX, or RIS formats
- CrossRef metadata enrichment for incomplete records

### PDF Download
- Three-tiered download strategy (curl_cffi → Playwright → requests)
- PDF caching with optional host folder copy
- Text extraction for LLM consumption
- Publisher-specific routing optimization
- Anti-detection rate limiting and stealth modes

### Zotero Integration
- Save items directly to Zotero library
- Collection management and organization
- Multi-signal duplicate detection (title, DOI, ISBN)

### Production Features
- Circuit breaker pattern for fault tolerance
- Response caching with TTL-based LRU eviction
- Request queuing with semaphores
- Metrics logging (request count, latency)
- Health check endpoint for monitoring

## Development Activity

Recent commits focus on production hardening and deployment automation:

- **HTTP Transport Fixes** - Corrected `/mcp` routing to avoid Mount's 307 redirect; implemented StreamableHTTPSessionManager for proper transport handling
- **Monitoring & Operations** - Added log access and health monitoring scripts for production observability
- **Deployment Automation** - Created update and deploy scripts for TrueNAS with production docker-compose configuration
- **Security** - Removed hardcoded credentials from config, implemented secure credential loading via Docker secrets
- **Infrastructure** - Added production Docker image with `.env` autoloading and Playwright browser installation
- **Testing** - Fixed tests for credential removal and added secret reading tests
