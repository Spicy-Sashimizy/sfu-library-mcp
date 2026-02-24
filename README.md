The portfolio README is complete. Here is the final content:

---

# sfu-library-mcp

MCP server providing authenticated access to Simon Fraser University Library via Primo API for AI-assisted academic research

## Tech Stack

- **Python** (85%) - Core server implementation with async/await patterns
- **Model Context Protocol SDK** (8%) - MCP server framework and tool definitions
- **FastAPI/Uvicorn** (5%) - HTTP transport layer for remote access
- **Selenium/Playwright** (2%) - Browser automation for CAS + Duo MFA authentication

## Architecture

### Entry Points
- **STDIO MCP Server** - Primary interface for Claude Desktop integration
- **HTTP Transport Server** - FastAPI-based server for remote client access
- **Capture Server** - Browser-assisted PDF download coordination

### Core Modules
- **Authentication Layer** - CAS + Duo MFA authentication with JWT token management, automatic refresh, caching, and encryption at rest
- **Client Module** - Primo API communication with session management and cookie persistence
- **Tools Dispatcher** - 14 MCP tools with input validation and result formatting
- **Configuration System** - Environment-based config with Docker secrets support

### Supporting Systems
- **Search & Discovery** - Query building, execution, and multi-signal result re-ranking
- **Citation Engine** - Multi-format citation generation (APA, MLA, Chicago, BibTeX, RIS) with CrossRef enrichment
- **Content Downloader** - Tiered PDF retrieval strategy with anti-detection measures and per-domain learning
- **Zotero Integration** - Direct library management with PNX-to-Zotero mapping and duplicate detection
- **Cache Layer** - TTL-based LRU caching with memory bounds
- **Resilience Patterns** - Circuit breaker, exponential backoff, and rate limiting

## Key Features

- **Authenticated Library Access** - CAS + Duo MFA authentication with 24-hour token persistence
- **Comprehensive Search** - General, author, subject, ISBN/ISSN searches with electronic resource filtering
- **Multi-Format Citations** - Generate citations in APA, MLA, Chicago, BibTeX, and RIS formats
- **PDF Retrieval** - Intelligent tiered downloading with anti-bot detection and caching
- **Zotero Integration** - Save items directly to Zotero with duplicate detection
- **Result Re-ranking** - Multi-signal relevance scoring for improved search results
- **Production Ready** - Docker deployment with health monitoring, structured logging, and metrics

## Development Activity

Recent work focuses on production deployment and HTTP transport improvements:
- Fixed HTTP transport routing to eliminate 307 redirects
- Implemented StreamableHTTPSessionManager for robust session handling
- Added log access and health monitoring scripts
- Created update and deployment automation for TrueNAS
- Removed hardcoded credentials in favor of Docker secrets
- Added comprehensive logging across all modules

The project includes 41 Python modules (~12,600 lines) with full test coverage and production Docker configurations.

---

*Auto-generated portfolio view by ClaudeBox*
