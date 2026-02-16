# SFU Library MCP Server - outdated there where new stuff added check commits


A Model Context Protocol (MCP) server that provides Claude Desktop access to the SFU Library database through the Primo API.

## Features

- **Authenticated Access**: Automatic authentication via CAS + Duo MFA
- **JWT Token Caching**: Tokens cached for ~24 hours to avoid re-authentication
- **Multiple Search Types**: General search, author, subject, ISBN, electronic resources
- **Detailed Item Information**: Get full metadata including descriptions, subjects, and access links
- **Citation Generation**: APA, MLA, Chicago, and BibTeX citation formats
- **Export Functionality**: Export results to JSON, CSV, BibTeX, or RIS formats
- **Batch Operations**: Look up multiple ISBNs or generate multiple citations at once
- **Seamless Integration**: Works with Claude Desktop's research feature

## Available Tools (14 total)

### Search Tools
| Tool | Description |
|------|-------------|
| `search_library` | Search for books, articles, journals with filtering options |
| `search_by_author` | Search for works by a specific author |
| `search_by_subject` | Search by subject/topic |
| `search_by_isbn` | Look up books by ISBN |
| `search_electronic_resources` | Search for e-books and online resources only |
| `get_item_details` | Get full details for a specific record |

### Citation & Export Tools
| Tool | Description |
|------|-------------|
| `generate_citation` | Generate citation in APA, MLA, Chicago, or BibTeX format |
| `batch_generate_citations` | Generate citations for multiple items at once |
| `export_search_results` | Export search results to JSON, CSV, BibTeX, or RIS |
| `get_full_text_links` | Extract PDF/HTML/DOI access links for an item |
| `batch_isbn_lookup` | Look up multiple ISBNs at once (max 20) |

### Authentication Tools
| Tool | Description |
|------|-------------|
| `get_token_status` | Check authentication status |
| `authenticate` | Manually authenticate or refresh token |
| `clear_cache` | Clear cached authentication token |

## Installation

1. Install dependencies:
```bash
pip install mcp selenium requests pyotp
```

2. Ensure Chrome/ChromeDriver is installed for authentication

3. The Claude Desktop configuration has been added to:
   `%APPDATA%\Claude\claude_desktop_config.json`

4. Restart Claude Desktop to load the MCP server

## Usage in Claude Desktop

Once configured, you can ask Claude to:

### Search Examples
- "Search the SFU library for machine learning textbooks"
- "Find articles by author Einstein"
- "Look up ISBN 978-0-13-468599-1"
- "Search for electronic resources about Python programming"

### Citation Examples
- "Generate an APA citation for this book"
- "Create a BibTeX entry for this article"
- "Generate citations for all these sources in MLA format"

### Export Examples
- "Export search results for 'climate change' as BibTeX"
- "Export the top 20 results about data science to CSV"
- "Get the RIS export for these articles so I can import to Zotero"

### Batch Operations
- "Look up these 5 ISBNs and tell me which ones are available"
- "Get the full text links for this journal article"

### Authentication
- "Check my library authentication status"

## JWT Token Management

- Tokens are valid for approximately 24 hours
- Automatically refreshed 5 minutes before expiry
- Cached in `token_cache.json` in the server directory
- Use `clear_cache` tool if experiencing authentication issues

## Files

| File | Description |
|------|-------------|
| `sfu_library_mcp_server.py` | Main MCP server implementation |
| `test_client.py` | Test script for client functionality |
| `token_cache.json` | Cached JWT token (auto-generated) |
| `requirements.txt` | Python dependencies |

## Troubleshooting

- **Authentication fails**: Run `clear_cache` and try again
- **Token expired**: Server auto-refreshes; if persistent, restart Claude Desktop
- **No results**: Check search query and try different field/scope options
