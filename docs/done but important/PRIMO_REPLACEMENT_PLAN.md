# Primo-Free MCP Server Implementation Plan

## Goal

Replace the Ex Libris Primo API dependency with open, free academic APIs and the SFU Library's own public Solr endpoint. No JWT tokens, no browser automation, no proprietary APIs.

---

## Architecture Overview

```
Claude Desktop / Remote Client
    │
    ├─ Stdio (local dev)
    │  └→ sfu_library_mcp_server.py
    │
    └─ HTTP (remote via mcp-remote)
       └→ sfu_library_mcp_http.py (port 8080)
          │
          ├─→ SFU Database Registry (Layer 1)
          │   └─→ Solr API at databases.lib.sfu.ca (public, no auth)
          │        → Cached locally, refreshed daily/weekly
          │
          ├─→ Open Academic Search (Layer 2)
          │   ├─→ OpenAlex API (primary search/discovery)
          │   ├─→ Semantic Scholar API (citation context, TLDRs)
          │   ├─→ CrossRef API (DOI enrichment — already implemented)
          │   ├─→ Unpaywall API (OA link resolution)
          │   └─→ Europe PMC (biomedical, optional)
          │
          ├─→ Access Resolution (Layer 3)
          │   ├─→ Unpaywall check → free full-text?
          │   ├─→ SFU subscription match → EZProxy URL for user
          │   └─→ No auth needed — user clicks link in browser
          │
          └─→ Zotero Integration (keep as-is)
              └─→ Zotero API (user's API key)
```

---

## Layer 1: SFU Database Registry

### Source: Solr Endpoint (Public, No Auth)

**Base URL:**
```
https://databases.lib.sfu.ca/solr/sfu_databases/select
```

**Fetch all records (one request):**
```
GET https://databases.lib.sfu.ca/solr/sfu_databases/select?q=*:*&rows=1000&wt=json
```

Returns all **764 database records** in a single response (~200KB). No authentication required. Response times: 0-83ms.

### Complete Field Reference

| Field | Type | Present In | Description |
|-------|------|------------|-------------|
| `id` | string | 764/764 (100%) | Alma MMS ID, e.g. `"61228559990003610"` |
| `name` | string | 764/764 (100%) | Primary database name, e.g. `"JSTOR"` |
| `names` | string[] | 496/764 (65%) | Alternative names/abbreviations, e.g. `["CEWBJ", "Works of Ben Jonson online"]` |
| `description` | text | 764/764 (100%) | Prose description of the database |
| `url` | string | 764/764 (100%) | Access URL. **11 records have empty string** (physical/terminal-only resources) |
| `provider` | string | 537/764 (70%) | Platform provider, e.g. `"EBSCOhost"`. 227 records have no provider |
| `subjects` | string[] | 748/764 (98%) | Subject classifications, e.g. `["Chemistry", "Health Sciences"]` |
| `subjectRanks` | string[] | 745/764 (97%) | Format: `"SubjectName\|rank"` where rank 0-6 (display ordering weight) |
| `contentTypes` | string[] | 754/764 (99%) | Content type labels, e.g. `["Full-text database", "Index"]` |
| `free` | boolean | 764/764 (100%) | `true` = freely available (242 records), `false` = SFU subscription (522 records) |
| `proxy` | boolean | 764/764 (100%) | `true` = requires EZProxy (487 records), `false` = direct access (277 records) |
| `publicNote` | text | 168/764 (22%) | Access restrictions/notes. **May contain HTML** (`<strong>`, `<a>`, `<p>` tags) |
| `firstChar` | string | 764/764 (100%) | Lowercase first letter of sort-name, for alphabetical browsing |
| `loaded` | date | 764/764 (100%) | Timestamp of last batch load, e.g. `"2026-03-31T13:05:27.563Z"` |
| `_version_` | long | 764/764 (100%) | Solr internal version (ignore) |

### Query Capabilities

All standard Solr query features work on the `/select` endpoint:

```bash
# Fetch all records
?q=*:*&rows=1000&wt=json

# Text search in name
?q=name:JSTOR&wt=json

# Text search in description (tokenized, partial match)
?q=description:psychology&wt=json

# Wildcard search
?q=name:*bio*&wt=json

# Filter by provider (exact match)
?q=*:*&fq=provider:EBSCOhost&wt=json

# Filter by subject
?q=*:*&fq=subjects:Chemistry&wt=json

# Filter by content type
?q=*:*&fq=contentTypes:Datasets&wt=json

# Free databases only
?q=*:*&fq=free:true&wt=json

# Multiple filters (AND logic)
?q=*:*&fq=free:true&fq=subjects:Chemistry&wt=json

# Select specific fields (bandwidth optimization)
?q=*:*&fl=id,name,url,free,proxy&rows=1000&wt=json

# Pagination
?q=*:*&rows=50&start=100&wt=json

# Faceted counts
?q=*:*&rows=0&wt=json&facet=true&facet.field=subjects&facet.limit=200
```

**Note:** `sort=name asc` sorts by tokenized first token, not full string. Use `firstChar` for alphabetical browsing.

### Solr Admin Endpoints

NOT exposed (404):
- `/solr/sfu_databases/schema`
- `/solr/sfu_databases/schema/fields`
- `/solr/sfu_databases/admin/luke`

Only `/select` is publicly accessible.

### Complete Taxonomy

#### Providers (107 unique, top 20)

| Provider | Count |
|----------|-------|
| EBSCOhost | 71 |
| SFU Library Digital Collections | 67 |
| Galegroup | 50 |
| ProQuest | 49 |
| Wharton Research Data Services | 34 |
| Alexander Street Press | 33 |
| Adam Matthew Digital | 26 |
| Abacus Dataverse Network | 12 |
| Ovid | 11 |
| CHASS at University of Toronto | 10 |
| Web of Science | 10 |
| SAGE | 9 |
| Taylor & Francis eBooks | 9 |
| Oxford University Press | 8 |
| Elsevier | 7 |
| Springer | 6 |
| East View | 5 |
| Wiley Online Library | 5 |
| Hein Online | 4 |
| Westlaw | 4 |

Plus 87 more with 1-3 records each (Cambridge Core, JSTOR, IEEE Xplore, ACM Digital Library, etc.)

#### Subjects (107 unique, top 30)

| Subject | Count |
|---------|-------|
| History | 124 |
| General & Multidisciplinary | 111 |
| English - General | 64 |
| Finance | 64 |
| Canadian Studies | 58 |
| Economics | 56 |
| Political Science | 56 |
| History - Social & Cultural | 50 |
| History - Canada | 48 |
| Health Sciences | 46 |
| Indigenous Studies | 46 |
| Sociology | 46 |
| Contemporary Arts | 44 |
| Criminology | 44 |
| Interactive Arts & Technology (SIAT) | 44 |
| Business Administration | 43 |
| Gender, Sexuality, and Women's Studies | 43 |
| Biomedical Physiology and Kinesiology (BPK) | 41 |
| Resource & Environmental Management | 41 |
| Biological Sciences | 39 |
| Communication | 38 |
| Education | 35 |
| Psychology | 33 |
| Chemistry | 27 |
| Law | 27 |
| Computing Science | 23 |
| Geography | 19 |
| Philosophy | 13 |
| Dance | 11 |
| Forensics | 10 |

#### Content Types (17 total — complete)

| Content Type | Count |
|-------------|-------|
| Index | 147 |
| Ejournal collection | 137 |
| Ebook collection | 95 |
| Full-text database | 94 |
| Datasets | 89 |
| Partial full-text database | 77 |
| Primary sources | 64 |
| Digital collection | 61 |
| News sources | 43 |
| Ereference collection | 38 |
| Image collection | 30 |
| Streaming video | 22 |
| Streaming audio | 20 |
| Theses | 13 |
| Geospatial | 8 |
| Statistical sources | 8 |
| Partial database | 5 |

### Edge Cases

- **11 records have empty URLs** — physical/on-premises resources (Bloomberg terminals, LSEG Workspace, PCensus, etc.). These have `publicNote` explaining access.
- **16 records have no subjects** — Regional Business News, various SFU digital collections
- **10 records have no contentTypes** — Dictionary of Old English, SFU special collections
- **227 records have no provider** — many free/open resources and SFU-specific collections
- **`publicNote` may contain raw HTML** — must be sanitized/stripped when displayed

### Caching Strategy

- Fetch all 764 records on startup and cache in memory
- Refresh on a configurable interval (daily recommended — the `loaded` timestamp shows batch loads)
- Cache file on disk as JSON for cold-start fallback
- Total payload: ~200KB, negligible memory footprint

---

## Layer 2: Open Academic Search APIs

### Primary: OpenAlex

**Replaces:** Primo search/discovery for articles, papers, books

| Property | Detail |
|----------|--------|
| Base URL | `https://api.openalex.org/` |
| Key Endpoints | `/works` (search), `/works/{id}` (detail), `/authors`, `/sources`, `/institutions` |
| Authentication | None required. Add `mailto=you@sfu.ca` to enter polite pool |
| Rate Limits | Polite pool: ~10 req/s. Without mailto: ~1 req/s |
| Coverage | 250M+ scholarly works, all disciplines |
| License | CC0 (fully open data) |

**Search Example:**
```
GET https://api.openalex.org/works?search=machine+learning&mailto=you@sfu.ca&per_page=25
```

**Rich Filtering:**
```
GET https://api.openalex.org/works?filter=default.search:neural+networks,publication_year:2024-2026,open_access.is_oa:true&sort=cited_by_count:desc&per_page=25&mailto=you@sfu.ca
```

**Metadata Returned:**
- `id` — OpenAlex ID
- `doi` — DOI URL
- `title` — Work title
- `authorships[]` — Authors with names, affiliations, ORCID
- `publication_date` — ISO date
- `primary_location.source` — Journal/source with ISSN
- `open_access.is_oa` — Boolean
- `open_access.oa_url` — Direct link to OA copy
- `cited_by_count` — Citation count
- `abstract_inverted_index` — Abstract (requires reconstruction from inverted index)
- `concepts[]` / `topics[]` — Subject classification with scores
- `type` — article, book, dataset, etc.
- `biblio` — Volume, issue, first/last page
- `referenced_works[]` — Outgoing references
- `related_works[]` — Similar works

**Filters available:** `publication_year`, `open_access.is_oa`, `type`, `authorships.author.id`, `primary_location.source.id`, `concepts.id`, `topics.id`, `cited_by_count`, `from_publication_date`, `to_publication_date`, `has_doi`, `has_abstract`, and many more.

### Secondary: Semantic Scholar

**Complements OpenAlex with:** citation graphs, TLDR summaries, SPECTER embeddings

| Property | Detail |
|----------|--------|
| Base URL | `https://api.semanticscholar.org/graph/v1/` |
| Key Endpoints | `/paper/search`, `/paper/{id}`, `/paper/{id}/citations`, `/paper/{id}/references` |
| Authentication | Optional free API key (request at semanticscholar.org) |
| Rate Limits | No key: ~100 req/5min. With key: ~1 req/s sustained |
| Coverage | 215M+ papers, strong in CS, biomedical, STEM |

**Search Example:**
```
GET https://api.semanticscholar.org/graph/v1/paper/search?query=machine+learning&limit=25&fields=title,authors,abstract,year,citationCount,openAccessPdf,tldr
```

**Unique Features:**
- `tldr` — AI-generated one-line summary
- `influentialCitationCount` — Citations that are particularly important
- `embedding` — SPECTER vector for similarity search
- Batch endpoint: up to 500 papers per request

### DOI Enrichment: CrossRef (Keep As-Is)

Already implemented in current codebase. No changes needed.

| Property | Detail |
|----------|--------|
| Base URL | `https://api.crossref.org/` |
| Key Endpoints | `/works/{doi}`, `/works?query=...` |
| Authentication | None (use `mailto` in User-Agent for polite pool) |
| Rate Limits | Polite: ~50 req/s |
| Coverage | 150M+ DOI records |

### OA Link Resolution: Unpaywall

**Purpose:** Given a DOI, find if a free full-text copy exists anywhere

| Property | Detail |
|----------|--------|
| Base URL | `https://api.unpaywall.org/v2/` |
| Endpoint | `/v2/{doi}?email=you@sfu.ca` |
| Authentication | Email parameter only (no registration) |
| Rate Limits | 100,000 req/day, ~10 req/s |
| Coverage | Checks ~130M DOIs against 50M+ OA copies |

**Response includes:**
- `is_oa` — Boolean
- `best_oa_location.url_for_pdf` — Direct PDF link
- `best_oa_location.url_for_landing_page` — Landing page
- `best_oa_location.version` — publishedVersion, acceptedManuscript, submittedVersion
- `best_oa_location.license` — CC license if applicable
- `oa_locations[]` — All known OA copies

### Biomedical: Europe PMC (Optional)

| Property | Detail |
|----------|--------|
| Base URL | `https://www.ebi.ac.uk/europepmc/webservices/rest/` |
| Endpoint | `/search?query={query}&resultType=core&format=json` |
| Authentication | None |
| Rate Limits | ~3 req/s recommended |
| Coverage | 43M+ records (PubMed, PMC, preprints) |

**Unique:** Returns full-text XML for PMC-indexed articles, MeSH terms, text-mined annotations.

### Preprints: arXiv API (Optional)

| Property | Detail |
|----------|--------|
| Base URL | `http://export.arxiv.org/api/query` |
| Format | Atom/XML (not JSON) |
| Authentication | None |
| Rate Limits | 1 req/3s (strict) |
| Coverage | 2.5M+ preprints (physics, math, CS, biology, economics) |

---

## Layer 3: Access Resolution

### Flow for Each Search Result

```
1. User searches via MCP tool
2. OpenAlex returns results with DOI and OA status
    │
    ├─ If OpenAlex says is_oa=true → return oa_url directly
    │
    ├─ If not OA → check Unpaywall for alternative OA copy
    │   ├─ OA copy found → return Unpaywall URL
    │   └─ No OA copy → continue to step 3
    │
    └─ 3. Match against SFU Database Registry
        ├─ Publisher/source found in SFU subscriptions?
        │   ├─ Yes, proxy=true → construct EZProxy URL:
        │   │   https://proxy.lib.sfu.ca/login?url={article_url}
        │   │   Return URL — user authenticates in their browser
        │   ├─ Yes, proxy=false → return direct URL
        │   └─ No match → return DOI link (user may have personal access)
        │
        └─ Always include: DOI link as universal fallback
```

### EZProxy URL Construction

```python
EZPROXY_BASE = "https://proxy.lib.sfu.ca/login?url="

def get_access_url(article_url: str, needs_proxy: bool) -> str:
    if needs_proxy:
        return f"{EZPROXY_BASE}{article_url}"
    return article_url
```

### Matching Articles to SFU Subscriptions

To determine if an article's source is available through SFU:

1. Extract the publisher/source domain from the article URL
2. Match against `url` domains in the SFU Database Registry
3. Or match the journal name against known databases (e.g., if source is "Nature", check if Nature is in the registry)
4. Use the `proxy` field to determine if EZProxy URL is needed

**Publisher domain mapping example:**
```python
# Extract domain from SFU database URLs
# e.g., "https://www.jstor.org" → "jstor.org"
# Then match article URLs against these domains
```

---

## Proposed MCP Tools

### New Tools (replacing Primo-based ones)

| Tool | Description | Primary API |
|------|-------------|-------------|
| `search_academic` | General academic search with filters | OpenAlex |
| `search_by_author` | Find works by author name/ORCID | OpenAlex |
| `search_by_doi` | Look up a specific DOI | OpenAlex + CrossRef |
| `search_by_topic` | Browse by academic topic/concept | OpenAlex |
| `get_citations` | Get papers that cite a given work | Semantic Scholar |
| `get_references` | Get a paper's reference list | Semantic Scholar |
| `get_paper_summary` | AI-generated TLDR for a paper | Semantic Scholar |
| `find_open_access` | Check if OA copy exists for a DOI | Unpaywall |
| `get_full_text_link` | Best available access URL (OA → EZProxy → DOI) | Unpaywall + Registry |
| `browse_sfu_databases` | Browse/search SFU's subscribed databases | Solr endpoint |
| `check_sfu_access` | Check if SFU subscribes to a given resource | Solr endpoint |
| `search_biomedical` | Search biomedical literature specifically | Europe PMC |

### Kept Tools (no changes)

| Tool | Description |
|------|-------------|
| `generate_citation` | APA, MLA, Chicago, BibTeX |
| `batch_generate_citations` | Multiple citations |
| `export_search_results` | JSON, CSV, BibTeX, RIS |
| `save_to_zotero` | Save to Zotero library |
| `list_zotero_collections` | List Zotero collections |
| `batch_save_to_zotero` | Save multiple to Zotero |
| `search_zotero` | Search user's Zotero library |
| `get_zotero_collection_items` | List collection items |
| `get_zotero_status` | Check Zotero credentials |

### Removed Tools

| Tool | Reason |
|------|--------|
| `search_library` | Replaced by `search_academic` (OpenAlex) |
| `search_by_subject` | Replaced by `search_by_topic` (OpenAlex concepts) |
| `search_by_isbn` | Covered by `search_by_doi` + OpenAlex ISBN filter |
| `search_electronic_resources` | Replaced by `browse_sfu_databases` (Solr) |
| `get_item_details` | Replaced by `search_by_doi` with full metadata |
| `batch_isbn_lookup` | Covered by OpenAlex batch lookup |

---

## Implementation Steps

### Phase 1: SFU Database Registry Client

1. Create `src/lib/sfu_databases.py`
   - Fetch all records from Solr endpoint
   - Parse into structured data models
   - Build publisher domain index for access matching
   - Implement local file cache with TTL
   - Strip HTML from `publicNote` fields

### Phase 2: OpenAlex Client

1. Create `src/lib/openalex.py`
   - Search endpoint with query building
   - Filter construction (year, OA, type, topic, author)
   - Abstract reconstruction from inverted index
   - Result normalization to common format
   - Polite pool (mailto parameter)

### Phase 3: Access Resolution

1. Create `src/lib/access_resolver.py`
   - Unpaywall lookup by DOI
   - SFU subscription matching (domain-based)
   - EZProxy URL construction
   - Cascading resolution: OA → SFU subscription → DOI fallback

### Phase 4: Semantic Scholar Client (Optional Enhancement)

1. Create `src/lib/semantic_scholar.py`
   - Citation/reference graph queries
   - TLDR summaries
   - Batch paper lookup

### Phase 5: Tool Rewiring

1. Update `src/lib/tools.py`
   - Replace Primo-based tool handlers with new API clients
   - Add new tools (browse_sfu_databases, check_sfu_access, etc.)
   - Update input schemas and descriptions
   - Keep citation and Zotero tools unchanged

### Phase 6: Remove Primo Dependencies

1. Remove `src/lib/client.py` (Primo client)
2. Remove Primo-specific configuration from `src/lib/config.py`
3. Update tests

---

## Configuration (Environment Variables)

```env
# OpenAlex (recommended but not required)
OPENALEX_MAILTO=your@email.com

# Semantic Scholar (optional, for higher rate limits)
SEMANTIC_SCHOLAR_API_KEY=your_key

# SFU Database Registry
SFU_DB_REGISTRY_CACHE_TTL=86400        # 24 hours in seconds
SFU_DB_REGISTRY_CACHE_FILE=sfu_databases_cache.json

# Unpaywall
UNPAYWALL_EMAIL=your@email.com

# EZProxy
SFU_EZPROXY_BASE=https://proxy.lib.sfu.ca/login?url=

# Zotero (keep existing)
SFU_ZOTERO_API_KEY=your_key
SFU_ZOTERO_USER_ID=your_id

# Feature Flags
SFU_FEATURE_SEMANTIC_SCHOLAR_ENABLED=true
SFU_FEATURE_EUROPE_PMC_ENABLED=false
SFU_FEATURE_CACHE_ENABLED=true
SFU_FEATURE_RERANK_ENABLED=true
SFU_FEATURE_ZOTERO_ENABLED=true
```

---

## Dependencies

### New (to add to requirements.txt)

None — all APIs are HTTP-based. The existing `requests` library handles everything.

### Existing (keep)

```
mcp>=1.26.0
requests>=2.31.0
starlette>=0.40.0
uvicorn>=0.40.0
pydantic>=2.0.0
pyzotero>=1.5.0
rank_bm25>=0.2.2
```

---

## Stability Considerations

### Solr Endpoint

- **Risk:** SFU could change/remove the Solr endpoint without notice
- **Mitigation:** Local cache with disk fallback. If Solr is unreachable, serve from last known good cache
- **Alternative:** Email `lib-systems@sfu.ca` to ask about a stable API or data export

### OpenAlex

- **Risk:** Low. Funded by grants, CC0 licensed, widely adopted
- **Mitigation:** CrossRef as fallback search (already implemented)

### Rate Limits

- **OpenAlex:** 10 req/s (polite) — more than sufficient for MCP usage
- **Unpaywall:** 100K/day — will never hit this with interactive use
- **Semantic Scholar:** Most restrictive — queue/debounce if needed

### Next.js Data Routes (NOT recommended for primary use)

The site also exposes per-record data at:
```
https://databases.lib.sfu.ca/_next/data/{buildId}/record/{id}.json
```
This includes extra fields (`licenseTerms`, `authenticationNote`, `collection_id`, `set_id`) but the `buildId` changes on every site rebuild, making it fragile. Only use Solr for production.

---

## Implementation Status & TODO (as of 2026-05-02)

### What Is Already Done

| Item | File | Notes |
|------|------|-------|
| CrossRef DOI enrichment | `src/lib/citations.py` | `enrich_metadata_from_crossref()` — fully functional |
| Semantic Scholar API key config | `src/lib/config.py` | `semantic_scholar_api_key` field wired with env/Docker secret |
| Caching infrastructure | `src/lib/cache.py` | TTL cache with memory limits — reusable for all new clients |
| Retry + circuit breaker | `src/lib/retry.py` | Reusable for all new HTTP clients |
| Access resolution test harness | `src/tests/test_access_resolution.py` | Standalone integration test hitting real APIs (not wired to server yet) |
| General infrastructure | `src/lib/logging_setup.py`, `src/lib/validators.py`, `src/lib/formatters.py` | Reusable as-is |

### What Has NOT Been Implemented

**All six phases are complete.** The server no longer depends on the Primo API.

---

### Implementation Checklist

#### Phase 1 — SFU Database Registry Client [x] COMPLETE
- [x] Create `src/lib/sfu_databases.py`
  - [x] Fetch all 764 records from `https://databases.lib.sfu.ca/solr/sfu_databases/select?q=*:*&rows=1000&wt=json`
  - [x] Parse into dataclass/typed dict models
  - [x] Build publisher domain index (`url` field → normalized domain) for access matching
  - [x] Strip HTML from `publicNote` fields (remove `<strong>`, `<a>`, `<p>` tags)
  - [x] Implement disk-backed JSON cache with configurable TTL (default 24h)
  - [x] Serve from disk cache if Solr is unreachable (cold-start fallback)
  - [x] Unit tests in `src/tests/test_sfu_databases.py`

#### Phase 2 — OpenAlex Client [x] COMPLETE
- [x] Create `src/lib/openalex.py`
  - [x] `search_works(query, filters, page, per_page)` — primary search
  - [x] Filter builder: `publication_year`, `open_access.is_oa`, `type`, `authorships.author.id`, `topics.id`
  - [x] Reconstruct abstract from `abstract_inverted_index` (OpenAlex stores it inverted)
  - [x] Normalize result to a common `WorkMetadata` dict shared with CrossRef/S2 output
  - [x] Include `mailto` param in all requests for polite pool (10 req/s vs 1 req/s)
  - [x] Unit tests in `src/tests/test_openalex.py`

#### Phase 3 — Access Resolution [x] COMPLETE
- [x] Create `src/lib/access_resolver.py`
  - [x] `resolve_access(doi, article_url, source_name)` → returns best URL + access type label
  - [x] Step 1: pass-through if OpenAlex already gives `is_oa=true` + `oa_url`
  - [x] Step 2: Unpaywall check by DOI (`https://api.unpaywall.org/v2/{doi}?email=...`)
  - [x] Step 3: domain match against SFU Database Registry (from Phase 1 index)
  - [x] Step 4: construct EZProxy URL (`https://proxy.lib.sfu.ca/login?url={article_url}`) if `proxy=true`
  - [x] Step 5: DOI link fallback
  - [x] Unit tests in `src/tests/test_access_resolver.py`
  - [x] Wire the standalone `src/tests/test_access_resolution.py` harness into this module

#### Phase 4 — Semantic Scholar Client [x] COMPLETE
- [x] Create `src/lib/semantic_scholar.py`
  - [x] `search_papers(query, fields, limit)` — basic search
  - [x] `get_paper(paper_id, fields)` — detail fetch
  - [x] `get_citations(paper_id, limit)` — papers citing this work
  - [x] `get_references(paper_id, limit)` — this paper's reference list
  - [x] `get_tldr(paper_id)` — AI-generated summary
  - [x] Batch endpoint: `POST /paper/batch` (up to 500 IDs)
  - [x] Use API key from `config.semantic_scholar_api_key` (1 req/s authenticated vs 100 req/5min anon)
  - [x] Unit tests in `src/tests/test_semantic_scholar.py`

#### Phase 5 — Tool Rewiring in `tools.py` [x] COMPLETE
- [x] Add 21 new tool definitions (see Proposed MCP Tools table above)
  - [x] `search_academic` (OpenAlex)
  - [x] `search_by_author` (OpenAlex)
  - [x] `search_by_doi` (OpenAlex + CrossRef)
  - [x] `search_by_topic` (OpenAlex concepts/topics)
  - [x] `get_citations` (Semantic Scholar)
  - [x] `get_references` (Semantic Scholar)
  - [x] `get_paper_summary` (Semantic Scholar TLDR)
  - [x] `find_open_access` (Unpaywall)
  - [x] `get_full_text_link` (access resolver cascade)
  - [x] `browse_sfu_databases` (Solr endpoint)
  - [x] `check_sfu_access` (Solr endpoint)
  - [x] `search_biomedical` (Europe PMC — optional, behind feature flag)
- [x] Add handler functions for each new tool
- [x] Update `handle_tool_call()` dispatch to route new tool names
- [x] Remove handlers for deprecated tools (see Phase 6)
- [x] Keep citation, export, and Zotero tool handlers unchanged
- [x] Update `src/tests/test_tools_unit.py` with new tool tests

#### Phase 6 — Remove Primo Dependencies [x] COMPLETE
- [x] Delete `src/lib/client.py` (the `SFULibraryClient` Primo REST client — 200+ lines)
- [x] Remove the 6 deprecated tool definitions from `tools.py`:
  - [x] `search_library` → replaced by `search_academic`
  - [x] `get_item_details` → replaced by `search_by_doi`
  - [x] `search_by_subject` → replaced by `search_by_topic`
  - [x] `search_by_isbn` → replaced by `search_by_doi` with ISBN filter
  - [x] `search_electronic_resources` → replaced by `browse_sfu_databases`
  - [x] `batch_isbn_lookup` → replaced by OpenAlex batch ISBN lookup
- [x] Remove `lib_client` / `SFULibraryClient` instantiation from server entry points
- [x] Repurpose `src/tests/test_client_unit.py` to test new OpenAlex/access resolver modules
- [x] Update `src/lib/config.py`:
  - [x] Remove any lingering Primo URL/credential fields
  - [x] Add `openalex_mailto: str`
  - [x] Add `unpaywall_email: str`
  - [x] Add `sfu_ezproxy_base: str` (default `https://proxy.lib.sfu.ca/login?url=`)
  - [x] Add `sfu_db_registry_cache_ttl: int` (default 86400)
  - [x] Add `sfu_db_registry_cache_file: str`
  - [x] Expand feature flags: `semantic_scholar_enabled`, `europe_pmc_enabled`
  - [x] Update `validate_config()` to warn when `openalex_mailto` / `unpaywall_email` are unset
- [x] Update `src/tests/test_config.py` for new fields

---

### Modules Deprecated (Deleted)

| File | Status | Reason |
|------|--------|--------|
| `src/lib/client.py` | **Deleted** | Primo REST API client — replaced by OpenAlex + Solr clients |
| `src/tests/test_client_unit.py` | **Repurposed** | Now tests OpenAlex helpers and access resolver utilities |

### Modules to Refactor (Keep but Modify Significantly)

| File | What Needs Changing |
|------|---------------------|
| `src/lib/tools.py` | Remove 6 Primo tool definitions/handlers; add 12 new tool definitions/handlers; update dispatch table |
| `src/lib/config.py` | Add 5 new config fields; expand feature flags; update `validate_config()` |
| `src/lib/formatters.py` | Add `format_openalex_result()` and `format_sfu_database()` output formatters |
| `src/tests/test_tools_unit.py` | Replace Primo-dependent tool tests with new tool tests |
| `src/tests/test_config.py` | Add coverage for new config fields |

### Modules to Keep As-Is

| File | Notes |
|------|-------|
| `src/lib/cache.py` | Reuse for Solr/OpenAlex response caching |
| `src/lib/citations.py` | CrossRef enrichment already works; keep all citation formatters |
| `src/lib/reranker.py` | Reusable for OpenAlex result reranking |
| `src/lib/retry.py` | Reuse for all new HTTP clients |
| `src/lib/validators.py` | Reuse query sanitizers |
| `src/lib/logging_setup.py` | No changes needed |
| `src/lib/zotero.py` | No changes needed |
| `src/sfu_library_mcp_server.py` | Minor: remove `SFULibraryClient` instantiation |
| `src/sfu_library_mcp_http.py` | Minor: remove `SFULibraryClient` instantiation |
| `src/tests/test_citations.py` | No changes needed |
| `src/tests/test_cache.py` | No changes needed |
| `src/tests/test_reranker.py` | No changes needed |
| `src/tests/test_zotero.py` | No changes needed |
