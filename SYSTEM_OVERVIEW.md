# SFU Library Search — Hypothetical System Overview

## Complete System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                           USER'S MACHINE (8-16 GB RAM)                             ║
║                                                                                    ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐    ║
║  │                    SEARCH UI (FAIRplexica Fork)                            │    ║
║  │                    Next.js 15 + Tailwind + Headless UI                     │    ║
║  │                                                                           │    ║
║  │  ┌─────────────────────────────────────────────────────────────────┐       │    ║
║  │  │  Search Bar: "recent articles on ML in healthcare"              │       │    ║
║  │  └─────────────────────────────────────────┬───────────────────────┘       │    ║
║  │                                            │                              │    ║
║  │  ┌─────────────────────────────────────────▼───────────────────────┐       │    ║
║  │  │  AI Interpretation Panel                                        │       │    ║
║  │  │  "Understood: machine learning AND healthcare"                  │       │    ║
║  │  │  "Expanded: deep learning, clinical, medical AI, diagnosis"     │       │    ║
║  │  └─────────────────────────────────────────────────────────────────┘       │    ║
║  │                                                                           │    ║
║  │  [Articles ▼] [2020-present ▼] [Sort: Date ▼] [Peer-reviewed ☑]          │    ║
║  │                                                                           │    ║
║  │  ─── 47 results ─────────────────────────────────────────────────         │    ║
║  │  ┌────────────────────────────────────────────────────────────┐            │    ║
║  │  │ 1. Deep Learning Applications in Clinical Medicine        │            │    ║
║  │  │    Smith, J. et al. (2024) · Nature Medicine · Vol 30     │            │    ║
║  │  │    "This review examines the application of deep..."      │            │    ║
║  │  │    [📋APA] [📋MLA] [📋BibTeX] [📥Zotero] [📄PDF]          │            │    ║
║  │  │    Article · Peer-reviewed · DOI: 10.1038/...             │            │    ║
║  │  └────────────────────────────────────────────────────────────┘            │    ║
║  │  ┌────────────────────────────────────────────────────────────┐            │    ║
║  │  │ 2. Machine Learning for Drug Discovery: A Review          │            │    ║
║  │  │    Chen, L. & Wang, R. (2023) · Science · Vol 382         │            │    ║
║  │  │    ...                                                    │            │    ║
║  │  └────────────────────────────────────────────────────────────┘            │    ║
║  │                                                                           │    ║
║  │  ┌──────────────────────────────────┐  ┌──────────────────────────┐       │    ║
║  │  │ Sidebar                          │  │ Settings (Admin Panel)   │       │    ║
║  │  │ • 🔍 Search                      │  │ • LLM Backend URL        │       │    ║
║  │  │ • ⭐ Saved Searches              │  │ • Model Selection        │       │    ║
║  │  │ • 📜 History                     │  │ • SFU Credentials        │       │    ║
║  │  │ • ⚙️ Settings                    │  │ • Zotero API Key         │       │    ║
║  │  └──────────────────────────────────┘  │ • Global Search Context  │       │    ║
║  │                                        └──────────────────────────┘       │    ║
║  └───────────────────────────────────────────────────────────────────────────┘    ║
║           │                                          │                            ║
║           │ HTTP (localhost)                          │ HTTP (localhost)            ║
║           ▼                                          ▼                            ║
║  ┌─────────────────────┐                ┌──────────────────────────┐              ║
║  │  OLLAMA (Default)   │                │  SFU Library Backend     │              ║
║  │  localhost:11434     │                │  (Python — existing MCP  │              ║
║  │                     │                │   server code, reused)   │              ║
║  │  Qwen3-1.7B Q4_K_M │                │                          │              ║
║  │  ~1.5 GB RAM        │                │  Modules reused:         │              ║
║  │                     │                │  ├─ client.py (Primo API)│              ║
║  │  Structured JSON    │                │  ├─ citations.py         │              ║
║  │  output via schema  │                │  ├─ formatters.py        │              ║
║  │  enforcement        │                │  ├─ validators.py        │              ║
║  │                     │                │  ├─ cache.py             │              ║
║  │  ┌───────────────┐  │                │  ├─ reranker.py          │              ║
║  │  │ Swappable:    │  │                │  ├─ zotero.py            │              ║
║  │  │ • llama.cpp   │  │                │  ├─ downloader.py        │              ║
║  │  │ • LM Studio   │  │                │  ├─ rate_limiter.py      │              ║
║  │  │ • llamafile   │  │                │  ├─ proxy_utils.py       │              ║
║  │  │ • Cloud API   │  │                │  ├─ stealth.py           │              ║
║  │  │ (change URL)  │  │                │  ├─ retry.py             │              ║
║  │  └───────────────┘  │                │  └─ publisher_router.py  │              ║
║  └─────────────────────┘                └──────────────┬───────────┘              ║
║                                                        │                          ║
╚════════════════════════════════════════════════════════╪══════════════════════════╝
                                                         │
                                          ┌──────────────┼──────────────┐
                                          │              │              │
                              ┌───────────▼──┐  ┌───────▼────┐  ┌─────▼──────┐
                              │ SFU Primo API │  │ SFU CAS    │  │ CrossRef   │
                              │ (Search)      │  │ + Duo MFA  │  │ API (free) │
                              │              │  │ + EZProxy   │  │            │
                              │ JWT-authed   │  │            │  │ DOI-based  │
                              │ REST API     │  │ Selenium   │  │ metadata   │
                              │ 6B+ records  │  │ + pyotp    │  │ enrichment │
                              └──────────────┘  └────────────┘  └────────────┘
                                                                       │
                              ┌──────────────┐  ┌────────────┐         │
                              │ Zotero API   │  │ Publisher  │         │
                              │ (optional)   │  │ Sites      │         │
                              │              │  │            │         │
                              │ Save items   │  │ PDF        │         │
                              │ Collections  │  │ downloads  │         │
                              │ PDF attach   │  │ via EZProxy│         │
                              └──────────────┘  └────────────┘         │
                                                                       │
```

---

## Data Flow: Natural Language Query → Search Results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SEARCH FLOW (Happy Path)                          │
└─────────────────────────────────────────────────────────────────────────────┘

 User types: "recent articles about climate change impacts on BC salmon"
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 1: QUERY PARSING (Local LLM — Ollama)                       │
 │                                                                     │
 │  Input:  Natural language string                                    │
 │  Model:  Qwen3-1.7B via OpenAI-compatible API                      │
 │  Method: response_format = { type: "json_schema", json_schema: {   │
 │            terms, boolean_query, author, date_from, date_to,        │
 │            material_type, sort, expanded_terms                      │
 │          }}                                                         │
 │  Output: GUARANTEED valid JSON (constrained decoding)               │
 │                                                                     │
 │  {                                                                  │
 │    "terms": ["climate change", "salmon", "British Columbia"],       │
 │    "boolean_query": "climate change AND salmon AND (BC OR           │
 │                      British Columbia)",                            │
 │    "author": null,                                                  │
 │    "date_from": "2020",                                             │
 │    "material_type": "article",                                      │
 │    "sort": "date_desc",                                             │
 │    "expanded_terms": ["sockeye", "Pacific salmon", "fisheries",     │
 │                        "environmental impact", "Fraser River"]      │
 │  }                                                                  │
 │                                                                     │
 │  FALLBACK: If Ollama unavailable → pass raw query as keyword search │
 └──────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 2: AUTHENTICATION CHECK (Existing: client.py)                │
 │                                                                     │
 │  Token cache hit? ──YES──► Use cached JWT + EZProxy cookies         │
 │       │                                                             │
 │       NO                                                            │
 │       │                                                             │
 │       ▼                                                             │
 │  Selenium Chrome (headless)                                         │
 │       │                                                             │
 │       ├─ Navigate to SFU CAS login                                  │
 │       ├─ Fill username + password                                   │
 │       ├─ Detect Duo MFA iframe                                      │
 │       ├─ Select device (pattern matching)                           │
 │       ├─ Generate TOTP code (pyotp)                                 │
 │       ├─ Submit MFA                                                 │
 │       ├─ Extract JWT from sessionStorage                            │
 │       ├─ Navigate to EZProxy login URL                              │
 │       ├─ Capture proxy.lib.sfu.ca cookies                           │
 │       └─ Cache token + cookies (file-locked, optional encryption)   │
 └──────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 3: PRIMO API SEARCH (Existing: client.py)                    │
 │                                                                     │
 │  Build request from parsed query:                                   │
 │    GET /primo-explore/v1/pnxs                                       │
 │      ?q=any,contains,{boolean_query}                                │
 │      &searchInFulltable=true                                        │
 │      &sortby={sort}                                                 │
 │      &limit={limit}                                                 │
 │      &offset={offset}                                               │
 │      &vid=SFUL                                                      │
 │    Headers: Authorization: Bearer {JWT}                             │
 │    Cookies: EZProxy session cookies                                 │
 │                                                                     │
 │  ┌──────────────────────────────────────────────────┐               │
 │  │ Resilience layers:                               │               │
 │  │  ├─ Check response cache (TTL 300s, LRU 100)     │               │
 │  │  ├─ Circuit breaker (5 failures → open 60s)      │               │
 │  │  ├─ Retry with backoff (3 attempts, exp delay)   │               │
 │  │  └─ Concurrency semaphore (max 8 parallel)       │               │
 │  └──────────────────────────────────────────────────┘               │
 │                                                                     │
 │  Response: PNX records with display/addata/links/delivery sections  │
 └──────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 4: POST-PROCESSING (Existing: reranker.py, formatters.py)    │
 │                                                                     │
 │  Reranking (5-signal weighted scoring):                             │
 │    ├─ Title relevance (35%) — query token overlap                   │
 │    ├─ Recency (20%) — year-based decay                              │
 │    ├─ Full-text available (20%) — has links?                        │
 │    ├─ Type match (15%) — article > book > conference                │
 │    └─ Completeness (10%) — DOI, authors, date present?              │
 │                                                                     │
 │  Optional: Embedding-based reranking                                │
 │    all-MiniLM-L6-v2 (43 MB) or BGE-small (66 MB)                   │
 │    Cosine similarity between query embedding and result embeddings  │
 │                                                                     │
 │  Format results: extract title, authors, date, source, DOI, links   │
 │  Cache by record_id for citation fallback (Strategy B)              │
 └──────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 5: RENDER IN SEARCH UI                                       │
 │                                                                     │
 │  AI Interpretation: show parsed query + expanded terms              │
 │  Results list: cards with metadata + action buttons                 │
 │  Action buttons call existing modules directly:                     │
 │    [APA] → citations.py generate_citation("apa", record)            │
 │    [MLA] → citations.py generate_citation("mla", record)            │
 │    [Zotero] → zotero.py save_item(record, collection)               │
 │    [PDF] → downloader.py download_article(url, cookies)             │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: PDF Download (Tiered Strategy)

```
 User clicks [PDF ↓] on a result
           │
           ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  URL RESOLUTION (Existing: client.py extract_full_text_links)   │
 │                                                                 │
 │  Priority chain:                                                │
 │    1. pnx.links.linktopdf (direct PDF URL)                     │
 │    2. pnx.addata.doi → https://doi.org/{doi}                   │
 │    3. pnx.links.linktorsrc (publisher source page)              │
 │    4. pnx.links.linktohtml (HTML full text)                     │
 │                                                                 │
 │  If not open-access:                                            │
 │    Wrap URL with EZProxy:                                       │
 │    wiley.com → wiley-com.proxy.lib.sfu.ca                      │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  RATE LIMIT CHECK (Existing: rate_limiter.py)                   │
 │                                                                 │
 │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
 │  │ Session: 12/15  │  │ Hourly: 18/20    │  │ wiley.com:    │  │
 │  │ remaining       │  │ remaining        │  │ 3/5 remaining │  │
 │  └─────────────────┘  └──────────────────┘  └───────────────┘  │
 │                                                                 │
 │  If ANY budget exhausted → reject with "limit reached" message  │
 │  Add humanized delay: random 3-8 seconds between downloads      │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  TIERED DOWNLOAD (Existing: downloader.py + publisher_router)   │
 │                                                                 │
 │  PublisherRouter checks domain history:                         │
 │    UNKNOWN → try all tiers in order                             │
 │    DIRECT_OK → prioritize Tier 1 + 3                            │
 │    BROWSER_ONLY → prioritize Tier 2                             │
 │                                                                 │
 │  ┌───────────────────────────────────────────────────────────┐   │
 │  │ TIER 1: curl_cffi                                        │   │
 │  │ • TLS fingerprint spoofing (Chrome 131 impersonation)    │   │
 │  │ • HTTP/2 with GREASE extensions                          │   │
 │  │ • EZProxy cookies attached                               │   │
 │  │ • Fastest (~1-3s)                                        │   │
 │  │ └─ Success? → validate PDF magic bytes → save            │   │
 │  │    Failure? ↓                                            │   │
 │  ├───────────────────────────────────────────────────────────┤   │
 │  │ TIER 2: Playwright (headless Chromium)                   │   │
 │  │ • Full browser rendering (handles JS redirects)          │   │
 │  │ • Anti-detection scripts injected:                       │   │
 │  │   - navigator.webdriver = false                          │   │
 │  │   - Fake chrome.runtime, plugins, hardware               │   │
 │  │   - WebGL vendor spoofing                                │   │
 │  │ • EZProxy cookies injected into browser context          │   │
 │  │ • Slower (~5-15s)                                        │   │
 │  │ └─ Success? → validate → save                           │   │
 │  │    Failure? ↓                                            │   │
 │  ├───────────────────────────────────────────────────────────┤   │
 │  │ TIER 3: requests (plain HTTP)                            │   │
 │  │ • Standard Python requests library                       │   │
 │  │ • Custom User-Agent header                               │   │
 │  │ • Session cookies                                        │   │
 │  │ • Fallback only                                          │   │
 │  │ └─ Success? → validate → save                           │   │
 │  │    Failure? → report failure, update publisher_router     │   │
 │  └───────────────────────────────────────────────────────────┘   │
 │                                                                 │
 │  PDF validation: check first 4 bytes = 0x25504446 (%PDF)        │
 │  Save to: /tmp/sfu-library-downloads/{hash}.pdf                 │
 │  Extract text: pdftotext (max 100K chars)                       │
 │  Update publisher_router: record success tier for domain        │
 └─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Citation Generation

```
 User clicks [APA] on a result
           │
           ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  CITATION PIPELINE (Existing: citations.py)                     │
 │                                                                 │
 │  4-Strategy Metadata Resolution:                                │
 │                                                                 │
 │  Strategy A: Extract from PNX record                            │
 │    ├─ pnx.display: title, creator, date, type, source           │
 │    ├─ pnx.addata: doi, isbn, issn, volume, issue, pages         │
 │    └─ pnx.search: additional author formats                     │
 │    └─ Has all required fields? → format citation                │
 │         │                                                       │
 │         NO (missing fields)                                     │
 │         ▼                                                       │
 │  Strategy B: Check search result cache                          │
 │    └─ _record_cache[record_id] (up to 500 cached)               │
 │    └─ Has missing fields? → merge and format                    │
 │         │                                                       │
 │         NO                                                      │
 │         ▼                                                       │
 │  Strategy C: Re-search for CDI records                          │
 │    └─ If recordid starts with "TN_cdi_"                         │
 │    └─ Search by title to find alternate PNX record               │
 │         │                                                       │
 │         STILL MISSING                                            │
 │         ▼                                                       │
 │  Strategy D: CrossRef API enrichment                            │
 │    └─ If DOI available: GET api.crossref.org/works/{doi}        │
 │    └─ Fill in: authors, container-title, published-date          │
 │                                                                 │
 │  Format output:                                                 │
 │    APA:     Smith, J. (2024). Title. Nature Medicine, 30(4),     │
 │             112-128. https://doi.org/10.1038/...                │
 │    MLA:     Smith, John. "Title." Nature Medicine, vol. 30,      │
 │             no. 4, 2024, pp. 112-128.                           │
 │    Chicago: Smith, John. "Title." Nature Medicine 30, no. 4      │
 │             (2024): 112-128.                                    │
 │    BibTeX:  @article{smith2024deep, author={Smith, J.}, ...}    │
 │    RIS:     TY  - JOUR\nAU  - Smith, J.\nT1  - Title\n...      │
 └─────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                           │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │ Search UI        │    │ MCP Server       │                      │
│  │ (NEW — Next.js)  │    │ (EXISTING)       │                      │
│  │                  │    │                  │                      │
│  │ FAIRplexica fork │    │ stdio transport  │                      │
│  │ Port 3000        │    │ OR               │                      │
│  │                  │    │ HTTP transport   │                      │
│  └────────┬─────────┘    └────────┬─────────┘                      │
│           │                       │                                │
│           │   Both use the same   │                                │
│           └───────────┬───────────┘                                │
│                       │                                            │
│                       ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    CORE LIBRARY LAYER                       │    │
│  │                    (src/lib/ — Python)                      │    │
│  │                                                             │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │                 client.py                             │   │    │
│  │  │           SFULibraryClient                           │   │    │
│  │  │                                                      │   │    │
│  │  │  • authenticate() → Selenium + pyotp                 │   │    │
│  │  │  • search() → Primo REST API                         │   │    │
│  │  │  • get_item_details() → Primo REST API               │   │    │
│  │  │  • ensure_authenticated() → token lifecycle          │   │    │
│  │  └──────┬──────────────┬────────────────┬───────────────┘   │    │
│  │         │              │                │                   │    │
│  │         ▼              ▼                ▼                   │    │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────────┐          │    │
│  │  │ cache.py │  │ retry.py     │  │ validators.py│          │    │
│  │  │          │  │              │  │              │          │    │
│  │  │ TTL-LRU  │  │ Backoff +   │  │ ISBN, ISSN,  │          │    │
│  │  │ 100 items│  │ Circuit     │  │ query        │          │    │
│  │  │ 50 MB    │  │ Breaker     │  │ sanitization │          │    │
│  │  └──────────┘  └──────────────┘  └──────────────┘          │    │
│  │                                                             │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │              citations.py                             │   │    │
│  │  │  • extract_metadata() (4 fallback strategies)         │   │    │
│  │  │  • format_citation() (APA/MLA/Chicago/BibTeX/RIS)     │   │    │
│  │  │  • CrossRef enrichment for missing fields             │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  │                                                             │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │              downloader.py                            │   │    │
│  │  │  ArticleDownloader (3-tier PDF retrieval)             │   │    │
│  │  │         │              │              │               │   │    │
│  │  │         ▼              ▼              ▼               │   │    │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │    │
│  │  │  │curl_cffi │  │Playwright│  │ requests │           │   │    │
│  │  │  │TLS spoof │  │headless  │  │ plain    │           │   │    │
│  │  │  │Tier 1    │  │Tier 2    │  │ Tier 3   │           │   │    │
│  │  │  └──────────┘  └──────────┘  └──────────┘           │   │    │
│  │  │         │              │              │               │   │    │
│  │  │         ▼              ▼              ▼               │   │    │
│  │  │  ┌──────────────────────────────────────────┐         │   │    │
│  │  │  │ stealth.py (anti-detection hardening)    │         │   │    │
│  │  │  │ proxy_utils.py (EZProxy URL transform)   │         │   │    │
│  │  │  │ publisher_router.py (learns per-domain)  │         │   │    │
│  │  │  │ rate_limiter.py (session/hourly/domain)  │         │   │    │
│  │  │  └──────────────────────────────────────────┘         │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  │                                                             │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │              zotero.py                                │   │    │
│  │  │  ZoteroClient (pyzotero wrapper)                      │   │    │
│  │  │  • save_item() with dedup detection                   │   │    │
│  │  │  • search_items(), list_collections()                 │   │    │
│  │  │  • add_pdf_attachment()                               │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  │                                                             │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │    │
│  │  │reranker.py  │  │formatters.py │  │logging_setup.py   │  │    │
│  │  │5-signal     │  │humanize PNX  │  │structured logging │  │    │
│  │  │weighted rank│  │results       │  │                   │  │    │
│  │  └─────────────┘  └──────────────┘  └───────────────────┘  │    │
│  │                                                             │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │              config.py                                │   │    │
│  │  │  ServerConfig dataclass (~50 properties)              │   │    │
│  │  │  Docker secrets → env vars → .env → defaults          │   │    │
│  │  │  Feature flags for all optional capabilities          │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Options

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT CONFIGURATIONS                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  OPTION 1: Desktop App (Individual Student/Researcher)              │
│  ═══════════════════════════════════════════════════                │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐                │
│  │ Search UI│  │ Ollama   │  │ Python Backend     │                │
│  │ (Tauri/  │──│ (1.5 GB  │  │ (existing lib/*)   │                │
│  │  Electron│  │  model)  │  │                    │                │
│  │  or web) │  │          │  │ PyInstaller → .exe │                │
│  └──────────┘  └──────────┘  └────────────────────┘                │
│                                                                     │
│  Total RAM: ~2 GB  |  Disk: ~1.7 GB  |  No GPU needed              │
│  FIPPA: Compliant (all processing local)                            │
│  Internet: Only for SFU Primo API + EZProxy calls                   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  OPTION 2: Docker Self-Hosted (Department/Lab)                      │
│  ═══════════════════════════════════════════════                    │
│                                                                     │
│  docker-compose.yml:                                                │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐                │
│  │ Search UI│  │ Ollama   │  │ Python Backend     │                │
│  │ Next.js  │──│ Qwen3-4B │──│ API server         │                │
│  │ :3000    │  │ :11434   │  │ :8080              │                │
│  └──────────┘  └──────────┘  └────────────────────┘                │
│                                                                     │
│  Serves multiple users via web browser                              │
│  Can use larger model on server hardware                            │
│  Admin dashboard for configuration                                  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  OPTION 3: MCP Only (Power Users with Claude/GPT)                   │
│  ═══════════════════════════════════════════════                    │
│                                                                     │
│  ┌──────────────────┐     ┌────────────────────┐                   │
│  │ Claude Desktop   │     │ MCP Server         │                   │
│  │ or LM Studio     │────►│ (existing, as-is)  │                   │
│  │ or Gemini CLI    │     │ stdio transport    │                   │
│  └──────────────────┘     └────────────────────┘                   │
│                                                                     │
│  No new development needed — ship existing MCP server               │
│  Full 23-tool access                                                │
│  Requires capable LLM for tool calling                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What's New vs. What's Reused

```
┌──────────────────────────────────────────────────────────────────┐
│                    BUILD vs. REUSE MAP                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ██████ = NEW code to write                                      │
│  ░░░░░░ = EXISTING code reused as-is                            │
│  ▓▓▓▓▓▓ = FORKED and modified                                   │
│                                                                  │
│  FRONTEND                                                        │
│  ▓▓▓▓▓▓ FAIRplexica fork (reshape chat → search)                │
│  ██████ SearchResultCard component                               │
│  ██████ FacetFilters component (date, type, sort)                │
│  ██████ AIInterpretation component                               │
│  ██████ CitationButton component (wraps Citation.js)             │
│  ██████ ZoteroSaveButton component                               │
│  ▓▓▓▓▓▓ Settings/Admin pages (add SFU-specific config)          │
│                                                                  │
│  QUERY PARSER (NEW — thin layer)                                 │
│  ██████ QueryParser class (~50 lines)                            │
│         - OpenAI SDK call to Ollama with JSON schema             │
│         - Fallback to keyword passthrough                        │
│                                                                  │
│  API BRIDGE (NEW — thin layer)                                   │
│  ██████ Next.js API routes that call Python backend (~100 lines) │
│         - /api/search → Python search endpoint                   │
│         - /api/cite → Python citation endpoint                   │
│         - /api/zotero → Python Zotero endpoint                   │
│         - /api/download → Python download endpoint               │
│                                                                  │
│  BACKEND (100% REUSED)                                           │
│  ░░░░░░ client.py — Primo API + authentication                  │
│  ░░░░░░ citations.py — 5 citation formats, 4 strategies         │
│  ░░░░░░ formatters.py — result formatting                       │
│  ░░░░░░ validators.py — input sanitization                      │
│  ░░░░░░ cache.py — response caching                             │
│  ░░░░░░ reranker.py — 5-signal result ranking                   │
│  ░░░░░░ zotero.py — Zotero integration                          │
│  ░░░░░░ downloader.py — 3-tier PDF download                     │
│  ░░░░░░ rate_limiter.py — anti-detection budgets                │
│  ░░░░░░ publisher_router.py — domain learning                   │
│  ░░░░░░ stealth.py — browser fingerprint evasion                │
│  ░░░░░░ proxy_utils.py — EZProxy URL transform                  │
│  ░░░░░░ retry.py — backoff + circuit breaker                    │
│  ░░░░░░ config.py — environment-based configuration             │
│  ░░░░░░ tools.py — MCP tool definitions (Track B)               │
│                                                                  │
│  INFRASTRUCTURE (REUSED + MODIFIED)                              │
│  ▓▓▓▓▓▓ Docker setup (swap SearXNG → Ollama)                    │
│  ░░░░░░ .devcontainer (Chrome, Playwright, Selenium)             │
│  ░░░░░░ requirements.txt                                         │
│                                                                  │
│  ────────────────────────────────────────────────                │
│  Estimated new code:  ~300-500 lines                             │
│  Existing code reused: ~5,000+ lines                             │
│  Forked/modified:      ~2,000 lines (FAIRplexica reshape)        │
│  Ratio: ~90% reuse, ~10% new                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## RAM Budget (16 GB System)

```
┌─────────────────────────────────────────────────────┐
│              MEMORY ALLOCATION                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ████████████████░░░░░░░░░░░░░░░░  16 GB Total     │
│  ├─ OS + Desktop          ░░░░░░░  ~2.0 GB         │
│  ├─ Qwen3-1.7B Q4_K_M    ████     ~1.5 GB         │
│  ├─ Ollama overhead       █        ~0.3 GB         │
│  ├─ Embedding model       ░        ~0.1 GB         │
│  ├─ Search UI (browser)   ██       ~0.5 GB         │
│  ├─ Python backend        █        ~0.3 GB         │
│  ├─ Chrome (Selenium)     ██       ~0.5 GB (when authing) │
│  ├─ Response cache        ░        ~0.05 GB        │
│  └─ Available headroom    ░░░░░░░  ~10.75 GB       │
│                                                     │
│  TOTAL USED: ~5.25 GB                               │
│  HEADROOM:   ~10.75 GB (67% free)                   │
│                                                     │
│  Works comfortably on 8 GB systems too              │
│  (skip embedding model, Selenium only during auth)  │
│                                                     │
└─────────────────────────────────────────────────────┘
```
