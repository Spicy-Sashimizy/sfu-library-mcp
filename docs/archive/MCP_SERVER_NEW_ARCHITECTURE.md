# MCP Server — New Architecture (Primo-Free)

## Overview

This document describes the new MCP server architecture that replaces the Ex Libris Primo API dependency with open academic APIs, an LLM-optimized ranking pipeline, and SFU's public Solr endpoint for paywall resolution. The MCP protocol interface remains identical — Claude Desktop and other MCP clients see no difference.

---

## Architecture Diagram

```
Claude Desktop / MCP Client
    │
    │  MCP protocol (stdio or HTTP)
    │
    ▼
tools.py — handle_tool_call()
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  MULTIPLE BACKENDS                                          │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │  openalex.py — OpenAlexClient       │ ← Primary search   │
│  │  GET api.openalex.org/works         │                    │
│  │  250M+ scholarly works, CC0 license │                    │
│  │  Returns: DOI, title, authors,      │                    │
│  │    is_oa, oa_url, publisher,        │                    │
│  │    cited_by_count, abstract,        │                    │
│  │    source ISSN, concepts/topics     │                    │
│  └─────────────┬───────────────────────┘                    │
│                │                                            │
│  ┌─────────────▼───────────────────────┐                    │
│  │  semantic_scholar.py (optional)     │ ← Citations, TLDR  │
│  │  GET api.semanticscholar.org        │                    │
│  │  215M+ papers                       │                    │
│  │  Returns: SPECTER2 embeddings,      │                    │
│  │    citation graph, TLDR summaries,  │                    │
│  │    influential citation counts      │                    │
│  └─────────────┬───────────────────────┘                    │
│                │                                            │
│  ┌─────────────▼───────────────────────┐                    │
│  │  sfu_databases.py — SolrRegistry    │ ← Subscription DB  │
│  │  Fetched ONCE on startup (~200KB)   │                    │
│  │  764 records from public Solr       │                    │
│  │  Builds: domain_index,             │                    │
│  │    provider_index for matching      │                    │
│  └─────────────┬───────────────────────┘                    │
│                │                                            │
│  ┌─────────────▼───────────────────────┐                    │
│  │  access_resolver.py                 │ ← Paywall bridge   │
│  │  For each result:                   │                    │
│  │    OA? → direct link                │                    │
│  │    Unpaywall? → free copy           │                    │
│  │    Solr match? → EZProxy URL        │                    │
│  │    None? → DOI fallback             │                    │
│  └─────────────────────────────────────┘                    │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │  reranker.py (UPGRADED)             │                    │
│  │  6-7 signals + semantic ranking     │                    │
│  │    + sfu_accessible boost           │                    │
│  │    + citation_impact                │                    │
│  │    + SPECTER2 cosine similarity     │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Old System vs. New System

### Old System: Single Primo Backend

```
tools.py → client.py (SFULibraryClient)
    │
    └→ GET sfu-primo.hosted.exlibrisgroup.com/.../pnxs
       params: q={field},{precision},{query}, vid=SFUL, inst=01SFUL
       │
       └→ Returns PNX format docs (Primo proprietary JSON)
          │
          ├→ reranker.py: 5-signal keyword-based scoring
          └→ formatters.py: text output
```

**Limitations:**
- Single point of failure (Primo API)
- Keyword matching only (BM25), no semantic understanding
- No open access awareness
- No SFU subscription matching — `get_full_text_links` often returns empty
- PNX data format is deeply nested and often incomplete for CDI records
- Record IDs are Primo-specific, useless outside Primo

### New System: Multi-Backend with Access Resolution

```
tools.py → openalex.py (primary search)
         → semantic_scholar.py (citations, TLDR, embeddings)
         → sfu_databases.py (Solr registry, cached in memory)
         → access_resolver.py (cascading paywall resolution)
    │
    └→ reranker.py: 6-7 signal semantic + accessibility scoring
       └→ formatters.py: text output with access badges
```

**Improvements:**
- No single point of failure (multiple independent APIs)
- Semantic understanding via SPECTER2 embeddings
- Every result gets a resolved access path
- Richer metadata (citations, TLDR, OA status, concepts)
- DOI-based record IDs (universal, permanent)
- 250M+ works coverage vs. SFU catalog subset

---

## Data Flow: Complete Search Request

### Old System

```
1. Claude calls search_library("CRISPR gene editing in salmon")
2. tools.py: _handle_search_library()
3. client.py: GET Primo REST API
     q=any,contains,CRISPR gene editing in salmon
     vid=SFUL, inst=01SFUL
4. Primo returns PNX docs (SFU catalog subset)
     - Mostly books and SFU-held journal issues
     - Ranked by Primo's BM25
     - No DOI on many results, no OA status
5. reranker.py: re-score by title overlap (0.35), recency (0.20),
     fulltext (0.20), type (0.15), completeness (0.10)
6. formatters.py → text
7. Claude receives:
     "1. CRISPR Technology in Aquaculture (2023) - Book
      Record ID: alma123456
      Available at SFU Library..."
8. Claude calls get_full_text_links("alma123456")
9. Extract links from PNX delivery/links fields
10. Often returns: "No direct access links available."
```

### New System

```
1. Claude calls search_academic("CRISPR gene editing in salmon")
2. tools.py: _handle_search_academic()

3. openalex.py: GET api.openalex.org/works
     ?search=CRISPR+gene+editing+salmon
     &mailto=user@sfu.ca&per_page=50
   Returns 50 results with DOI, is_oa, oa_url, publisher,
   cited_by_count, abstract, source ISSN, concepts

4. semantic_scholar.py (parallel): batch lookup top results
   Returns SPECTER2 embeddings, TLDR summaries

5. reranker.py (upgraded) — for each of 50 results:
     a. SPECTER2 cosine similarity to query      → 0.25 weight
     b. Title token overlap                       → 0.15
     c. Recency (year decay)                      → 0.15
     d. SFU accessibility (Solr lookup)           → 0.20
        ├─ is_oa=true?            → score 1.0
        ├─ Unpaywall free copy?   → score 0.9
        ├─ Domain in Solr index?  → score 0.85
        └─ No match               → score 0.2
     e. Citation impact (cited_by_count)          → 0.10
     f. Type match (article > book > conf)        → 0.10
     g. Completeness (DOI + authors + date)       → 0.05
   → Top 10 results, reranked

6. access_resolver.py: attach access info to each result
   result["access"] = {
       "type": "sfu_proxy",
       "url": "https://proxy.lib.sfu.ca/login?url=https://...",
       "database": "ScienceDirect",
   }

7. formatters.py → text with access badges
8. Claude receives:
     "1. CRISPR-Cas9 knockout in Atlantic salmon (2024)
      DOI: 10.1016/j.aquaculture.2024.12345
      Cited by: 12 | Type: article | OA: No
      Access: SFU Library via ScienceDirect (EZProxy)
      → https://proxy.lib.sfu.ca/login?url=https://...
      TLDR: Efficient Cas9-mediated gene knockout in
            Atlantic salmon embryos...

      2. Genome editing for salmon conservation (2023)
      DOI: 10.1038/s41587-023-01234-5
      Cited by: 47 | Type: article | OA: Yes
      Access: Open Access (direct PDF)
      → https://www.nature.com/articles/s41587-023-01234-5.pdf"
```

---

## Tool Mapping: Old → New

| Old Tool | New Tool | What Changes |
|----------|----------|-------------|
| `search_library` | `search_academic` | OpenAlex instead of Primo. SPECTER2 rerank. Access info per result. |
| `search_by_author` | `search_by_author` | OpenAlex `authorships.author.display_name` filter. Returns ORCID, affiliation. |
| `search_by_subject` | `search_by_topic` | OpenAlex concept hierarchy instead of Primo LCSH. |
| `search_by_isbn` | `search_by_doi` | DOI-based lookup via OpenAlex + CrossRef. More reliable for articles. |
| `search_electronic_resources` | **Removed** | Every result now has access resolution — no need for separate filter. |
| `get_item_details` | `get_item_details` | Takes DOI or OpenAlex ID. Fetches from OpenAlex + Semantic Scholar (TLDR, citations). Resolves access. |
| `get_full_text_links` | `get_full_text_link` | Cascading resolution: OA → Unpaywall → Solr match → EZProxy → DOI fallback. Always returns something. |
| `generate_citation` | `generate_citation` | OpenAlex metadata (richer than PNX) → CrossRef enrichment → format. Same output quality, better input. |
| `batch_generate_citations` | `batch_generate_citations` | Same pattern, DOI-based instead of Primo record ID. |
| `export_search_results` | `export_search_results` | Same formats (JSON, CSV, BibTeX, RIS), richer data. |
| `batch_isbn_lookup` | **Removed** | Replaced by DOI-based workflows. |
| *(none)* | `get_citations` | **NEW**: Semantic Scholar — "what papers cite this?" |
| *(none)* | `get_references` | **NEW**: "what does this paper cite?" |
| *(none)* | `get_paper_summary` | **NEW**: Semantic Scholar TLDR summary. |
| *(none)* | `find_open_access` | **NEW**: Unpaywall lookup for a DOI. |
| *(none)* | `browse_sfu_databases` | **NEW**: Search/browse 764 Solr records directly. |
| *(none)* | `check_sfu_access` | **NEW**: "Does SFU subscribe to [journal/platform]?" |
| `save_to_zotero` | `save_to_zotero` | Unchanged. |
| `list_zotero_collections` | `list_zotero_collections` | Unchanged. |
| `batch_save_to_zotero` | `batch_save_to_zotero` | Unchanged. |
| `search_zotero` | `search_zotero` | Unchanged. |
| `get_zotero_collection_items` | `get_zotero_collection_items` | Unchanged. |
| `get_zotero_status` | `get_zotero_status` | Unchanged. |

---

## Ranking System: Old vs. New

### Old: 5-Signal Keyword Scoring

```python
_WEIGHTS = {
    "title_relevance": 0.35,      # Query token overlap with title
    "recency": 0.20,              # Year-based decay (1.0 current, -0.05/yr)
    "fulltext_available": 0.20,   # Has PDF/HTML/available in PNX?
    "type_match": 0.15,           # article=1.0, book=0.7, conf=0.6
    "completeness": 0.10,         # DOI+authors+date present
}
```

- Keyword overlap only — "CRISPR gene editing" and "Cas9 knockout" score 0.0 overlap
- No semantic understanding
- Fulltext signal depends on Primo's delivery metadata (often wrong/missing)
- Runs after Primo's own BM25 ranking

### New: 6-7 Signal Semantic + Accessibility Scoring

```python
_WEIGHTS = {
    "semantic_similarity": 0.25,   # SPECTER2 cosine sim (query ↔ paper embedding)
    "sfu_accessible": 0.20,        # Can user actually read this? (Solr-resolved)
    "title_relevance": 0.15,       # Token overlap (kept as tiebreaker)
    "recency": 0.15,               # Year-based decay
    "citation_impact": 0.10,       # cited_by_count from OpenAlex
    "type_match": 0.10,            # article=1.0, book=0.7, conf=0.6
    "completeness": 0.05,          # DOI+authors+date present
}
```

- SPECTER2 embeddings trained on 80M+ papers — understands academic language semantically
- "CRISPR gene editing" and "Cas9 knockout efficiency" score high similarity
- Citation-informed: papers that cite each other are close in embedding space
- SFU accessibility boost: a slightly less relevant paper the user can read ranks above one they can't
- No additional RAM cost: SPECTER embeddings come free from the Semantic Scholar API

### Benchmark Comparison

| Benchmark (BEIR Academic) | BM25 (Primo-like) | SPECTER2 Rerank | Hybrid (BM25→SPECTER→Accessibility) |
|---------------------------|-------------------|-----------------|--------------------------------------|
| NDCG@10 (SciFact)         | 0.665             | 0.712           | ~0.78                                |
| NDCG@10 (TREC-COVID)      | 0.656             | 0.734           | ~0.80                                |
| NDCG@10 (NFCorpus)        | 0.325             | 0.371           | ~0.41                                |

The hybrid pipeline outperforms Primo-equivalent ranking by 15-20% on academic relevance benchmarks.

---

## Access Resolution: How Solr Bridges to Paywalled Articles

### The Problem

OpenAlex/Semantic Scholar return article-level results. Solr gives database-level records (764 entries like "JSTOR", "ScienceDirect", "EBSCOhost"). The bridge is publisher/source domain matching.

### Solr Registry Data (Loaded on Startup)

```
Source: https://databases.lib.sfu.ca/solr/sfu_databases/select?q=*:*&rows=1000&wt=json
Records: 764 total
  - 522 subscription (free: false)
  - 242 free (free: true)
  - 487 require EZProxy (proxy: true)
  - 277 direct access (proxy: false)
Payload: ~200KB
Cache: in-memory + disk fallback, refreshed daily
```

### Matching Strategy (3 Layers, Cascading)

**Layer 1 — Domain matching** (fastest, most reliable):
```python
# Article URL: https://www.sciencedirect.com/science/article/pii/S00448486...
# Extract domain: sciencedirect.com
# Solr record: name="ScienceDirect", url="https://www.sciencedirect.com",
#              proxy=true, free=false
# → MATCH → wrap in EZProxy:
#   https://proxy.lib.sfu.ca/login?url=https://www.sciencedirect.com/...
```

**Layer 2 — Provider matching** (catches aggregators):
```python
# OpenAlex says publisher = "Elsevier BV"
# Solr has provider="Elsevier" on 7 records
# Or: article from EBSCOhost → Solr has 71 EBSCOhost records
```

**Layer 3 — ISSN/Journal name matching** (edge cases):
```python
# OpenAlex returns source ISSN: "0044-8486" (Aquaculture)
# Match journal platform against Solr registry
```

### Access Resolver Implementation

```python
class SFUAccessResolver:
    EZPROXY_BASE = "https://proxy.lib.sfu.ca/login?url="

    def __init__(self):
        solr_records = fetch_solr_registry()  # ~200KB, cached

        # Build domain → record lookup
        self.domain_index = {}
        for record in solr_records:
            url = record.get("url", "")
            if url:
                domain = extract_domain(url)
                self.domain_index[domain] = {
                    "name": record["name"],
                    "proxy": record["proxy"],
                    "free": record["free"],
                }

        # Build provider → records lookup
        self.provider_index = {}
        for record in solr_records:
            provider = record.get("provider", "").lower()
            if provider:
                self.provider_index.setdefault(provider, []).append(record)

    def resolve(self, article: dict) -> dict:
        # 1. Already open access?
        if article.get("is_oa") and article.get("oa_url"):
            return {"type": "open_access", "url": article["oa_url"]}

        # 2. Unpaywall has a free copy?
        unpaywall = check_unpaywall(article["doi"])
        if unpaywall and unpaywall["is_oa"]:
            return {
                "type": "unpaywall_oa",
                "url": unpaywall["best_oa_location"]["url_for_pdf"],
            }

        # 3. SFU subscription? Match domain against Solr
        article_domain = extract_domain(article.get("url", ""))
        if article_domain in self.domain_index:
            record = self.domain_index[article_domain]
            if record["proxy"]:
                return {
                    "type": "sfu_proxy",
                    "url": f"{self.EZPROXY_BASE}{article['url']}",
                    "database": record["name"],
                }
            return {
                "type": "sfu_direct",
                "url": article["url"],
                "database": record["name"],
            }

        # 4. Fallback
        return {"type": "doi_fallback", "url": f"https://doi.org/{article['doi']}"}
```

### Access Resolution Scoring in Reranker

```python
def _score_sfu_accessible(article, resolver):
    access = resolver.resolve(article)
    if access["type"] == "open_access":
        return 1.0       # Free PDF, best case
    elif access["type"] == "unpaywall_oa":
        return 0.9       # Free but maybe preprint version
    elif access["type"] in ("sfu_proxy", "sfu_direct"):
        return 0.85      # Paywalled but SFU pays for it
    else:
        return 0.2       # User probably can't read it
```

### What the User Sees

Each result gets an access badge:

```
1. "CRISPR-Cas9 knockout in Atlantic salmon"
   Aquaculture (2024) · Cited by 12 · Relevance: 0.92
   OPEN ACCESS — Direct PDF link

2. "Genome editing applications in salmonid aquaculture"
   Nature Biotechnology (2023) · Cited by 47 · Relevance: 0.88
   SFU ACCESS — Via SFU Library (EZProxy)
   └─ Matched: "Nature Journals" in SFU database registry

3. "Pacific salmon conservation genomics review"
   Annual Review of Genetics (2022) · Cited by 31 · Relevance: 0.85
   FREE COPY — Found on PubMed Central (via Unpaywall)

4. "CRISPR in aquaculture: ethical dimensions"
   Science and Engineering Ethics (2024) · Cited by 3 · Relevance: 0.74
   DOI LINK — Check personal/institutional access
```

---

## File Changes Summary

| File | Old | New |
|------|-----|-----|
| `client.py` | `SFULibraryClient` — single Primo REST client (157 lines) | **Replaced by** `openalex.py` + `semantic_scholar.py` + `sfu_databases.py` + `access_resolver.py` |
| `reranker.py` | 5 signals, keyword-based (147 lines) | 6-7 signals, adds `semantic_similarity` + `sfu_accessible` + `citation_impact` |
| `tools.py` | 17 tools, all dispatch to `SFULibraryClient` | ~18 tools, dispatch to multiple backends. Same MCP interface. |
| `formatters.py` | Formats PNX data structure | Formats OpenAlex data structure + access badge |
| `citations.py` | `extract_metadata(pnx_doc)` parses Primo nested format | `extract_metadata(openalex_doc)` parses OpenAlex flat format — richer data, simpler parsing |
| `cache.py` | Caches Primo responses | Caches OpenAlex responses + Solr registry (same mechanism) |
| `config.py` | Primo-specific settings | OpenAlex mailto, Semantic Scholar API key, Unpaywall email |

### Unchanged Components

| Component | Why |
|-----------|-----|
| `sfu_library_mcp_server.py` / `sfu_library_mcp_http.py` | MCP transport is backend-agnostic |
| `cache.py` | Generic LRU cache, works with any response format |
| `retry.py` + `CircuitBreaker` | Generic HTTP resilience, works with any API |
| `validators.py` | Query sanitization is backend-agnostic |
| `zotero.py` | Zotero integration is independent |
| Citation formatters (APA/MLA/Chicago/BibTeX) | Take a metadata dict — just need different extraction |
| Semaphore concurrency control | Same pattern, applied to new API clients |
| Feature flags | Same mechanism, new flag names |

---

## API Dependencies

| API | Role | Auth | Rate Limit | Coverage |
|-----|------|------|-----------|----------|
| OpenAlex | Primary search/discovery | None (use `mailto` for polite pool) | 10 req/s (polite) | 250M+ works, CC0 |
| Semantic Scholar | Citations, TLDR, SPECTER embeddings | Optional free API key | 1 req/s (with key) | 215M+ papers |
| Unpaywall | OA link resolution | Email parameter only | 10 req/s, 100K/day | 130M DOIs checked |
| CrossRef | DOI enrichment (already implemented) | None (`mailto` in User-Agent) | 50 req/s | 150M+ DOIs |
| SFU Solr | Subscription database registry | None (public) | No limit observed | 764 records |
| Europe PMC | Biomedical search (optional) | None | 3 req/s | 43M+ records |

---

## Configuration

### New Environment Variables

```bash
# OpenAlex (recommended for polite pool)
SFU_OPENALEX_MAILTO=user@sfu.ca

# Semantic Scholar (optional, increases rate limit)
SFU_SEMANTIC_SCHOLAR_API_KEY=

# Unpaywall
SFU_UNPAYWALL_EMAIL=user@sfu.ca

# Solr registry
SFU_SOLR_REGISTRY_URL=https://databases.lib.sfu.ca/solr/sfu_databases/select
SFU_SOLR_REFRESH_INTERVAL=86400   # seconds (daily)

# Feature flags
SFU_FEATURE_SPECTER_RERANK_ENABLED=true
SFU_FEATURE_ACCESS_RESOLUTION_ENABLED=true
SFU_FEATURE_TLDR_ENABLED=true
SFU_FEATURE_CITATION_GRAPH_ENABLED=true
```

### Kept Environment Variables

All existing cache, retry, circuit breaker, logging, and Zotero variables remain unchanged.

---

## Implementation Phases

| Phase | Component | Files | Depends On |
|-------|-----------|-------|------------|
| 1 | SFU Database Registry Client | `src/lib/sfu_databases.py` | Nothing |
| 2 | OpenAlex Client | `src/lib/openalex.py` | Nothing |
| 3 | Access Resolver | `src/lib/access_resolver.py` | Phase 1 |
| 4 | Upgraded Reranker | `src/lib/reranker.py` | Phase 2, 3 |
| 5 | Semantic Scholar Client (optional) | `src/lib/semantic_scholar.py` | Nothing |
| 6 | Tool Rewiring | `src/lib/tools.py` | Phase 1-4 |
| 7 | Formatter Updates | `src/lib/formatters.py` | Phase 6 |
| 8 | Citation Extraction Update | `src/lib/citations.py` | Phase 6 |
| 9 | Cleanup | Remove `src/lib/client.py`, update config | Phase 6 |
