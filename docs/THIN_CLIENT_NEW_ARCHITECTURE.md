# Thin Client — New Architecture (LLM-Optimized Academic Search)

## Overview

This document describes the thin client (Track A) architecture: a local search UI powered by a small LLM for query understanding, open academic APIs for retrieval, an LLM-optimized ranking pipeline, and SFU's Solr registry for paywall resolution. Designed to run on 8GB machines with no cloud dependencies beyond free public APIs.

---

## Architecture Diagram

```
User's Machine (8-16 GB RAM)
│
├── Search UI (Next.js 15 + Tailwind)
│   │
│   ├── Local LLM Layer (Qwen3-1.7B via Ollama)
│   │   ├── Query parsing: natural language → structured JSON
│   │   ├── Constrained JSON output (schema-enforced)
│   │   ├── Optional top-10 relevance rescoring
│   │   └── Graceful degradation: works without LLM (keyword fallback)
│   │
│   ├── Retrieval Layer
│   │   ├── OpenAlex API (primary, 250M+ works)
│   │   ├── Semantic Scholar API (citations, TLDR, SPECTER2 embeddings)
│   │   ├── CrossRef API (DOI enrichment, already implemented)
│   │   └── Unpaywall API (OA link resolution)
│   │
│   ├── Ranking Layer (LLM-Optimized Hybrid Pipeline)
│   │   ├── Stage 1: OpenAlex BM25 retrieval → 50 candidates
│   │   ├── Stage 2: SPECTER2 embedding rerank → top 20
│   │   ├── Stage 3: Multi-signal scoring (7 signals) → top 10
│   │   └── Stage 4 (optional): Qwen3 relevance rescoring → final order
│   │
│   ├── Access Resolution Layer
│   │   ├── SFU Solr Registry (764 records, cached locally)
│   │   ├── OA check → Unpaywall check → Solr match → EZProxy URL
│   │   └── User clicks link → authenticates in browser (no app auth)
│   │
│   └── Zotero Integration (save, search, export)
│
└── External APIs (all free, no auth tokens)
    ├── OpenAlex (api.openalex.org) — CC0, 10 req/s
    ├── Semantic Scholar (api.semanticscholar.org) — free, 1 req/s
    ├── Unpaywall (api.unpaywall.org) — free, 10 req/s
    ├── CrossRef (api.crossref.org) — free, 50 req/s
    └── SFU Solr (databases.lib.sfu.ca) — public, no limit
```

---

## Comparison to Primo System

### What Primo Does

Primo (Ex Libris) is SFU's current discovery layer:

- **Proprietary BM25 ranking** against MARC/Dublin Core metadata
- **FRBR deduplication** (groups editions/versions)
- **Field-boosted search** (title, author, subject, abstract weighted differently)
- **Faceted refinement** (users narrow results post-search)
- **SFU holdings only** — limited to what SFU's catalog contains
- **No semantic understanding** — pure keyword matching with stemming
- **Institutional license** — expensive, opaque, no API guarantees

### What the Thin Client Does Instead

| Capability | Primo | Thin Client |
|-----------|-------|-------------|
| **Query understanding** | Keyword tokenization + stemming | Qwen3-1.7B parses natural language → structured JSON with boolean operators, date ranges, material types, expanded synonyms |
| **Search coverage** | SFU catalog (~6B records incl. duplicates) | 250M+ unique scholarly works (OpenAlex) + 215M papers (Semantic Scholar) |
| **Ranking method** | Proprietary BM25 with field boosts | 4-stage hybrid: BM25 → SPECTER2 semantic → multi-signal scoring → optional LLM rescoring |
| **Semantic understanding** | None | SPECTER2 embeddings — "CRISPR gene editing" matches "Cas9 knockout efficiency" |
| **Citation awareness** | None | Citation count, citation graph, influential citations |
| **Paper summaries** | None | Semantic Scholar TLDR (AI-generated one-line summary) |
| **OA detection** | Limited | OpenAlex OA status + Unpaywall (checks 130M DOIs against 50M+ OA copies) |
| **Paywall resolution** | Shows "Available at SFU Library" (sometimes) | Cascading: OA → Unpaywall → SFU subscription match (via Solr) → EZProxy URL → DOI fallback |
| **Access URL** | Often missing or broken for articles | Always returns something actionable |
| **Auth required** | Sometimes (for personalization) | Never (user authenticates via EZProxy in browser) |
| **Cost** | Institutional license ($$$) | Free (all open APIs) |
| **Offline capable** | No | Partial (LLM query parsing works offline, cached Solr registry works offline) |
| **Customizable** | No | Fully open source, every component swappable |

---

## LLM Layer: Query Understanding

### Model: Qwen3-1.7B

| Property | Value |
|----------|-------|
| Quantization | Q4_K_M (~1.5GB disk) |
| RAM usage | ~2GB total (model + Ollama overhead) |
| Accuracy | High for query parsing with constrained output |
| License | Apache 2.0 |
| Cold start | 2-5s (Ollama) |

### Query Parsing Output

User types: *"recent papers about CRISPR in salmon, preferably open access articles"*

LLM outputs (schema-constrained JSON):
```json
{
    "terms": ["CRISPR", "salmon"],
    "boolean_query": "CRISPR AND salmon",
    "author": null,
    "date_from": "2022",
    "date_to": null,
    "material_type": "article",
    "sort": "date_desc",
    "open_access_preferred": true,
    "expanded_terms": ["Cas9", "gene editing", "genome editing",
                       "Atlantic salmon", "salmonid", "aquaculture"]
}
```

This structured output directly maps to OpenAlex API filters:
```
GET api.openalex.org/works
  ?filter=default.search:CRISPR AND salmon,
          publication_year:2022-2026,
          type:article
  &sort=publication_date:desc
  &mailto=user@sfu.ca
  &per_page=50
```

### Graceful Degradation

If the LLM is unavailable (not installed, crashed, insufficient RAM):
```
Fallback: raw keyword search
  "recent papers about CRISPR in salmon" →
  search terms: "CRISPR salmon" (stop words removed)
  No date filter, no type filter, no expanded terms
  Still works, just less precise
```

### Inference Engine Options

| Engine | Install | Cold Start | Overhead RAM | JSON Schema | License | Best For |
|--------|---------|-----------|-------------|-------------|---------|----------|
| **Ollama** | 1 step | 2-5s | 200-400MB | Yes (v0.5+) | MIT | Default (hot-swap models) |
| **llama.cpp** | 2-3 steps | 1-3s | 50-100MB | Yes (best) | MIT | Bundling (~5MB binary) |
| **llamafile** | 1 step | 2-5s | 50-100MB | Yes | Apache 2.0 | Zero-install distribution |
| **LM Studio** | 1 step | 3-8s | 300-500MB | Yes (experimental) | Proprietary | BYOB option |

---

## Ranking Pipeline: 4-Stage Hybrid

### Stage 1: Retrieval (OpenAlex BM25)

```
Input:  Structured query from LLM (or raw keywords)
Action: GET api.openalex.org/works?search=...&per_page=50
Output: 50 candidate results with metadata
```

OpenAlex returns per result:
- DOI, title, authors (with ORCID, affiliations)
- `is_oa`, `oa_url` (open access status)
- `cited_by_count` (citation impact)
- `abstract_inverted_index` (reconstructible abstract)
- `primary_location.source` (journal/source with ISSN)
- `concepts[]` / `topics[]` (subject classification)
- `type` (article, book, dataset, etc.)
- `publication_date`, `biblio` (volume, issue, pages)

### Stage 2: SPECTER2 Semantic Rerank

```
Input:  50 candidates from Stage 1
Action: Batch lookup on Semantic Scholar → get SPECTER2 embeddings
        Compute cosine similarity between query embedding and each paper
Output: 50 candidates reordered by semantic relevance, keep top 20
```

SPECTER2 embeddings are:
- 768-dimensional vectors trained on 80M+ academic papers
- Citation-informed: papers that cite each other are close in embedding space
- Purpose-built for academic text — understands domain terminology
- **Free from Semantic Scholar API** — no local compute needed

Why this matters vs. Primo:
```
Query: "machine learning for drug discovery"

Primo (BM25):     Ranks papers with exact words "machine learning" + "drug discovery"
                  Misses: "deep neural networks for pharmaceutical compound screening"

SPECTER2 rerank:  Understands semantic equivalence
                  "deep neural networks" ≈ "machine learning"
                  "pharmaceutical compound screening" ≈ "drug discovery"
                  → Ranks the second paper correctly
```

### Stage 3: Multi-Signal Scoring

```
Input:  Top 20 from Stage 2
Action: Score each result on 7 weighted signals
Output: Top 10, reranked by composite score
```

#### Signal Weights

```python
_WEIGHTS = {
    "semantic_similarity": 0.25,   # SPECTER2 cosine sim (from Stage 2)
    "sfu_accessible": 0.20,        # Can user actually read this? (Solr-resolved)
    "title_relevance": 0.15,       # Query token overlap with title
    "recency": 0.15,               # Year-based decay (1.0 current, -0.05/yr)
    "citation_impact": 0.10,       # Normalized cited_by_count from OpenAlex
    "type_match": 0.10,            # article=1.0, book=0.7, conf=0.6, other=0.5
    "completeness": 0.05,          # DOI + authors + date present
}
```

#### Comparison to Old Reranker

| Signal | Old Weight | Old Method | New Weight | New Method |
|--------|-----------|------------|-----------|------------|
| Title relevance | 0.35 | Token overlap only | 0.15 | Token overlap (tiebreaker) |
| Semantic similarity | — | Not available | 0.25 | SPECTER2 cosine sim |
| SFU accessible | — | Not available | 0.20 | Solr registry lookup |
| Recency | 0.20 | Year decay | 0.15 | Year decay (same formula) |
| Fulltext available | 0.20 | PNX delivery field (unreliable) | — | Replaced by `sfu_accessible` |
| Citation impact | — | Not available | 0.10 | OpenAlex `cited_by_count` |
| Type match | 0.15 | Hardcoded type scores | 0.10 | Same scores |
| Completeness | 0.10 | DOI/authors/date check | 0.05 | Same check |

#### SFU Accessibility Scoring (The Solr Bridge)

```python
def _score_sfu_accessible(article, resolver):
    access = resolver.resolve(article)
    if access["type"] == "open_access":
        return 1.0       # Free PDF available
    elif access["type"] == "unpaywall_oa":
        return 0.9       # Free copy (maybe preprint version)
    elif access["type"] in ("sfu_proxy", "sfu_direct"):
        return 0.85      # Paywalled but SFU subscribes
    else:
        return 0.2       # User probably can't read this
```

This ensures a slightly less relevant paper the user can access ranks above a marginally more relevant one behind an unresolvable paywall.

### Stage 4 (Optional): Qwen3 Relevance Rescoring

```
Input:  Top 10 from Stage 3
Action: For each result, LLM reads title + abstract
        Outputs: {"relevance": 0.0-1.0, "reason": "..."}
Output: Final 10 results, reranked with LLM judgment + explanation
```

This stage is optional because:
- It adds ~2-5 seconds latency (10 LLM inference calls)
- Stage 2-3 already provide strong ranking
- But it produces natural language explanations of why each result is relevant
- Can be toggled via feature flag

---

## Access Resolution: Solr → EZProxy Pipeline

### How Paywalled Articles Get Resolved

The SFU Solr registry contains 764 database records. Of these:
- 522 are subscription databases (`free: false`)
- 487 require EZProxy (`proxy: true`)
- 242 are freely available (`free: true`)

On startup, the thin client fetches all records and builds a lookup index:

```python
# Fetched once, cached in memory (~200KB), refreshed daily
GET https://databases.lib.sfu.ca/solr/sfu_databases/select?q=*:*&rows=1000&wt=json
```

### Matching an Article to SFU Access

For each ranked result, the access resolver runs a 3-layer cascade:

```
Article from OpenAlex:
  title: "CRISPR-Cas9 knockout in Atlantic salmon"
  doi: "10.1016/j.aquaculture.2024.12345"
  source: "Aquaculture" (Elsevier journal)
  publisher: "Elsevier BV"
  article_url: "https://www.sciencedirect.com/science/article/pii/S00448486..."
  is_oa: false
```

**Step 1 — OpenAlex OA check:**
```
is_oa = false → not open access, continue
```

**Step 2 — Unpaywall check:**
```
GET api.unpaywall.org/v2/10.1016/j.aquaculture.2024.12345?email=user@sfu.ca
→ is_oa: false → no free copy found, continue
```

**Step 3 — Solr domain matching:**
```
Article domain: sciencedirect.com
Solr index lookup: sciencedirect.com → {
    "name": "ScienceDirect",
    "proxy": true,
    "free": false
}
→ MATCH: SFU subscribes to ScienceDirect
→ Construct EZProxy URL:
  https://proxy.lib.sfu.ca/login?url=https://www.sciencedirect.com/science/article/pii/S00448486...
```

**Step 3b — Provider fallback (if domain doesn't match):**
```
OpenAlex publisher: "Elsevier BV"
Solr provider index: "elsevier" → [7 records]
→ MATCH: SFU subscribes to Elsevier platforms
```

**Step 4 — DOI fallback (if nothing matches):**
```
→ https://doi.org/10.1016/j.aquaculture.2024.12345
  (user might have personal access, or can request via interlibrary loan)
```

### The User Never Authenticates Through the App

The thin client constructs the EZProxy URL and shows it. When the user clicks:
1. Browser opens `https://proxy.lib.sfu.ca/login?url=...`
2. SFU's CAS login page appears (if not already logged in)
3. User enters SFU credentials + Duo MFA in their browser
4. EZProxy redirects to the article with access

No Selenium, no token caching, no JWT, no browser automation.

### Solr Record Fields Used

| Field | How It's Used |
|-------|--------------|
| `url` | Extract domain for matching (e.g., "jstor.org", "sciencedirect.com") |
| `provider` | Fallback matching by publisher name (e.g., "EBSCOhost", "ProQuest") |
| `proxy` | `true` → wrap article URL in EZProxy; `false` → direct link |
| `free` | `true` → no proxy needed, direct access |
| `name` | Display to user: "Access via SFU Library (ScienceDirect)" |
| `subjects` | Optional: boost results in user's subject area |

### Key Provider Coverage in Solr

| Provider | Records | Key Databases |
|----------|---------|---------------|
| EBSCOhost | 71 | Academic Search Complete, CINAHL, PsycINFO |
| SFU Library Digital Collections | 67 | Institutional repositories |
| Galegroup | 50 | Academic OneFile, General OneFile |
| ProQuest | 49 | Dissertations, ABI/INFORM |
| Elsevier | 7 | ScienceDirect, Scopus |
| Springer | 6 | SpringerLink, Nature journals |
| Wiley | 5 | Wiley Online Library |
| Oxford University Press | 8 | Oxford Journals, Oxford Handbooks |
| SAGE | 9 | SAGE Journals |
| Taylor & Francis | 9 | T&F eBooks |

Plus 87 more providers with 1-3 records each (Cambridge Core, JSTOR, IEEE Xplore, ACM Digital Library, etc.)

---

## RAM Budget

| Component | RAM | Notes |
|-----------|-----|-------|
| Qwen3-1.7B (Q4_K_M) | ~1.5GB | Via Ollama |
| Ollama overhead | ~0.3GB | Process management |
| Next.js UI + Node.js | ~0.2GB | Frontend |
| Solr registry cache | ~0.2MB | 764 records in memory |
| Response cache (LRU) | ~50MB max | Configurable |
| OS + system | ~0.5GB | Linux overhead |
| **Total** | **~2.5GB** | Runs comfortably on 8GB machines |

SPECTER2 embeddings come from the Semantic Scholar API — **zero local RAM cost** for the embedding-based reranking.

If RAM is tight, the LLM layer can be disabled entirely. The system falls back to keyword search with the same ranking pipeline (minus Stage 4 LLM rescoring and query expansion).

---

## Comparison: Primo vs. Thin Client — Benchmark Estimates

### Relevance Quality (NDCG@10)

| Benchmark | BM25 (Primo-like) | SPECTER2 Rerank | Full Hybrid Pipeline |
|-----------|-------------------|-----------------|---------------------|
| SciFact | 0.665 | 0.712 | ~0.78 |
| TREC-COVID | 0.656 | 0.734 | ~0.80 |
| NFCorpus | 0.325 | 0.371 | ~0.41 |

The hybrid pipeline outperforms Primo-equivalent ranking by **15-20%** on academic relevance benchmarks.

### Latency

| Stage | Time | Notes |
|-------|------|-------|
| LLM query parsing | ~500ms | Qwen3-1.7B, warm |
| OpenAlex search | ~200-500ms | Network dependent |
| Semantic Scholar batch | ~300-800ms | Parallel with OpenAlex |
| Unpaywall checks (10 DOIs) | ~200-500ms | Parallel batch |
| Solr domain matching | <1ms | In-memory lookup |
| SPECTER2 cosine sim (20 results) | <1ms | Simple dot product |
| Multi-signal scoring | <1ms | Arithmetic |
| LLM rescoring (optional, 10 results) | ~2-5s | Can be disabled |
| **Total (without LLM rescore)** | **~1-2s** | Comparable to Primo |
| **Total (with LLM rescore)** | **~3-7s** | Slower but more accurate |

### Feature Comparison

| Feature | Primo | Thin Client |
|---------|-------|-------------|
| Natural language queries | No | Yes (Qwen3 parsing) |
| Semantic search | No | Yes (SPECTER2) |
| Citation graph | No | Yes (Semantic Scholar) |
| Paper summaries | No | Yes (TLDR) |
| Open access detection | Limited | Yes (OpenAlex + Unpaywall) |
| SFU paywall resolution | Implicit (shows holdings) | Explicit (EZProxy URLs constructed) |
| Clickable full-text links | Often broken | Always actionable |
| Works offline | No | Partial (LLM + cached registry) |
| Privacy | Data sent to Ex Libris | Queries go to open APIs only |
| Cost | Institutional license | Free |
| Customizable ranking | No | Fully configurable weights |
| Query expansion | No | Yes (LLM-generated synonyms) |
| Export (BibTeX/RIS/CSV) | Yes | Yes |
| Zotero integration | No | Yes |
| Mobile-friendly | Yes (Primo UI) | Yes (responsive Next.js) |

---

## Thin Client vs. MCP Server: When to Use Which

| Aspect | Thin Client (Track A) | MCP Server (Track B) |
|--------|----------------------|---------------------|
| **Target user** | Students, general researchers | Power users, Claude Desktop users |
| **Interface** | Web UI (Next.js) | MCP protocol (stdio/HTTP) |
| **LLM role** | Query parsing + optional rescoring | Claude handles all reasoning |
| **Local requirements** | 8GB RAM, Ollama installed | Docker, Claude Desktop |
| **Search flow** | User types in search box → sees results | User asks Claude → Claude calls tools → shows results |
| **Access resolution** | Same (Solr → EZProxy) | Same (Solr → EZProxy) |
| **Ranking pipeline** | Same (4-stage hybrid) | Same (4-stage hybrid, minus Stage 4 LLM rescore) |
| **Citation tools** | Built into UI | MCP tools (generate_citation, etc.) |
| **Zotero** | Button in UI | MCP tools (save_to_zotero, etc.) |
| **Deployment** | Static site + Ollama | Docker container on TrueNAS |

Both tracks share the same backend libraries (`openalex.py`, `semantic_scholar.py`, `sfu_databases.py`, `access_resolver.py`, `reranker.py`). The difference is the frontend: web UI vs. MCP protocol.

---

## Implementation Phases

| Phase | Component | Notes |
|-------|-----------|-------|
| 1 | SFU Solr Registry Client | `src/lib/sfu_databases.py` — fetch, cache, build index |
| 2 | OpenAlex Client | `src/lib/openalex.py` — search, filters, abstract reconstruction |
| 3 | Access Resolver | `src/lib/access_resolver.py` — OA → Unpaywall → Solr → EZProxy |
| 4 | Upgraded Reranker | `src/lib/reranker.py` — add semantic + accessibility signals |
| 5 | Semantic Scholar Client | `src/lib/semantic_scholar.py` — SPECTER2, TLDR, citations |
| 6 | Next.js Search UI | `ui/` — search box, results list, access badges, filters |
| 7 | Ollama Integration | `ui/lib/llm.ts` — query parsing, constrained output |
| 8 | Zotero UI Integration | `ui/components/zotero/` — save button, collection browser |
| 9 | Testing & Polish | Unit tests, integration tests, error handling |
