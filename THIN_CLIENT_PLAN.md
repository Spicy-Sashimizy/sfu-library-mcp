# SFU Library MCP — Thin Client Deployment Plan (v2)

## Executive Summary

**Revised approach:** Instead of building a full MCP chat client that requires a large LLM capable of general conversation + tool calling (unreliable on local models), build a **contextual search engine** where a tiny local LLM (~1.5 GB) handles only query understanding, and deterministic code does everything else. The full MCP server remains available for power users who want to connect Claude, GPT, or other capable models.

**Key insight:** A 1.7B model with constrained JSON output can reliably translate natural language into structured API parameters. It doesn't need to "think" — it just needs to parse intent. This uses ~2 GB total RAM instead of 8-9 GB, runs on any laptop, and eliminates the tool-calling reliability problem entirely.

---

## Table of Contents

1. [Architecture: Two Tracks](#1-architecture-two-tracks)
2. [Track A: Contextual Search Engine (Primary Product)](#2-track-a-contextual-search-engine)
3. [Track B: Full MCP Server (Power Users)](#3-track-b-full-mcp-server-power-users)
4. [Local LLM Selection for Query Understanding](#4-local-llm-selection-for-query-understanding)
5. [Inference Engine Comparison](#5-inference-engine-comparison)
6. [Constrained Output & Structured Generation](#6-constrained-output--structured-generation)
7. [Frontend: Search UI (Not Chat UI)](#7-frontend-search-ui-not-chat-ui)
8. [Packaging as .exe / Desktop App](#8-packaging-as-exe--desktop-app)
9. [Free Cloud LLM Alternatives](#9-free-cloud-llm-alternatives)
10. [University Pain Points & Compliance](#10-university-pain-points--compliance)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Architecture: Two Tracks

```
┌─────────────────────────────────────────────────────────────┐
│                    SFU Library Platform                      │
│                                                             │
│  TRACK A: Contextual Search Engine          TRACK B: MCP    │
│  (Primary — for all users)                  (Optional —     │
│                                              power users)   │
│  ┌───────────────────────────┐                              │
│  │  Search UI (web/desktop)  │         ┌──────────────────┐ │
│  │  ┌─────────────────────┐  │         │  MCP Server      │ │
│  │  │ Qwen3-1.7B (Ollama) │  │         │  (stdio/HTTP)    │ │
│  │  │ Query Understanding  │  │         │  24 tools        │ │
│  │  │ ~1.5 GB RAM         │  │         │  Connect Claude, │ │
│  │  └────────┬────────────┘  │         │  GPT, Gemini,    │ │
│  │           │ structured    │         │  or any MCP      │ │
│  │           │ JSON          │         │  client           │ │
│  │  ┌────────▼────────────┐  │         └──────────────────┘ │
│  │  │ Deterministic Code  │  │                              │
│  │  │ • Primo API calls   │  │                              │
│  │  │ • Result ranking    │  │                              │
│  │  │ • Citation gen      │  │                              │
│  │  │ • PDF retrieval     │  │                              │
│  │  └─────────────────────┘  │                              │
│  └───────────────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

**Why two tracks?**
- Track A works for everyone — no LLM expertise needed, tiny resource footprint
- Track B preserves all 24 MCP tools for users who already have Claude/GPT/etc.
- They share the same backend (Primo API client, auth, Zotero, citations)
- The search engine can work **without any LLM at all** (graceful degradation to keyword search)

---

## 2. Track A: Contextual Search Engine

### How It Works

```
User types: "recent articles about climate change impacts on BC salmon populations"
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Qwen3-1.7B via Ollama        │
                    │  Constrained JSON output       │
                    │                               │
                    │  Input: natural language query │
                    │  Output (guaranteed valid):    │
                    │  {                             │
                    │    "terms": ["climate change", │
                    │      "salmon", "British        │
                    │      Columbia"],               │
                    │    "boolean": "climate change  │
                    │      AND salmon AND (BC OR     │
                    │      British Columbia)",       │
                    │    "author": null,             │
                    │    "date_from": "2020",        │
                    │    "material_type": "article", │
                    │    "sort": "date_desc",        │
                    │    "expanded_terms": [         │
                    │      "sockeye", "Pacific       │
                    │      salmon", "fisheries",     │
                    │      "environmental impact"]   │
                    │  }                             │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Deterministic Python Code     │
                    │  (NO LLM involved)            │
                    │                               │
                    │  1. Build Primo API request    │
                    │  2. Authenticate (SFU CAS)    │
                    │  3. Execute search             │
                    │  4. Rerank with embeddings     │
                    │  5. Generate citations          │
                    │  6. Format results              │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Search Results UI             │
                    │  • Ranked results with metadata│
                    │  • Citation buttons (APA/MLA)  │
                    │  • "Save to Zotero" buttons    │
                    │  • PDF download links          │
                    │  • Facet filters (date, type)  │
                    │  • "Did you mean..." suggestions│
                    └───────────────────────────────┘
```

### What the LLM Does (narrow, well-defined)

| Task | LLM Needed? | Why |
|------|------------|-----|
| Parse natural language → structured query | **Yes** | Core value-add over keyword search |
| Query expansion (synonyms, related terms) | **Yes** | Improves recall significantly |
| Intent classification (search vs. citation vs. download) | **Yes** | Routes to correct handler |
| Build API request parameters | No | Deterministic code |
| Execute API calls | No | Deterministic code |
| Rank/rerank results | Optional | Embedding model or deterministic scoring |
| Generate citations (APA/MLA/etc.) | No | Template-based, already implemented |
| Download PDFs | No | Deterministic, already implemented |
| Zotero operations | No | Deterministic, already implemented |
| Summarize a result | Optional (enhancement) | Nice-to-have, not core |

### Graceful Degradation (No LLM Available)

If Ollama isn't running or the model isn't downloaded, the search engine still works:
- Falls back to **keyword search** (pass user input directly as search terms)
- No query expansion, no intent classification
- Results still ranked, citations still generated, Zotero still works
- User sees a subtle banner: "Install Ollama for smarter search"

This is critical for university adoption — the core tool has zero AI dependencies.

---

## 3. Track B: Full MCP Server (Power Users)

The existing MCP server (24 tools) continues to work as-is. Power users connect via:

| Client | How |
|--------|-----|
| Claude Desktop | `.mcpb` bundle or `claude_desktop_config.json` |
| LM Studio | `mcp.json` config or deeplink |
| Gemini CLI | `~/.gemini/settings.json` |
| Any MCP client | stdio or HTTP transport |
| Claude Code | Direct MCP server connection |

No changes needed to the MCP server. It's already built. Track B is "deploy what exists."

---

## 4. Local LLM Selection for Query Understanding

### The Key Shift: You Don't Need a Big Model

For query understanding (not conversation), tiny models with constrained output work:

| Model | Params | Quantized Size | RAM Usage | Query Translation Accuracy | Notes |
|-------|--------|---------------|-----------|---------------------------|-------|
| **Qwen3-1.7B** | 1.7B | Q4_K_M ~1.5 GB | ~2 GB total | High (with constrained output) | **Top pick.** Native function calling, Apache 2.0 |
| **Qwen3-4B** | 4B | Q4_K_M ~2.5 GB | ~3.5 GB total | Higher | Step-up if 1.7B isn't accurate enough |
| **Ministral-3B** | 3B | Q4_K_M ~2 GB | ~3 GB total | Good | Mistral's small model, native function calling |
| **Qwen2.5-3B-Instruct** | 3B | Q4_K_M ~2 GB | ~3 GB total | Good | Explicitly optimized for JSON output |
| **Phi-3.5-mini** | 3.8B | Q4_K_M ~2.4 GB | ~3.5 GB total | Good | Microsoft, community function-calling fine-tunes exist |

### Why NOT a Big MoE Model Anymore

| Concern | Big MoE (8-9 GB) | Small Query Model (1.5 GB) |
|---------|------------------|---------------------------|
| RAM on 16GB system | Tight, no headroom | Leaves 14 GB free |
| Tool calling reliability | 70-90% (unreliable) | N/A — constrained output guarantees valid JSON |
| Startup time | 10-30 seconds | 1-3 seconds |
| Works on 8GB RAM? | No | **Yes** |
| Needs GPU? | Helps a lot | No, CPU is fine |
| Can hold a conversation? | Yes (poorly) | No (not the goal) |

### Embedding Model for Result Reranking

Run alongside the LLM, adds negligible overhead:

| Model | Size | Purpose | RAM |
|-------|------|---------|-----|
| **all-MiniLM-L6-v2** | 43 MB | Fast query-result similarity | ~80 MB loaded |
| **BGE-small-en-v1.5** | 66 MB | Better accuracy | ~130 MB loaded |
| **Qwen3-Reranker-0.6B** | ~600 MB | Best reranking quality | ~800 MB loaded |

**Total RAM budget (Track A):**
- Qwen3-1.7B: ~1.5 GB
- Embedding model: ~0.1 GB
- App + Ollama overhead: ~0.5 GB
- **Total: ~2.1 GB** (runs on any machine made in the last decade)

---

## 5. Inference Engine Comparison

For serving a 1.7B model to a single-user desktop app:

| Engine | Overhead | Structured Output | CPU-Only | Bundleable | Verdict |
|--------|----------|-------------------|----------|------------|---------|
| **Ollama** | Low (Go binary) | Native JSON schema (v0.5+) | Yes | Single binary | **Best choice** |
| **llama.cpp server** | Lowest | GBNF grammars | Yes | Compile from source | Best if embedding in C++ app |
| **vLLM** | High (Python, GPU) | Yes (via Outlines) | No | Difficult | **Overkill, don't use** |
| **SGLang** | High | Yes | No | Difficult | **Overkill, don't use** |
| **TGI** | Medium | Yes | Limited | Docker-only | Not for desktop |

### Why Ollama Wins for This Use Case

- **Single binary** — `curl -fsSL https://ollama.com/install.sh | sh` (or Windows installer)
- **Model management built in** — `ollama pull qwen3:1.7b` downloads and caches the model
- **OpenAI-compatible API** — your Python code calls `http://localhost:11434/v1/chat/completions`
- **Structured output** — pass a JSON schema via `format` parameter, get guaranteed valid JSON
- **Hot-swappable models** — users can upgrade to a bigger model without changing any code
- **CPU inference works** — no GPU needed for a 1.7B model
- **Memory for Qwen3-1.7B Q4:** ~1.5 GB model + ~200 MB Ollama overhead

### vLLM — When To Use It Instead

vLLM makes sense only for **multi-user server deployments** (Track B, Tier 2):
- PagedAttention gives 19x throughput vs Ollama at 128+ concurrent users
- GPU required (not suitable for student laptops)
- Python ecosystem makes it easy to deploy on a department server
- Supports structured output via Outlines integration

---

## 6. Constrained Output & Structured Generation

This is what makes small models viable. Instead of hoping the model outputs valid JSON, you **guarantee** it.

### How Constrained Decoding Works

At each token generation step, the engine masks all tokens that would create invalid JSON. The model can only generate tokens that conform to your schema. Result: **100% structurally valid output, always.**

### Ollama's Built-in JSON Schema Mode

```python
import ollama

# Define the search query schema
schema = {
    "type": "object",
    "properties": {
        "terms": {"type": "array", "items": {"type": "string"}},
        "boolean_query": {"type": "string"},
        "author": {"type": ["string", "null"]},
        "date_from": {"type": ["string", "null"]},
        "date_to": {"type": ["string", "null"]},
        "material_type": {
            "type": "string",
            "enum": ["any", "book", "article", "thesis", "conference", "journal"]
        },
        "sort": {
            "type": "string",
            "enum": ["relevance", "date_desc", "date_asc", "title"]
        },
        "expanded_terms": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["terms", "material_type", "sort"]
}

response = ollama.chat(
    model="qwen3:1.7b",
    messages=[{
        "role": "system",
        "content": "You are a library search query parser. Extract structured search parameters from the user's natural language query. Expand terms with synonyms and related concepts."
    }, {
        "role": "user",
        "content": "find me recent articles about machine learning in healthcare"
    }],
    format=schema  # Ollama enforces this schema via constrained decoding
)
# Output is GUARANTEED to match the schema
```

### Reliability Data

| Approach | JSON Validity | Semantic Accuracy | Source |
|----------|--------------|-------------------|--------|
| Small model (1-3B), no constraints | ~80% valid | ~40-60% correct fields | SLOT paper |
| Small model + constrained decoding | **100% valid** | ~70-85% correct fields | JSONSchemaBench |
| Small model + fine-tuning + constraints | **100% valid** | **~85-95% correct fields** | SLOT paper (1B fine-tuned) |
| Qwen3-1.7B + Ollama JSON schema (no fine-tuning) | **100% valid** | ~80-90% (native function calling training) | Qwen3 benchmarks |

**Bottom line:** Qwen3-1.7B with Ollama's JSON schema mode should give you ~85%+ correct query translations out of the box. Fine-tuning on a few hundred examples of your specific query patterns would push this to ~95%+.

---

## 7. Frontend: Search UI (Not Chat UI)

### Why a Search UI, Not a Chat UI

| Chat UI (LM Studio, Jan, etc.) | Search UI (what you should build) |
|-------------------------------|----------------------------------|
| Expects conversational LLM | Query box → results list |
| Tool calling must work perfectly | LLM is invisible to the user |
| User sees raw JSON tool calls | User sees formatted results |
| Confusing when model hallucinates | Model errors → fallback to keyword search |
| Generic interface for everything | Purpose-built for library search |
| Requires LLM literacy | Familiar Google/Primo-like UX |

### Existing Projects to Fork/Study

**Perplexica** (32.6k GitHub stars, MIT license)
- Next.js + TypeScript, search-focused UI with citations
- Supports Ollama for local models
- Three search modes (Speed/Balanced/Quality)
- Architecture: SearXNG search → LLM synthesis → cited results
- **Most relevant:** Replace SearXNG with your Primo API calls
- Repo: `github.com/ItzCrazyKns/Perplexica`

**Farfalle** (FastAPI + Next.js)
- Lighter weight than Perplexica
- Supports Ollama for fully local deployment
- Multiple search providers

**Khoj** (33.3k stars, AGPL)
- Python backend, supports Ollama/vLLM
- Semantic search over documents
- Desktop + web clients
- More of a personal assistant than pure search

### Proposed UI Layout

```
┌──────────────────────────────────────────────────────────┐
│  SFU Library Search                    [Settings] [Help] │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │  Search: recent articles on ML in healthcare   │ [🔍] │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  Filters: [Articles ▼] [2020-present ▼] [Sort: Date ▼]  │
│  AI understood: "machine learning" AND "healthcare"      │
│  Expanded with: "deep learning", "medical AI", "clinical"│
│                                                          │
│  ─── 47 results ─────────────────────────────────────── │
│                                                          │
│  1. Deep Learning Applications in Clinical Medicine      │
│     Smith, J. et al. (2024) · Nature Medicine · Vol 30   │
│     [APA] [MLA] [BibTeX] [Save to Zotero] [PDF ↓]      │
│                                                          │
│  2. Machine Learning for Drug Discovery: A Review        │
│     Chen, L. & Wang, R. (2023) · Science · Vol 382      │
│     [APA] [MLA] [BibTeX] [Save to Zotero] [Full Text]  │
│                                                          │
│  3. AI-Assisted Diagnosis in Radiology                   │
│     ...                                                  │
│                                                          │
│  [Load more results]                                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Key UX Decisions

1. **Show the AI's interpretation** — "AI understood: X, expanded with: Y" builds trust and lets users correct mistakes
2. **Facet filters are deterministic** — date, type, sort are dropdowns, not LLM-generated
3. **Citation buttons are one-click** — no LLM needed, template-based
4. **No streaming chat** — results appear as a list, like Google Scholar
5. **Works without LLM** — just a search box + results, like regular Primo

---

## 8. Packaging as .exe / Desktop App

### Recommended: Tauri or Wails (Not Electron)

For a search-focused app (not a chat app), smaller frameworks are better:

| Framework | Binary Size | RAM Usage | Backend Language | Verdict |
|-----------|------------|-----------|-----------------|---------|
| **Tauri** | ~600 KB - 5 MB | ~30-50 MB | Rust + any JS frontend | **Best for smallest footprint** |
| **Wails** | ~10-15 MB | ~40-60 MB | Go + any JS frontend | **Best if backend is Go** |
| **Electron** | ~100-150 MB | ~150-300 MB | Node.js | Overkill for a search app |
| **PyInstaller** (no GUI framework) | ~30-50 MB | ~50-100 MB | Python | Simplest if keeping Python |

### Distribution Package Contents

```
SFU-Library-Search-Setup.exe (or .dmg / .AppImage)
├── SFU Library Search app          (~5 MB with Tauri)
├── Ollama installer (optional)     (bundled or download link)
├── Qwen3-1.7B model               (download on first run, ~1.5 GB)
└── Config
    ├── Default Ollama model config
    └── SFU authentication setup wizard
```

### First-Run Experience

1. User runs installer → app installs (~5 MB)
2. App opens → "Welcome to SFU Library Search"
3. SFU login prompt → user enters NetID + password + TOTP secret
4. "Would you like to enable AI-powered search?" → Yes/No
   - **Yes:** Downloads Ollama (~200 MB) + Qwen3-1.7B (~1.5 GB). Progress bar.
   - **No:** App works immediately with keyword search only
5. Search box appears. Done.

**Total install size:**
- Without AI: ~5 MB (just the search app)
- With AI: ~1.7 GB (app + Ollama + model)

Compare to the original plan's ~9 GB for a MoE model. This is a **5x reduction**.

---

## 9. Free Cloud LLM Alternatives

### For the Search Engine (Track A) — Lightweight Query Parsing

Since the LLM only does query parsing (not conversation), cloud API costs are minimal:

| Service | Cost for Query Parsing | MCP Support | FIPPA Issue? |
|---------|----------------------|-------------|-------------|
| **Ollama (local)** | Free | N/A (embedded) | **No** — all local |
| **Gemini API (free tier)** | Free (15 RPM, 1K req/day) | N/A (direct API) | Yes — US servers |
| **Claude API (free tier)** | Limited | N/A (direct API) | Yes — US servers |
| **Groq (free tier)** | Free (30 RPM) | N/A (direct API) | Yes — US servers |

**For FIPPA compliance:** Local Ollama is the only safe option. Cloud APIs could work for a non-authenticated, metadata-only mode (no personal data in queries).

### For the MCP Server (Track B) — Full LLM Needed

| Client | Cost | MCP Support | Best For |
|--------|------|------------|---------|
| **Gemini CLI** | Free | Native | Most generous free tier |
| **Claude Desktop (free)** | Free | Native | Best tool calling quality, tight limits |
| **LM Studio + local model** | Free | Native | Privacy-conscious users |
| **ChatGPT (Education)** | University license | Native | If SFU has an agreement |

---

## 10. University Pain Points & Compliance

### How Track A (Search Engine) Changes the Risk Profile

| Concern | Full MCP + Chat LLM (old plan) | Contextual Search Engine (new plan) |
|---------|-------------------------------|-------------------------------------|
| **FIPPA** | LLM processes full conversations → PII risk | LLM only parses search terms → minimal PII |
| **Hallucination** | LLM generates results → fabrication risk | LLM only structures queries → results come from real catalog |
| **Academic integrity** | "AI is doing the research" | "AI helps you search, like Google" |
| **Accessibility** | Chat UI accessibility is complex | Search UI is standard web — well-understood WCAG patterns |
| **Cost** | 8-9 GB model, needs beefy hardware | 1.5 GB model, runs on any laptop |
| **LLM dependency** | Entire tool breaks without LLM | Core search works without any LLM |
| **Governance pitch** | "We're deploying an AI assistant" | "We're improving our search bar" |

### Remaining Pain Points

#### Critical: FIPPA (if using cloud LLM)
- **Solved by local Ollama** — query parsing stays on the user's machine
- If using cloud API for query parsing: search terms alone are lower risk than full conversations, but still technically PII if they reveal research interests
- **Recommendation:** Default to local Ollama; offer cloud as opt-in with privacy warning

#### High: Licensed Content / Copyright
- Unchanged from previous analysis
- **Mitigation:** LLM never sees full-text content; only processes user queries and metadata
- The search engine returns links to licensed resources, not reproductions

#### High: Privacy Impact Assessment
- Still required, but **much simpler PIA** for a search tool vs. a conversational AI assistant
- The PIA essentially covers: "We parse search queries locally using a small language model to improve search accuracy. No user data leaves the machine. The model does not retain or learn from user queries."

#### Medium: OIPC AI Oversight
- Still need transparency about AI use
- **Mitigation:** Show "AI understood: ..." in the UI → users see exactly what the AI did
- The AI's role is so narrow (query parsing) that explainability is straightforward

#### Medium: Governance Approval
- **Much easier pitch** than "we're deploying an AI chatbot"
- Frame as: "search quality improvement using local NLP"
- Similar to how Primo's own Research Assistant (Ex Libris beta) is positioned
- **Note:** Ex Libris has already launched a beta AI-powered Primo Research Assistant — SFU's vendor is moving this direction anyway

#### Low: Academic Integrity
- A search engine doesn't write papers or summarize sources
- It's equivalent to Google Scholar with better understanding of your query
- No faculty objection expected

### Approval Complexity Comparison

| Step | AI Chat Assistant (old) | Search Engine (new) |
|------|------------------------|-------------------|
| PIA | Complex (full conversations, PII) | Simple (search terms only, local processing) |
| AIFPIA | Required | May not be required (narrow AI use, no decisions about people) |
| IT Security Review | Complex (LLM infra, API keys, data flows) | Simple (local binary, standard HTTPS to Primo) |
| Legal Review | FIPPA deep dive needed | Routine if using local LLM |
| Accessibility Audit | Chat UI is novel | Search UI follows established patterns |
| **Estimated timeline** | **6-18 months** | **2-6 months** |

---

## 11. Implementation Roadmap

### Phase 1: Core Search Backend (1-2 weeks)

- [ ] Extract search logic from MCP tools into standalone Python module
- [ ] Create a `SearchEngine` class that:
  - Accepts structured query parameters (dict/JSON)
  - Calls Primo API with authentication
  - Returns formatted results with metadata
  - Generates citations on demand
  - Handles Zotero save operations
- [ ] Add keyword-search fallback (no LLM needed)
- [ ] Test standalone without any LLM

### Phase 2: LLM Query Parser (1 week)

- [ ] Create `QueryParser` class that:
  - Connects to Ollama (`http://localhost:11434`)
  - Sends natural language query with JSON schema constraint
  - Returns structured search parameters
  - Falls back to keyword passthrough if Ollama unavailable
- [ ] Test with Qwen3-1.7B via Ollama
- [ ] Define JSON schema for search parameters
- [ ] Benchmark accuracy: test 50+ natural language queries, measure correct field extraction
- [ ] If accuracy < 85%: try Qwen3-4B or fine-tune on ~200 example queries

### Phase 3: Search Frontend (2-3 weeks)

- [ ] Choose framework: Tauri (smallest) or Electron (fastest to build)
- [ ] Build search UI:
  - Search bar with auto-submit
  - "AI interpretation" display
  - Results list with metadata
  - Citation buttons (APA/MLA/Chicago/BibTeX)
  - Save to Zotero buttons
  - PDF download links
  - Facet filters (date, material type, sort)
- [ ] SFU authentication flow (first-run setup wizard)
- [ ] Optional Ollama setup wizard
- [ ] WCAG accessibility pass

### Phase 4: Packaging & Distribution (1-2 weeks)

- [ ] Compile to platform installers:
  - Windows: `.exe` installer (Tauri/NSIS or PyInstaller)
  - macOS: `.dmg`
  - Linux: `.AppImage` or `.deb`
- [ ] Bundle Ollama download (or prompt to install)
- [ ] Auto-download Qwen3-1.7B on first "enable AI" click
- [ ] Test on clean Windows 10/11, macOS, Ubuntu machines
- [ ] Write 1-page setup guide for students

### Phase 5: MCP Server Polish (1 week, parallel)

- [ ] Ensure MCP server works standalone via stdio
- [ ] Test with Claude Desktop, LM Studio, Gemini CLI
- [ ] Write MCP connection guide for power users
- [ ] Package as `.mcpb` bundle (for Claude Desktop users)

### Phase 6: University Pilot (2-4 weeks)

- [ ] Deploy to 5-10 test users (library staff, grad students)
- [ ] Collect query accuracy metrics (AI interpretation vs. user intent)
- [ ] Collect search satisfaction data
- [ ] Draft PIA document
- [ ] Present to SFU Library IT as "search quality improvement"

---

## Appendix A: Model & Tool Comparison

### Query Understanding Models

| Model | Size (Q4) | RAM | Function Calling | License | Best For |
|-------|-----------|-----|-----------------|---------|----------|
| Qwen3-1.7B | 1.5 GB | ~2 GB | Native | Apache 2.0 | **Default pick** — best accuracy/size ratio |
| Qwen3-4B | 2.5 GB | ~3.5 GB | Native | Apache 2.0 | Step-up if 1.7B isn't enough |
| Ministral-3B | 2 GB | ~3 GB | Native | Apache 2.0 | Alternative to Qwen |
| Qwen2.5-Coder-1.5B | 1.2 GB | ~1.8 GB | Community fine-tunes | Apache 2.0 | If structured output focus |

### Serving Engines

| Engine | Use Case | RAM Overhead | GPU Needed | Structured Output |
|--------|----------|-------------|------------|-------------------|
| Ollama | Desktop (Track A) | ~200 MB | No | JSON schema mode |
| llama.cpp | Embedded in app | ~100 MB | No | GBNF grammars |
| vLLM | Multi-user server (Track B Tier 2) | ~2 GB | Yes | Outlines integration |

### Embedding Models for Reranking

| Model | Size | RAM | Speed | Accuracy (MTEB) |
|-------|------|-----|-------|-----------------|
| all-MiniLM-L6-v2 | 43 MB | ~80 MB | 14.7ms/1K tok | ~80% |
| BGE-small-en-v1.5 | 66 MB | ~130 MB | Fast | ~82% |
| Nomic Embed v1.5 | 274 MB | ~400 MB | Moderate | ~81.2% |

## Appendix B: Reference Projects

| Project | Stars | Architecture | Relevance |
|---------|-------|-------------|-----------|
| [Perplexica](https://github.com/ItzCrazyKns/Perplexica) | 32.6k | Next.js + SearXNG + Ollama | Fork frontend for search UI |
| [Khoj](https://github.com/khoj-ai/khoj) | 33.3k | Python + Ollama + semantic search | Study BYOM architecture |
| [STORM](https://github.com/stanford-oval/storm) | — | DSPy + multi-source retrieval | Academic search patterns |
| [Outlines](https://github.com/dottxt-ai/outlines) | — | Structured generation library | If you need finer control than Ollama |
| [Haystack](https://haystack.deepset.ai) | — | Modular RAG framework | Production pipeline patterns |

## Appendix C: Key Sources

**Models & Inference:**
- [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
- [Qwen2.5 Blog](https://qwenlm.github.io/blog/qwen2.5/)
- [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [SLOT Paper (1B model structured output)](https://arxiv.org/html/2505.04016v1)
- [JSONSchemaBench](https://arxiv.org/html/2501.10868v3)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)

**Architecture:**
- [Perplexica Architecture](https://deepwiki.com/ItzCrazyKns/Perplexica)
- [Haystack RAG Pipelines](https://haystack.deepset.ai)
- [Intent-Driven NL Interface (hybrid LLM approach)](https://medium.com/data-science-collective/intent-driven-natural-language-interface)
- [Ex Libris Primo Research Assistant Beta](https://exlibrisgroup.com/announcement/ex-libris-launches-a-beta-program-of-generative-ai-powered-primo-research-assistant/)

**Serving:**
- [Red Hat: vLLM vs llama.cpp](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp)
- [Red Hat: Ollama vs vLLM Benchmarks](https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm-deep-dive-performance-benchmarking)
- [Constrained Decoding Guide](https://www.aidancooper.co.uk/constrained-decoding/)

**Compliance:**
- [BC FIPPA](https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/96165_00)
- [OIPC "Getting Ahead of the Curve"](https://www.oipc.bc.ca/reports/special-reports/)
- [OPC AI Principles (Canada)](https://www.priv.gc.ca/en/privacy-topics/technology/artificial-intelligence/)

**Desktop Frameworks:**
- [Tauri](https://tauri.app/)
- [Wails](https://wails.io/)
- [PyInstaller](https://pyinstaller.org/)
