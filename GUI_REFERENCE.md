# GUI Reference: What To Reuse & How To Build It

## TL;DR Decision

**Fork FAIRplexica** (UB-Mannheim's academic Perplexica fork) and strip it down from a chat UI to a search UI. It gives you:
- Next.js 15 + Tailwind + Headless UI (modern, well-structured)
- 11 LLM provider integrations including Ollama, LM Studio, OpenAI
- Admin dashboard with JWT auth (institutional deployment ready)
- GLOBAL_CONTEXT config (set to library search instead of "research data management")
- Docker setup, SQLite persistence, file upload support
- MIT license, actively maintained

Then replace SearXNG with your Primo API backend and reshape the chat-first UI into a search-first UI.

---

## 1. FAIRplexica: What You Get For Free

**Repo:** https://github.com/UB-Mannheim/FAIRplexica
**License:** MIT | **Stars:** 14 | **Last active:** Oct 2025 | **Commits:** 580

### Architecture (Unified Next.js 15)
```
src/
  app/                        # Next.js App Router
    about/page.tsx            # Institutional about page
    admin/page.tsx            # Admin dashboard (JWT protected)
    api/                      # API routes (search, chat, models, config)
    c/[chatId]/               # Chat sessions
    discover/page.tsx         # Trending content
    library/page.tsx          # History
    settings/page.tsx         # User settings
  components/
    Chat.tsx                  # Active conversation
    ChatWindow.tsx            # State machine (loading/error/empty/active)
    EmptyChat.tsx             # Landing page ("Research begins here.")
    EmptyChatMessageInput.tsx # Search input with action bar
    MessageBox.tsx            # Individual message with markdown + citations
    MessageSources.tsx        # Citation card grid (2-col → 4-col responsive)
    MessageInput.tsx          # Follow-up input
    ThinkBox.tsx              # AI reasoning accordion
    Sidebar.tsx               # Icon sidebar (desktop) / bottom bar (mobile)
    Layout.tsx                # Main content wrapper
    MessageInputActions/
      Attach.tsx              # File upload
      Optimization.tsx        # Speed/Balanced/Quality modes
  lib/
    providers/                # LLM backends
      ollama.ts               # ← Already integrated
      lmstudio.ts             # ← Already integrated
      openai.ts
      anthropic.ts
      groq.ts
      gemini.ts
      deepseek.ts
      + 4 more
    search/
      metaSearchAgent.ts      # SearXNG integration ← REPLACE with Primo API
    config.ts                 # config.toml reader
    auth.ts                   # JWT admin auth
    db/                       # SQLite via Drizzle ORM
```

### What To Keep vs. Replace vs. Remove

| Component | Keep? | Action |
|-----------|-------|--------|
| **Next.js 15 + Tailwind** | Keep | Foundation |
| **Admin dashboard** (/admin) | Keep | Configure LLM backend, API keys, global context |
| **LLM providers** (lib/providers/) | Keep | Ollama, LM Studio, OpenAI all pre-built |
| **JWT auth** (lib/auth.ts) | Keep | Good for institutional deployment |
| **Settings page** | Keep | Let users configure their LLM backend |
| **About page** | Keep | Replace branding with SFU Library |
| **SQLite + Drizzle** | Keep | Search history, saved searches |
| **File upload** | Maybe | Could be useful for PDF search |
| **GLOBAL_CONTEXT** | Keep | Change from "research data management" to library search context |
| **Docker setup** | Keep | 2 containers (app + Ollama instead of SearXNG) |
| **EmptyChat landing** | Reshape | Change from chat-first to search-first |
| **MessageSources** | Reshape | Becomes the primary results display, not secondary |
| **Chat flow** (Chat.tsx, MessageBox.tsx) | Reshape | Results list instead of conversation |
| **ThinkBox** | Reshape | Becomes "AI understood: ..." display |
| **SearXNG integration** | **Replace** | Replace with Primo API calls |
| **Discover page** | **Remove** | Not relevant for library search |
| **Weather/News widgets** | **Remove** | Not relevant |
| **Video/Image search** | **Remove** | Not relevant (unless searching multimedia catalog) |
| **Copilot mode** | **Remove** | Complexity not needed |

### Docker: Replace SearXNG with Ollama

**Before (FAIRplexica):**
```yaml
services:
  searxng:                    # Web metasearch
    image: searxng/searxng
    ports: 4001:8080
  app:
    build: .
    ports: 3000:3000
```

**After (SFU Library Search):**
```yaml
services:
  ollama:                     # Local LLM inference
    image: ollama/ollama
    ports: 11434:11434
    volumes: ollama-models:/root/.ollama
  app:
    build: .
    ports: 3000:3000
    environment:
      - OLLAMA_URL=http://ollama:11434
```

---

## 2. The Reshape: Chat UI → Search UI

### What Changes Visually

**FAIRplexica (current):**
```
┌──────────────────────────────────────────┐
│ [Sidebar] │  "Research begins here."     │
│ Home      │  ┌──────────────────────┐    │
│ Discover  │  │ Ask anything...      │    │
│ Library   │  └──────────────────────┘    │
│           │  [Sources] [Model] [Speed]   │
│           │                              │
│           │  ← Chat-first: empty page    │
│           │     until first message       │
└──────────────────────────────────────────┘
```

**After reshape (SFU Library Search):**
```
┌──────────────────────────────────────────────────────────────┐
│ [Sidebar] │  SFU Library Search                [Settings]    │
│ Search    │                                                  │
│ Saved     │  ┌──────────────────────────────────────┐        │
│ History   │  │ Search the library...                │  [🔍]  │
│           │  └──────────────────────────────────────┘        │
│           │  [Articles ▼] [Date ▼] [Sort: Relevance ▼]      │
│           │                                                  │
│           │  AI: "climate change" AND "salmon" expanded with │
│           │  "sockeye", "fisheries", "Pacific salmon"        │
│           │                                                  │
│           │  ─── 47 results ──────────────────────────────── │
│           │                                                  │
│           │  1. Title of First Result                        │
│           │     Smith, J. et al. (2024) · Nature · Vol 30    │
│           │     [APA] [MLA] [BibTeX] [Zotero] [PDF ↓]       │
│           │                                                  │
│           │  2. Title of Second Result                       │
│           │     Chen, L. (2023) · Science · Vol 382          │
│           │     [APA] [MLA] [BibTeX] [Zotero] [Full Text]   │
│           │                                                  │
│           │  [Page 1] [2] [3] ... [47]                       │
└──────────────────────────────────────────────────────────────┘
```

### Component Mapping

| FAIRplexica Component | Becomes | What Changes |
|----------------------|---------|-------------|
| `EmptyChat.tsx` | `SearchLanding.tsx` | Search bar prominent, no "ask anything" framing |
| `EmptyChatMessageInput.tsx` | `SearchInput.tsx` | Replace Sources/Model/Optimization with facet filters |
| `MessageSources.tsx` | `SearchResults.tsx` | **Promoted from secondary to primary display.** Full result cards with metadata, citations, actions |
| `MessageBox.tsx` | `AIInterpretation.tsx` | Shows "AI understood: X, expanded with: Y" — compact, above results |
| `ThinkBox.tsx` | Part of `AIInterpretation.tsx` | Shows query expansion reasoning |
| `Chat.tsx` | `SearchPage.tsx` | Results list replaces conversation thread |
| `Sidebar.tsx` | Keep mostly as-is | Change nav: Search, Saved Searches, History |
| `MessageInputActions/Optimization.tsx` | `FacetFilters.tsx` | Date range, material type, sort order |
| `Layout.tsx` | Keep as-is | Same responsive layout |
| `Navbar.tsx` | Keep mostly as-is | Add SFU branding |
| Settings dialog | Keep as-is | Already has LLM provider configuration |
| Admin dashboard | Keep as-is | Perfect for institutional configuration |

---

## 3. Reusable UI Components From The Ecosystem

### What FAIRplexica Already Includes (No Extra Dependencies)

| Component | Library | Already In FAIRplexica |
|-----------|---------|----------------------|
| Popovers, switches, dialogs | Headless UI v2 | Yes |
| Tooltips | (custom) | Yes |
| Icons | Lucide React | Yes |
| Toasts | Sonner | Yes |
| Theme switching | next-themes | Yes |
| Markdown rendering | markdown-to-jsx | Yes |
| Auto-resize textarea | react-textarea-autosize | Yes |
| Animation | (none needed for search) | — |

### What To Add For Search-Specific Features

| Feature | Recommended Library | Why |
|---------|-------------------|-----|
| **Citation formatting** | **Citation.js** (`citation-js`) | Converts metadata → APA/MLA/Chicago/BibTeX. Format-independent, browser + Node.js. Most mature option. |
| **Faceted search filters** | Build with Headless UI (already available) | Popover + checkbox groups for material type, date range picker for dates. Don't need a search framework. |
| **Pagination** | Build with Tailwind (simple) | Primo API already returns paginated results. Just render page buttons. |
| **Autocomplete/suggestions** | Headless UI Combobox | Already in the dependency tree. Use for search suggestions. |
| **Copy citation button** | Already have Copy in MessageActions | Reuse the copy-to-clipboard pattern |

### What NOT To Add

- **Algolia InstantSearch / ReactiveSearch** — overkill. You're not searching a local index; you're calling one API. These are designed for client-side faceted search over Elasticsearch/Typesense.
- **shadcn/ui** — FAIRplexica doesn't use it and adding it would mean reworking the existing Headless UI components. Stick with what's there.
- **TanStack Table** — you're displaying a results list, not a data table.

---

## 4. Search Result Card Design

### Inspired By: Google Scholar + Semantic Scholar + Primo

```
┌────────────────────────────────────────────────────────────┐
│ 1. Deep Learning Applications in Clinical Medicine         │  ← Title (link to detail)
│                                                            │
│ Smith, J. · Wang, L. · Chen, R.                            │  ← Authors
│ Nature Medicine · 2024 · Vol 30, Issue 4, pp. 112-128      │  ← Source, year, volume
│                                                            │
│ "This review examines the application of deep learning     │  ← Abstract snippet
│  models to clinical diagnosis, focusing on radiology..."   │     (first 150 chars)
│                                                            │
│ [📋 APA] [📋 MLA] [📋 Chicago] [📥 Zotero] [📄 PDF]       │  ← Action buttons
│                                                            │
│ Article · Peer-reviewed · DOI: 10.1038/s41591-024-...      │  ← Metadata badges
└────────────────────────────────────────────────────────────┘
```

### Tailwind Implementation (sketch)

```tsx
// SearchResultCard.tsx — builds on FAIRplexica's existing Tailwind theme
<div className="border-b border-light-200 dark:border-dark-200 py-4">
  {/* Title */}
  <h3 className="text-lg font-medium text-light-primary dark:text-dark-primary hover:underline cursor-pointer">
    {result.title}
  </h3>

  {/* Authors + Source */}
  <p className="text-sm text-light-secondary dark:text-dark-secondary mt-1">
    {result.authors.join(' · ')} · {result.source} · {result.year}
  </p>

  {/* Abstract snippet */}
  <p className="text-sm text-light-secondary dark:text-dark-secondary mt-2 line-clamp-2">
    {result.abstract}
  </p>

  {/* Action buttons */}
  <div className="flex gap-2 mt-3">
    <CitationButton format="apa" record={result} />
    <CitationButton format="mla" record={result} />
    <ZoteroSaveButton record={result} />
    {result.pdfUrl && <PDFDownloadButton url={result.pdfUrl} />}
  </div>

  {/* Badges */}
  <div className="flex gap-2 mt-2">
    <Badge>{result.type}</Badge>
    {result.peerReviewed && <Badge variant="success">Peer-reviewed</Badge>}
    {result.doi && <Badge variant="muted">DOI: {result.doi}</Badge>}
  </div>
</div>
```

---

## 5. How FAIRplexica Connects to Ollama (Already Built)

**File:** `src/lib/providers/ollama.ts`

FAIRplexica already has full Ollama integration:
- Reads Ollama URL from `config.toml`
- Lists available models via Ollama API
- Creates LangChain `ChatOllama` instances
- Creates embedding models via `OllamaEmbeddings`
- Supports model selection in the UI (searchable dropdown grouped by provider)

**What you'd change:** Instead of using LangChain's `ChatOllama` for conversation, you'd use the OpenAI SDK with `base_url` pointing to Ollama for structured JSON output (as described in the plan). LangChain is overkill for a single JSON extraction call.

---

## 6. Comparison: Start Fresh vs. Fork FAIRplexica

| Factor | Start Fresh (Next.js + shadcn) | Fork FAIRplexica |
|--------|-------------------------------|------------------|
| **Time to first working UI** | 2-3 weeks | 3-5 days |
| **LLM provider integration** | Build from scratch | 11 providers pre-built |
| **Admin dashboard** | Build from scratch | Pre-built with JWT |
| **Dark mode** | Set up theme system | Already working |
| **Mobile responsive** | Build from scratch | Already working |
| **Docker setup** | Configure from scratch | Pre-built, just swap SearXNG for Ollama |
| **Database/persistence** | Set up ORM | SQLite + Drizzle pre-configured |
| **Code you'd delete** | N/A | ~40% (chat flow, widgets, discover, SearXNG) |
| **Technical debt** | None | Some (LangChain dependency, chat-first architecture) |
| **Component library** | Choose freely (shadcn/ui) | Headless UI (already working) |
| **Full control** | Yes | Yes (MIT license, fork = your repo) |

**Recommendation:** Fork FAIRplexica. The amount of infrastructure you get for free (auth, settings, providers, Docker, DB, theming, responsive layout) far outweighs the effort of deleting the chat-specific components. You'd spend 3-5 days reshaping vs. 2-3 weeks building from zero.

---

## 7. Reference Projects Worth Studying

### For UI Patterns

| Project | What To Study | URL |
|---------|--------------|-----|
| **FAIRplexica** | Overall architecture, admin panel, provider integration | github.com/UB-Mannheim/FAIRplexica |
| **Perplexica/Vane** | MessageSources citation cards, ThinkBox reasoning display | github.com/ItzCrazyKns/Perplexica |
| **Google Scholar** | Minimal homepage, result card layout, citation modal | scholar.google.com |
| **Semantic Scholar** | TLDR summaries, citation velocity, compact result cards | semanticscholar.org |
| **VuFind** | Facet layouts, record detail pages, saved search UX | vufind.org |
| **Primo NDE** | Gradual disclosure, advanced search slide-out | exlibrisgroup.com/products/primo-discovery-service |

### For Technical Patterns

| Project | What To Study | URL |
|---------|--------------|-----|
| **Typesense v29.0** | NL → structured query (production implementation) | typesense.org/docs/guide/natural-language-search.html |
| **queelius/elasticsearch-lm** | Fine-tuned 1.1B model for NL → query DSL | github.com/queelius/elasticsearch-lm |
| **Citation.js** | Citation formatting library (APA/MLA/Chicago/BibTeX) | citation.js.org |
| **react-citation (Minitex)** | Citation display React component | github.com/Minitex/react-citation |
| **Algolia InstantSearch** | Search widget patterns (study, don't use directly) | github.com/algolia/instantsearch |

---

## 8. Optional Feature: Custom Dictionary Management UI

A settings page where users and admins can manage domain-specific terms that improve query parsing without retraining the LLM.

### UI Layout

```
┌──────────────────────────────────────────────────────────────┐
│ Settings > Custom Dictionary                    [Import JSON] │
├──────────────────────────────────────────────────────────────┤
│ [Abbreviations] [Synonyms] [Stop Expansions] [Preferred]     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Abbreviations                              [+ Add Entry]    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Abbreviation  │ Expansion                    │ Actions│   │
│  │───────────────│──────────────────────────────│────────│   │
│  │ REM           │ Resource & Environmental Mgmt │ ✏️ 🗑️  │   │
│  │ FASS          │ Faculty of Arts & Soc. Sci.  │ ✏️ 🗑️  │   │
│  │ ILL           │ Interlibrary Loan            │ ✏️ 🗑️  │   │
│  │ LCSH          │ Library of Congress Subject   │ ✏️ 🗑️  │   │
│  │               │ Headings                     │        │   │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  [Export JSON]  [Reset to Defaults]                           │
│                                                              │
│  ── Scope ──                                                 │
│  (•) My dictionary (personal)                                │
│  ( ) Institutional dictionary (admin only)                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Component Mapping (Builds on FAIRplexica)

| New Component | Builds On | Notes |
|---------------|-----------|-------|
| `DictionaryPage.tsx` | Settings page pattern | New route: `/settings/dictionary` |
| `DictionaryTable.tsx` | Headless UI Table | Editable rows with inline edit/delete |
| `DictionaryTabs.tsx` | Headless UI Tab Group | Abbreviations / Synonyms / Stop Expansions / Preferred |
| `ImportExportButtons.tsx` | File upload pattern (Attach.tsx) | JSON import/export |
| `DictionaryEntryModal.tsx` | Headless UI Dialog | Add/edit entry form |

### API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/dictionary` | GET | Get user's merged dictionary (personal + institutional) |
| `/api/dictionary` | PUT | Update user's personal dictionary |
| `/api/dictionary/entry` | POST | Add single entry |
| `/api/dictionary/entry` | DELETE | Remove single entry |
| `/api/admin/dictionary` | GET/PUT | Institutional dictionary (admin JWT required) |
| `/api/dictionary/export` | GET | Download as JSON |
| `/api/dictionary/import` | POST | Upload JSON file |

### Storage

- Personal dictionaries: SQLite (Drizzle ORM, already in FAIRplexica)
- Institutional dictionary: `config/institutional_dictionary.json` (admin-managed, version controlled)
- At query time, merge: institutional defaults → user overrides

---

## 9. Optional Feature: Model Improvement Dashboard (Admin)

An admin page for monitoring query accuracy and triggering model improvements.

### UI Layout

```
┌──────────────────────────────────────────────────────────────┐
│ Admin > Model Management                        [JWT Auth]    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ── Query Accuracy (last 30 days) ──                         │
│  Queries logged: 1,247                                       │
│  User refinements: 312 (25%)  ← users who changed search    │
│  Estimated accuracy: ~75%                                    │
│                                                              │
│  ── Active Model ──                                          │
│  Model: qwen3-1.7b-sfu-v3 (fine-tuned 2026-02-15)          │
│  Base: qwen3:1.7b                                            │
│  Training examples: 487                                      │
│  [Rollback to v2] [View changelog]                           │
│                                                              │
│  ── Improvement Options ──                                   │
│                                                              │
│  [ RAG Examples ]                                            │
│  Good query→JSON pairs: 234 curated                          │
│  [Review & Curate Queries]  [Test Current Accuracy]          │
│                                                              │
│  [ Fine-Tune (requires GPU) ]                                │
│  Status: Ready (487 training examples)                       │
│  [Start Fine-Tune Job]  [Schedule: Monthly ▼]                │
│  Last run: 2026-02-15 — accuracy 75% → 82%                  │
│                                                              │
│  ── Model Versions ──                                        │
│  │ Version │ Date       │ Examples │ Accuracy │ Status   │   │
│  │ v3      │ 2026-02-15 │ 487      │ 82%      │ Active   │   │
│  │ v2      │ 2026-01-10 │ 312      │ 78%      │ Archived │   │
│  │ v1      │ 2025-12-01 │ 200      │ 71%      │ Archived │   │
│  │ base    │ —          │ —        │ 65%      │ Fallback │   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Query Curation Sub-Page

```
┌──────────────────────────────────────────────────────────────┐
│ Admin > Curate Queries                                       │
├──────────────────────────────────────────────────────────────┤
│ Filter: [Refined only ▼] [Last 7 days ▼]                    │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Query: "REM thesis on salmon in BC"                    │  │
│  │ LLM Output:                                            │  │
│  │   terms: ["REM", "thesis", "salmon", "BC"]             │  │
│  │   material_type: "thesis"                              │  │
│  │ User refined to: "Resource Environmental Management    │  │
│  │   salmon British Columbia"                             │  │
│  │                                                        │  │
│  │ Correct JSON: [Edit]                                   │  │
│  │ [✓ Add to training set] [✗ Skip] [Flag for review]    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Curated: 23/156 pending                                     │
│  [Save & Continue]                                           │
└──────────────────────────────────────────────────────────────┘
```

This page fits naturally into FAIRplexica's existing admin dashboard (JWT-protected, already built).

---

## Appendix: FAIRplexica Config (What Gets Customized)

```toml
[GENERAL]
SIMILARITY_MEASURE = "cosine"
KEEP_ALIVE = "5m"
GLOBAL_CONTEXT = "SFU library catalog academic research"  # Changed from "research data management"
SYSTEM_PROMPT = "You are a library search assistant for Simon Fraser University. Parse the user's search query and extract structured search parameters."

[ADMIN]
USERNAME = "admin"
PASSWORD = ""
JWT_SECRET = ""

[MODELS]
# Users configure their preferred LLM backend here
[MODELS.OLLAMA]
API_URL = "http://localhost:11434"

[MODELS.LM_STUDIO]
API_URL = "http://localhost:1234/v1"

[MODELS.OPENAI]
API_KEY = ""

[API_ENDPOINTS]
# Replace SearXNG with Primo API
PRIMO_API_URL = "https://sfu-primo.hosted.exlibrisgroup.com/primo_library/libweb/webservices/rest/primo-explore/v1/pnxs"
```
