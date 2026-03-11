# SFU Library MCP — Thin Client Deployment Plan

## Executive Summary

This plan details how to package the SFU Library MCP server as a lightweight, locally-hosted desktop application that runs with a local MoE LLM on 16GB RAM systems. It covers model selection, GUI/exe packaging, free cloud alternatives, and university compliance pain points.

**Key assumption:** The SFU CAS login workaround has been stripped out and the tool authenticates directly via SFU's SSO (CAS + Duo MFA).

---

## Table of Contents

1. [Current MCP Server Capabilities](#1-current-mcp-server-capabilities)
2. [Local MoE LLM Selection (16GB RAM)](#2-local-moe-llm-selection-16gb-ram)
3. [GUI Desktop Application & .exe Packaging](#3-gui-desktop-application--exe-packaging)
4. [Free Online LLMs with MCP Support](#4-free-online-llms-with-mcp-support)
5. [University Pain Points & Compliance](#5-university-pain-points--compliance)
6. [Recommended Architecture](#6-recommended-architecture)
7. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. Current MCP Server Capabilities

The server exposes **24 MCP tools** across these categories:

| Category | Tools | Count |
|----------|-------|-------|
| **Search** | `search_library`, `search_by_author`, `search_by_subject`, `search_by_isbn`, `search_electronic_resources`, `get_item_details` | 6 |
| **Citations & Export** | `generate_citation` (APA/MLA/Chicago/BibTeX), `batch_generate_citations`, `export_search_results`, `get_full_text_links`, `batch_isbn_lookup`, `get_metrics` | 6 |
| **PDF Download** | `download_article`, `read_article`, `download_from_url` | 3 |
| **Authentication** | `authenticate`, `get_token_status`, `clear_cache` | 3 |
| **Zotero Integration** | `save_to_zotero`, `list_zotero_collections`, `batch_save_to_zotero`, `search_zotero`, `get_zotero_collection_items`, `backfill_collection_pdfs`, `get_zotero_status`, `zotero_authenticate` | 8 |
| **Diagnostics** | `get_diagnostics` | 1 |

**Authentication flow:** SFU CAS SSO → Duo MFA (TOTP) → EZProxy session → JWT extraction from Primo sessionStorage. Direct SFU login assumed (no workaround).

**Key dependencies:** Selenium (Chrome), curl_cffi, Playwright, pyzotero, CrossRef API, SFU Primo REST API, EZProxy.

---

## 2. Local MoE LLM Selection (16GB RAM)

### Recommended Models (ranked)

| Rank | Model | Total/Active Params | Quantization | Size | Speed (est.) | Tool Calling |
|------|-------|-------------------|-------------|------|-------------|-------------|
| 1 | **Qwen1.5-MoE-A2.7B** | 14.3B / 2.7B | Q4_K_M | ~8-9 GB | 10-20 tok/s | Yes |
| 2 | **DeepSeek-Coder-V2-Lite** | 16B / 2.4B | Q4_K_M | ~10.4 GB | 5-15 tok/s | Yes |
| 3 | **GPT-OSS-20B** | 20B MoE | MXFP4 | ~12-13 GB | ~42 tok/s (GPU) | Yes |
| 4 | **Qwen 3.5 35B-A3B** | 35B / 3B | Q2_K / IQ3_XXS | ~10-12 GB | 3-8 tok/s | Yes |

**Why NOT Mixtral 8x7B:** At Q4_K_M it's ~26 GB — doesn't fit. Even Q2_K (~15.6 GB) leaves no room for OS/context.

**Top pick for this use case:** **Qwen1.5-MoE-A2.7B Q4_K_M** — fits comfortably in 8-9 GB, leaves 7+ GB for OS/context, and has reliable tool calling support. The MCP server needs a model that can reliably generate tool calls, not necessarily the smartest model.

### Local LLM Runners with MCP Client Support

| Runner | MCP Client Support | Recommended? | Notes |
|--------|-------------------|-------------|-------|
| **LM Studio** (v0.3.17+) | Native MCP client | **Best for end users** | GUI model manager, configure MCP via `mcp.json`, tool call confirmation dialogs |
| **Jan.ai** | Native MCP host | Good | Open-source (AGPLv3), ships with default MCP servers |
| **Ollama** + mcp-client-for-ollama | Via external wrapper | Most flexible | Broadest model ecosystem, TUI client |
| **AnythingLLM** (v1.8.0+) | MCP tool support | Good | Desktop app, agent-based MCP integration |
| **Continue.dev** | MCP client | Good for devs | VS Code/JetBrains extension, works with Ollama/LM Studio backends |

### Recommended Stack for 16GB

```
┌──────────────────────────────┐
│  LM Studio (GUI Frontend)   │  ← User-facing chat interface
│  + Qwen1.5-MoE-A2.7B Q4_K_M│  ← Local model, ~8-9 GB
├──────────────────────────────┤
│  MCP Client (built into LM  │  ← Connects to MCP server via stdio
│  Studio v0.3.17+)           │
├──────────────────────────────┤
│  SFU Library MCP Server     │  ← Compiled to .exe via PyInstaller
│  (stdio transport)          │
└──────────────────────────────┘
```

---

## 3. GUI Desktop Application & .exe Packaging

### Option A: Off-the-Shelf GUI + .mcpb Bundle (Recommended — Least Development)

**Architecture:**
1. Users install **Msty**, **Jan.ai**, or **LM Studio** (one-time)
2. Your MCP server is packaged as a `.mcpb` bundle (official MCP distribution format)
3. Users one-click install the `.mcpb` file → MCP server auto-configures

**MCP Bundle Format (.mcpb):**
- ZIP archive containing your MCP server + `manifest.json`
- Official standard from the Model Context Protocol project
- Supports Python servers (bundled or requiring runtime)
- One-click install in compatible clients (Claude Desktop, LM Studio, Jan, etc.)
- Similar model to Chrome extensions (.crx) or VS Code extensions (.vsix)

**Packaging the MCP server as .mcpb:**
```bash
# Install mcpb CLI
npm install -g @anthropic-ai/mcpb

# Package the server
mcpb pack --manifest manifest.json --output sfu-library-mcp.mcpb
```

**Pros:** Minimal custom development, users get a polished maintained UI, model updates handled by the GUI app.
**Cons:** Users install two things; depends on third-party app.

### Option B: Single Installer — Electron + Compiled MCP Server

**Architecture:**
1. Compile MCP server with **PyInstaller** into a standalone `.exe`
2. Use **chat-mcp** (Electron-based MCP chat client) or fork it
3. Bundle with **electron-builder + NSIS** into a single Windows installer
4. Installer deploys both the GUI and MCP server, wires up config automatically

**Compiling the MCP server:**
```bash
# Using PyInstaller
pip install pyinstaller
pyinstaller --onefile --name sfu-library-mcp src/sfu_library_mcp_server.py

# Alternative: Nuitka (better performance, harder to decompile)
pip install nuitka
nuitka --standalone --onefile --output-filename=sfu-library-mcp src/sfu_library_mcp_server.py
```

**Key technical consideration:** The MCP stdio transport works by spawning the server as a subprocess and communicating via stdin/stdout. A PyInstaller `.exe` works for this — the GUI app just spawns `sfu-library-mcp.exe` as a child process.

**Pros:** Single installer, full UX control, no external dependencies.
**Cons:** Large installer (~150-200 MB with Electron + Python exe + model), you maintain the frontend.

### Option C: Lightweight Tauri App (Smallest Footprint)

- Tauri uses OS native webview (~10-20 MB vs Electron's ~100+ MB)
- `tauri-mcp` plugin exists but ecosystem is immature
- Would produce the smallest distributable but requires the most development effort
- **Verdict:** Watch this space but don't build on it yet.

### Recommended GUI Apps (Ranked for University Users)

| App | License | MCP Support | Ease of Use | Size | Best For |
|-----|---------|------------|------------|------|----------|
| **Msty** | Proprietary (free) | 113+ tools | Excellent | ~100 MB | Non-technical users, privacy-first |
| **Jan.ai** | AGPLv3 (open-source) | Native | Good | ~150 MB | University preference for open-source |
| **LM Studio** | Proprietary (free personal) | Native | Excellent | ~200 MB | Best integrated local model experience |
| **AnythingLLM** | MIT | Agent-based | Good | ~120 MB | Document/RAG-focused workflows |

---

## 4. Free Online LLMs with MCP Support

### Cloud Options with Native MCP Client Support

| Service | Cost | MCP Support | Rate Limits (Free) | Data Sovereignty | University Fit |
|---------|------|------------|-------------------|-----------------|---------------|
| **Gemini CLI** | Free | Native | 5 RPM Pro, ~1K req/day | US (Google) | Best free option for individuals |
| **Claude Desktop** | Free tier | Native | 10-45 msg/5hr | US (Anthropic) | Very limited scale |
| **ChatGPT** | Paid only | Native (Dev Mode) | N/A | US (OpenAI) | Only with Education license |

**Most generous free option:** **Gemini CLI** — free access to Gemini 2.5 Pro with 1M token context, native MCP support via `~/.gemini/settings.json`. However, data goes to Google US servers.

### Self-Hosted Free Stacks (Best for University)

| Stack | Cost | MCP Support | Privacy | Scalability |
|-------|------|------------|---------|-------------|
| **Ollama + LibreChat** | Free (self-hosted) | Via config | Excellent (on-prem) | Limited by hardware |
| **Ollama + Open WebUI** | Free (self-hosted) | Via mcpo proxy | Excellent | Limited by hardware |
| **Ollama + mcp-client-for-ollama** | Free | Native TUI | Excellent | Single user |

### Gateway/Proxy Layer

**LiteLLM** — Free, open-source AI gateway that:
- Provides a single MCP endpoint for all tools
- Supports 100+ LLM backends (local and cloud)
- Handles API key management, cost tracking, usage caps
- Ideal for university deployment to manage access across multiple providers

### ⚠️ Critical Warning for University Use

**ALL free cloud LLM services process data in the US.** This likely violates BC FIPPA (see Section 5). For university deployment, **self-hosted is the only compliant option** unless queries are fully anonymized before leaving Canada.

---

## 5. University Pain Points & Compliance

### Critical Blockers

#### 5.1 FIPPA Compliance (SEVERITY: CRITICAL)

BC's Freedom of Information and Protection of Privacy Act creates hard legal constraints:

- **Section 33.1** — Restricts disclosure of personal information outside Canada
- **Section 33(2)(u)** — Permits processing outside Canada only if "temporary"
- **Section 69** — Mandates Privacy Impact Assessments (PIAs) for systems handling personal info
- **SFU Policy I 10.04** — Requires employees to maintain confidentiality of personal information

**Impact:** If student search queries (which constitute personal information) are sent to US-based LLM APIs, SFU is likely in violation. This is the **#1 architectural constraint**.

**Mitigations:**
1. **Self-hosted local LLM** — All processing stays on-campus (best option)
2. **Azure OpenAI Canada regions** — Canada Central/East data centers (adds cost)
3. **Full query anonymization** — Strip all PII before sending to cloud LLMs (complex, brittle)

#### 5.2 OIPC AI Oversight Requirements (SEVERITY: HIGH)

BC Office of the Information and Privacy Commissioner requires:
- **AI Fairness and Privacy Impact Assessment (AIFPIA)** for all AI programs
- **User notification** when AI is used in decision-making
- **Transparency** — must explain how the AI system operates
- **Special restrictions** on highly sensitive information

#### 5.3 Privacy Impact Assessment (SEVERITY: HIGH)

Mandatory under FIPPA Section 69. Must be completed before deployment. Covers:
- What personal information is collected
- How it flows through the system
- Where it's stored and processed
- Who has access
- Retention and disposal policies

### Significant Concerns

#### 5.4 Licensed Content & Copyright (SEVERITY: HIGH)

- Library database licenses (JSTOR, ProQuest, EBSCOhost) typically **prohibit automated bulk access** and feeding content to AI systems
- Canadian copyright law has narrower fair dealing provisions than US fair use
- Summarizing copyrighted full-text via LLM may violate publisher agreements
- **Mitigation:** Only process metadata (titles, abstracts, DOIs), never full-text content through the LLM

#### 5.5 LLM Hallucination & Citation Accuracy (SEVERITY: HIGH)

- LLMs fabricate citations, invent authors, generate plausible but nonexistent sources
- For a library tool, a fabricated citation is worse than no result
- **Mitigation:** The MCP architecture grounds the LLM in real catalog data — the model doesn't generate search results, it only interprets user queries and calls MCP tools that return real data. This is the key advantage.

#### 5.6 Authentication & SSO (SEVERITY: MEDIUM)

- SFU uses enterprise SSO (CAS + Duo MFA)
- Tool must integrate with existing identity management
- Different user types have different access levels
- Audit trails required under FIPPA
- **Direct SFU login integration (current approach) addresses this well**

#### 5.7 Academic Integrity (SEVERITY: MEDIUM)

- Faculty may object that AI-powered search undermines information literacy learning outcomes
- Students may cite AI summaries instead of reading primary sources
- **Mitigation:** Position as a "search assistant" not a "research assistant" — it finds sources, doesn't analyze them

#### 5.8 WCAG Accessibility (SEVERITY: MEDIUM)

- BC policy requires WCAG 2.1 AA compliance for public-facing tools
- Chat interfaces must be keyboard-navigable, screen-reader compatible
- Streaming LLM responses need ARIA live regions
- **Mitigation:** Using an established GUI (Msty, Jan, LM Studio) inherits their accessibility work

#### 5.9 Cost Unpredictability (SEVERITY: MEDIUM)

- LLM API costs scale with usage (~35,000 students)
- University budgets are fixed annually
- **Mitigation:** Local models have zero per-query cost; if using cloud APIs, LiteLLM gateway provides usage caps

#### 5.10 Governance Approval Timeline

Deploying at SFU would likely require:
1. Privacy Impact Assessment (PIA) — mandatory
2. AI Fairness and Privacy Impact Assessment (AIFPIA) — recommended by OIPC
3. IT Security review
4. Library committee approval
5. Accessibility audit
6. General Counsel review

**Estimated timeline: 6-18 months** through the approval pipeline.

### Pain Points Summary

| Concern | Severity | Local LLM Solves It? |
|---------|----------|---------------------|
| FIPPA data sovereignty | Critical | **Yes** — data never leaves campus |
| OIPC AI oversight | High | Partially — still need AIFPIA |
| Privacy Impact Assessment | High | Simplifies it significantly |
| Licensed content/copyright | High | No — separate concern |
| Hallucination risk | High | **MCP architecture mitigates** — grounded in real data |
| SSO integration | Medium | No — separate concern (already handled) |
| Academic integrity | Medium | No — policy/positioning issue |
| WCAG accessibility | Medium | Partially — depends on GUI choice |
| Cost | Medium | **Yes** — zero per-query cost |
| Governance timeline | Medium | No — process is process |

---

## 6. Recommended Architecture

### Tier 1: Individual Researcher / Student (Thin Client)

```
┌─────────────────────────────────────────────────┐
│            User's Machine (16GB RAM)            │
│                                                 │
│  ┌──────────────────────┐  ┌─────────────────┐  │
│  │  Jan.ai or LM Studio │  │  SFU Library    │  │
│  │  (GUI + Local LLM)   │◄─┤  MCP Server     │  │
│  │                      │  │  (.exe or .mcpb)│  │
│  │  Qwen1.5-MoE-A2.7B  │  │                 │  │
│  │  Q4_K_M (~9GB)       │  │  stdio transport│  │
│  └──────────────────────┘  └────────┬────────┘  │
│                                     │           │
└─────────────────────────────────────┼───────────┘
                                      │ HTTPS
                              ┌───────▼───────┐
                              │  SFU Primo API │
                              │  EZProxy       │
                              │  Zotero API    │
                              │  CrossRef API  │
                              └───────────────┘
```

**Distribution:**
1. User installs Jan.ai (open-source, ~150 MB, one-time)
2. User downloads Qwen1.5-MoE-A2.7B Q4_K_M via Jan's model manager
3. User installs `sfu-library-mcp.mcpb` (one-click)
4. User enters SFU credentials on first run

**Privacy:** All LLM processing is local. Only library API calls (search queries, not chat content) leave the machine. FIPPA-compliant because no personal information goes to LLM cloud providers.

### Tier 2: Department / Lab Deployment

```
┌──────────────────────────────────────────────────────────┐
│               Department Server (GPU, 64GB+)             │
│                                                          │
│  ┌───────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Ollama       │  │  LiteLLM     │  │  SFU Library  │  │
│  │  (LLM Server) │◄─┤  (AI Gateway)│  │  MCP Server   │  │
│  │               │  │              │  │               │  │
│  │  Qwen 3.5     │  │  Usage caps  │  │  HTTP/SSE     │  │
│  │  35B-A3B      │  │  Auth mgmt   │  │  transport    │  │
│  └───────────────┘  └──────────────┘  └───────────────┘  │
│                            ▲                              │
└────────────────────────────┼──────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼────┐  ┌─────▼────┐  ┌─────▼────┐
        │ LibreChat │  │ Open     │  │ Jan.ai   │
        │ (Web)     │  │ WebUI    │  │ (Desktop)│
        └──────────┘  └──────────┘  └──────────┘
              Users connect via browser or desktop app
```

### Tier 3: Free Cloud Alternative (For Testing / Low-Volume Use)

```
  User Machine
  ┌──────────────────────┐
  │  Gemini CLI          │  ← Free tier, native MCP
  │  + SFU Library MCP   │  ← .mcpb or local server
  │    Server             │
  └──────────────────────┘
```

**Warning:** Data goes to Google US servers. Not FIPPA-compliant for production use with personal information.

---

## 7. Implementation Roadmap

### Phase 1: Package MCP Server (1-2 weeks)

- [ ] Strip authentication workaround, ensure direct SFU CAS login works
- [ ] Test MCP server with stdio transport standalone
- [ ] Compile with PyInstaller: `pyinstaller --onefile src/sfu_library_mcp_server.py`
- [ ] Test compiled .exe with LM Studio / Jan.ai
- [ ] Package as .mcpb bundle with manifest.json
- [ ] Write installation guide for non-technical users

### Phase 2: Local LLM Integration Testing (1-2 weeks)

- [ ] Test Qwen1.5-MoE-A2.7B Q4_K_M with tool calling via LM Studio
- [ ] Test DeepSeek-Coder-V2-Lite as fallback model
- [ ] Benchmark tool-call reliability (does the model correctly invoke MCP tools?)
- [ ] Test on minimum spec hardware (16GB RAM, no GPU)
- [ ] Document model-specific prompt templates for reliable tool calling

### Phase 3: GUI & Distribution (2-3 weeks)

- [ ] Choose GUI: Jan.ai (open-source) or Msty (polish) based on testing
- [ ] Create Windows installer (Inno Setup) bundling:
  - GUI app (portable)
  - Compiled MCP server (.exe)
  - Pre-configured mcp.json
  - Model download instructions
- [ ] Create macOS .dmg equivalent
- [ ] Test end-to-end on clean Windows 10/11 machines
- [ ] WCAG accessibility review of chosen GUI

### Phase 4: University Compliance (Ongoing, 3-6 months)

- [ ] Draft Privacy Impact Assessment (PIA)
- [ ] Prepare AIFPIA documentation
- [ ] Engage SFU Library IT for security review
- [ ] Engage SFU General Counsel for FIPPA sign-off
- [ ] Position tool as "search assistant" in academic integrity context
- [ ] Prepare faculty-facing documentation explaining LLM grounding via MCP

### Phase 5: Pilot Deployment (2-4 weeks)

- [ ] Deploy to 5-10 test users (library staff, graduate researchers)
- [ ] Collect feedback on usability, model quality, tool reliability
- [ ] Monitor for hallucination incidents
- [ ] Iterate on prompt templates and tool descriptions

---

## Appendix A: Model Download Sizes & URLs

| Model | GGUF File | Download Size | HuggingFace Repo |
|-------|-----------|--------------|-----------------|
| Qwen1.5-MoE-A2.7B Q4_K_M | `qwen1_5-moe-a2.7b-chat.Q4_K_M.gguf` | ~8.5 GB | `Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF` |
| DeepSeek-Coder-V2-Lite Q4_K_M | `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf` | ~10.4 GB | `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` |
| GPT-OSS-20B | varies | ~12 GB | `openai/gpt-oss-20b` |

## Appendix B: Key Sources

- [LM Studio MCP Docs](https://lmstudio.ai/docs/app/mcp)
- [Jan.ai MCP Docs](https://www.jan.ai/docs/desktop/mcp)
- [MCP Bundle Format (.mcpb)](https://github.com/modelcontextprotocol/mcpb)
- [Msty Desktop](https://msty.ai/)
- [AnythingLLM MCP](https://docs.anythingllm.com/mcp-compatibility/overview)
- [Gemini CLI](https://google-gemini.github.io/gemini-cli/)
- [mcp-client-for-ollama](https://github.com/jonigl/mcp-client-for-ollama)
- [LiteLLM MCP Gateway](https://docs.litellm.ai/docs/mcp)
- [BC FIPPA](https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/96165_00)
- [OIPC "Getting Ahead of the Curve"](https://www.oipc.bc.ca/reports/special-reports/)
- [OPC AI Principles](https://www.priv.gc.ca/en/privacy-topics/technology/artificial-intelligence/)
- [PyInstaller](https://pyinstaller.org/)
- [Nuitka](https://nuitka.net/)
- [chat-mcp (Electron)](https://github.com/AI-QL/chat-mcp)
