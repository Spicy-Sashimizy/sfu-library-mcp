# docs/ — thin-client testbed documentation

Active documents live directly in this folder; everything else is filed in a
subdirectory. Reorganized 2026-06-11 alongside the OpenSearch → thin-client
swap (see `THIN_CLIENT_SWAP.md` for what this container now runs).

## Active (root)

| Doc | What it is |
|---|---|
| `THIN_CLIENT_SWAP.md` | **Start here** — this container's architecture: engines (incl. era sub-sections, abstracts v3, meta v2, dense warm cache), serving swap, hot/cold structure, migration, removals |
| `STORAGE_BUDGET_150M.md` | Whole-DB storage budget: measured levers + implementation status, all-levers totals (300.9→188.8 GB), mainline hypothetical (274→~112 GB), degradation accounting, BMP quirks |
| `LEXICAL_STORAGE_RESEARCH.md` | Storage/compression research + measured results (§8 sidecar variants incl. SLM decode, §7 multilingual + front coding, §9 all-levers matrix) |
| `DENSE_WARMCACHE_RESEARCH.md` | Query-driven dense warm cache: literature lineage (database cracking, CrackIVF) + the v1 design implemented in `lib/thinclient/dense_cache.py` |
| `THIN_CLIENT_STACK_RESEARCH.md` | Engine research + measured POC that selected tantivy+BMP+usearch; hot/cold persona metrics; abstract policy |
| `SEARCH_ENGINE_ALTERNATIVES.md` | Per-engine verdicts (why leave OpenSearch on laptops, why BMP over Seismic) |
| `COMPRESSION_EVAL_RESULTS.md` | The original measured lossless levers (Lucene 274→142.5 GB) + dense quantization (binary+rescore 32×) |
| `LOCALIZED_DEPLOYMENT_PLAN.md` | Deployment tiers (15M laptop / warm-cache / 150M server); partially superseded — see its 2026-06-11 status note |
| `BENCHMARK_METHODOLOGY.md` | How the LLM-judged NDCG@10 eval harness works (harness is kept and active) |

## Subdirectories

- `archive/` — historical but kept: MASTER_TODO (pre-swap phase tracker),
  session notes, dated benchmark results, completed/superseded plan docs
  (incl. `THIN_CLIENT_PLAN.md` and `THIN_CLIENT_NEW_ARCHITECTURE.md` — the
  pre-local-index client-app plans built around the Primo API and live-API
  retrieval; superseded by the local thin-client stack, kept for the client
  UI / packaging / compliance research).
- `deprecated/` — obsolete for THIS container (OpenSearch-era: upgrade/SPLADE
  integration/optimization roadmap, cloud offload, old setup guides). The
  original sfu-library-mcp container still uses several of these.
- `infrastructure/` — environment/access docs (remote MCP access, GUI
  reference).
