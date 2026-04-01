#!/usr/bin/env python3
"""
test_query_parsing.py — Compare LLM-based query expansion vs raw keyword search.

Uses OpenAlex API as the search backend and Claude (Haiku) for query expansion.
Self-contained: no project imports required.

Usage:
    python3 src/tests/test_query_parsing.py              # full run
    python3 src/tests/test_query_parsing.py --dry-run    # show queries, skip API calls
"""

import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENALEX_BASE = "https://api.openalex.org/works"
OPENALEX_MAILTO = "test@sfu.ca"
OPENALEX_PER_PAGE = 20

# Rate-limit tokens (calls per second)
CLAUDE_RATE_LIMIT = 5  # max 5 calls/sec
OPENALEX_RATE_LIMIT = 10  # max 10 calls/sec

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "query_parsing_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("query_parsing")

# ---------------------------------------------------------------------------
# 25 test queries
# ---------------------------------------------------------------------------

TEST_QUERIES: list[str] = [
    # Provided in spec
    "how does CRISPR work",
    "Vancouver housing crisis causes",
    "machine learning in healthcare",
    "Indigenous land rights Canada",
    "climate change effects on BC forests",
    "why are antibiotics becoming less effective",
    "history of residential schools",
    "quantum computing explained",
    "mental health support for grad students",
    "sustainable urban planning",
    # 15 additional diverse academic queries relevant to SFU students
    "ocean acidification impacts on Pacific salmon",
    "feminist approaches to artificial intelligence ethics",
    "wildfire smoke exposure respiratory health",
    "decolonizing university curricula",
    "blockchain applications in supply chain management",
    "second language acquisition in multilingual classrooms",
    "urban heat island effect mitigation strategies",
    "misinformation spread on social media platforms",
    "microplastics in freshwater ecosystems",
    "renewable energy storage battery technology",
    "cognitive behavioral therapy effectiveness for anxiety",
    "autonomous vehicles ethical decision making",
    "truth and reconciliation commission outcomes",
    "affordable housing policy comparative analysis",
    "BERT and transformer models for natural language processing",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WorkResult:
    title: str = ""
    doi: str = ""
    publication_year: Optional[int] = None
    cited_by_count: int = 0
    is_oa: bool = False


@dataclass
class SearchOutcome:
    query_sent: str = ""
    total_count: int = 0
    top_results: list[dict] = field(default_factory=list)
    elapsed_sec: float = 0.0
    error: Optional[str] = None


@dataclass
class QueryExpansion:
    terms: list[str] = field(default_factory=list)
    boolean_query: str = ""
    synonyms: list[str] = field(default_factory=list)
    academic_phrasing: str = ""
    raw_response: str = ""
    elapsed_sec: float = 0.0
    error: Optional[str] = None


@dataclass
class ComparisonMetrics:
    raw_result_count: int = 0
    expanded_result_count: int = 0
    raw_avg_citations: float = 0.0
    expanded_avg_citations: float = 0.0
    raw_avg_year: float = 0.0
    expanded_avg_year: float = 0.0
    doi_overlap: int = 0
    raw_title_keyword_match_pct: float = 0.0
    expanded_title_keyword_match_pct: float = 0.0
    raw_oa_rate: float = 0.0
    expanded_oa_rate: float = 0.0


@dataclass
class QueryResult:
    original_query: str = ""
    expansion: Optional[QueryExpansion] = None
    raw_search: Optional[SearchOutcome] = None
    expanded_search: Optional[SearchOutcome] = None
    metrics: Optional[ComparisonMetrics] = None


# ---------------------------------------------------------------------------
# Rate limiter (token-bucket style)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple async rate limiter."""

    def __init__(self, calls_per_second: float):
        self._interval = 1.0 / calls_per_second
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._last_call + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


claude_limiter = RateLimiter(CLAUDE_RATE_LIMIT)
openalex_limiter = RateLimiter(OPENALEX_RATE_LIMIT)

# ---------------------------------------------------------------------------
# SSL context (some CI environments lack certs)
# ---------------------------------------------------------------------------

def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    return ctx


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _http_request(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[dict] = None,
    body: Optional[bytes] = None,
    timeout: int = 30,
) -> tuple[int, str]:
    """Synchronous HTTP request via urllib. Returns (status_code, body_text)."""
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


async def _async_http(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[dict] = None,
    body: Optional[bytes] = None,
    timeout: int = 30,
) -> tuple[int, str]:
    """Run blocking HTTP in executor so we don't block the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: _http_request(url, method=method, headers=headers, body=body, timeout=timeout)
    )


# ---------------------------------------------------------------------------
# OpenAlex search
# ---------------------------------------------------------------------------

async def search_openalex(query: str) -> SearchOutcome:
    """Search OpenAlex and return structured outcome."""
    await openalex_limiter.acquire()

    params = urllib.parse.urlencode({
        "search": query,
        "per_page": OPENALEX_PER_PAGE,
        "mailto": OPENALEX_MAILTO,
    })
    url = f"{OPENALEX_BASE}?{params}"

    t0 = time.monotonic()
    status, body = await _async_http(url, timeout=30)
    elapsed = time.monotonic() - t0

    outcome = SearchOutcome(query_sent=query, elapsed_sec=round(elapsed, 3))

    if status != 200:
        outcome.error = f"HTTP {status}: {body[:200]}"
        log.warning("OpenAlex error for %r: %s", query[:40], outcome.error[:120])
        return outcome

    try:
        data = json.loads(body)
        meta = data.get("meta", {})
        outcome.total_count = meta.get("count", 0)
        results = data.get("results", [])
        for w in results[:10]:
            outcome.top_results.append({
                "title": w.get("display_name", "") or "",
                "doi": w.get("doi", "") or "",
                "publication_year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count", 0) or 0,
                "is_oa": w.get("open_access", {}).get("is_oa", False),
            })
    except (json.JSONDecodeError, KeyError) as e:
        outcome.error = f"Parse error: {e}"
        log.warning("OpenAlex parse error for %r: %s", query[:40], e)

    return outcome


# ---------------------------------------------------------------------------
# Claude query expansion
# ---------------------------------------------------------------------------

EXPANSION_PROMPT = """\
You are an academic search query expansion assistant. Given a natural language query from a university student, produce a JSON object with exactly these fields:

- "terms": a list of 3-8 key search terms extracted or inferred from the query
- "boolean_query": an expanded boolean search query suitable for an academic database (use AND/OR, parentheses, and quoted phrases). This should broaden recall while keeping precision. Maximum 200 characters.
- "synonyms": a list of 3-6 synonyms or closely related terms that a researcher might use
- "academic_phrasing": a single sentence rephrasing the query as a researcher would state it

Return ONLY valid JSON, no markdown fences, no extra text."""


async def expand_query_claude(query: str, api_key: str) -> QueryExpansion:
    """Call Claude Haiku to expand a search query."""
    await claude_limiter.acquire()

    expansion = QueryExpansion()
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": f"Query: {query}\n\n{EXPANSION_PROMPT}"}
        ],
    }).encode("utf-8")

    t0 = time.monotonic()
    status, body = await _async_http(
        ANTHROPIC_API_URL, method="POST", headers=headers, body=payload, timeout=30
    )
    expansion.elapsed_sec = round(time.monotonic() - t0, 3)

    if status != 200:
        expansion.error = f"HTTP {status}: {body[:300]}"
        log.warning("Claude API error for %r: %s", query[:40], expansion.error[:120])
        return expansion

    try:
        resp_data = json.loads(body)
        text = resp_data.get("content", [{}])[0].get("text", "")
        expansion.raw_response = text

        # Strip markdown fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        parsed = json.loads(cleaned)
        expansion.terms = parsed.get("terms", [])
        expansion.boolean_query = parsed.get("boolean_query", "")
        expansion.synonyms = parsed.get("synonyms", [])
        expansion.academic_phrasing = parsed.get("academic_phrasing", "")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        expansion.error = f"Parse error: {e}"
        log.warning("Claude parse error for %r: %s", query[:40], e)

    return expansion


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _title_keyword_match_pct(results: list[dict], query: str) -> float:
    """Percentage of results whose title contains at least one query term."""
    if not results:
        return 0.0
    terms = [t.lower() for t in query.split() if len(t) > 2]
    if not terms:
        return 0.0
    matches = 0
    for r in results:
        title_lower = r.get("title", "").lower()
        if any(t in title_lower for t in terms):
            matches += 1
    return round(100.0 * matches / len(results), 1)


def _oa_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return round(100.0 * sum(1 for r in results if r.get("is_oa")) / len(results), 1)


def compute_metrics(
    original_query: str,
    raw: SearchOutcome,
    expanded: Optional[SearchOutcome],
) -> ComparisonMetrics:
    m = ComparisonMetrics()
    m.raw_result_count = raw.total_count

    raw_top = raw.top_results[:10]
    m.raw_avg_citations = _avg([r["cited_by_count"] for r in raw_top])
    m.raw_avg_year = _avg([r["publication_year"] for r in raw_top if r.get("publication_year")])
    m.raw_title_keyword_match_pct = _title_keyword_match_pct(raw_top, original_query)
    m.raw_oa_rate = _oa_rate(raw_top)

    if expanded and not expanded.error:
        exp_top = expanded.top_results[:10]
        m.expanded_result_count = expanded.total_count
        m.expanded_avg_citations = _avg([r["cited_by_count"] for r in exp_top])
        m.expanded_avg_year = _avg([r["publication_year"] for r in exp_top if r.get("publication_year")])
        m.expanded_title_keyword_match_pct = _title_keyword_match_pct(exp_top, original_query)
        m.expanded_oa_rate = _oa_rate(exp_top)

        # DOI overlap
        raw_dois = {r["doi"] for r in raw_top if r.get("doi")}
        exp_dois = {r["doi"] for r in exp_top if r.get("doi")}
        m.doi_overlap = len(raw_dois & exp_dois)

    return m


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_separator(char: str = "-", width: int = 90):
    print(char * width)


def print_query_result(idx: int, qr: QueryResult):
    print_separator("=")
    print(f"  Query {idx + 1}: {qr.original_query}")
    print_separator("=")

    if qr.expansion and not qr.expansion.error:
        print(f"  Expanded boolean query : {qr.expansion.boolean_query[:100]}")
        print(f"  Terms                  : {', '.join(qr.expansion.terms[:8])}")
        print(f"  Expansion time         : {qr.expansion.elapsed_sec}s")
    elif qr.expansion and qr.expansion.error:
        print(f"  Expansion ERROR        : {qr.expansion.error[:100]}")

    print()
    header = f"  {'Metric':<30} {'Raw':>12} {'Expanded':>12} {'Delta':>10}"
    print(header)
    print_separator("-")

    if qr.metrics:
        m = qr.metrics
        rows = [
            ("Result count", m.raw_result_count, m.expanded_result_count),
            ("Avg citations (top 10)", m.raw_avg_citations, m.expanded_avg_citations),
            ("Avg pub year (top 10)", m.raw_avg_year, m.expanded_avg_year),
            ("Title keyword match %", m.raw_title_keyword_match_pct, m.expanded_title_keyword_match_pct),
            ("Open access rate %", m.raw_oa_rate, m.expanded_oa_rate),
        ]
        for label, raw_v, exp_v in rows:
            delta = ""
            if isinstance(raw_v, (int, float)) and isinstance(exp_v, (int, float)) and exp_v != 0:
                d = exp_v - raw_v
                sign = "+" if d >= 0 else ""
                if isinstance(d, float):
                    delta = f"{sign}{d:.1f}"
                else:
                    delta = f"{sign}{d}"
            if isinstance(raw_v, float):
                print(f"  {label:<30} {raw_v:>12.1f} {exp_v:>12.1f} {delta:>10}")
            else:
                print(f"  {label:<30} {raw_v:>12} {exp_v:>12} {delta:>10}")

        print(f"  {'DOI overlap (of top 10)':<30} {m.doi_overlap:>12}")

    # Timing
    raw_t = qr.raw_search.elapsed_sec if qr.raw_search else 0
    exp_t = qr.expanded_search.elapsed_sec if qr.expanded_search else 0
    print(f"\n  Timing: raw={raw_t}s  expanded={exp_t}s")
    print()


def print_summary(results: list[QueryResult]):
    print_separator("=")
    print("  SUMMARY STATISTICS (averages across all queries)")
    print_separator("=")

    valid = [r for r in results if r.metrics]
    if not valid:
        print("  No valid results to summarize.")
        return

    n = len(valid)
    has_expanded = any(r.expanded_search and not r.expanded_search.error for r in valid)

    def avg_field(field_name: str) -> float:
        vals = [getattr(r.metrics, field_name) for r in valid if r.metrics]
        return _avg(vals)

    print(f"  Queries tested: {n}")
    print()
    header = f"  {'Metric':<35} {'Raw':>12} {'Expanded':>12}"
    print(header)
    print_separator("-")

    rows = [
        ("Avg result count", "raw_result_count", "expanded_result_count"),
        ("Avg citations (top 10)", "raw_avg_citations", "expanded_avg_citations"),
        ("Avg pub year (top 10)", "raw_avg_year", "expanded_avg_year"),
        ("Avg title keyword match %", "raw_title_keyword_match_pct", "expanded_title_keyword_match_pct"),
        ("Avg open access rate %", "raw_oa_rate", "expanded_oa_rate"),
    ]

    for label, raw_f, exp_f in rows:
        rv = avg_field(raw_f)
        ev = avg_field(exp_f) if has_expanded else 0.0
        print(f"  {label:<35} {rv:>12.1f} {ev:>12.1f}")

    avg_overlap = _avg([r.metrics.doi_overlap for r in valid if r.metrics])
    print(f"  {'Avg DOI overlap (of top 10)':<35} {avg_overlap:>12.1f}")

    # Timing
    raw_times = [r.raw_search.elapsed_sec for r in valid if r.raw_search]
    exp_times = [r.expanded_search.elapsed_sec for r in valid if r.expanded_search and not r.expanded_search.error]
    expand_times = [r.expansion.elapsed_sec for r in valid if r.expansion and not r.expansion.error]

    print()
    print(f"  Avg raw search time       : {_avg(raw_times)}s")
    if exp_times:
        print(f"  Avg expanded search time  : {_avg(exp_times)}s")
    if expand_times:
        print(f"  Avg expansion (LLM) time  : {_avg(expand_times)}s")
    print()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def process_query(
    query: str,
    api_key: Optional[str],
    dry_run: bool,
) -> QueryResult:
    """Process a single query: expand (if key available), search both paths, compute metrics."""
    qr = QueryResult(original_query=query)

    if dry_run:
        log.info("[DRY-RUN] Would process: %s", query)
        return qr

    # Raw keyword search
    log.info("Raw search: %s", query[:50])
    qr.raw_search = await search_openalex(query)

    # Claude expansion + expanded search
    if api_key:
        log.info("Expanding: %s", query[:50])
        qr.expansion = await expand_query_claude(query, api_key)

        if qr.expansion.boolean_query and not qr.expansion.error:
            log.info("Expanded search: %s", qr.expansion.boolean_query[:60])
            qr.expanded_search = await search_openalex(qr.expansion.boolean_query)
        elif qr.expansion.error:
            log.warning("Skipping expanded search due to expansion error")
            qr.expanded_search = SearchOutcome(error="Skipped — expansion failed")
        else:
            # boolean_query was empty; fall back to academic_phrasing or terms
            fallback = qr.expansion.academic_phrasing or " ".join(qr.expansion.terms)
            if fallback:
                log.info("Expanded search (fallback): %s", fallback[:60])
                qr.expanded_search = await search_openalex(fallback)
            else:
                qr.expanded_search = SearchOutcome(error="No expanded query produced")

    # Compute metrics
    if qr.raw_search:
        qr.metrics = compute_metrics(query, qr.raw_search, qr.expanded_search)

    return qr


async def run_all(queries: list[str], api_key: Optional[str], dry_run: bool) -> list[QueryResult]:
    """Run all queries with controlled concurrency."""
    # Process queries with limited concurrency to respect rate limits.
    # We use a semaphore to limit parallel in-flight tasks.
    sem = asyncio.Semaphore(3)

    async def bounded(q: str) -> QueryResult:
        async with sem:
            return await process_query(q, api_key, dry_run)

    tasks = [bounded(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to empty results
    clean: list[QueryResult] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            log.error("Query %d failed: %s", i + 1, r)
            qr = QueryResult(original_query=queries[i])
            clean.append(qr)
        else:
            clean.append(r)

    return clean


def save_results(results: list[QueryResult]):
    """Save full results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _serialize(obj: Any) -> Any:
        if hasattr(obj, "__dict__"):
            return {k: _serialize(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, list):
            return [_serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        return obj

    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": ANTHROPIC_MODEL,
        "query_count": len(results),
        "results": [_serialize(r) for r in results],
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    log.info("Results saved to %s", RESULTS_FILE)


def main():
    parser = argparse.ArgumentParser(description="Compare LLM query expansion vs raw keyword search")
    parser.add_argument("--dry-run", action="store_true", help="Show queries without making API calls")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning(
            "ANTHROPIC_API_KEY not set. Claude expansion tests will be SKIPPED. "
            "Only raw keyword searches will run."
        )

    print()
    print_separator("*")
    print("  LLM Query Expansion vs Raw Keyword Search — Comparison Test")
    print(f"  Queries: {len(TEST_QUERIES)}   Model: {ANTHROPIC_MODEL}")
    print(f"  Claude expansion: {'ENABLED' if api_key else 'DISABLED (no API key)'}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print_separator("*")
    print()

    if args.dry_run:
        for i, q in enumerate(TEST_QUERIES, 1):
            print(f"  [{i:>2}] {q}")
        print(f"\n  Total: {len(TEST_QUERIES)} queries (dry-run, no API calls made)")
        return

    t_start = time.monotonic()
    results = asyncio.run(run_all(TEST_QUERIES, api_key, args.dry_run))
    total_time = time.monotonic() - t_start

    # Display per-query results
    for i, qr in enumerate(results):
        print_query_result(i, qr)

    # Summary
    print_summary(results)
    print(f"  Total wall-clock time: {total_time:.1f}s")
    print()

    # Save JSON
    save_results(results)


if __name__ == "__main__":
    main()
