#!/usr/bin/env python3
"""
Pipeline Comparison Test: Primo vs OpenAlex + Semantic Scholar

Compares the OLD Primo-based search pipeline against the NEW OpenAlex + Semantic
Scholar pipeline across 25 SFU-specific academic queries. Measures coverage,
DOI overlap, recency, open access rates, citation counts, and metadata
completeness.

Usage:
    python3 src/tests/test_pipeline_comparison.py
    python3 src/tests/test_pipeline_comparison.py --dry-run
    python3 src/tests/test_pipeline_comparison.py --queries 5   # run first N queries only
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# Attempt aiohttp; fall back to synchronous requests
# ---------------------------------------------------------------------------
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    try:
        import requests
    except ImportError:
        print("ERROR: Neither aiohttp nor requests is installed.", file=sys.stderr)
        sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_comparison")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRIMO_BASE = "https://sfu-primo.hosted.exlibrisgroup.com"
PRIMO_SEARCH = f"{PRIMO_BASE}/primo_library/libweb/webservices/rest/primo-explore/v1/pnxs"
OPENALEX_SEARCH = "https://api.openalex.org/works"
SEMSCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "pipeline_comparison_results.json"

BROWSER_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": f"{PRIMO_BASE}/primo-explore/search?vid=SFUL",
    "Origin": PRIMO_BASE,
}

# Semantic Scholar enforces ~1 req/s for unauthenticated callers
SEMSCHOLAR_DELAY = 1.1  # seconds between requests

# 25 SFU-specific academic queries
QUERIES: list[str] = [
    # CS / Engineering (3)
    "machine learning neural networks",
    "quantum computing algorithms",
    "software engineering testing",
    # Sciences (3)
    "CRISPR gene editing therapy",
    "climate change carbon sequestration",
    "marine biology Pacific Northwest",
    # Social Sciences (3)
    "Indigenous reconciliation Canada education",
    "housing affordability Vancouver",
    "mental health university students",
    # Humanities (2)
    "postcolonial literature analysis",
    "digital humanities text mining",
    # Health (2)
    "antibiotic resistance mechanisms",
    "public health COVID-19 vaccination",
    # Business (1)
    "sustainable supply chain management",
    # Interdisciplinary (2)
    "artificial intelligence ethics bias",
    "open access scholarly publishing",
    # Additional 9 SFU-relevant queries
    "wildfire smoke air quality British Columbia",
    "autonomous vehicle safety reinforcement learning",
    "Second Language acquisition immersion programs",
    "microplastics ocean pollution remediation",
    "renewable energy grid storage batteries",
    "urban planning transit oriented development Vancouver",
    "cybersecurity ransomware detection",
    "salmon conservation Fraser River ecology",
    "feminist theory intersectionality workplace",
]


# ---------------------------------------------------------------------------
# Data classes for per-query results
# ---------------------------------------------------------------------------
@dataclass
class SourceResult:
    """Normalized result from a single source for one query."""
    source: str
    query: str
    num_results: int = 0
    titles: list[str] = field(default_factory=list)
    dois: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    open_access_count: int = 0
    citation_counts: list[int] = field(default_factory=list)
    has_title: int = 0
    has_authors: int = 0
    has_year: int = 0
    has_doi: int = 0
    error: str | None = None
    raw_response: Any = field(default=None, repr=False)


@dataclass
class QueryComparison:
    """Comparison metrics for one query across all three sources."""
    query: str
    primo: SourceResult | None = None
    openalex: SourceResult | None = None
    semscholar: SourceResult | None = None
    doi_overlap_primo_openalex: int = 0
    doi_overlap_primo_semscholar: int = 0
    doi_overlap_openalex_semscholar: int = 0


# ---------------------------------------------------------------------------
# API callers (async with aiohttp)
# ---------------------------------------------------------------------------
async def fetch_primo_async(session: "aiohttp.ClientSession", query: str) -> SourceResult:
    """Call Primo REST API for a query."""
    result = SourceResult(source="primo", query=query)
    params = {
        "q": f"any,contains,{query}",
        "vid": "SFUL",
        "inst": "01SFUL",
        "tab": "default_tab",
        "scope": "default_scope",
        "lang": "en_US",
        "offset": 0,
        "limit": 10,
        "sort": "rank",
        "skipDelivery": "Y",
        "blendFacetsSeparately": "true",
        "pcAvailability": "false",
        "getMore": 0,
        "rtaLinks": "true",
        "newspapersActive": "true",
        "newspapersSearch": "false",
    }
    try:
        async with session.get(PRIMO_SEARCH, params=params, headers=BROWSER_HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}"
                logger.warning("Primo returned %d for query '%s'", resp.status, query[:40])
                return result
            data = await resp.json(content_type=None)
            result.raw_response = data
    except Exception as exc:
        result.error = str(exc)
        logger.error("Primo error for '%s': %s", query[:40], exc)
        return result

    docs = data.get("docs", [])
    result.num_results = len(docs)
    for doc in docs:
        pnx = doc.get("pnx", {})
        display = pnx.get("display", {})
        addata = pnx.get("addata", {})
        control = pnx.get("control", {})
        search_section = pnx.get("search", {})

        # Title
        title_list = display.get("title", [])
        title = title_list[0] if title_list else ""
        result.titles.append(title)
        if title:
            result.has_title += 1

        # Authors
        creators = display.get("creator", []) or display.get("contributor", [])
        if creators:
            result.has_authors += 1

        # Year
        date_list = addata.get("date", []) or display.get("creationdate", [])
        year = None
        if date_list:
            try:
                year = int(str(date_list[0])[:4])
                result.years.append(year)
                result.has_year += 1
            except (ValueError, IndexError):
                pass

        # DOI
        doi_list = addata.get("doi", [])
        if doi_list and doi_list[0]:
            result.dois.append(doi_list[0].lower().strip())
            result.has_doi += 1

        # Open access heuristic: check for openaccess / oa fields or fulltext delivery
        links = doc.get("delivery", {}).get("link", []) or []
        oa_flag = any("openaccess" in json.dumps(link).lower() for link in links)
        oa_field = pnx.get("display", {}).get("oa", [])
        if oa_flag or oa_field:
            result.open_access_count += 1

    return result


async def fetch_openalex_async(session: "aiohttp.ClientSession", query: str) -> SourceResult:
    """Call OpenAlex API for a query."""
    result = SourceResult(source="openalex", query=query)
    params = {
        "search": query,
        "per_page": 10,
        "mailto": "test@sfu.ca",
    }
    try:
        async with session.get(OPENALEX_SEARCH, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}"
                logger.warning("OpenAlex returned %d for query '%s'", resp.status, query[:40])
                return result
            data = await resp.json(content_type=None)
            result.raw_response = data
    except Exception as exc:
        result.error = str(exc)
        logger.error("OpenAlex error for '%s': %s", query[:40], exc)
        return result

    works = data.get("results", [])
    result.num_results = len(works)
    for work in works:
        # Title
        title = work.get("display_name", "") or work.get("title", "") or ""
        result.titles.append(title)
        if title:
            result.has_title += 1

        # Authors
        authorships = work.get("authorships", [])
        if authorships:
            result.has_authors += 1

        # Year
        year = work.get("publication_year")
        if year:
            try:
                result.years.append(int(year))
                result.has_year += 1
            except (ValueError, TypeError):
                pass

        # DOI
        doi = work.get("doi", "") or ""
        if doi:
            # OpenAlex returns full URL like https://doi.org/10.xxxx
            doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").lower().strip()
            if doi_clean:
                result.dois.append(doi_clean)
                result.has_doi += 1

        # Open access
        oa_info = work.get("open_access", {})
        if oa_info.get("is_oa"):
            result.open_access_count += 1

        # Citation count
        cited = work.get("cited_by_count")
        if cited is not None:
            result.citation_counts.append(int(cited))

    return result


async def fetch_semscholar_async(
    session: "aiohttp.ClientSession",
    query: str,
    semaphore: asyncio.Semaphore,
) -> SourceResult:
    """Call Semantic Scholar API (rate-limited) for a query."""
    result = SourceResult(source="semscholar", query=query)
    params = {
        "query": query,
        "limit": 10,
        "fields": "title,year,citationCount,isOpenAccess,externalIds,tldr,authors",
    }
    async with semaphore:
        try:
            async with session.get(
                SEMSCHOLAR_SEARCH,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 429:
                    result.error = "Rate limited (429)"
                    logger.warning("Semantic Scholar rate-limited for '%s'", query[:40])
                    return result
                if resp.status != 200:
                    result.error = f"HTTP {resp.status}"
                    logger.warning("Semantic Scholar returned %d for '%s'", resp.status, query[:40])
                    return result
                data = await resp.json(content_type=None)
                result.raw_response = data
        except Exception as exc:
            result.error = str(exc)
            logger.error("Semantic Scholar error for '%s': %s", query[:40], exc)
            return result
        finally:
            # Rate limiting: sleep after each request
            await asyncio.sleep(SEMSCHOLAR_DELAY)

    papers = data.get("data", [])
    result.num_results = len(papers)
    for paper in papers:
        # Title
        title = paper.get("title", "") or ""
        result.titles.append(title)
        if title:
            result.has_title += 1

        # Authors
        authors = paper.get("authors", [])
        if authors:
            result.has_authors += 1

        # Year
        year = paper.get("year")
        if year:
            try:
                result.years.append(int(year))
                result.has_year += 1
            except (ValueError, TypeError):
                pass

        # DOI from externalIds
        ext_ids = paper.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "") or ""
        if doi:
            result.dois.append(doi.lower().strip())
            result.has_doi += 1

        # Open access
        if paper.get("isOpenAccess"):
            result.open_access_count += 1

        # Citation count
        cite_count = paper.get("citationCount")
        if cite_count is not None:
            result.citation_counts.append(int(cite_count))

    return result


# ---------------------------------------------------------------------------
# Synchronous fallback callers (when aiohttp unavailable)
# ---------------------------------------------------------------------------
def fetch_primo_sync(query: str) -> SourceResult:
    """Synchronous Primo fetch using requests."""
    result = SourceResult(source="primo", query=query)
    params = {
        "q": f"any,contains,{query}",
        "vid": "SFUL",
        "inst": "01SFUL",
        "tab": "default_tab",
        "scope": "default_scope",
        "lang": "en_US",
        "offset": 0,
        "limit": 10,
        "sort": "rank",
        "skipDelivery": "Y",
        "blendFacetsSeparately": "true",
        "pcAvailability": "false",
        "getMore": 0,
        "rtaLinks": "true",
        "newspapersActive": "true",
        "newspapersSearch": "false",
    }
    try:
        resp = requests.get(PRIMO_SEARCH, params=params, headers=BROWSER_HEADERS, timeout=30)
        if resp.status_code != 200:
            result.error = f"HTTP {resp.status_code}"
            return result
        data = resp.json()
    except Exception as exc:
        result.error = str(exc)
        return result

    docs = data.get("docs", [])
    result.num_results = len(docs)
    for doc in docs:
        pnx = doc.get("pnx", {})
        display = pnx.get("display", {})
        addata = pnx.get("addata", {})

        title_list = display.get("title", [])
        title = title_list[0] if title_list else ""
        result.titles.append(title)
        if title:
            result.has_title += 1

        creators = display.get("creator", []) or display.get("contributor", [])
        if creators:
            result.has_authors += 1

        date_list = addata.get("date", []) or display.get("creationdate", [])
        if date_list:
            try:
                result.years.append(int(str(date_list[0])[:4]))
                result.has_year += 1
            except (ValueError, IndexError):
                pass

        doi_list = addata.get("doi", [])
        if doi_list and doi_list[0]:
            result.dois.append(doi_list[0].lower().strip())
            result.has_doi += 1

        links = doc.get("delivery", {}).get("link", []) or []
        oa_flag = any("openaccess" in json.dumps(link).lower() for link in links)
        oa_field = pnx.get("display", {}).get("oa", [])
        if oa_flag or oa_field:
            result.open_access_count += 1

    return result


def fetch_openalex_sync(query: str) -> SourceResult:
    """Synchronous OpenAlex fetch using requests."""
    result = SourceResult(source="openalex", query=query)
    params = {"search": query, "per_page": 10, "mailto": "test@sfu.ca"}
    try:
        resp = requests.get(OPENALEX_SEARCH, params=params, timeout=30)
        if resp.status_code != 200:
            result.error = f"HTTP {resp.status_code}"
            return result
        data = resp.json()
    except Exception as exc:
        result.error = str(exc)
        return result

    works = data.get("results", [])
    result.num_results = len(works)
    for work in works:
        title = work.get("display_name", "") or work.get("title", "") or ""
        result.titles.append(title)
        if title:
            result.has_title += 1

        if work.get("authorships"):
            result.has_authors += 1

        year = work.get("publication_year")
        if year:
            try:
                result.years.append(int(year))
                result.has_year += 1
            except (ValueError, TypeError):
                pass

        doi = work.get("doi", "") or ""
        if doi:
            doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").lower().strip()
            if doi_clean:
                result.dois.append(doi_clean)
                result.has_doi += 1

        oa_info = work.get("open_access", {})
        if oa_info.get("is_oa"):
            result.open_access_count += 1

        cited = work.get("cited_by_count")
        if cited is not None:
            result.citation_counts.append(int(cited))

    return result


def fetch_semscholar_sync(query: str) -> SourceResult:
    """Synchronous Semantic Scholar fetch using requests (with rate limiting)."""
    result = SourceResult(source="semscholar", query=query)
    params = {
        "query": query,
        "limit": 10,
        "fields": "title,year,citationCount,isOpenAccess,externalIds,tldr,authors",
    }
    try:
        resp = requests.get(SEMSCHOLAR_SEARCH, params=params, timeout=30)
        if resp.status_code == 429:
            result.error = "Rate limited (429)"
            return result
        if resp.status_code != 200:
            result.error = f"HTTP {resp.status_code}"
            return result
        data = resp.json()
    except Exception as exc:
        result.error = str(exc)
        return result

    papers = data.get("data", [])
    result.num_results = len(papers)
    for paper in papers:
        title = paper.get("title", "") or ""
        result.titles.append(title)
        if title:
            result.has_title += 1

        if paper.get("authors"):
            result.has_authors += 1

        year = paper.get("year")
        if year:
            try:
                result.years.append(int(year))
                result.has_year += 1
            except (ValueError, TypeError):
                pass

        ext_ids = paper.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "") or ""
        if doi:
            result.dois.append(doi.lower().strip())
            result.has_doi += 1

        if paper.get("isOpenAccess"):
            result.open_access_count += 1

        cite_count = paper.get("citationCount")
        if cite_count is not None:
            result.citation_counts.append(int(cite_count))

    return result


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------
def compute_doi_overlap(a: SourceResult, b: SourceResult) -> int:
    """Count DOIs that appear in both source results."""
    if not a or not b:
        return 0
    set_a = set(a.dois)
    set_b = set(b.dois)
    return len(set_a & set_b)


def safe_avg(values: list[int | float], precision: int = 2) -> float | None:
    """Return the average or None if empty."""
    if not values:
        return None
    return round(sum(values) / len(values), precision)


def metadata_completeness_pct(sr: SourceResult) -> float | None:
    """Percentage of results with title + authors + year + DOI."""
    if sr.num_results == 0:
        return None
    complete = 0
    for i in range(sr.num_results):
        has_t = i < len(sr.titles) and sr.titles[i]
        has_y = i < len(sr.years)
        has_d = i < len(sr.dois)
        # Authors tracked as aggregate count; approximate per-result
        has_a = sr.has_authors > i  # rough check
        if has_t and has_a and has_y and has_d:
            complete += 1
    return round(100.0 * complete / sr.num_results, 1)


def build_comparison(query: str, primo: SourceResult, openalex: SourceResult, semscholar: SourceResult) -> QueryComparison:
    comp = QueryComparison(
        query=query,
        primo=primo,
        openalex=openalex,
        semscholar=semscholar,
        doi_overlap_primo_openalex=compute_doi_overlap(primo, openalex),
        doi_overlap_primo_semscholar=compute_doi_overlap(primo, semscholar),
        doi_overlap_openalex_semscholar=compute_doi_overlap(openalex, semscholar),
    )
    return comp


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------
def build_summary(comparisons: list[QueryComparison]) -> dict:
    """Aggregate metrics across all queries."""
    n = len(comparisons)
    if n == 0:
        return {}

    sources = ["primo", "openalex", "semscholar"]
    summary: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_queries": n,
        "queries": [c.query for c in comparisons],
        "per_source": {},
        "doi_overlap": {
            "primo_openalex": {"total": 0, "avg": 0.0},
            "primo_semscholar": {"total": 0, "avg": 0.0},
            "openalex_semscholar": {"total": 0, "avg": 0.0},
        },
        "per_query": [],
    }

    for src in sources:
        all_results_counts: list[int] = []
        all_years: list[int] = []
        total_oa = 0
        total_results = 0
        all_citations: list[int] = []
        all_completeness: list[float] = []
        error_count = 0

        for comp in comparisons:
            sr: SourceResult | None = getattr(comp, src)
            if sr is None or sr.error:
                error_count += 1
                continue
            all_results_counts.append(sr.num_results)
            all_years.extend(sr.years)
            total_oa += sr.open_access_count
            total_results += sr.num_results
            all_citations.extend(sr.citation_counts)
            mc = metadata_completeness_pct(sr)
            if mc is not None:
                all_completeness.append(mc)

        summary["per_source"][src] = {
            "avg_results": safe_avg(all_results_counts),
            "total_results": sum(all_results_counts),
            "avg_year": safe_avg(all_years),
            "median_year": sorted(all_years)[len(all_years) // 2] if all_years else None,
            "oa_rate_pct": round(100.0 * total_oa / total_results, 1) if total_results else None,
            "avg_citation_count": safe_avg(all_citations),
            "median_citation_count": sorted(all_citations)[len(all_citations) // 2] if all_citations else None,
            "avg_metadata_completeness_pct": safe_avg(all_completeness),
            "errors": error_count,
        }

    # DOI overlap totals
    for comp in comparisons:
        summary["doi_overlap"]["primo_openalex"]["total"] += comp.doi_overlap_primo_openalex
        summary["doi_overlap"]["primo_semscholar"]["total"] += comp.doi_overlap_primo_semscholar
        summary["doi_overlap"]["openalex_semscholar"]["total"] += comp.doi_overlap_openalex_semscholar

    for key in summary["doi_overlap"]:
        summary["doi_overlap"][key]["avg"] = round(summary["doi_overlap"][key]["total"] / n, 2)

    # Per-query detail
    for comp in comparisons:
        entry: dict[str, Any] = {"query": comp.query}
        for src in sources:
            sr = getattr(comp, src)
            if sr is None:
                entry[src] = {"error": "not fetched"}
                continue
            entry[src] = {
                "num_results": sr.num_results,
                "num_dois": len(sr.dois),
                "avg_year": safe_avg(sr.years),
                "oa_rate_pct": round(100.0 * sr.open_access_count / sr.num_results, 1) if sr.num_results else None,
                "avg_citations": safe_avg(sr.citation_counts),
                "metadata_completeness_pct": metadata_completeness_pct(sr),
                "error": sr.error,
            }
        entry["doi_overlap"] = {
            "primo_openalex": comp.doi_overlap_primo_openalex,
            "primo_semscholar": comp.doi_overlap_primo_semscholar,
            "openalex_semscholar": comp.doi_overlap_openalex_semscholar,
        }
        summary["per_query"].append(entry)

    return summary


def print_table(summary: dict) -> None:
    """Print a human-readable comparison table to stdout."""
    sep = "=" * 110
    print()
    print(sep)
    print("  PIPELINE COMPARISON: Primo  vs  OpenAlex + Semantic Scholar")
    print(f"  {summary.get('num_queries', 0)} queries  |  {summary.get('timestamp', '')}")
    print(sep)

    # Aggregate table
    header = f"{'Metric':<35} {'Primo':>15} {'OpenAlex':>15} {'Sem.Scholar':>15}"
    print()
    print(header)
    print("-" * len(header))

    ps = summary.get("per_source", {})
    metrics = [
        ("Avg results / query", "avg_results"),
        ("Total results", "total_results"),
        ("Avg publication year", "avg_year"),
        ("Median publication year", "median_year"),
        ("Open Access rate (%)", "oa_rate_pct"),
        ("Avg citation count", "avg_citation_count"),
        ("Median citation count", "median_citation_count"),
        ("Metadata completeness (%)", "avg_metadata_completeness_pct"),
        ("API errors", "errors"),
    ]
    for label, key in metrics:
        vals = []
        for src in ("primo", "openalex", "semscholar"):
            v = ps.get(src, {}).get(key)
            vals.append(str(v) if v is not None else "N/A")
        print(f"  {label:<33} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    # DOI overlap
    do = summary.get("doi_overlap", {})
    print()
    print("  DOI Overlap (avg per query):")
    print(f"    Primo <-> OpenAlex:       {do.get('primo_openalex', {}).get('avg', 'N/A')}")
    print(f"    Primo <-> Sem.Scholar:    {do.get('primo_semscholar', {}).get('avg', 'N/A')}")
    print(f"    OpenAlex <-> Sem.Scholar: {do.get('openalex_semscholar', {}).get('avg', 'N/A')}")

    # Per-query mini table
    print()
    print(f"  {'Query':<45} {'P#':>4} {'OA#':>4} {'SS#':>4} {'DOI-ovlp':>9}")
    print("  " + "-" * 70)
    for pq in summary.get("per_query", []):
        q = pq["query"][:43]
        p_n = pq.get("primo", {}).get("num_results", "E")
        o_n = pq.get("openalex", {}).get("num_results", "E")
        s_n = pq.get("semscholar", {}).get("num_results", "E")
        ovlp = pq.get("doi_overlap", {}).get("primo_openalex", 0)
        print(f"  {q:<45} {str(p_n):>4} {str(o_n):>4} {str(s_n):>4} {str(ovlp):>9}")

    print()
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main runners
# ---------------------------------------------------------------------------
async def run_async(queries: list[str]) -> list[QueryComparison]:
    """Run all queries concurrently (Primo + OpenAlex in parallel, Sem.Scholar rate-limited)."""
    comparisons: list[QueryComparison] = []
    # Semaphore limits Semantic Scholar to 1 concurrent request
    ss_semaphore = asyncio.Semaphore(1)

    async with aiohttp.ClientSession() as session:
        for i, query in enumerate(queries):
            logger.info("[%d/%d] Querying: %s", i + 1, len(queries), query)

            # Fire Primo and OpenAlex concurrently; Semantic Scholar sequentially
            primo_task = asyncio.create_task(fetch_primo_async(session, query))
            oa_task = asyncio.create_task(fetch_openalex_async(session, query))
            ss_task = asyncio.create_task(fetch_semscholar_async(session, query, ss_semaphore))

            primo_result, oa_result, ss_result = await asyncio.gather(
                primo_task, oa_task, ss_task, return_exceptions=False
            )

            comp = build_comparison(query, primo_result, oa_result, ss_result)
            comparisons.append(comp)

            logger.info(
                "  -> Primo=%d  OpenAlex=%d  SemScholar=%d  DOI-overlap(P/OA)=%d",
                primo_result.num_results,
                oa_result.num_results,
                ss_result.num_results,
                comp.doi_overlap_primo_openalex,
            )

    return comparisons


def run_sync(queries: list[str]) -> list[QueryComparison]:
    """Run all queries synchronously (fallback when aiohttp is unavailable)."""
    comparisons: list[QueryComparison] = []
    for i, query in enumerate(queries):
        logger.info("[%d/%d] Querying: %s", i + 1, len(queries), query)

        primo_result = fetch_primo_sync(query)
        oa_result = fetch_openalex_sync(query)

        # Rate-limit Semantic Scholar
        time.sleep(SEMSCHOLAR_DELAY)
        ss_result = fetch_semscholar_sync(query)

        comp = build_comparison(query, primo_result, oa_result, ss_result)
        comparisons.append(comp)

        logger.info(
            "  -> Primo=%d  OpenAlex=%d  SemScholar=%d  DOI-overlap(P/OA)=%d",
            primo_result.num_results,
            oa_result.num_results,
            ss_result.num_results,
            comp.doi_overlap_primo_openalex,
        )

    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Primo vs OpenAlex+Semantic Scholar pipelines")
    parser.add_argument("--dry-run", action="store_true", help="Print queries without making API calls")
    parser.add_argument("--queries", type=int, default=None, help="Run only the first N queries")
    args = parser.parse_args()

    queries = QUERIES
    if args.queries:
        queries = queries[: args.queries]

    if args.dry_run:
        print(f"\n[DRY RUN] Would execute {len(queries)} queries:\n")
        for i, q in enumerate(queries, 1):
            print(f"  {i:>2}. {q}")
        print(f"\nAPIs: Primo, OpenAlex, Semantic Scholar")
        print(f"Results would be saved to: {RESULTS_FILE}")
        print(f"Using {'aiohttp (async)' if HAS_AIOHTTP else 'requests (sync fallback)'}")
        return

    logger.info("Starting pipeline comparison with %d queries", len(queries))
    logger.info("HTTP library: %s", "aiohttp (async)" if HAS_AIOHTTP else "requests (sync)")

    start = time.monotonic()

    if HAS_AIOHTTP:
        comparisons = asyncio.run(run_async(queries))
    else:
        comparisons = run_sync(queries)

    elapsed = time.monotonic() - start
    logger.info("All queries completed in %.1f seconds", elapsed)

    summary = build_summary(comparisons)
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["http_library"] = "aiohttp" if HAS_AIOHTTP else "requests"

    # Save JSON results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Results saved to %s", RESULTS_FILE)

    # Print formatted table
    print_table(summary)


if __name__ == "__main__":
    main()
