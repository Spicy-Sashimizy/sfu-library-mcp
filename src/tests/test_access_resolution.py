#!/usr/bin/env python3
"""
Access Resolution Hit-Rate Test
================================
Measures how well SFU's Solr database registry can resolve access for
academic articles discovered via OpenAlex.

Resolution cascade (per article):
  1. OpenAlex OA flag
  2. Unpaywall best_oa_location
  3. Solr domain match (SFU subscription DB url field)
  4. Solr provider match (fuzzy publisher -> provider)
  5. DOI fallback (no resolved access)

Usage:
    python3 src/tests/test_access_resolution.py            # full run
    python3 src/tests/test_access_resolution.py --dry-run   # fetch Solr only, skip article resolution
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Dependency check -- requests is the only external requirement
# ---------------------------------------------------------------------------
try:
    import requests
except ImportError:
    sys.exit(
        "ERROR: 'requests' is not installed.\n"
        "  pip install requests   (or use the project venv)"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOLR_URL = (
    "https://databases.lib.sfu.ca/solr/sfu_databases/select"
    "?q=*:*&rows=1000&wt=json"
)

OPENALEX_BASE = "https://api.openalex.org/works"
OPENALEX_MAILTO = "test@sfu.ca"
OPENALEX_PER_PAGE = 20
OPENALEX_DELAY = 0.12  # ~8 req/sec (under 10 limit, with margin)

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
UNPAYWALL_EMAIL = "test@sfu.ca"
UNPAYWALL_DELAY = 1.05  # slightly over 1 s to stay under 1 req/sec

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "access_resolution_results.json"

# Diverse academic queries covering different disciplines
SEARCH_QUERIES = [
    "machine learning",
    "climate change mitigation",
    "CRISPR gene editing",
    "quantum computing algorithms",
    "indigenous rights Canada",
    "antibiotic resistance",
    "deep learning natural language processing",
    "ocean acidification coral reefs",
    "supply chain management",
    "cognitive behavioral therapy",
    "renewable energy storage",
    "urban planning sustainability",
    "financial risk modeling",
    "protein folding prediction",
    "social media misinformation",
    "childhood education equity",
    "dark matter detection",
    "biodiversity conservation",
    "cybersecurity threat detection",
    "public health pandemic preparedness",
]

# Common corporate suffixes stripped for fuzzy provider matching
PROVIDER_STRIP_PATTERNS = re.compile(
    r"\b(inc\.?|ltd\.?|llc\.?|publishing|press|group|corporation|corp\.?"
    r"|co\.?|gmbh|s\.?a\.?|plc|limited|verlag)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_domain(url: str) -> str | None:
    """Return the bare domain (no www.) from a URL, or None."""
    if not url:
        return None
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        domain = (parsed.hostname or "").lower().strip(".")
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or None
    except Exception:
        return None


def normalize_provider(name: str) -> str:
    """Lowercase, strip corporate suffixes, collapse whitespace."""
    if not name:
        return ""
    name = name.lower().strip()
    name = PROVIDER_STRIP_PATTERNS.sub("", name)
    name = re.sub(r"[.,;:]+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def safe_get(url: str, params: dict | None = None,
             timeout: int = 30) -> requests.Response | None:
    """GET with retry (once) and error handling."""
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 5))
                print(f"  [rate-limited] waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == 0:
                time.sleep(2)
                continue
            print(f"  [error] {exc}")
            return None
    return None


def fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * n / total:.1f}%"


def print_table(headers: list[str], rows: list[list], col_widths: list[int] | None = None):
    """Simple ASCII table printer."""
    if col_widths is None:
        col_widths = [
            max(len(str(h)), *(len(str(r[i])) for r in rows) if rows else [len(str(h))])
            for i, h in enumerate(headers)
        ]
    # Ensure minimums
    col_widths = [max(w, len(str(h))) for w, h in zip(col_widths, headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    def fmt_row(cells):
        return "| " + " | ".join(
            str(c).ljust(w) for c, w in zip(cells, col_widths)
        ) + " |"
    print(sep)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)


# ===================================================================
# Step 1: Fetch and analyze Solr registry
# ===================================================================

def fetch_solr_registry() -> list[dict]:
    """Fetch all database records from SFU Solr."""
    print("\n[Step 1] Fetching SFU Solr database registry ...")
    t0 = time.time()
    resp = safe_get(SOLR_URL, timeout=60)
    if resp is None:
        sys.exit("FATAL: Could not fetch Solr registry.")
    data = resp.json()
    docs = data.get("response", {}).get("docs", [])
    elapsed = time.time() - t0
    print(f"  Fetched {len(docs)} records in {elapsed:.1f}s")
    return docs


def build_solr_indices(docs: list[dict]) -> tuple[dict, dict, list[dict]]:
    """
    Returns (domain_index, provider_index, enriched_docs).

    domain_index:   domain_str -> list[record]
    provider_index: normalized_provider -> list[record]
    """
    domain_index: dict[str, list[dict]] = defaultdict(list)
    provider_index: dict[str, list[dict]] = defaultdict(list)

    for doc in docs:
        # Extract domain from url field
        url = doc.get("url", "")
        if isinstance(url, list):
            url = url[0] if url else ""
        domain = extract_domain(url)
        if domain:
            doc["_domain"] = domain
            domain_index[domain].append(doc)

        # Provider index
        provider_raw = doc.get("provider", "")
        if isinstance(provider_raw, list):
            provider_raw = provider_raw[0] if provider_raw else ""
        if provider_raw:
            norm = normalize_provider(provider_raw)
            if norm:
                provider_index[norm].append(doc)

    return dict(domain_index), dict(provider_index), docs


def report_solr_analysis(docs: list[dict], domain_index: dict, provider_index: dict):
    """Print analysis of the Solr registry."""
    total = len(docs)
    free_count = sum(1 for d in docs if d.get("free") is True or str(d.get("free", "")).lower() == "true")
    sub_count = total - free_count
    proxy_count = sum(1 for d in docs if d.get("proxy") is True or str(d.get("proxy", "")).lower() == "true")
    direct_count = total - proxy_count

    print(f"\n{'='*60}")
    print("  SOLR REGISTRY ANALYSIS")
    print(f"{'='*60}")
    print(f"  Total records:        {total}")
    print(f"  Subscription (paid):  {sub_count} ({fmt_pct(sub_count, total)})")
    print(f"  Free:                 {free_count} ({fmt_pct(free_count, total)})")
    print(f"  Require EZProxy:      {proxy_count} ({fmt_pct(proxy_count, total)})")
    print(f"  Direct access:        {direct_count} ({fmt_pct(direct_count, total)})")
    print(f"  Unique domains:       {len(domain_index)}")
    print(f"  Unique providers:     {len(provider_index)}")

    # Top providers
    provider_counter = Counter()
    for doc in docs:
        p = doc.get("provider", "")
        if isinstance(p, list):
            p = p[0] if p else ""
        if p:
            provider_counter[p] += 1
    print(f"\n  Top 15 Providers:")
    print_table(
        ["Provider", "Records"],
        [[p, c] for p, c in provider_counter.most_common(15)],
        [45, 8],
    )

    # Top subjects
    subject_counter = Counter()
    for doc in docs:
        subjects = doc.get("subjects", [])
        if isinstance(subjects, str):
            subjects = [subjects]
        for s in subjects:
            subject_counter[s] += 1
    print(f"\n  Top 15 Subjects:")
    print_table(
        ["Subject", "Records"],
        [[s, c] for s, c in subject_counter.most_common(15)],
        [45, 8],
    )


# ===================================================================
# Step 2: Fetch sample articles from OpenAlex
# ===================================================================

def fetch_openalex_articles() -> list[dict]:
    """Run diverse queries and collect up to 400 unique articles."""
    print(f"\n[Step 2] Fetching articles from OpenAlex ({len(SEARCH_QUERIES)} queries) ...")
    seen_dois: set[str] = set()
    articles: list[dict] = []
    t0 = time.time()

    for i, query in enumerate(SEARCH_QUERIES, 1):
        print(f"  [{i:2d}/{len(SEARCH_QUERIES)}] '{query}' ", end="", flush=True)
        resp = safe_get(
            OPENALEX_BASE,
            params={
                "search": query,
                "per_page": OPENALEX_PER_PAGE,
                "mailto": OPENALEX_MAILTO,
            },
        )
        if resp is None:
            print("-> FAILED")
            time.sleep(OPENALEX_DELAY)
            continue

        results = resp.json().get("results", [])
        added = 0
        for work in results:
            doi = work.get("doi", "") or ""
            # Normalize DOI
            if doi.startswith("https://doi.org/"):
                doi = doi[len("https://doi.org/"):]
            if not doi:
                continue
            if doi in seen_dois:
                continue
            seen_dois.add(doi)

            # Extract useful fields
            # primary_location is the modern field; fall back to host_venue
            location = work.get("primary_location") or {}
            source = location.get("source") or {}
            host_venue = work.get("host_venue") or {}

            # Safely extract publisher, handling empty authorships/institutions lists
            _fallback_pub = ""
            if work.get("authorships"):
                _auths = work["authorships"]
                if _auths and _auths[0].get("institutions"):
                    _fallback_pub = _auths[0]["institutions"][0].get("display_name", "")
            publisher = (
                source.get("host_organization_name")
                or host_venue.get("publisher")
                or _fallback_pub
            )
            if not publisher:
                publisher = ""

            source_url = (
                location.get("landing_page_url")
                or location.get("pdf_url")
                or host_venue.get("url")
                or ""
            )

            oa_info = work.get("open_access") or {}
            is_oa = oa_info.get("is_oa", False)

            article = {
                "doi": doi,
                "title": (work.get("title") or "")[:120],
                "publisher": publisher,
                "source_name": source.get("display_name") or host_venue.get("display_name") or "",
                "source_url": source_url,
                "is_oa": bool(is_oa),
                "oa_url": oa_info.get("oa_url", ""),
                "type": work.get("type", ""),
                "publication_year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count", 0),
                "query": query,
            }
            articles.append(article)
            added += 1

        print(f"-> {added} new articles (total: {len(articles)})")
        time.sleep(OPENALEX_DELAY)

        if len(articles) >= 400:
            break

    elapsed = time.time() - t0
    print(f"  Collected {len(articles)} unique articles in {elapsed:.1f}s")
    return articles


# ===================================================================
# Step 3: Resolve access for each article
# ===================================================================

def resolve_access(
    articles: list[dict],
    domain_index: dict[str, list[dict]],
    provider_index: dict[str, list[dict]],
) -> list[dict]:
    """
    For each article, attempt resolution cascade.
    Returns articles enriched with resolution info.
    """
    print(f"\n[Step 3] Resolving access for {len(articles)} articles ...")
    t0 = time.time()

    # Pre-compute: set of all provider normalized names for fuzzy match
    provider_norms = set(provider_index.keys())

    unpaywall_calls = 0
    for i, art in enumerate(articles):
        art["resolution"] = None
        art["resolution_detail"] = {}

        # --- 3a: OA via OpenAlex ---
        if art.get("is_oa"):
            art["resolution"] = "oa"
            art["resolution_detail"] = {"source": "openalex", "oa_url": art.get("oa_url", "")}
            continue

        # --- 3b: Unpaywall ---
        doi = art["doi"]
        if doi:
            time.sleep(UNPAYWALL_DELAY)
            unpaywall_calls += 1
            if unpaywall_calls % 20 == 0:
                print(f"  ... processed {i+1}/{len(articles)} "
                      f"(unpaywall calls: {unpaywall_calls})")
            resp = safe_get(
                f"{UNPAYWALL_BASE}/{doi}",
                params={"email": UNPAYWALL_EMAIL},
                timeout=15,
            )
            if resp is not None:
                try:
                    uw = resp.json()
                    best = uw.get("best_oa_location")
                    if best and best.get("url"):
                        art["resolution"] = "unpaywall"
                        art["resolution_detail"] = {
                            "source": "unpaywall",
                            "url": best.get("url", ""),
                            "version": best.get("version", ""),
                            "host_type": best.get("host_type", ""),
                        }
                        continue
                except (json.JSONDecodeError, AttributeError):
                    pass

        # --- 3c: Solr domain match ---
        source_url = art.get("source_url", "")
        domain = extract_domain(source_url)
        if domain:
            # Try exact domain first, then parent domain
            matched_records = domain_index.get(domain)
            if not matched_records:
                # Try one level up (e.g., journals.sagepub.com -> sagepub.com)
                parts = domain.split(".")
                if len(parts) > 2:
                    parent = ".".join(parts[-2:])
                    matched_records = domain_index.get(parent)
            if matched_records:
                rec = matched_records[0]
                proxy_needed = (
                    rec.get("proxy") is True
                    or str(rec.get("proxy", "")).lower() == "true"
                )
                art["resolution"] = "solr_domain"
                art["resolution_detail"] = {
                    "source": "solr_domain",
                    "matched_domain": domain,
                    "solr_db_name": rec.get("name", ""),
                    "solr_db_id": rec.get("id", ""),
                    "proxy_needed": proxy_needed,
                }
                continue

        # --- 3d: Solr provider match (fuzzy) ---
        pub = art.get("publisher", "")
        if pub:
            pub_norm = normalize_provider(pub)
            matched_provider = None
            # Exact normalized match
            if pub_norm in provider_norms:
                matched_provider = pub_norm
            else:
                # Substring containment in either direction
                for pn in provider_norms:
                    if pub_norm and pn and (pub_norm in pn or pn in pub_norm):
                        matched_provider = pn
                        break
            if matched_provider:
                recs = provider_index[matched_provider]
                rec = recs[0]
                proxy_needed = (
                    rec.get("proxy") is True
                    or str(rec.get("proxy", "")).lower() == "true"
                )
                art["resolution"] = "solr_provider"
                art["resolution_detail"] = {
                    "source": "solr_provider",
                    "article_publisher": pub,
                    "matched_provider_norm": matched_provider,
                    "solr_db_name": rec.get("name", ""),
                    "proxy_needed": proxy_needed,
                    "provider_record_count": len(recs),
                }
                continue

        # --- 3e: DOI fallback ---
        art["resolution"] = "doi_fallback"
        art["resolution_detail"] = {"source": "doi_fallback", "doi": doi}

    elapsed = time.time() - t0
    print(f"  Resolution complete in {elapsed:.1f}s "
          f"(unpaywall API calls: {unpaywall_calls})")
    return articles


# ===================================================================
# Step 4: Report statistics
# ===================================================================

def report_results(
    articles: list[dict],
    domain_index: dict,
    provider_index: dict,
):
    """Print formatted resolution statistics."""
    total = len(articles)
    if total == 0:
        print("\n  No articles to report on.")
        return {}

    # --- Resolution breakdown ---
    res_counts = Counter(a["resolution"] for a in articles)
    labels = {
        "oa":             "OA (OpenAlex)",
        "unpaywall":      "Unpaywall resolved",
        "solr_domain":    "Solr domain match",
        "solr_provider":  "Solr provider match",
        "doi_fallback":   "DOI fallback only",
    }

    print(f"\n{'='*60}")
    print("  ACCESS RESOLUTION RESULTS")
    print(f"{'='*60}")
    print(f"  Total articles tested: {total}\n")

    rows = []
    for key in ["oa", "unpaywall", "solr_domain", "solr_provider", "doi_fallback"]:
        n = res_counts.get(key, 0)
        rows.append([labels[key], n, fmt_pct(n, total)])
    print_table(["Resolution Method", "Count", "Pct"], rows, [28, 8, 8])

    # Resolved (non-fallback) rate
    resolved = total - res_counts.get("doi_fallback", 0)
    print(f"\n  Overall resolved: {resolved}/{total} ({fmt_pct(resolved, total)})")
    solr_total = res_counts.get("solr_domain", 0) + res_counts.get("solr_provider", 0)
    print(f"  Solr-matched:     {solr_total}/{total} ({fmt_pct(solr_total, total)})")

    # --- EZProxy analysis for Solr-matched ---
    solr_articles = [a for a in articles if a["resolution"] in ("solr_domain", "solr_provider")]
    proxy_needed = sum(1 for a in solr_articles if a["resolution_detail"].get("proxy_needed"))
    direct = len(solr_articles) - proxy_needed
    if solr_articles:
        print(f"\n  Of {len(solr_articles)} Solr-matched articles:")
        print(f"    Require EZProxy: {proxy_needed} ({fmt_pct(proxy_needed, len(solr_articles))})")
        print(f"    Direct access:   {direct} ({fmt_pct(direct, len(solr_articles))})")

    # --- Publisher coverage ---
    pub_counter = Counter()
    pub_resolved = defaultdict(lambda: Counter())
    for a in articles:
        pub = a.get("publisher") or "(unknown)"
        pub_counter[pub] += 1
        pub_resolved[pub][a["resolution"]] += 1

    print(f"\n  Top 20 Publishers and Resolution Breakdown:")
    pub_rows = []
    for pub, count in pub_counter.most_common(20):
        rc = pub_resolved[pub]
        oa_n = rc.get("oa", 0)
        uw_n = rc.get("unpaywall", 0)
        sd_n = rc.get("solr_domain", 0)
        sp_n = rc.get("solr_provider", 0)
        fb_n = rc.get("doi_fallback", 0)
        has_solr = "Yes" if (sd_n + sp_n) > 0 else "No"
        display_pub = (pub[:38] + "..") if len(pub) > 40 else pub
        pub_rows.append([display_pub, count, oa_n, uw_n, sd_n + sp_n, fb_n, has_solr])
    print_table(
        ["Publisher", "N", "OA", "UW", "Solr", "Fallback", "InSolr?"],
        pub_rows,
        [40, 5, 4, 4, 5, 8, 7],
    )

    # --- Subject coverage (from Solr matches) ---
    matched_subjects = Counter()
    for a in solr_articles:
        detail = a.get("resolution_detail", {})
        db_name = detail.get("solr_db_name", "")
        # We don't have direct subject info on the article side,
        # but we report the Solr DB subjects
        matched_norm = detail.get("matched_provider_norm", "")
        matched_domain = detail.get("matched_domain", "")
        # Just tally which Solr DBs are being hit
        if db_name:
            matched_subjects[db_name] += 1

    if matched_subjects:
        print(f"\n  Most-Matched Solr Databases:")
        db_rows = [[name, count] for name, count in matched_subjects.most_common(15)]
        print_table(["Database Name", "Matches"], db_rows, [45, 8])

    # --- Build JSON results ---
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_articles": total,
        "resolution_counts": dict(res_counts),
        "resolution_percentages": {
            k: round(100 * v / total, 2) for k, v in res_counts.items()
        },
        "overall_resolved": resolved,
        "overall_resolved_pct": round(100 * resolved / total, 2),
        "solr_matched": solr_total,
        "solr_proxy_needed": proxy_needed,
        "solr_direct": direct,
        "top_publishers": [
            {"name": p, "count": c} for p, c in pub_counter.most_common(20)
        ],
        "articles": [
            {
                "doi": a["doi"],
                "title": a["title"],
                "publisher": a["publisher"],
                "is_oa": a["is_oa"],
                "resolution": a["resolution"],
                "resolution_detail": a["resolution_detail"],
            }
            for a in articles
        ],
    }
    return results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Access Resolution Hit-Rate Test against SFU Solr registry"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and analyze Solr registry only; skip article resolution.",
    )
    args = parser.parse_args()

    t_start = time.time()

    # Step 1: Solr
    docs = fetch_solr_registry()
    domain_index, provider_index, enriched_docs = build_solr_indices(docs)
    report_solr_analysis(docs, domain_index, provider_index)

    if args.dry_run:
        print(f"\n[dry-run] Skipping OpenAlex fetch and access resolution.")
        print(f"  Total time: {time.time() - t_start:.1f}s")
        return

    # Step 2: OpenAlex
    articles = fetch_openalex_articles()
    if not articles:
        print("\n  No articles collected. Exiting.")
        return

    # Step 3: Resolve
    articles = resolve_access(articles, domain_index, provider_index)

    # Step 4: Report
    results = report_results(articles, domain_index, provider_index)

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {RESULTS_FILE}")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
