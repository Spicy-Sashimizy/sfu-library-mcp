#!/usr/bin/env python3
"""Build the LambdaMART training dataset (Phase N).

Joins the LLM-judged relevance labels (data/eval_results/llm_judge_cache.json,
TREC 0-3 grades) to OpenAlex document metadata, computes the SHARED feature vector
(lib.lambdamart_features.doc_features — the same one used at inference), and emits a
JSONL dataset that scripts/train_lambdamart.py consumes.

Why this exists
---------------
The original Phase O.3 plan trained on a query log that was never collected and on a
weak citation-count label. We now have 4,034 real graded (query, doc) judgments across
120 eval queries — a far better signal. This builder turns those judgments into a
features+labels+qid table.

Metadata sources (in priority order, per work):
  1. OpenAlex batch fetch (OpenAlexClient.get_works_batch) — yields full metadata
     INCLUDING the DOI that the has_doi feature needs. Requires network.
  2. data/openalex_eval_cache.json — offline cache (no DOI). Used under --no-fetch
     for dry runs only; a model trained this way has a degenerate has_doi feature
     and MUST NOT ship (see plan Risk 2).
  cited_by_count is always backfilled from data/openalex_cite_cache.json when present.

Output JSONL rows: {"qid": int, "workid": str, "query": str, "grade": int,
                    "features": [6 floats], "meta": {has_doi, source}}

Usage:
  # Offline dry run — reports coverage, writes nothing that should ship:
  python scripts/build_lambdamart_dataset.py --no-fetch --dry-run

  # Full build (network; ~80 OpenAlex calls for ~4k works):
  python scripts/build_lambdamart_dataset.py --out data/lambdamart_dataset.jsonl \
      --embed-model models/sfu-academic-embed-v5
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_lambdamart_dataset")

DEFAULT_JUDGE_CACHE = REPO_ROOT / "data/eval_results/llm_judge_cache.json"
DEFAULT_CITE_CACHE = REPO_ROOT / "data/openalex_cite_cache.json"
DEFAULT_EVAL_CACHE = REPO_ROOT / "data/openalex_eval_cache.json"
DEFAULT_OUT = REPO_ROOT / "data/lambdamart_dataset.jsonl"

RELEVANT_THRESHOLD = 2   # judge grade >= this == relevant   (matches eval_cross_encoder)
MIN_CANDIDATES = 2       # a query needs >=2 judged candidates


# ── Label loading (ported from scripts/eval_cross_encoder.py for consistency) ──

def load_judge_grades(path: Path) -> dict[str, dict[str, int]]:
    """Return grades_by_query[query][bare_work_id] = grade (int)."""
    cache = json.loads(path.read_text())
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    for key, grade in cache.items():
        query, doc_id = key.rsplit("||", 1)
        if isinstance(grade, int):
            grades[query][_bare(doc_id)] = grade
    return grades


def qualifies(cands: dict[str, int]) -> bool:
    """A query qualifies if it has >=MIN_CANDIDATES judged docs and >=1 relevant."""
    if len(cands) < MIN_CANDIDATES:
        return False
    return any(g >= RELEVANT_THRESHOLD for g in cands.values())


def _bare(wid: str) -> str:
    return wid.rsplit("/", 1)[-1] if wid else wid


# ── Metadata resolution ────────────────────────────────────────────────────────

def load_cite_cache(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {_bare(k): int(v) for k, v in raw.items() if isinstance(v, int)}


def load_eval_cache_meta(path: Path) -> dict[str, dict]:
    """Flatten data/openalex_eval_cache.json into {bare_id: raw_work}."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for _key, works in raw.items():
        if not isinstance(works, list):
            continue
        for w in works:
            if isinstance(w, dict) and w.get("id"):
                out[_bare(w["id"])] = w
    return out


def _doc_from_eval_cache(w: dict) -> dict:
    """Build a normalize_work-shaped doc from an eval-cache record (no DOI)."""
    year = w.get("publication_year")
    return {
        "title": w.get("title") or "",
        "abstract": w.get("abstract") or "",
        "date": str(year) if year else "",
        "type": w.get("type") or "",
        "doi": "",  # eval cache lacks DOI — degenerate has_doi (see Risk 2)
        "authors": [],
        "is_oa": False,
        "oa_url": "",
        "cited_by_count": w.get("cited_by_count", 0),
        "openalex_id": w.get("id", ""),
    }


def resolve_metadata(
    work_ids: set[str], eval_meta: dict[str, dict], cite_cache: dict[str, int],
    no_fetch: bool,
) -> tuple[dict[str, dict], str]:
    """Return ({bare_id: normalize_work-shaped doc}, source_label).

    Prefers a network batch fetch (has DOI); falls back to the offline eval cache.
    cited_by_count is backfilled from cite_cache regardless of source.
    """
    docs: dict[str, dict] = {}
    source = "eval_cache"

    if not no_fetch:
        try:
            from lib.config import load_config
            from lib.openalex import OpenAlexClient
            cfg = load_config()
            client = OpenAlexClient(
                mailto=cfg.openalex_mailto, api_key=cfg.openalex_api_key,
                daily_call_limit=cfg.openalex_daily_call_limit,
                tracker_path=cfg.openalex_tracker_path,
            )
            logger.info("Fetching %d works from OpenAlex (~%d calls)...",
                        len(work_ids), (len(work_ids) + 49) // 50)
            fetched = client.get_works_batch(sorted(work_ids))
            docs.update(fetched)
            source = "openalex_fetch"
            logger.info("Fetched %d/%d works from OpenAlex", len(fetched), len(work_ids))
        except Exception as e:
            logger.warning("OpenAlex fetch failed (%s); falling back to eval cache", e)

    # Backfill any still-missing works from the offline eval cache.
    for wid in work_ids:
        if wid not in docs and wid in eval_meta:
            docs[wid] = _doc_from_eval_cache(eval_meta[wid])

    # Backfill citation counts from the cite cache where richer than the doc's own.
    for wid, doc in docs.items():
        if not doc.get("cited_by_count") and wid in cite_cache:
            doc["cited_by_count"] = cite_cache[wid]

    return docs, source


# ── Dataset construction ───────────────────────────────────────────────────────

def build_rows(grades_by_query, docs, embed_model_path):
    """Yield dataset rows. Embeds per query so cosine matches the inference path."""
    from lib.reranker import _normalize_for_rerank, _compute_semantic_scores
    from lib.lambdamart_features import doc_features

    qid = 0
    stats = {"queries": 0, "rows": 0, "missing_meta": 0, "with_doi": 0}
    for query in sorted(grades_by_query):
        cands = grades_by_query[query]
        if not qualifies(cands):
            continue
        wids = [w for w in cands if w in docs]
        if len(wids) < MIN_CANDIDATES:
            stats["missing_meta"] += len(cands) - len(wids)
            continue

        cand_docs = [docs[w] for w in wids]
        # Reuse the exact inference cosine path so f5 is computed identically.
        cosines = _compute_semantic_scores(query, cand_docs, embed_model_path)
        if not cosines:
            cosines = [0.0] * len(cand_docs)

        for w, doc, cos in zip(wids, cand_docs, cosines):
            norm = _normalize_for_rerank(doc)
            feats = doc_features(norm, doc.get("cited_by_count", 0), cos)
            has_doi = bool(norm.get("doi"))
            stats["with_doi"] += int(has_doi)
            stats["rows"] += 1
            yield {
                "qid": qid,
                "workid": w,
                "query": query,
                "grade": cands[w],
                "features": feats,
                "meta": {"has_doi": has_doi},
            }
        stats["queries"] += 1
        qid += 1
    build_rows.stats = stats  # attach for the caller


def main():
    parser = argparse.ArgumentParser(description="Build LambdaMART training dataset (Phase N)")
    parser.add_argument("--judge-cache", default=str(DEFAULT_JUDGE_CACHE))
    parser.add_argument("--cite-cache", default=str(DEFAULT_CITE_CACHE))
    parser.add_argument("--eval-cache", default=str(DEFAULT_EVAL_CACHE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--embed-model", default=str(REPO_ROOT / "models/sfu-academic-embed-v5"),
                        help="SFU embedding model for the embed_cosine feature")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Offline: use eval+cite caches only (no DOI — dry-run quality)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report coverage stats; do not write the dataset")
    args = parser.parse_args()

    judge_path = Path(args.judge_cache)
    if not judge_path.exists():
        logger.error("Judge cache not found: %s", judge_path)
        sys.exit(1)

    grades_by_query = load_judge_grades(judge_path)
    qualifying = {q: c for q, c in grades_by_query.items() if qualifies(c)}
    work_ids = {w for c in qualifying.values() for w in c}
    logger.info("Judge cache: %d queries (%d qualifying), %d distinct works",
                len(grades_by_query), len(qualifying), len(work_ids))

    eval_meta = load_eval_cache_meta(Path(args.eval_cache))
    cite_cache = load_cite_cache(Path(args.cite_cache))
    docs, source = resolve_metadata(work_ids, eval_meta, cite_cache, args.no_fetch)
    logger.info("Resolved metadata for %d/%d works (source: %s)",
                len(docs), len(work_ids), source)

    embed_model = args.embed_model if Path(args.embed_model).exists() else None
    if embed_model is None:
        logger.warning("Embed model %s not found — embed_cosine will be 0.0", args.embed_model)

    rows = list(build_rows(qualifying, docs, embed_model))
    stats = build_rows.stats
    print(f"\nDataset summary:")
    print(f"  Qualifying queries used:   {stats['queries']}")
    print(f"  (query, doc) rows:         {stats['rows']}")
    print(f"  Rows with a DOI:           {stats['with_doi']} "
          f"({100 * stats['with_doi'] / max(1, stats['rows']):.0f}%)")
    print(f"  Works missing metadata:    {stats['missing_meta']}")
    print(f"  Metadata source:           {source}")
    if source == "eval_cache":
        print("  ⚠  Offline build: has_doi is degenerate (all 0). For a shippable")
        print("     model, run without --no-fetch so DOIs are fetched (Risk 2).")

    if args.dry_run:
        print("\nDry run — no dataset written.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
