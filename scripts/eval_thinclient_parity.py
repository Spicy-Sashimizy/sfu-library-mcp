#!/usr/bin/env python3
"""Eval the thin-client stack (tantivy+BMP+usearch) against the OpenSearch
baseline: per-leg overlap, latency, and LLM-judged NDCG@10.

Compares, per query (data/eval_results/diverse_queries.json):
  - thinclient bm25f  vs  OpenSearch bm25f   (top-k id overlap)
  - thinclient splade vs  OpenSearch splade  (top-k id overlap)
  - thinclient RRF    vs  OpenSearch RRF     (overlap + NDCG@10 via the
    LLM-judge cache where judged docs exist)

Overlap vs the full-corpus baseline is only apples-to-apples when the
thin-client index holds the SAME corpus (post-migration). For subset builds
(--limit validation), pass --no-baseline / skip the os record: the run then
reports thin-client functional health only (legs return results, latency,
leg complementarity).

OOM-safe record-then-replay (default; use this at 150M scale)
─────────────────────────────────────────────────────────────
At 150M docs the thin-client serving set (~104 GB mmap) and the OpenSearch
150M cluster cannot both stay hot on the 31 GB host — interleaving both
engines in ONE process (the legacy `combined` mode) OOM-kills. Instead, run
each engine in its OWN process and join offline. The retriever has no close()
hook, so only a fresh process truly releases the mmap working set; the phases
are therefore separate subcommands, sequenced by scripts/run_parity_safe.sh:

    # phase A — thin-client only (no OpenSearch in this process)
    SFU_DENSE_WARMCACHE=0 .venv/bin/python3 scripts/eval_thinclient_parity.py \
        record-tc --index-root data/thinclient_index --queries 40 \
        --output data/eval_results/parity_record_tc.json

    # phase B — OpenSearch only (thin-client process already exited)
    .venv/bin/python3 scripts/eval_thinclient_parity.py \
        record-os --baseline-url http://host.docker.internal:9200 --queries 40 \
        --output data/eval_results/parity_record_os.json

    # phase C — pure offline join (few MB of RAM)
    .venv/bin/python3 scripts/eval_thinclient_parity.py compare \
        --tc-record data/eval_results/parity_record_tc.json \
        --os-record data/eval_results/parity_record_os.json

`compare` emits the same summary schema as the legacy single-process run, so
nothing downstream changes. Omit --os-record for a thin-client-only summary.

Legacy single-process mode (may OOM at 150M, kept for small indices):
    .venv/bin/python3 scripts/eval_thinclient_parity.py combined \
        --index-root data/thinclient_index [--queries 40] [--no-baseline]
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_thinclient")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

TOP_K = 50
NDCG_K = 10


def load_queries(n: int) -> list[dict]:
    records = json.loads((REPO_ROOT / "data/eval_results/diverse_queries.json").read_text())
    out, seen = [], set()
    for rec in records:
        if rec["paraphrase"] not in seen:
            seen.add(rec["paraphrase"])
            out.append(rec)
        if len(out) >= n:
            break
    return out


def load_judge() -> dict[tuple[str, str], int]:
    cache = json.loads((REPO_ROOT / "data/eval_results/llm_judge_cache.json").read_text())
    out = {}
    for key, grade in cache.items():
        qk, _, did = key.rpartition("||")
        out[(qk, did)] = grade
    return out


def ndcg_at_k(ranked_ids: list[str], judge: dict, judge_key: str, k: int = NDCG_K) -> float | None:
    grades = [judge.get((judge_key, did)) for did in ranked_ids[:k]]
    known = [g for g in grades if g is not None]
    if not known:
        return None
    dcg = sum((2 ** g - 1) / math.log2(i + 2)
              for i, g in enumerate(grades) if g is not None)
    ideal = sorted((g for (qk, _), g in judge.items() if qk == judge_key),
                   reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else None


def rrf(lists: list[list[str]], k: int = 60, top: int = TOP_K) -> list[str]:
    scores: dict[str, float] = {}
    for lst in lists:
        for rank, did in enumerate(lst, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:top]


# ── memory guard ───────────────────────────────────────────────────────────
# The 150M build's OOM kills left "no traceback, no OOM event" (THIN_CLIENT_
# SWAP.md). Sample MemAvailable and abort cleanly *before* the kernel killer
# fires, flushing whatever was recorded so the run is resumable/diagnosable.

def mem_available_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        return None
    return None


def guard_memory(floor_gb: float, on_abort) -> None:
    avail = mem_available_gb()
    if avail is not None and avail < floor_gb:
        logger.error("MemAvailable %.2f GB < floor %.2f GB — aborting cleanly "
                     "before OOM-kill; flushing partial record", avail, floor_gb)
        on_abort()
        sys.exit(2)


def _ts_default(prefix: str) -> Path:
    return REPO_ROOT / f"data/eval_results/{prefix}_{time.strftime('%Y%m%d_%H%M')}.json"


# ── phase A / B: record one engine's ranked id lists + latency ───────────────

def _record(engine: str, retriever, queries: list[dict], floor_gb: float,
            out_path: Path, config: dict) -> None:
    rows = []

    def flush():
        out_path.write_text(json.dumps(
            {"engine": engine, "config": {**config, "recorded": len(rows),
                                          "requested": len(queries)},
             "queries": rows}, indent=2))

    for i, rec in enumerate(queries):
        guard_memory(floor_gb, flush)
        q = rec["paraphrase"]
        jk = rec.get("judge_key", q)
        t0 = time.perf_counter()
        bm = [d["openalex_id"] for d in retriever.search(q, TOP_K, mode="bm25f")]
        lat_bm = time.perf_counter() - t0
        t0 = time.perf_counter()
        sp = [d["openalex_id"] for d in retriever.search(q, TOP_K, mode="splade")]
        lat_sp = time.perf_counter() - t0
        rows.append({"query": q, "judge_key": jk, "bm25f": bm, "splade": sp,
                     "lat_bm25f_ms": round(lat_bm * 1000, 2),
                     "lat_splade_ms": round(lat_sp * 1000, 2)})
        if (i + 1) % 10 == 0:
            avail = mem_available_gb()
            logger.info("[%s] %d/%d queries (MemAvailable %.1f GB)", engine,
                        i + 1, len(queries), avail if avail is not None else -1)
    flush()
    logger.info("[%s] wrote %d records -> %s", engine, len(rows), out_path)


def cmd_record_tc(args) -> None:
    # The dense warm cache holds an up-to-500k-doc RAM delta; shed it for the
    # eval unless the caller explicitly asked to keep it on.
    os.environ.setdefault("SFU_DENSE_WARMCACHE", "0")
    from lib.thinclient.retriever import ThinClientRetriever
    tc = ThinClientRetriever(index_root=args.index_root, remote_abstracts=False)
    assert tc.is_available(), f"thin-client index at {args.index_root} not available"
    logger.info("thin-client live sections: %s", tc.live_sections())
    out = Path(args.output) if args.output else _ts_default("parity_record_tc")
    _record("tc", tc, load_queries(args.queries), args.mem_floor_gb, out,
            {"index_root": args.index_root, "top_k": TOP_K,
             "warmcache": os.environ.get("SFU_DENSE_WARMCACHE")})


def cmd_record_os(args) -> None:
    from lib.opensearch_retriever import OpenSearchRetriever
    baseline = OpenSearchRetriever(
        url=args.baseline_url, index=args.baseline_index,
        splade_model_path=str(REPO_ROOT / "models/splade_onnx"), timeout=30)
    out = Path(args.output) if args.output else _ts_default("parity_record_os")
    _record("os", baseline, load_queries(args.queries), args.mem_floor_gb, out,
            {"baseline": f"{args.baseline_url}/{args.baseline_index}", "top_k": TOP_K})


# ── phase C: offline join of the two records into the parity summary ─────────

def _summarize(tc_rows: list[dict], os_rows: list[dict] | None,
               judge: dict, config: dict) -> dict:
    os_by_q = {r["query"]: r for r in (os_rows or [])}
    per_query, lat = [], {"tc_bm25f": [], "tc_splade": [], "os_bm25f": [], "os_splade": []}
    ndcgs = {"tc_rrf": [], "os_rrf": []}
    overlaps = {"bm25f": [], "splade": [], "rrf": []}
    leg_overlap_tc = []

    for tr in tc_rows:
        q, jk = tr["query"], tr.get("judge_key", tr["query"])
        tc_bm, tc_sp = tr["bm25f"], tr["splade"]
        lat["tc_bm25f"].append(tr["lat_bm25f_ms"] / 1000)
        lat["tc_splade"].append(tr["lat_splade_ms"] / 1000)
        tc_rrf = rrf([tc_bm, tc_sp])
        leg_overlap_tc.append(len(set(tc_bm) & set(tc_sp)))

        row = {"query": q, "tc_bm25f_n": len(tc_bm), "tc_splade_n": len(tc_sp)}
        n = ndcg_at_k(tc_rrf, judge, jk)
        if n is not None:
            ndcgs["tc_rrf"].append(n)
            row["tc_rrf_ndcg@10"] = round(n, 4)

        osr = os_by_q.get(q)
        if osr:
            os_bm, os_sp = osr["bm25f"], osr["splade"]
            lat["os_bm25f"].append(osr["lat_bm25f_ms"] / 1000)
            lat["os_splade"].append(osr["lat_splade_ms"] / 1000)
            os_rrf = rrf([os_bm, os_sp])
            overlaps["bm25f"].append(len(set(tc_bm) & set(os_bm)) / max(len(os_bm), 1))
            overlaps["splade"].append(len(set(tc_sp) & set(os_sp)) / max(len(os_sp), 1))
            overlaps["rrf"].append(len(set(tc_rrf) & set(os_rrf)) / max(len(os_rrf), 1))
            n = ndcg_at_k(os_rrf, judge, jk)
            if n is not None:
                ndcgs["os_rrf"].append(n)
                row["os_rrf_ndcg@10"] = round(n, 4)
        per_query.append(row)

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    have_baseline = bool(os_rows)
    return {
        "config": {**config, "queries": len(tc_rows), "top_k": TOP_K,
                   "baseline": config.get("baseline") if have_baseline else None},
        "latency_ms": {k: round(avg(v) * 1000, 2) if v else None for k, v in lat.items()},
        "ndcg@10": {k: {"mean": avg(v), "judged_queries": len(v)} for k, v in ndcgs.items()},
        "overlap_vs_baseline@50": {k: avg(v) for k, v in overlaps.items()},
        "tc_leg_complementarity": {"avg_bm25f_splade_overlap@50": avg(leg_overlap_tc)},
        "per_query": per_query,
    }


def _print_summary(summary: dict, have_baseline: bool, n: int) -> None:
    print("\n" + "=" * 88)
    print(f"THIN-CLIENT PARITY EVAL — {n} queries vs "
          f"{summary['config']['baseline'] or '(no baseline)'}")
    print("=" * 88)
    print("latency ms/query:", {k: v for k, v in summary["latency_ms"].items() if v})
    print("NDCG@10:", summary["ndcg@10"])
    if have_baseline:
        print("overlap@50 vs baseline:", summary["overlap_vs_baseline@50"])
    print("tc leg overlap@50 (bm25f∩splade):",
          summary["tc_leg_complementarity"]["avg_bm25f_splade_overlap@50"])
    print("=" * 88)


def cmd_compare(args) -> None:
    tc = json.loads(Path(args.tc_record).read_text())
    tc_rows = tc["queries"]
    os_rows = None
    config = {"index_root": tc.get("config", {}).get("index_root")}
    if args.os_record:
        osd = json.loads(Path(args.os_record).read_text())
        os_rows = osd["queries"]
        config["baseline"] = osd.get("config", {}).get("baseline")
    judge = load_judge()
    summary = _summarize(tc_rows, os_rows, judge, config)
    out_path = Path(args.output) if args.output else _ts_default("thinclient_parity")
    out_path.write_text(json.dumps(summary, indent=2))
    _print_summary(summary, bool(os_rows), len(tc_rows))
    logger.info("wrote %s", out_path)


# ── legacy single-process mode (may OOM at 150M) ─────────────────────────────

def cmd_combined(args) -> None:
    logger.warning("combined mode loads BOTH engines in one process — at 150M "
                   "scale this OOM-kills on a 31 GB host; prefer record-tc/"
                   "record-os/compare via scripts/run_parity_safe.sh")
    from lib.thinclient.retriever import ThinClientRetriever
    tc = ThinClientRetriever(index_root=args.index_root, remote_abstracts=False)
    assert tc.is_available(), f"thin-client index at {args.index_root} not available"
    logger.info("thin-client live sections: %s", tc.live_sections())

    tc_rows, os_rows = [], None if args.no_baseline else []
    baseline = None
    if not args.no_baseline:
        from lib.opensearch_retriever import OpenSearchRetriever
        baseline = OpenSearchRetriever(
            url=args.baseline_url, index=args.baseline_index,
            splade_model_path=str(REPO_ROOT / "models/splade_onnx"), timeout=30)

    for rec in load_queries(args.queries):
        q, jk = rec["paraphrase"], rec.get("judge_key", rec["paraphrase"])
        t0 = time.perf_counter()
        bm = [d["openalex_id"] for d in tc.search(q, TOP_K, mode="bm25f")]
        lbm = time.perf_counter() - t0
        t0 = time.perf_counter()
        sp = [d["openalex_id"] for d in tc.search(q, TOP_K, mode="splade")]
        lsp = time.perf_counter() - t0
        tc_rows.append({"query": q, "judge_key": jk, "bm25f": bm, "splade": sp,
                        "lat_bm25f_ms": round(lbm * 1000, 2),
                        "lat_splade_ms": round(lsp * 1000, 2)})
        if baseline is not None:
            t0 = time.perf_counter()
            obm = [d["openalex_id"] for d in baseline.search(q, TOP_K, mode="bm25f")]
            olbm = time.perf_counter() - t0
            t0 = time.perf_counter()
            osp = [d["openalex_id"] for d in baseline.search(q, TOP_K, mode="splade")]
            olsp = time.perf_counter() - t0
            os_rows.append({"query": q, "judge_key": jk, "bm25f": obm, "splade": osp,
                            "lat_bm25f_ms": round(olbm * 1000, 2),
                            "lat_splade_ms": round(olsp * 1000, 2)})

    config = {"index_root": args.index_root}
    if not args.no_baseline:
        config["baseline"] = f"{args.baseline_url}/{args.baseline_index}"
    summary = _summarize(tc_rows, os_rows, load_judge(), config)
    out_path = Path(args.output) if args.output else _ts_default("thinclient_parity")
    out_path.write_text(json.dumps(summary, indent=2))
    _print_summary(summary, not args.no_baseline, len(tc_rows))
    logger.info("wrote %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--queries", type=int, default=40)
        p.add_argument("--mem-floor-gb", type=float, default=2.0,
                       help="abort cleanly if MemAvailable drops below this")
        p.add_argument("--output", default=None)

    p_tc = sub.add_parser("record-tc", help="record thin-client ranked ids (engine A)")
    p_tc.add_argument("--index-root", default=str(REPO_ROOT / "data/thinclient_index"))
    add_common(p_tc)
    p_tc.set_defaults(func=cmd_record_tc)

    p_os = sub.add_parser("record-os", help="record OpenSearch ranked ids (engine B)")
    p_os.add_argument("--baseline-url",
                      default=os.environ.get("SFU_MIGRATION_SOURCE",
                                             "http://host.docker.internal:9200"))
    p_os.add_argument("--baseline-index", default="openalex_works")
    add_common(p_os)
    p_os.set_defaults(func=cmd_record_os)

    p_cmp = sub.add_parser("compare", help="offline join of records -> parity summary")
    p_cmp.add_argument("--tc-record", required=True)
    p_cmp.add_argument("--os-record", default=None)
    p_cmp.add_argument("--output", default=None)
    p_cmp.set_defaults(func=cmd_compare)

    p_comb = sub.add_parser("combined", help="legacy single-process (may OOM at 150M)")
    p_comb.add_argument("--index-root", default=str(REPO_ROOT / "data/thinclient_index"))
    p_comb.add_argument("--no-baseline", action="store_true")
    p_comb.add_argument("--baseline-url",
                        default=os.environ.get("SFU_MIGRATION_SOURCE",
                                               "http://host.docker.internal:9200"))
    p_comb.add_argument("--baseline-index", default="openalex_works")
    add_common(p_comb)
    p_comb.set_defaults(func=cmd_combined)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
