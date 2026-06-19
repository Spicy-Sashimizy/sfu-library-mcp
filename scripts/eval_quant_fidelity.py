#!/usr/bin/env python3
"""Off-BMP quality gate: does BMP's 8-bit ×70 quantization change SPLADE ranking
vs Qdrant's raw f32?

This isolates the ONLY thing that differs between the two sparse engines when they
score the same precomputed SPLADE vectors over the same docs:

  - BMP (`bmp.Searcher`):   doc impact  = min(255, max(1, round(w*70)))   [builder.py:213]
                            query impact = max(1, round(w*70)) over top-64 [retriever.py:475-476]
  - Qdrant (on_disk sparse): doc value   = raw f32 weight
                            query value  = raw f32 weight (same top-64 term set)

Everything else favors Qdrant: its sparse search is an EXACT posting-list dot
product, while BMP is block-max *approximate* (WAND pruning) — so on recall Qdrant
can only match or beat BMP. The residual risk is therefore purely "does ×70 8-bit
rounding reorder the top-k?", and that is corpus-size-independent: proving it on a
real N-doc sample generalizes to 150M.

Method: load N real docs (sparse_field) from spool_backup into two CSR matrices
sharing one sparsity pattern (f32 values vs ×70-clamped int values). Score the 40
diverse eval queries against the pool both ways, top-50 each, and compare:
  - overlap@10 / overlap@50          (set agreement of the returned ids)
  - Kendall-tau over the union top-50 (ordering agreement)
  - NDCG@10 under each scoring, from the LLM-judge cache, where judged W-ids land
    in the pool (the DELTA is valid even when absolute coverage is low, since both
    scorings see the identical pool).

Token space matches the Qdrant ingest exactly: every token -> its bert-base-uncased
vocab id (the same bijection scripts/spike_qdrant_sparse.py uses).

Usage:
  scripts/eval_quant_fidelity.py --n 500000 --queries 40 \
      --output data/eval_results/quant_fidelity.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quant_fidelity")

QUANT_SCALE = 70          # builder.QUANT_SCALE / retriever.QUANT_SCALE
BMP_IMPACT_MAX = 255      # builder.BMP_IMPACT_MAX (8-bit cap)
SPLADE_QUERY_TERMS = 64   # retriever.SPLADE_QUERY_TERMS
SPOOL = REPO_ROOT / "data" / "thinclient_index" / "spool_backup"
QUERIES_JSON = REPO_ROOT / "data" / "eval_results" / "diverse_queries.json"
JUDGE_CACHE = REPO_ROOT / "data" / "eval_results" / "llm_judge_cache.json"
SPLADE_MODEL = str(REPO_ROOT / "models" / "splade_onnx")


def _vocab():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    v = tok.get_vocab()
    logger.info("loaded bert-base-uncased vocab (%d tokens)", len(v))
    return v


def _load_pool(n: int, vocab: dict, judged_wids: set, harvest_scan: int):
    """Stream spool_backup -> (wids, CSR f32, CSR quant) sharing one sparsity
    pattern. quant = min(255, max(1, round(w*70))), matching the BMP build.

    The pool = the first `n` docs (a realistic competition set) PLUS every doc
    whose W-id is in the LLM-judge cache encountered within the first
    `harvest_scan` docs — so judged docs actually compete in the ranking and the
    head-to-head NDCG@10 is meaningful, not coverage-starved."""
    import zstandard
    slices = sorted(glob.glob(str(SPOOL / "*" / "slice_*.jsonl.zst")))
    if not slices:
        raise SystemExit(f"no spool slices under {SPOOL}")
    V = max(vocab.values()) + 1
    wids: list[str] = []
    indptr = [0]
    indices: list[int] = []
    data_f32: list[float] = []
    data_q: list[int] = []
    dctx = zstandard.ZstdDecompressor()
    scanned = kept = harvested = 0
    added: set[str] = set()
    t0 = time.perf_counter()
    for sl in slices:
        for_break = False
        with open(sl, "rb") as fh, dctx.stream_reader(fh) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8"):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sf = rec.get("sparse_field")
                wid = rec.get("openalex_id") or rec.get("id")
                if not sf or not wid:
                    continue
                scanned += 1
                is_random = kept < n
                is_judged = wid in judged_wids and wid not in added
                if not (is_random or is_judged):
                    if scanned >= harvest_scan and kept >= n:
                        for_break = True
                        break
                    continue
                row_n = 0
                for tok, w in sf.items():
                    if w <= 0:
                        continue
                    i = vocab.get(tok)
                    if i is None:
                        continue
                    indices.append(i)
                    data_f32.append(float(w))
                    data_q.append(min(BMP_IMPACT_MAX, max(1, int(round(float(w) * QUANT_SCALE)))))
                    row_n += 1
                if row_n == 0:
                    continue
                wids.append(wid)
                added.add(wid)
                indptr.append(len(indices))
                kept += 1
                if is_judged:
                    harvested += 1
                if kept % 100000 == 0:
                    logger.info("  kept %d docs (scanned %d, %d judged-harvested, %.0f/s)",
                                kept, scanned, harvested, scanned / (time.perf_counter() - t0))
                if scanned >= harvest_scan and kept >= n:
                    for_break = True
                    break
        if for_break:
            break
    logger.info("harvested %d judged docs into pool", harvested)
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int32)
    D_f32 = sp.csr_matrix((np.asarray(data_f32, dtype=np.float32), indices, indptr),
                          shape=(len(wids), V))
    D_q = sp.csr_matrix((np.asarray(data_q, dtype=np.float32), indices, indptr),
                        shape=(len(wids), V))
    logger.info("pool: %d docs, %d nnz, vocab dim %d, %.1fs",
                len(wids), len(indices), V, time.perf_counter() - t0)
    return np.asarray(wids), D_f32, D_q, V


def _load_queries(n: int):
    recs = json.loads(QUERIES_JSON.read_text())
    seen, qs = set(), []
    for rec in recs:
        p = rec["paraphrase"]
        if p in seen:
            continue
        seen.add(p)
        qs.append(p)
        if len(qs) >= n:
            break
    return qs


def _load_judge():
    """cache key 'query[:80]||W-id' -> grade. Return {qprefix: {wid: grade}}."""
    cache = json.loads(JUDGE_CACHE.read_text())
    by_q: dict[str, dict[str, int]] = {}
    for k, g in cache.items():
        pre, _, wid = k.rpartition("||")
        if not wid:
            continue
        by_q.setdefault(pre, {})[wid] = int(g)
    return by_q


def _ndcg_at_k(ranked_wids, judged: dict[str, int], k: int = 10):
    """Exponential-gain NDCG@k (TREC), matching eval_thinclient_parity.ndcg_at_k:
    unjudged docs contribute 0 but hold their slot; iDCG from the judged pool."""
    if not judged:
        return None
    dcg = 0.0
    for i, wid in enumerate(ranked_wids[:k]):
        g = judged.get(wid)
        if g:
            dcg += (2 ** g - 1) / math.log2(i + 2)
    ideal = sorted(judged.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal) if g)
    if idcg == 0:
        return None
    return dcg / idcg


def _kendall_tau_union(rank_a, rank_b):
    """Kendall-tau over the union of the two top-k lists; missing docs get rank=len."""
    from scipy.stats import kendalltau
    union = list(dict.fromkeys(list(rank_a) + list(rank_b)))
    pa = {w: i for i, w in enumerate(rank_a)}
    pb = {w: i for i, w in enumerate(rank_b)}
    big = len(union)
    a = [pa.get(w, big) for w in union]
    b = [pb.get(w, big) for w in union]
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    tau, _ = kendalltau(a, b)
    return tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000, help="random docs in the pool")
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--harvest-scan", type=int, default=5_000_000,
                    help="scan this many docs to harvest judged docs into the pool")
    ap.add_argument("--output", default=str(REPO_ROOT / "data/eval_results/quant_fidelity.json"))
    args = ap.parse_args()

    from lib.opensearch_retriever import encode_splade

    vocab = _vocab()
    queries = _load_queries(args.queries)
    judge = _load_judge()
    # union of judged W-ids across the selected queries -> harvest these into the pool
    judged_wids: set[str] = set()
    for q in queries:
        judged_wids |= set(judge.get(q[:80], {}).keys())
    logger.info("queries=%d judge-prefixes=%d judged-wids-for-these-queries=%d",
                len(queries), len(judge), len(judged_wids))
    wids, D_f32, D_q, V = _load_pool(args.n, vocab, judged_wids, args.harvest_scan)
    wid_set = set(wids.tolist())

    per_q = []
    for q in queries:
        wt = encode_splade(q, SPLADE_MODEL)
        if not wt:
            continue
        top = sorted(wt.items(), key=lambda x: -x[1])[:SPLADE_QUERY_TERMS]
        qf = np.zeros(V, dtype=np.float32)
        qq = np.zeros(V, dtype=np.float32)
        for tok, w in top:
            i = vocab.get(tok)
            if i is None:
                continue
            qf[i] = float(w)
            qq[i] = max(1, int(round(float(w) * QUANT_SCALE)))
        s_f = D_f32.dot(qf)
        s_q = D_q.dot(qq)
        kk = args.topk
        top_f = wids[np.argsort(-s_f, kind="stable")[:kk]]
        top_q = wids[np.argsort(-s_q, kind="stable")[:kk]]
        sf, sq = set(top_f), set(top_q)
        ov10 = len(set(top_f[:10]) & set(top_q[:10])) / 10.0
        ov50 = len(sf & sq) / float(kk)
        tau = _kendall_tau_union(list(top_f), list(top_q))
        judged = judge.get(q[:80], {})
        judged_in_pool = {w: g for w, g in judged.items() if w in wid_set}
        # NDCG is meaningful only where judged docs actually compete in the pool;
        # the f32-vs-quant DELTA is the parity signal.
        ndcg_f = _ndcg_at_k(list(top_f), judged_in_pool, 10)
        ndcg_q = _ndcg_at_k(list(top_q), judged_in_pool, 10)
        per_q.append({
            "query": q[:80], "overlap@10": ov10, "overlap@50": ov50,
            "kendall_tau": tau, "ndcg@10_f32": ndcg_f, "ndcg@10_quant": ndcg_q,
            "n_judged": len(judged), "n_judged_in_pool": len(judged_in_pool),
        })

    def _mean(key, sub=per_q):
        vals = [r[key] for r in sub if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    ndcg_sub = [r for r in per_q if r["ndcg@10_f32"] is not None and r["ndcg@10_quant"] is not None]
    summary = {
        "pool_docs": int(len(wids)), "queries": len(per_q),
        "mean_overlap@10": _mean("overlap@10"),
        "mean_overlap@50": _mean("overlap@50"),
        "mean_kendall_tau": _mean("kendall_tau"),
        "ndcg_queries_with_coverage": len(ndcg_sub),
        "mean_ndcg@10_f32": _mean("ndcg@10_f32", ndcg_sub),
        "mean_ndcg@10_quant": _mean("ndcg@10_quant", ndcg_sub),
    }
    if summary["mean_ndcg@10_f32"] is not None:
        summary["ndcg_delta_f32_minus_quant"] = summary["mean_ndcg@10_f32"] - summary["mean_ndcg@10_quant"]
    out = {"config": vars(args), "summary": summary, "per_query": per_q}
    Path(args.output).write_text(json.dumps(out, indent=2))

    logger.info("=" * 64)
    logger.info("QUANT FIDELITY — f32 (Qdrant) vs ×70 8-bit (BMP), %d docs, %d queries",
                len(wids), len(per_q))
    logger.info("rank agreement: overlap@10=%.3f overlap@50=%.3f kendall_tau=%.3f",
                summary["mean_overlap@10"], summary["mean_overlap@50"],
                summary["mean_kendall_tau"] or float("nan"))
    if summary["mean_ndcg@10_f32"] is not None:
        logger.info("NDCG@10 (%d judged-coverage queries): f32=%.4f quant=%.4f delta=%.4f",
                    summary["ndcg_queries_with_coverage"], summary["mean_ndcg@10_f32"],
                    summary["mean_ndcg@10_quant"], summary["ndcg_delta_f32_minus_quant"])
    else:
        logger.info("NDCG@10: no judged docs landed in the pool (rank-agreement is the signal)")
    logger.info("wrote %s", args.output)
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
