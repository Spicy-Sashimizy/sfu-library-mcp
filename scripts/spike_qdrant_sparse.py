#!/usr/bin/env python3
"""SPIKE: does Qdrant serve a SPLADE sparse index from disk (on_disk=true) at low
resident RAM and acceptable latency? Decides whether Qdrant replaces the BMP
sparse leg (which is load-into-RAM only — ~211 GB resident at 150M; see
docs/THIN_CLIENT_SWAP.md). Research/validation only — not production code.

Pipeline (subcommands):
  build   — stream N real OpenAlex works -> GPU SPLADE encode -> upsert into a
            Qdrant collection whose sparse index is on_disk=true.
  measure — run the eval queries against the collection; report p50/p95 latency
            and the Qdrant process resident RSS (the number that matters).

Index space: both the GPU doc encoder (Splade_PP_en_v1) and the query encoder
(lib.opensearch_retriever.encode_splade) emit BERT wordpiece token strings; we
map every token -> its bert-base-uncased vocab id (a bijection) so doc and query
share one u32 sparse-index space.

Usage:
  scripts/spike_qdrant_sparse.py build   --n 500000 --collection splade_spike
  scripts/spike_qdrant_sparse.py measure --collection splade_spike --queries 40
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import logging
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qdrant_spike")

QDRANT_URL = "http://localhost:6333"
SNAPSHOT_DIR = REPO_ROOT / "data" / "openalex_snapshot"


# ── shared token -> u32 vocab-id map (bert-base-uncased; both encoders use it) ──
_VOCAB = None
def _vocab():
    global _VOCAB
    if _VOCAB is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        _VOCAB = tok.get_vocab()  # {token_str: id}
        logger.info("loaded bert-base-uncased vocab (%d tokens)", len(_VOCAB))
    return _VOCAB


def _to_sparse(weights: dict, vocab: dict):
    """{token: weight} -> (indices[u32], values[f32]) in the shared id space."""
    idx, val = [], []
    for tok, w in weights.items():
        i = vocab.get(tok)
        if i is not None:
            idx.append(int(i)); val.append(float(w))
    return idx, val


def _invert_abstract(inv):
    from lib.thinclient.retriever import _invert_abstract as f
    return f(inv)


def _stream_works(n: int):
    """Yield (openalex_id, text, payload) from the snapshot until n docs."""
    parts = sorted(glob.glob(str(SNAPSHOT_DIR / "works_part_*.jsonl.gz")))
    if not parts:
        raise SystemExit(f"no works_part_*.jsonl.gz under {SNAPSHOT_DIR}")
    seen = 0
    for part in parts:
        with gzip.open(part, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    w = json.loads(line)
                except Exception:
                    continue
                wid = (w.get("id") or "").rsplit("/", 1)[-1]
                if not wid:
                    continue
                title = w.get("title") or w.get("display_name") or ""
                inv = w.get("abstract_inverted_index")
                abstract = _invert_abstract(inv) if inv else ""
                text = (title + " " + abstract).strip()
                if not text:
                    continue
                year = w.get("publication_year")
                payload = {"oaid": wid, "year": year,
                           "type": w.get("type"),
                           "is_oa": bool((w.get("open_access") or {}).get("is_oa"))}
                yield wid, text, payload
                seen += 1
                if seen >= n:
                    return


def qdrant_rss_mb() -> float:
    import subprocess
    out = subprocess.run(["pgrep", "-f", "tools/qdrant/qdrant"],
                         capture_output=True, text=True)
    pids = [p for p in out.stdout.split() if p.isdigit()]
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/status") as f:
                for ln in f:
                    if ln.startswith("VmRSS"):
                        total += int(ln.split()[1])  # kB
        except OSError:
            pass
    return total / 1024


def _stream_spool(n: int):
    """Yield (oaid, sparse_field_dict, payload) from spool_backup slices —
    the SPLADE vectors are ALREADY computed here (no GPU re-encode needed)."""
    import zstandard
    sb = REPO_ROOT / "data" / "thinclient_index" / "spool_backup"
    slices = sorted(glob.glob(str(sb / "*" / "slice_*.jsonl.zst")))
    if not slices:
        raise SystemExit(f"no slices under {sb}")
    seen = 0
    dctx = zstandard.ZstdDecompressor()
    for sl in slices:
        with open(sl, "rb") as fh, dctx.stream_reader(fh) as r:
            text = io.TextIOWrapper(r, encoding="utf-8")
            for line in text:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sf = rec.get("sparse_field")
                wid = rec.get("openalex_id") or rec.get("id")
                if not sf or not wid:
                    continue
                payload = {"oaid": wid, "year": rec.get("publication_year"),
                           "type": rec.get("type")}
                yield wid, sf, payload
                seen += 1
                if seen >= n:
                    return


def cmd_build_spool(args):
    """GPU-free build: reuse precomputed sparse_field from spool_backup."""
    from qdrant_client import QdrantClient, models
    client = QdrantClient(url=QDRANT_URL, timeout=120)
    name = args.collection
    if client.collection_exists(name) and not args.append:
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name, vectors_config={},
            sparse_vectors_config={"splade": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True))},
            on_disk_payload=True)
        logger.info("created collection %s (sparse on_disk=True)", name)
    vocab = _vocab()
    rss0 = qdrant_rss_mb()
    logger.info("qdrant RSS before ingest: %.0f MB", rss0)
    t0 = time.perf_counter()
    points, pid_seq, done = [], args.start_id, 0
    for wid, sf, pay in _stream_spool(args.n):
        idx, val = _to_sparse(sf, vocab)
        if not idx:
            continue
        points.append(models.PointStruct(
            id=pid_seq, payload=pay,
            vector={"splade": models.SparseVector(indices=idx, values=val)}))
        pid_seq += 1
        if len(points) >= 1000:
            client.upsert(collection_name=name, points=points, wait=False)
            done += len(points); points = []
            if done % 100000 < 1000:
                el = time.perf_counter() - t0
                logger.info("ingested %d (%.0f docs/s) | qdrant RSS %.0f MB",
                            done, done / el, qdrant_rss_mb())
    if points:
        client.upsert(collection_name=name, points=points, wait=False); done += len(points)
    el = time.perf_counter() - t0
    logger.info("DONE spool build: %d docs in %.0fs (%.0f docs/s)", done, el, done / max(el, 1))
    logger.info("qdrant RSS after ingest: %.0f MB (was %.0f MB)", qdrant_rss_mb(), rss0)
    info = client.get_collection(name)
    logger.info("collection points=%s status=%s", info.points_count, info.status)


def _parallel_worker(task):
    """One ingest worker: owns a subset of slices, upserts with wait=True (bounded
    in-flight -> no server backlog, no runaway RAM) over gRPC."""
    import zstandard
    from qdrant_client import QdrantClient, models
    worker_id, slice_files, n_cap, collection, batch = task
    client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True, timeout=600)
    vocab = _vocab()
    id_base = worker_id * 100_000_000
    local = done = 0
    points = []
    dctx = zstandard.ZstdDecompressor()
    for sl in slice_files:
        with open(sl, "rb") as fh, dctx.stream_reader(fh) as r:
            text = io.TextIOWrapper(r, encoding="utf-8")
            for line in text:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sf = rec.get("sparse_field")
                wid = rec.get("openalex_id") or rec.get("id")
                if not sf or not wid:
                    continue
                idx, val = _to_sparse(sf, vocab)
                if not idx:
                    continue
                points.append(models.PointStruct(
                    id=id_base + local,
                    payload={"oaid": wid, "year": rec.get("publication_year"),
                             "type": rec.get("type")},
                    vector={"splade": models.SparseVector(indices=idx, values=val)}))
                local += 1
                if len(points) >= batch:
                    client.upsert(collection_name=collection, points=points, wait=True)
                    done += len(points); points = []
                    if done >= n_cap:
                        if points:
                            client.upsert(collection_name=collection, points=points, wait=True)
                        return done
    if points:
        client.upsert(collection_name=collection, points=points, wait=True); done += len(points)
    return done


def cmd_build_spool_parallel(args):
    """Bounded + parallel + gRPC ingest from spool_backup. wait=True per batch
    means the measured rate is the TRUE end-to-end (server-indexed) rate, and
    RAM stays bounded (no async backlog)."""
    import multiprocessing as mp
    from qdrant_client import QdrantClient, models
    client = QdrantClient(url=QDRANT_URL, timeout=120)
    name = args.collection
    if client.collection_exists(name) and not args.append:
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name, vectors_config={},
            sparse_vectors_config={"splade": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True))},
            on_disk_payload=True,
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=args.segments,
                max_optimization_threads=args.opt_threads))
        logger.info("created %s (sparse on_disk, segments=%d, opt_threads=%d)",
                    name, args.segments, args.opt_threads)
    slices = sorted(glob.glob(str(REPO_ROOT / "data/thinclient_index/spool_backup"
                                  / "*" / "slice_*.jsonl.zst")))
    if not slices:
        raise SystemExit("no spool_backup slices")
    W = args.workers
    n_cap = args.n // W
    tasks = [(w, slices[w::W], n_cap, name, args.batch) for w in range(W)]
    rss0 = qdrant_rss_mb()
    logger.info("PARALLEL build: %d workers, ~%d docs/worker, %d slices, qdrant RSS before=%.0f MB",
                W, n_cap, len(slices), rss0)
    t0 = time.perf_counter()
    peak = rss0
    with mp.Pool(W) as pool:
        res = pool.map_async(_parallel_worker, tasks)
        while not res.ready():
            time.sleep(30)
            try:
                info = client.get_collection(name)
                rss = qdrant_rss_mb(); peak = max(peak, rss)
                el = time.perf_counter() - t0
                logger.info("  +%.0fs points=%s status=%s qdrant RSS=%.0f MB (peak %.0f) ~%.0f docs/s",
                            el, info.points_count, info.status, rss, peak,
                            info.points_count / max(el, 1))
            except Exception as e:
                logger.warning("poll err: %s", e)
        results = res.get()
    el = time.perf_counter() - t0
    total = sum(results)
    logger.info("DONE parallel: %d docs in %.0fs (%.0f docs/s end-to-end, wait=True) | peak RSS %.0f MB",
                total, el, total / max(el, 1), peak)
    info = client.get_collection(name)
    logger.info("points=%s status=%s", info.points_count, info.status)


def cmd_build(args):
    from qdrant_client import QdrantClient, models
    from lib.thinclient.doc_encoder import make_doc_encoder

    client = QdrantClient(url=QDRANT_URL, timeout=120)
    name = args.collection
    if client.collection_exists(name):
        if not args.append:
            client.delete_collection(name)
        else:
            logger.info("appending to existing collection %s", name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={},  # no dense vectors in this spike
            sparse_vectors_config={
                "splade": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True))},
            on_disk_payload=True,
        )
        logger.info("created collection %s (sparse on_disk=True)", name)

    vocab = _vocab()
    logger.info("loading GPU SPLADE doc encoder (TensorRT)...")
    enc = make_doc_encoder(backend="auto", batch_size=args.batch)
    rss0 = qdrant_rss_mb()
    logger.info("qdrant RSS before ingest: %.0f MB", rss0)

    t0 = time.perf_counter()
    buf_ids, buf_text, buf_pay = [], [], []
    pid_seq = args.start_id
    done = 0

    def flush():
        nonlocal pid_seq, done
        if not buf_text:
            return
        vecs = enc.encode_batch(buf_text)
        points = []
        for wid, wt, pay in zip(buf_ids, vecs, buf_pay):
            idx, val = _to_sparse(wt, vocab)
            if not idx:
                continue
            points.append(models.PointStruct(
                id=pid_seq, payload=pay,
                vector={"splade": models.SparseVector(indices=idx, values=val)}))
            pid_seq += 1
        if points:
            client.upsert(collection_name=name, points=points, wait=False)
            done += len(points)
        buf_ids.clear(); buf_text.clear(); buf_pay.clear()

    for wid, text, pay in _stream_works(args.n):
        buf_ids.append(wid); buf_text.append(text); buf_pay.append(pay)
        if len(buf_text) >= args.batch:
            flush()
            if done and done % 50000 < args.batch:
                el = time.perf_counter() - t0
                logger.info("ingested %d docs (%.0f docs/s) | qdrant RSS %.0f MB",
                            done, done / el, qdrant_rss_mb())
    flush()
    el = time.perf_counter() - t0
    logger.info("DONE build: %d docs in %.0fs (%.0f docs/s)", done, el, done / max(el, 1))
    logger.info("qdrant RSS after ingest: %.0f MB (was %.0f MB)", qdrant_rss_mb(), rss0)
    logger.info("waiting for collection to settle / index on disk...")
    info = client.get_collection(name)
    logger.info("collection points=%s status=%s", info.points_count, info.status)


def cmd_measure(args):
    from qdrant_client import QdrantClient, models
    from lib.opensearch_retriever import encode_splade

    client = QdrantClient(url=QDRANT_URL, timeout=120)
    name = args.collection
    vocab = _vocab()
    info = client.get_collection(name)
    logger.info("collection %s: points=%s status=%s", name, info.points_count, info.status)
    logger.info("qdrant RSS at rest: %.0f MB", qdrant_rss_mb())

    queries = json.loads((REPO_ROOT / "data/eval_results/diverse_queries.json").read_text())
    seen, qs = set(), []
    for rec in queries:
        if rec["paraphrase"] not in seen:
            seen.add(rec["paraphrase"]); qs.append(rec["paraphrase"])
        if len(qs) >= args.queries:
            break

    model = str(REPO_ROOT / "models" / "splade_onnx")
    lats, peak_rss = [], qdrant_rss_mb()
    for q in qs:
        wt = encode_splade(q, model)
        idx, val = _to_sparse(wt, vocab)
        if not idx:
            continue
        t0 = time.perf_counter()
        client.query_points(
            collection_name=name,
            query=models.SparseVector(indices=idx, values=val),
            using="splade", limit=50, with_payload=False)
        lats.append((time.perf_counter() - t0) * 1000)
        peak_rss = max(peak_rss, qdrant_rss_mb())

    lats.sort()
    def pct(p): return lats[min(len(lats) - 1, int(len(lats) * p))]
    logger.info("=" * 60)
    logger.info("QDRANT on_disk SPARSE SPIKE — %d queries", len(lats))
    logger.info("latency ms: p50=%.1f p95=%.1f max=%.1f mean=%.1f",
                pct(0.5), pct(0.95), lats[-1], sum(lats) / len(lats))
    logger.info("qdrant resident RSS during query load: %.0f MB", peak_rss)
    logger.info("collection points=%s", info.points_count)
    logger.info("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=500_000)
    b.add_argument("--collection", default="splade_spike")
    b.add_argument("--batch", type=int, default=256)
    b.add_argument("--start-id", type=int, default=0)
    b.add_argument("--append", action="store_true")
    b.set_defaults(func=cmd_build)
    bs = sub.add_parser("build-spool")
    bs.add_argument("--n", type=int, default=5_000_000)
    bs.add_argument("--collection", default="splade_spool")
    bs.add_argument("--start-id", type=int, default=0)
    bs.add_argument("--append", action="store_true")
    bs.set_defaults(func=cmd_build_spool)
    bp = sub.add_parser("build-spool-par")
    bp.add_argument("--n", type=int, default=30_000_000)
    bp.add_argument("--collection", default="splade_par")
    bp.add_argument("--workers", type=int, default=8)
    bp.add_argument("--batch", type=int, default=1000)
    bp.add_argument("--segments", type=int, default=16)
    bp.add_argument("--opt-threads", type=int, default=8)
    bp.add_argument("--append", action="store_true")
    bp.set_defaults(func=cmd_build_spool_parallel)
    m = sub.add_parser("measure")
    m.add_argument("--collection", default="splade_spike")
    m.add_argument("--queries", type=int, default=40)
    m.set_defaults(func=cmd_measure)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
