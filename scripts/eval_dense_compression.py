#!/usr/bin/env python3
"""Dense-ANN HNSW compression eval — derived from eval_dense_poc.py (600k POC).

Question: how much can the dense leg's vectors be compressed before search
quality degrades, and what does each option save at 600k / 150M scale?

Method
──────
1. Extract ALL 600k vectors (384-dim fp32, L2-normalized, sfu-academic-embed-v5)
   from the live `openalex_works_dense` index (the exact corpus the dense POC
   was built on) and cache them locally.
2. Encode the same 360 diverse paraphrase queries used by eval_dense_poc.py.
3. Build faiss HNSW indexes (m=16, efConstruction=256 — mirroring the lucene
   HNSW config in index_template_dense.json) over several compressed
   representations of the SAME vectors:

     fp32_hnsw          baseline (what openalex_works_dense uses today)
     fp16_hnsw          scalar fp16            (2x,  ≈ faiss SQfp16 / lucene fp16)
     int8_hnsw          scalar int8            (4x,  ≈ lucene int8 quantization)
     int4_hnsw          scalar int4            (8x,  ≈ lucene int4)  [+fp32 rescore]
     pq48_hnsw          product quantization 48 bytes (32x)          [+fp32 rescore]
     binary_flat        1 bit/dim sign binarization (32x, BBQ-style) [+fp32 rescore]
     trunc192_fp32      Matryoshka-style truncation 384→192 dims (2x)
     trunc192_int8      truncation + int8 (8x)

   "+fp32 rescore" mirrors OpenSearch disk-based / BBQ mode: the compressed
   index produces an oversampled candidate list (4x) that is re-ranked with the
   original fp32 vectors kept on disk.
4. Score every variant three ways:
     a. ANN fidelity:  recall@10 / recall@50 vs exact fp32 brute-force search
        (pure vector-level damage from compression + HNSW).
     b. Dense-leg-only NDCG@10 / MRR@10 / Recall@50 against the LLM-judge cache
        (same metrics + gain function as eval_dense_poc.py).
     c. End-to-end 3-leg RRF(BM25F + SPLADE + dense) NDCG@10, re-fusing each
        variant's dense list with CACHED BM25F/SPLADE rankings from the full
        150M-doc openalex_works index — i.e. the original POC pipeline with
        only the dense leg's compression varied.
5. Report per-variant: bytes/vector, measured faiss index size at 600k,
   extrapolated vector storage at 150.4M docs, recall vs exact, NDCG deltas.

Caveats inherited from eval_dense_poc.py still apply (subset dense coverage,
paraphrase ground-truth reuse, keyword-seeded judgments). Additional caveat:
faiss HNSW is a stand-in for lucene HNSW — same algorithm and parameters, but
implementation details (graph pruning, int8 calibration) differ slightly.

Usage
─────
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    SFU_OPENSEARCH_URL=http://...:9200 \
    .venv/bin/python3 scripts/eval_dense_compression.py \
        [--top-k 50] [--ef-search 128] [--rebuild] [--skip-lexical]
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_dense_compression")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_JUDGE_CACHE = REPO_ROOT / "data/eval_results/llm_judge_cache.json"
DEFAULT_DIVERSE = REPO_ROOT / "data/eval_results/diverse_queries.json"
DEFAULT_OUTPUT = REPO_ROOT / "data/eval_results/dense_compression_eval.json"
CACHE_DIR = REPO_ROOT / "data/dense_compression"
DENSE_MODEL = str(REPO_ROOT / "models/sfu-academic-embed-v5")
SPLADE_ONNX = str(REPO_ROOT / "models/splade_onnx")

DIM = 384
FULL_SCALE_DOCS = 150_413_098  # openalex_works doc count (2026-05-29)
HNSW_M = 16                    # mirrors index_template_dense.json
HNSW_EF_CONSTRUCTION = 256
RESCORE_OVERSAMPLE = 4

RELEVANT_THRESHOLD = 2
K_NDCG = 10
K_RECALL = 50
RRF_K = 60
JUDGE_KEY_LEN = 80


def opensearch_url() -> str:
    return os.environ.get(
        "SFU_OPENSEARCH_URL",
        "http://claudebox-sfu-library-mcp-training-opensearch:9200",
    ).rstrip("/")


# ── Metrics (identical to eval_dense_poc.py) ────────────────────────────────────

def dcg(gains: list[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_grades: list[int], ideal_grades: list[int], k: int) -> float:
    idcg = dcg(sorted(ideal_grades, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return dcg(ranked_grades, k) / idcg


def mrr_at_k(ranked_grades: list[int], k: int) -> float:
    for i, g in enumerate(ranked_grades[:k]):
        if g >= RELEVANT_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


def score_ranking(fused_ids: list[str], gradebook: dict[str, int]) -> tuple[float, float, float]:
    ranked_grades = [gradebook.get(i, 0) for i in fused_ids]
    ideal = list(gradebook.values())
    ndcg = ndcg_at_k(ranked_grades, ideal, K_NDCG)
    mrr = mrr_at_k(ranked_grades, K_NDCG)
    relevant = {i for i, g in gradebook.items() if g >= RELEVANT_THRESHOLD}
    if relevant:
        hit = sum(1 for i in fused_ids[:K_RECALL] if i in relevant)
        recall = hit / len(relevant)
    else:
        recall = 0.0
    return ndcg, mrr, recall


def rrf_fuse_ids(ranked_id_lists: list[list[str]], top_k: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_id_lists:
        for rank, did in enumerate(lst, start=1):
            if did:
                scores[did] += 1.0 / (RRF_K + rank)
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]


def load_judge_grades(path: Path) -> dict[str, dict[str, int]]:
    cache = json.loads(path.read_text())
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    for key, grade in cache.items():
        qkey, did = key.rsplit("||", 1)
        if isinstance(grade, int):
            grades[qkey][did] = grade
    return grades


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ── Step 1: corpus vectors ──────────────────────────────────────────────────────

def fetch_corpus(url: str, index: str) -> tuple[np.ndarray, list[str]]:
    """Scroll all vectors + openalex_ids out of the dense POC index (cached)."""
    vec_path = CACHE_DIR / "vectors_600k.npy"
    ids_path = CACHE_DIR / "ids_600k.json"
    if vec_path.exists() and ids_path.exists():
        logger.info("Loading cached corpus vectors from %s", vec_path)
        return np.load(vec_path), json.loads(ids_path.read_text())

    import requests
    logger.info("Scrolling %s/%s for vectors (one-time, cached afterwards)…", url, index)
    vectors: list[list[float]] = []
    ids: list[str] = []
    body = {"size": 2000, "_source": ["openalex_id", "embedding"],
            "query": {"match_all": {}}}
    resp = requests.post(f"{url}/{index}/_search?scroll=5m", json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    scroll_id = data["_scroll_id"]
    while True:
        hits = data["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            src = h["_source"]
            vectors.append(src["embedding"])
            ids.append(src.get("openalex_id") or h["_id"])
        if len(ids) % 50000 < 2000:
            logger.info("  scrolled %d docs", len(ids))
        resp = requests.post(f"{url}/_search/scroll",
                             json={"scroll": "5m", "scroll_id": scroll_id}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        scroll_id = data["_scroll_id"]
    requests.delete(f"{url}/_search/scroll", json={"scroll_id": scroll_id}, timeout=30)

    X = np.asarray(vectors, dtype=np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, X)
    ids_path.write_text(json.dumps(ids))
    logger.info("Cached %d vectors (%.1f MB fp32) -> %s",
                len(ids), X.nbytes / 1e6, vec_path)
    return X, ids


# ── Step 2: queries ─────────────────────────────────────────────────────────────

def load_scorable_queries(judge_cache: Path, diverse: Path) -> tuple[list[dict], dict]:
    gradebooks = load_judge_grades(judge_cache)
    records = json.loads(diverse.read_text())
    scorable = []
    for rec in records:
        key = rec.get("judge_key", rec["original_query"][:JUDGE_KEY_LEN])
        gb = gradebooks.get(key, {})
        if any(g >= RELEVANT_THRESHOLD for g in gb.values()):
            scorable.append({"paraphrase": rec["paraphrase"],
                             "query_type": rec["query_type"], "judge_key": key})
    logger.info("Scorable queries: %d / %d records", len(scorable), len(records))
    return scorable, gradebooks


def encode_queries(queries: list[dict]) -> np.ndarray:
    q_path = CACHE_DIR / "query_vectors.npy"
    sig_path = CACHE_DIR / "query_vectors.sig"
    sig = str(len(queries)) + "|" + queries[0]["paraphrase"] + "|" + queries[-1]["paraphrase"]
    if q_path.exists() and sig_path.exists() and sig_path.read_text() == sig:
        logger.info("Loading cached query vectors")
        return np.load(q_path)
    from sentence_transformers import SentenceTransformer
    logger.info("Encoding %d queries with %s", len(queries), DENSE_MODEL)
    model = SentenceTransformer(DENSE_MODEL)
    Q = model.encode([q["paraphrase"] for q in queries], batch_size=64,
                     normalize_embeddings=True, show_progress_bar=False)
    Q = np.asarray(Q, dtype=np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(q_path, Q)
    sig_path.write_text(sig)
    return Q


# ── Step 3: cached lexical legs (BM25F + SPLADE from the full 150M index) ───────

def fetch_lexical_legs(queries: list[dict], top_k: int) -> dict[str, dict[str, list[str]]]:
    cache_path = CACHE_DIR / f"lexical_legs_top{top_k}.json"
    if cache_path.exists():
        logger.info("Loading cached BM25F/SPLADE rankings")
        return json.loads(cache_path.read_text())
    from lib.opensearch_retriever import OpenSearchRetriever
    retriever = OpenSearchRetriever(url=opensearch_url(), index="openalex_works",
                                    splade_model_path=SPLADE_ONNX, timeout=30)

    def ids_of(docs: list[dict]) -> list[str]:
        return [d.get("openalex_id") or d.get("doi") or d.get("title") or "" for d in docs]

    legs: dict[str, dict[str, list[str]]] = {}
    for i, q in enumerate(queries):
        text = q["paraphrase"]
        legs[text] = {
            "bm25": ids_of(retriever.search(text, top_k=top_k, mode="bm25f")),
            "splade": ids_of(retriever.search(text, top_k=top_k, mode="splade")),
        }
        if (i + 1) % 25 == 0:
            logger.info("  lexical legs %d/%d", i + 1, len(queries))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(legs))
    return legs


# ── Step 4: compressed index variants ───────────────────────────────────────────

def build_variant(name: str, X: np.ndarray, rebuild: bool):
    """Build (or load) one faiss index over a compressed representation of X.

    Returns (index, search_matrix_transform, codes_bytes_per_vec, build_secs,
             file_size_bytes, is_binary).
    search_matrix_transform maps fp32 query vectors into this index's space.
    """
    import faiss
    faiss.omp_set_num_threads(os.cpu_count() or 8)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    idx_path = CACHE_DIR / f"{name}.faiss"
    n, d = X.shape

    def trunc(M: np.ndarray) -> np.ndarray:
        T = np.ascontiguousarray(M[:, : d // 2])
        norms = np.linalg.norm(T, axis=1, keepdims=True)
        return T / np.maximum(norms, 1e-12)

    specs = {
        # name: (factory/builder, bytes per vector code, query transform)
        "fp32_hnsw":     (lambda: faiss.IndexHNSWFlat(d, HNSW_M), d * 4, None),
        "fp16_hnsw":     (lambda: faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_fp16, HNSW_M), d * 2, None),
        "int8_hnsw":     (lambda: faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, HNSW_M), d, None),
        "int4_hnsw":     (lambda: faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_4bit, HNSW_M), d // 2, None),
        "pq48_hnsw":     (lambda: faiss.IndexHNSWPQ(d, 48, HNSW_M), 48, None),
        "binary_flat":   (None, d // 8, None),  # special-cased below
        "trunc192_fp32": (lambda: faiss.IndexHNSWFlat(d // 2, HNSW_M), (d // 2) * 4, trunc),
        "trunc192_int8": (lambda: faiss.IndexHNSWSQ(d // 2, faiss.ScalarQuantizer.QT_8bit, HNSW_M), d // 2, trunc),
    }
    builder, code_bytes, q_transform = specs[name]
    Xv = q_transform(X) if q_transform else X

    if name == "binary_flat":
        codes = np.packbits((Xv > 0).astype(np.uint8), axis=1)
        if idx_path.exists() and not rebuild:
            index = faiss.read_index_binary(str(idx_path))
            build_secs = 0.0
        else:
            t0 = time.perf_counter()
            index = faiss.IndexBinaryFlat(d)
            index.add(codes)
            build_secs = time.perf_counter() - t0
            faiss.write_index_binary(index, str(idx_path))
        return index, q_transform, code_bytes, build_secs, idx_path.stat().st_size, True

    if idx_path.exists() and not rebuild:
        logger.info("  loading cached index %s", idx_path.name)
        index = faiss.read_index(str(idx_path))
        build_secs = 0.0
    else:
        index = builder()
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        t0 = time.perf_counter()
        if not index.is_trained:
            train_n = min(100_000, n)
            index.train(Xv[:train_n])
        index.add(Xv)
        build_secs = time.perf_counter() - t0
        faiss.write_index(index, str(idx_path))
        logger.info("  built %s in %.0fs", name, build_secs)
    return index, q_transform, code_bytes, build_secs, idx_path.stat().st_size, False


def search_variant(index, Q: np.ndarray, q_transform, is_binary: bool,
                   top_k: int, ef_search: int, rescore_X: np.ndarray | None):
    """Search all queries; optionally rescore an oversampled candidate set with
    the original fp32 vectors (disk-based / BBQ-style two-phase search)."""
    import faiss
    k = top_k * RESCORE_OVERSAMPLE if rescore_X is not None else top_k
    Qv = q_transform(Q) if q_transform else Q
    if is_binary:
        Qb = np.packbits((Qv > 0).astype(np.uint8), axis=1)
        t0 = time.perf_counter()
        _, I = index.search(Qb, k)
        secs = time.perf_counter() - t0
    else:
        faiss.downcast_index(index).hnsw.efSearch = max(ef_search, k)
        t0 = time.perf_counter()
        _, I = index.search(Qv, k)
        secs = time.perf_counter() - t0
    if rescore_X is not None:
        out = np.empty((I.shape[0], top_k), dtype=I.dtype)
        for qi in range(I.shape[0]):
            cand = I[qi][I[qi] >= 0]
            scores = rescore_X[cand] @ Q[qi]
            out[qi, : len(cand[:top_k])] = cand[np.argsort(-scores)][:top_k]
        I = out
    return I[:, :top_k], secs


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Dense HNSW compression eval (600k POC corpus)")
    parser.add_argument("--judge-cache", default=str(DEFAULT_JUDGE_CACHE))
    parser.add_argument("--diverse-queries", default=str(DEFAULT_DIVERSE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dense-index", default="openalex_works_dense")
    parser.add_argument("--top-k", type=int, default=K_RECALL)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--rebuild", action="store_true", help="rebuild cached faiss indexes")
    parser.add_argument("--skip-lexical", action="store_true",
                        help="skip 3-leg RRF scoring (no OpenSearch lexical queries)")
    args = parser.parse_args()

    url = opensearch_url()
    X, ids = fetch_corpus(url, args.dense_index)
    queries, gradebooks = load_scorable_queries(Path(args.judge_cache), Path(args.diverse_queries))
    Q = encode_queries(queries)
    legs = None if args.skip_lexical else fetch_lexical_legs(queries, args.top_k)

    import faiss
    faiss.omp_set_num_threads(os.cpu_count() or 8)

    # Exact fp32 ground truth for ANN-fidelity recall.
    logger.info("Computing exact fp32 brute-force ground truth…")
    flat = faiss.IndexFlatIP(X.shape[1])
    flat.add(X)
    _, GT = flat.search(Q, args.top_k)

    # Rescore variants mirror OpenSearch's two-phase disk mode (fp32 on disk).
    variant_plan = [
        ("fp32_hnsw", False), ("fp16_hnsw", False), ("int8_hnsw", False),
        ("int4_hnsw", False), ("int4_hnsw", True), ("pq48_hnsw", False),
        ("pq48_hnsw", True), ("binary_flat", False), ("binary_flat", True),
        ("trunc192_fp32", False), ("trunc192_int8", False),
    ]

    results: list[dict] = []
    built: dict[str, tuple] = {}
    baseline_3leg: dict[str, float] = {}

    for name, rescore in variant_plan:
        label = name + ("+rescore" if rescore else "")
        logger.info("Variant: %s", label)
        if name not in built:
            built[name] = build_variant(name, X, args.rebuild)
        index, q_transform, code_bytes, build_secs, file_size, is_binary = built[name]
        I, search_secs = search_variant(index, Q, q_transform, is_binary,
                                        args.top_k, args.ef_search,
                                        X if rescore else None)

        # (a) ANN fidelity vs exact fp32.
        rec10 = mean([len(set(I[i, :10]) & set(GT[i, :10])) / 10 for i in range(len(Q))])
        rec50 = mean([len(set(I[i]) & set(GT[i])) / args.top_k for i in range(len(Q))])

        # (b)+(c) judge-cache quality, per slice.
        acc = {s: {"nd": [], "nm": [], "nr": [], "n3": []}
               for s in ("keyword", "natural", "overall")}
        for qi, q in enumerate(queries):
            gb = gradebooks[q["judge_key"]]
            dense_ids = [ids[j] for j in I[qi] if j >= 0]
            nd, nm, nr = score_ranking(dense_ids, gb)
            for s in (q["query_type"], "overall"):
                acc[s]["nd"].append(nd)
                acc[s]["nm"].append(nm)
                acc[s]["nr"].append(nr)
            if legs is not None:
                leg = legs[q["paraphrase"]]
                fused3 = rrf_fuse_ids([leg["bm25"], leg["splade"], dense_ids], args.top_k)
                n3, _, _ = score_ranking(fused3, gb)
                for s in (q["query_type"], "overall"):
                    acc[s]["n3"].append(n3)

        row = {
            "variant": label,
            "code_bytes_per_vector": code_bytes,
            "compression_vs_fp32": round(DIM * 4 / code_bytes, 1),
            "index_file_size_mb_600k": round(file_size / 1e6, 1),
            "est_vector_storage_gb_150M": round(code_bytes * FULL_SCALE_DOCS / 1e9, 1),
            "est_hnsw_graph_gb_150M": round(HNSW_M * 2 * 4 * FULL_SCALE_DOCS / 1e9, 1),
            "needs_fp32_on_disk_for_rescore": rescore,
            "build_secs": round(build_secs, 1),
            "search_ms_per_query": round(search_secs * 1000 / len(Q), 2),
            "ann_recall@10_vs_exact": round(rec10, 4),
            "ann_recall@50_vs_exact": round(rec50, 4),
        }
        for s in ("keyword", "natural", "overall"):
            row[f"dense_ndcg@10_{s}"] = round(mean(acc[s]["nd"]), 4)
            row[f"dense_recall@50_{s}"] = round(mean(acc[s]["nr"]), 4)
            if legs is not None:
                row[f"rrf3_ndcg@10_{s}"] = round(mean(acc[s]["n3"]), 4)
        if legs is not None:
            if label == "fp32_hnsw":
                baseline_3leg = {s: row[f"rrf3_ndcg@10_{s}"] for s in ("keyword", "natural", "overall")}
            if baseline_3leg:
                for s in ("keyword", "natural", "overall"):
                    row[f"rrf3_ndcg@10_delta_{s}"] = round(
                        row[f"rrf3_ndcg@10_{s}"] - baseline_3leg[s], 4)
        results.append(row)

    out = {
        "config": {
            "derived_from": "scripts/eval_dense_poc.py (600k dense POC)",
            "corpus": {"index": args.dense_index, "n_vectors": len(ids), "dim": DIM,
                       "model": "sfu-academic-embed-v5 (L2-normalized)"},
            "hnsw": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION,
                     "ef_search": args.ef_search},
            "rescore_oversample": RESCORE_OVERSAMPLE,
            "top_k": args.top_k, "rrf_k": RRF_K, "ndcg_k": K_NDCG,
            "n_queries_scored": len(queries),
            "lexical_legs": "cached BM25F+SPLADE top-50 from openalex_works (150M)"
                            if legs is not None else "skipped",
            "full_scale_docs": FULL_SCALE_DOCS,
            "gain": "2^grade - 1",
        },
        "caveats": [
            "faiss HNSW stands in for lucene HNSW (same M/efConstruction; minor impl differences).",
            "Dense indexes a SUBSET -> dense contribution is a lower bound (as in eval_dense_poc).",
            "int8/int4 here use faiss ScalarQuantizer (per-dim min/max training); lucene's "
            "confidence-interval quantization typically performs slightly better.",
            "Graph overhead estimate = M*2*4 bytes/vector (base layer); upper layers add ~5%.",
            "+rescore rows require original fp32 vectors on disk (not in RAM), like OpenSearch "
            "disk-based vector search / BBQ two-phase retrieval.",
        ],
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 132)
    print("DENSE HNSW COMPRESSION EVAL — 600k POC corpus, %d queries" % len(queries))
    print("=" * 132)
    hdr = (f"{'variant':<22} {'B/vec':>6} {'x':>5} {'600k MB':>8} {'150M GB':>8} "
           f"{'R@10ex':>7} {'R@50ex':>7} {'dNDCG ovr':>9} {'dNDCG nat':>9} "
           f"{'3leg ovr':>8} {'d3leg':>7} {'ms/q':>6}")
    print(hdr)
    print("-" * 132)
    for r in results:
        print(f"{r['variant']:<22} {r['code_bytes_per_vector']:>6} "
              f"{r['compression_vs_fp32']:>4.0f}x {r['index_file_size_mb_600k']:>8.1f} "
              f"{r['est_vector_storage_gb_150M']:>8.1f} "
              f"{r['ann_recall@10_vs_exact']:>7.4f} {r['ann_recall@50_vs_exact']:>7.4f} "
              f"{r['dense_ndcg@10_overall']:>9.4f} {r['dense_ndcg@10_natural']:>9.4f} "
              f"{r.get('rrf3_ndcg@10_overall', float('nan')):>8.4f} "
              f"{r.get('rrf3_ndcg@10_delta_overall', float('nan')):>+7.4f} "
              f"{r['search_ms_per_query']:>6.2f}")
    print("-" * 132)
    print("B/vec = compressed vector code size. 150M GB = vector storage extrapolated to the full "
          f"{FULL_SCALE_DOCS / 1e6:.0f}M-doc corpus (graph adds ~{HNSW_M * 2 * 4 * FULL_SCALE_DOCS / 1e9:.0f} GB at M={HNSW_M}).")
    print("R@k ex = ANN recall vs exact fp32 search. dNDCG = dense-leg-only NDCG@10 (judge cache).")
    print("3leg = RRF(BM25F+SPLADE+dense) NDCG@10; d3leg = delta vs fp32_hnsw baseline.")
    print("=" * 132)
    logger.info("Wrote results -> %s", args.output)


if __name__ == "__main__":
    main()
