#!/usr/bin/env python3
"""Benchmark the SPLADE indexer: measures encode and OpenSearch upload throughput separately.

Runs 5000 docs through the real pipeline (GPU encode + live OpenSearch bulk),
reports per-phase timings, and extrapolates a full-run ETA.

Usage:
    python scripts/benchmark_indexer.py
    python scripts/benchmark_indexer.py --n-docs 10000 --batch-size 128
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import requests

SNAPSHOT_DIR = Path(__file__).parent.parent / "data" / "openalex_snapshot"
DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"
OPENSEARCH_URL = os.environ.get("SFU_OPENSEARCH_URL", "http://localhost:9200")
INDEX = os.environ.get("SFU_OPENSEARCH_INDEX", "openalex_works")
SPARSE_TOP_K = 256
MAX_DOC_LENGTH = 512


def load_docs(n: int) -> list[dict]:
    """Load n docs from snapshot files, skipping already-indexed ones."""
    checkpoint_path = SNAPSHOT_DIR / "indexer_checkpoint.json"
    start_file = 0
    start_offset = 0
    if checkpoint_path.exists():
        cp = json.loads(checkpoint_path.read_text())
        start_file = cp.get("file_index", 0)
        start_offset = cp.get("doc_offset", 0)

    files = sorted(SNAPSHOT_DIR.glob("works_part_*.jsonl.gz"))
    docs = []
    for file_idx, path in enumerate(files):
        if file_idx < start_file:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if file_idx == start_file and line_no < start_offset:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(docs) >= n:
                    break
        if len(docs) >= n:
            break
    return docs


def load_encoder(model_name: str):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SPLADE model '{model_name}' on {device} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    load_secs = time.time() - t0
    print(f"Model loaded in {load_secs:.1f}s — vocab: {len(vocab)} tokens")
    if device == "cuda":
        import torch
        print(f"GPU VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    return tokenizer, model, device, id_to_token


def encode_batch(tokenizer, model, device, id_to_token, texts: list[str]) -> list[dict]:
    import torch

    tokens = tokenizer(
        texts,
        max_length=MAX_DOC_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**tokens)

    splade_vecs = torch.log1p(torch.relu(output.logits))
    splade_vecs = torch.max(splade_vecs, dim=1).values

    results = []
    for vec in splade_vecs:
        nonzero = vec.nonzero(as_tuple=True)[0]
        if len(nonzero) == 0:
            results.append({})
            continue
        weights = vec[nonzero]
        if len(nonzero) > SPARSE_TOP_K:
            topk = torch.topk(weights, SPARSE_TOP_K)
            nonzero = nonzero[topk.indices]
            weights = topk.values
        sparse_dict = {}
        for idx, weight in zip(nonzero.cpu().tolist(), weights.cpu().tolist()):
            token = id_to_token.get(idx, "")
            if token and not token.startswith("[") and weight > 0.01:
                sparse_dict[token] = round(weight, 4)
        results.append(sparse_dict)
    return results


def bulk_upload(session: requests.Session, docs: list[dict]) -> float:
    """Upload docs to OpenSearch. Returns seconds taken."""
    lines = []
    for doc in docs:
        lines.append(json.dumps({"index": {"_index": INDEX, "_id": doc.get("id", "")}}))
        lines.append(json.dumps(doc))
    payload = "\n".join(lines) + "\n"

    t0 = time.time()
    resp = session.post(
        f"{OPENSEARCH_URL}/_bulk",
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
        timeout=120,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    return elapsed


def run_benchmark(n_docs: int, batch_size: int, bulk_size: int):
    print(f"\n{'='*60}")
    print(f"SPLADE Indexer Benchmark")
    print(f"  n_docs={n_docs}, batch_size={batch_size}, bulk_size={bulk_size}")
    print(f"  OpenSearch: {OPENSEARCH_URL}/{INDEX}")
    print(f"{'='*60}\n")

    # ── Load docs ────────────────────────────────────────────────────────────
    print(f"Loading {n_docs} docs from snapshot...")
    t0 = time.time()
    docs = load_docs(n_docs)
    load_time = time.time() - t0
    print(f"  Loaded {len(docs)} docs in {load_time:.1f}s\n")

    if len(docs) < n_docs:
        print(f"  Warning: only {len(docs)} docs available, continuing with that.")
    n_docs = len(docs)

    # ── Load model ───────────────────────────────────────────────────────────
    tokenizer, model, device, id_to_token = load_encoder(DEFAULT_MODEL)

    # ── OpenSearch connectivity ───────────────────────────────────────────────
    session = requests.Session()
    try:
        r = session.get(f"{OPENSEARCH_URL}/_cluster/health", timeout=10)
        health = r.json()
        print(f"\nOpenSearch: status={health['status']}, nodes={health['number_of_nodes']}")
    except Exception as e:
        print(f"\nOpenSearch unreachable: {e}")
        print("Benchmark will measure encode-only throughput.")
        session = None

    # Check current index settings
    if session:
        try:
            r = session.get(f"{OPENSEARCH_URL}/{INDEX}/_settings", timeout=10)
            s = r.json().get(INDEX, {}).get("settings", {}).get("index", {})
            print(f"  Index settings: refresh_interval={s.get('refresh_interval','default')}, "
                  f"replicas={s.get('number_of_replicas','?')}, shards={s.get('number_of_shards','?')}")
        except Exception:
            pass

    print()

    # ── Warmup (1 batch, not measured) ───────────────────────────────────────
    print("Warming up (1 batch)...")
    warmup_texts = [
        f"{d.get('title', '')} {d.get('abstract', '')}".strip()
        for d in docs[:batch_size] if d.get('title') or d.get('abstract')
    ][:batch_size]
    if warmup_texts:
        encode_batch(tokenizer, model, device, id_to_token, warmup_texts)
    print("  Done.\n")

    # ── Timed encode pass ─────────────────────────────────────────────────────
    print(f"Encoding {n_docs} docs in batches of {batch_size}...")
    encode_times = []
    all_sparse = []
    all_valid_docs = []

    encode_start = time.time()
    i = 0
    while i < n_docs:
        batch = docs[i: i + batch_size]
        texts, valid = [], []
        for d in batch:
            t = f"{d.get('title', '')} {d.get('abstract', '')}".strip()
            if t:
                texts.append(t)
                valid.append(d)
        if texts:
            t0 = time.time()
            sparse = encode_batch(tokenizer, model, device, id_to_token, texts)
            encode_times.append(time.time() - t0)
            all_sparse.extend(sparse)
            all_valid_docs.extend(valid)
        i += len(batch)

    total_encode_time = time.time() - encode_start
    encoded_docs = len(all_valid_docs)
    encode_dps = encoded_docs / total_encode_time if total_encode_time > 0 else 0
    avg_batch_encode = (sum(encode_times) / len(encode_times)) if encode_times else 0

    print(f"\n  Encoded: {encoded_docs:,} docs")
    print(f"  Total encode time: {total_encode_time:.2f}s")
    print(f"  Encode throughput: {encode_dps:.0f} docs/sec")
    print(f"  Avg batch encode time: {avg_batch_encode*1000:.1f}ms ({batch_size} docs/batch)")

    # ── Timed upload pass ─────────────────────────────────────────────────────
    upload_dps = None
    total_upload_time = None
    if session:
        # Build OS docs
        os_docs = []
        for d, sparse in zip(all_valid_docs, all_sparse):
            if sparse:
                os_docs.append({
                    "id": d.get("id", ""),
                    "title": d.get("title", ""),
                    "abstract": d.get("abstract", ""),
                    "publication_year": d.get("publication_year"),
                    "type": d.get("type", ""),
                    "openalex_id": d.get("id", ""),
                    "sparse_field": sparse,
                })

        print(f"\nUploading {len(os_docs):,} docs to OpenSearch in bulks of {bulk_size}...")
        upload_times = []
        upload_start = time.time()
        j = 0
        while j < len(os_docs):
            bulk = os_docs[j: j + bulk_size]
            try:
                t = bulk_upload(session, bulk)
                upload_times.append(t)
            except Exception as e:
                print(f"  Upload error at doc {j}: {e}")
            j += len(bulk)

        total_upload_time = time.time() - upload_start
        upload_dps = len(os_docs) / total_upload_time if total_upload_time > 0 else 0
        avg_bulk_time = (sum(upload_times) / len(upload_times)) if upload_times else 0
        print(f"\n  Uploaded: {len(os_docs):,} docs")
        print(f"  Total upload time: {total_upload_time:.2f}s")
        print(f"  Upload throughput: {upload_dps:.0f} docs/sec")
        print(f"  Avg bulk time: {avg_bulk_time*1000:.1f}ms ({bulk_size} docs/bulk = {bulk_size//batch_size} batches/bulk)")

    # ── Combined pipeline throughput ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("THROUGHPUT SUMMARY")
    print(f"{'='*60}")
    print(f"  Encode-only:    {encode_dps:>8.0f} docs/sec  ({total_encode_time:.1f}s for {encoded_docs:,} docs)")
    if upload_dps:
        combined_time = total_encode_time + total_upload_time
        combined_dps = encoded_docs / combined_time
        print(f"  Upload-only:    {upload_dps:>8.0f} docs/sec  ({total_upload_time:.1f}s for {encoded_docs:,} docs)")
        print(f"  Combined (sequential): {combined_dps:>6.0f} docs/sec")
        print(f"  Upload is {total_upload_time/total_encode_time:.1f}x the encode time")
        bottleneck = "UPLOAD" if total_upload_time > total_encode_time else "ENCODE"
        print(f"  Bottleneck: {bottleneck}")
    print()

    # ── ETA extrapolation ─────────────────────────────────────────────────────
    REMAINING = 146_475_000
    print(f"ETA EXTRAPOLATION  (remaining ≈ {REMAINING/1e6:.1f}M docs)")
    print(f"{'='*60}")

    def fmt_hours(secs):
        h = secs / 3600
        return f"{h:.1f} hours ({h/24:.1f} days)"

    if upload_dps:
        combined_time = total_encode_time + total_upload_time
        combined_dps = encoded_docs / combined_time
        print(f"  Current pipeline (encode+upload sequential):")
        print(f"    {combined_dps:.0f} docs/sec → {fmt_hours(REMAINING / combined_dps)}")
        print()
        # With async overlap: bottleneck becomes max(encode, upload)
        bottleneck_dps = min(encode_dps, upload_dps)
        print(f"  With async encode/upload overlap:")
        print(f"    ~{bottleneck_dps:.0f} docs/sec → {fmt_hours(REMAINING / bottleneck_dps)}")
        print()
        # With 10x larger bulk (accumulate batches before uploading)
        n_bulks_now = encoded_docs / bulk_size
        overhead_per_bulk = 0.005  # ~5ms HTTP overhead per bulk request
        upload_overhead_now = n_bulks_now * overhead_per_bulk
        new_bulk_size = bulk_size * 16
        n_bulks_new = encoded_docs / new_bulk_size
        upload_overhead_new = n_bulks_new * overhead_per_bulk
        saved = upload_overhead_now - upload_overhead_new
        print(f"  With bulk_size {bulk_size} → {new_bulk_size} (batching {16} GPU batches):")
        bulk_reduction_factor = bulk_size / new_bulk_size
        # Upload time scales roughly linearly with number of HTTP requests for small payloads
        upload_time_new = total_upload_time * bulk_reduction_factor + (encoded_docs / new_bulk_size) * overhead_per_bulk
        combined_time_new = max(total_encode_time, upload_time_new)  # if async
        if combined_time_new > 0:
            new_combined_dps = encoded_docs / (total_encode_time + upload_time_new)
            print(f"    ~{new_combined_dps:.0f} docs/sec → {fmt_hours(REMAINING / new_combined_dps)}")

    else:
        print(f"  Encode-only (no OpenSearch):  {fmt_hours(REMAINING / encode_dps)}")

    print()
    print(f"  Note: These are conservative estimates assuming constant throughput.")
    print(f"  Actual may vary with doc length distribution across files.")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"n_docs": n_docs, "batch_size": batch_size, "bulk_size": bulk_size, "device": device},
        "encode_dps": round(encode_dps, 1),
        "encode_total_seconds": round(total_encode_time, 2),
        "upload_dps": round(upload_dps, 1) if upload_dps else None,
        "upload_total_seconds": round(total_upload_time, 2) if total_upload_time else None,
        "remaining_docs": REMAINING,
    }
    out_path = Path(__file__).parent.parent / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SPLADE indexer throughput")
    parser.add_argument("--n-docs", type=int, default=5000, help="Docs to benchmark (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=64, help="GPU encode batch size (default: 64)")
    parser.add_argument("--bulk-size", type=int, default=64, help="OpenSearch bulk request size (default: 64 = 1 batch)")
    args = parser.parse_args()
    run_benchmark(args.n_docs, args.batch_size, args.bulk_size)


if __name__ == "__main__":
    main()
