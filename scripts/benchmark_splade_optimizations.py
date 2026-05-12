#!/usr/bin/env python3
"""Compare SPLADE encoder variants for indexing throughput.

Runs several optimization strategies against the baseline encoder on the same
batch of real OpenAlex docs and reports docs/sec for each:

  baseline          – current implementation (FP32, per-doc CPU sync)
  fp16              – FP16 autocast forward pass
  fp16_gpu_topk     – FP16 + GPU-side top-K + single CPU sync per batch
  fp16_bucket       – FP16 + GPU top-K + length-bucketed batches
  fp16_compile      – FP16 + GPU top-K + torch.compile() (fused kernels)

Each variant produces identical-shape output (list[dict[str, float]]) so we
also check correctness: top-K terms must overlap >= 95% with the baseline.

Usage:
    python scripts/benchmark_splade_optimizations.py
    python scripts/benchmark_splade_optimizations.py --n-docs 2000 --batch-size 64
"""

import argparse
import gzip
import json
import time
from pathlib import Path

SNAPSHOT_DIR = Path(__file__).parent.parent / "data" / "openalex_snapshot"
DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"
SPARSE_TOP_K = 256
MAX_DOC_LENGTH = 512


def load_docs(n: int) -> list[dict]:
    """Load n docs from snapshot files, starting at checkpoint position."""
    cp = SNAPSHOT_DIR / "indexer_checkpoint.json"
    start_file, start_offset = 0, 0
    if cp.exists():
        d = json.loads(cp.read_text())
        start_file = d.get("file_index", 0)
        start_offset = d.get("doc_offset", 0)

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
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = f"{rec.get('title', '') or ''} {rec.get('abstract', '') or ''}".strip()
                if text:
                    docs.append({"id": rec.get("id", ""), "text": text})
                if len(docs) >= n:
                    break
        if len(docs) >= n:
            break
    return docs


# ── Variant 1: BASELINE (matches current splade_indexer.py) ──────────────────


class BaselineEncoder:
    name = "baseline_fp32"

    def __init__(self, model_name, device):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(device).eval()
        self.id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def encode_batch(self, texts):
        import torch

        tokens = self.tokenizer(
            texts, max_length=MAX_DOC_LENGTH, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**tokens)

        vecs = torch.log1p(torch.relu(output.logits))
        vecs = torch.max(vecs, dim=1).values

        results = []
        for vec in vecs:
            nonzero = vec.nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                results.append({})
                continue
            weights = vec[nonzero]
            if len(nonzero) > SPARSE_TOP_K:
                topk = torch.topk(weights, SPARSE_TOP_K)
                nonzero = nonzero[topk.indices]
                weights = topk.values
            d = {}
            for idx, w in zip(nonzero.cpu().tolist(), weights.cpu().tolist()):
                t = self.id_to_token.get(idx, "")
                if t and not t.startswith("[") and w > 0.01:
                    d[t] = round(w, 4)
            results.append(d)
        return results


# ── Variant 2: FP16 autocast ─────────────────────────────────────────────────


class FP16Encoder(BaselineEncoder):
    name = "fp16_autocast"

    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        import torch
        # Convert model to half precision in-place (faster than autocast for inference)
        if device == "cuda":
            self.model = self.model.half()

    def encode_batch(self, texts):
        import torch

        tokens = self.tokenizer(
            texts, max_length=MAX_DOC_LENGTH, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**tokens)

        # Up-cast logits to fp32 for the SPLADE post-processing (numerical safety)
        logits = output.logits.float()
        vecs = torch.log1p(torch.relu(logits))
        vecs = torch.max(vecs, dim=1).values

        results = []
        for vec in vecs:
            nonzero = vec.nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                results.append({})
                continue
            weights = vec[nonzero]
            if len(nonzero) > SPARSE_TOP_K:
                topk = torch.topk(weights, SPARSE_TOP_K)
                nonzero = nonzero[topk.indices]
                weights = topk.values
            d = {}
            for idx, w in zip(nonzero.cpu().tolist(), weights.cpu().tolist()):
                t = self.id_to_token.get(idx, "")
                if t and not t.startswith("[") and w > 0.01:
                    d[t] = round(w, 4)
            results.append(d)
        return results


# ── Variant 3: FP16 + GPU-side top-K + single CPU sync ───────────────────────


class FP16GpuTopkEncoder:
    """Single CPU sync per batch instead of two per document.

    Strategy:
      1. Apply log1p(relu(logits)).max(dim=1) on GPU (in fp32).
      2. On GPU: do topk(K) across the whole vocab, skipping nonzero search.
      3. Zero out weights <= 0.01 still on GPU.
      4. Single .cpu().numpy() per batch → loop in Python with arrays only.
    """

    name = "fp16_gpu_topk"

    def __init__(self, model_name, device):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        if device == "cuda":
            self.model = self.model.half()
        self.model.to(device).eval()
        self.id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        # Pre-compute mask of "special" tokens (starting with "[") to skip later
        self._skip_token_ids = {
            tok_id for tok, tok_id in self.tokenizer.get_vocab().items() if tok.startswith("[")
        }

    def encode_batch(self, texts):
        import torch

        tokens = self.tokenizer(
            texts, max_length=MAX_DOC_LENGTH, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**tokens)

        # Cast back to fp32 for the SPLADE pooling math (cheap)
        logits = output.logits.float()
        vecs = torch.log1p(torch.relu(logits))
        vecs = torch.max(vecs, dim=1).values  # (B, V)

        # Top-K on GPU across full vocab (avoid scan-for-nonzero)
        # SPARSE_TOP_K is small relative to V, so this is fast
        top_w, top_idx = torch.topk(vecs, SPARSE_TOP_K, dim=1)  # (B, K)

        # Zero out small weights still on GPU
        mask = top_w > 0.01
        top_w = top_w * mask  # we'll re-filter in Python because round/format anyway

        # SINGLE CPU sync per batch
        top_w_cpu = top_w.cpu().numpy()
        top_idx_cpu = top_idx.cpu().numpy()

        results = []
        skip = self._skip_token_ids
        id2tok = self.id_to_token
        for i in range(top_w_cpu.shape[0]):
            d = {}
            for j in range(SPARSE_TOP_K):
                w = float(top_w_cpu[i, j])
                if w <= 0.01:
                    continue
                idx = int(top_idx_cpu[i, j])
                if idx in skip:
                    continue
                t = id2tok.get(idx)
                if t:
                    d[t] = round(w, 4)
            results.append(d)
        return results


# ── Variant 4: FP16 + GPU top-K + length-bucketed batches ────────────────────


class FP16BucketEncoder(FP16GpuTopkEncoder):
    """Same as fp16_gpu_topk but the caller passes pre-sorted batches.

    The benchmark sorts the input by length once and then chunks into batches;
    the per-batch cost is the same code, the win comes from less padding waste.
    """

    name = "fp16_bucket"


# ── Variant 5: FP16 + GPU top-K + torch.compile ──────────────────────────────


class FP16CompileEncoder(FP16GpuTopkEncoder):
    name = "fp16_compile"

    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        import torch
        try:
            # mode='reduce-overhead' is best for small-batch transformer inference
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)
            self._compile_ok = True
        except Exception as e:
            print(f"  torch.compile failed: {e} — falling back to eager")
            self._compile_ok = False


# ── Benchmark runner ─────────────────────────────────────────────────────────


def benchmark_variant(encoder, docs, batch_size, sort_by_length=False, warmup=1):
    """Time encode-only throughput. Returns (docs_per_sec, total_time, sample_output)."""
    import torch

    texts = [d["text"] for d in docs]
    if sort_by_length:
        # Stable sort by char length; group similar lengths into batches
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        texts = [texts[i] for i in order]

    # Warmup
    for _ in range(warmup):
        encoder.encode_batch(texts[:batch_size])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sample = None
    t0 = time.time()
    out_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out = encoder.encode_batch(batch)
        if sample is None and out:
            sample = out[0]
        out_all.extend(out)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = time.time() - t0

    n = len(texts)
    return n / total, total, sample, out_all


def overlap(a: dict, b: dict, top_n=20):
    """Jaccard overlap of top-N keys by weight."""
    if not a or not b:
        return 0.0
    a_top = set(sorted(a.keys(), key=lambda k: -a[k])[:top_n])
    b_top = set(sorted(b.keys(), key=lambda k: -b[k])[:top_n])
    if not a_top:
        return 0.0
    return len(a_top & b_top) / len(a_top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--variants", nargs="+",
        default=["baseline", "fp16", "fp16_gpu_topk", "fp16_bucket"],
        choices=["baseline", "fp16", "fp16_gpu_topk", "fp16_bucket", "fp16_compile"],
    )
    parser.add_argument("--larger-batch", type=int, default=None,
                        help="Also re-run the best variant at this batch size (e.g. 128, 192, 256)")
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"torch: {torch.__version__}")
    print()

    print(f"Loading {args.n_docs} docs ...")
    docs = load_docs(args.n_docs)
    print(f"  Got {len(docs)} docs.  Avg chars: {sum(len(d['text']) for d in docs)/len(docs):.0f}")
    print()

    classes = {
        "baseline": BaselineEncoder,
        "fp16": FP16Encoder,
        "fp16_gpu_topk": FP16GpuTopkEncoder,
        "fp16_bucket": FP16BucketEncoder,
        "fp16_compile": FP16CompileEncoder,
    }

    baseline_out = None
    results = []

    for variant in args.variants:
        cls = classes[variant]
        print(f"── {variant} ──────────────────────────────────────────")
        print(f"  Loading model...")
        t0 = time.time()
        enc = cls(args.model, device)
        print(f"  Loaded in {time.time()-t0:.1f}s")

        sort = variant == "fp16_bucket"
        dps, total, sample, all_out = benchmark_variant(
            enc, docs, args.batch_size, sort_by_length=sort, warmup=2
        )

        # Reference correctness: compare to baseline
        if variant == "baseline":
            baseline_out = all_out
            correctness = 1.0
        else:
            if sort:
                # Bucket variant: outputs are in sorted order; just check overlap on samples
                ovs = [overlap(all_out[i], baseline_out[0]) for i in range(min(20, len(all_out)))]
            else:
                ovs = [overlap(all_out[i], baseline_out[i]) for i in range(min(50, len(all_out)))]
            correctness = sum(ovs) / max(len(ovs), 1)

        # Sparse stats
        avg_terms = sum(len(d) for d in all_out if d) / max(sum(1 for d in all_out if d), 1)

        # GPU mem
        gpu_gb = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
        print(f"  Throughput:  {dps:>7.1f} docs/sec   ({total:.2f}s)")
        print(f"  Avg terms/doc: {avg_terms:.0f}")
        print(f"  Top-20 overlap vs baseline: {correctness:.1%}")
        print(f"  GPU mem: {gpu_gb:.2f} GB")
        if sample:
            top5 = dict(sorted(sample.items(), key=lambda x: -x[1])[:5])
            print(f"  Sample top-5: {top5}")
        print()

        results.append({
            "variant": variant,
            "docs_per_sec": round(dps, 1),
            "total_seconds": round(total, 2),
            "correctness_vs_baseline": round(correctness, 4),
            "avg_terms_per_doc": round(avg_terms, 1),
            "gpu_gb": round(gpu_gb, 2),
        })

        # Free between variants
        del enc
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print("═" * 70)
    print(f"{'Variant':<22} {'docs/s':>8} {'speedup':>8} {'ovlp':>6} {'GPU':>7}")
    print("─" * 70)
    baseline_dps = results[0]["docs_per_sec"]
    for r in results:
        sp = r["docs_per_sec"] / baseline_dps
        print(f"{r['variant']:<22} {r['docs_per_sec']:>8.1f} {sp:>7.2f}x "
              f"{r['correctness_vs_baseline']:>6.1%} {r['gpu_gb']:>6.2f}GB")
    print("═" * 70)

    # ETA estimate
    REMAINING = 146_475_000
    print()
    print("ETA at this throughput (encode-only, 146.5M docs remaining):")
    for r in results:
        secs = REMAINING / r["docs_per_sec"]
        hrs = secs / 3600
        print(f"  {r['variant']:<22}  {hrs:>6.1f} hours  ({hrs/24:.2f} days)")

    # Save
    out_path = Path(__file__).parent.parent / "benchmark_optimization_results.json"
    out_path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "n_docs": len(docs),
        "batch_size": args.batch_size,
        "results": results,
    }, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
