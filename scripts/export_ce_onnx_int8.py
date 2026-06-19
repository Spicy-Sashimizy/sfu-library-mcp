#!/usr/bin/env python3
"""Export the SFU cross-encoder (rerank) to ONNX + dynamic int8, and benchmark it
against the torch-CPU baseline.

Why: the demo's CPU rerank (~5 s warm/query reported) is the serving bottleneck at
5-15 concurrent (HYBRID_DEMO_DEPLOYMENT.md risk #4). int8 ONNX is the highest-leverage
pre-demo throughput lever and is fully independent of the BMP->Qdrant migration.

Measures (records to data/eval_results/):
  * latency/pair + throughput: torch fp32 CPU vs ONNX int8
  * score parity: Pearson r + max|Δlogit| between the two (sanity that int8 doesn't
    wreck ranking). NOTE: this is NOT an NDCG eval — full quality parity needs the
    LLM-judge harness (eval_cross_encoder.py); that is out of scope here.

Usage:
  .venv/bin/python3 scripts/export_ce_onnx_int8.py [--model DIR] [--out DIR] [--pairs N]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone

MODEL_DIR = "models/sfu-cross-encoder-v1"
OUT_DIR = "models/sfu-cross-encoder-v1-onnx-int8"
RESULTS_DIR = "data/eval_results"


def _pairs(n: int) -> list[tuple[str, str]]:
    """Realistic academic (query, title+abstract) rerank pairs."""
    queries = [
        "transformer models for protein structure prediction",
        "carbon capture using metal organic frameworks",
        "reinforcement learning for robotic manipulation",
        "social determinants of mental health in adolescents",
        "quantum error correction surface codes",
        "CRISPR off-target effects in gene therapy",
        "federated learning privacy guarantees",
        "machine translation low resource languages",
    ]
    docs = [
        "AlphaFold2 attention-based architecture for predicting 3D protein folds from sequence.",
        "Porous MOFs with high CO2 uptake and selectivity for post-combustion capture.",
        "Sim-to-real transfer of dexterous grasping policies with domain randomization.",
        "Longitudinal cohort study linking socioeconomic status to depression in teens.",
        "Threshold theorems and decoding of topological surface codes under noise.",
        "Genome-wide profiling of Cas9 cleavage at mismatched target sites.",
        "Differential privacy bounds for gradient aggregation across distributed clients.",
        "Cross-lingual pretraining and back-translation for under-resourced NMT.",
    ]
    out = []
    for i in range(n):
        out.append((queries[i % len(queries)], docs[(i * 3) % len(docs)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 8)
    tok = AutoTokenizer.from_pretrained(args.model)
    pairs = _pairs(args.pairs)

    # --- 1. export fp32 ONNX ---
    print(f">> exporting {args.model} -> ONNX (fp32) ...")
    ort_fp32 = ORTModelForSequenceClassification.from_pretrained(args.model, export=True)
    ort_fp32.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    # --- 2. dynamic int8 (AVX2; this host has no avx512_vnni) ---
    print(">> quantizing -> dynamic int8 (avx2) ...")
    quantizer = ORTQuantizer.from_pretrained(args.out)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=args.out, quantization_config=qconfig)
    # optimum writes model_quantized.onnx; load it explicitly
    ort_int8 = ORTModelForSequenceClassification.from_pretrained(
        args.out, file_name="model_quantized.onnx")

    # --- 3. torch baseline ---
    torch_model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()

    def enc(batch):
        a = [p[0] for p in batch]
        b = [p[1] for p in batch]
        return tok(a, b, padding=True, truncation=True, max_length=256, return_tensors="pt")

    def torch_scores(batch):
        with torch.no_grad():
            return torch_model(**enc(batch)).logits.squeeze(-1).cpu().numpy()

    def onnx_scores(batch):
        return ort_int8(**enc(batch)).logits.squeeze(-1)

    # --- 4. score parity ---
    s_torch = np.asarray(torch_scores(pairs), dtype=np.float64).ravel()
    s_onnx = np.asarray(onnx_scores(pairs), dtype=np.float64).ravel()
    pear = float(np.corrcoef(s_torch, s_onnx)[0, 1])
    max_abs = float(np.max(np.abs(s_torch - s_onnx)))
    # ranking parity: do the two agree on the top-half ordering?
    rank_torch = np.argsort(-s_torch)
    rank_onnx = np.argsort(-s_onnx)
    topk = max(1, len(pairs) // 2)
    jaccard = len(set(rank_torch[:topk]) & set(rank_onnx[:topk])) / topk

    # --- 5. latency (per-pair, batch=1, single-stream) ---
    def bench(fn) -> dict:
        for _ in range(args.warmup):
            fn(pairs[:1])
        lat = []
        for _ in range(args.iters):
            for p in pairs:
                t0 = time.perf_counter()
                fn([p])
                lat.append((time.perf_counter() - t0) * 1000.0)
        lat.sort()
        return {
            "p50_ms": round(statistics.median(lat), 3),
            "p95_ms": round(lat[int(len(lat) * 0.95)], 3),
            "mean_ms": round(statistics.mean(lat), 3),
            "throughput_pairs_s": round(1000.0 / statistics.mean(lat), 1),
        }

    print(">> benchmarking torch fp32 ...")
    b_torch = bench(torch_scores)
    print(">> benchmarking onnx int8 ...")
    b_onnx = bench(onnx_scores)

    def dirsize(path, pattern):
        import glob
        return round(sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, pattern))) / 1e6, 1)

    result = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": args.model,
        "out": args.out,
        "host": {"cpu_count": os.cpu_count(), "isa": "avx2", "quant": "dynamic int8 per-channel"},
        "n_pairs": len(pairs),
        "size_mb": {
            "torch_safetensors": dirsize(args.model, "*.safetensors"),
            "onnx_fp32": dirsize(args.out, "model.onnx"),
            "onnx_int8": dirsize(args.out, "model_quantized.onnx"),
        },
        "score_parity": {"pearson_r": round(pear, 5), "max_abs_logit_diff": round(max_abs, 5),
                          "top_half_jaccard": round(jaccard, 4)},
        "latency": {"torch_fp32": b_torch, "onnx_int8": b_onnx,
                    "speedup_p50": round(b_torch["p50_ms"] / b_onnx["p50_ms"], 2)},
        "note": "Latency/throughput + logit parity only. NDCG quality parity is a "
                "SEPARATE eval (eval_cross_encoder.py, LLM-judge) and is NOT measured here.",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    path = os.path.join(RESULTS_DIR, f"ce_onnx_int8_bench_{stamp}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\n>> results -> {path}")


if __name__ == "__main__":
    main()
