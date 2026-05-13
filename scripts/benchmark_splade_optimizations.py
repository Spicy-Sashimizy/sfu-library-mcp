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

        with torch.inference_mode():
            output = self.model(**tokens)

        # SPLADE: log1p(relu(logits)) * attention_mask — must mask padding
        # before max-pool, otherwise padding positions leak into the sparse vec.
        # Source: NAVER splade transformer_rep.py + Splade_PP_en_v1 model card §6d.
        vecs = torch.log1p(torch.relu(output.logits))
        vecs = vecs * tokens["attention_mask"].unsqueeze(-1)
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

        with torch.inference_mode():
            output = self.model(**tokens)

        logits = output.logits.float()
        vecs = torch.log1p(torch.relu(logits))
        vecs = vecs * tokens["attention_mask"].unsqueeze(-1)  # mask padding
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

        with torch.inference_mode():
            output = self.model(**tokens)

        logits = output.logits.float()
        vecs = torch.log1p(torch.relu(logits))
        vecs = vecs * tokens["attention_mask"].unsqueeze(-1)  # mask padding
        vecs = torch.max(vecs, dim=1).values  # (B, V)

        # Top-K on GPU across full vocab (avoid scan-for-nonzero)
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


# ── Variant 4: BF16 + GPU top-K ──────────────────────────────────────────────


class BF16GpuTopkEncoder(FP16GpuTopkEncoder):
    """BF16 variant — Ada Lovelace (sm_89) handles BF16 at full FP16 throughput
    with a much larger exponent range, so it's a safer drop-in than FP16.
    """

    name = "bf16_gpu_topk"

    def __init__(self, model_name, device):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        if device == "cuda":
            self.model = self.model.to(torch.bfloat16)
        self.model.to(device).eval()
        self.id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self._skip_token_ids = {
            tok_id for tok, tok_id in self.tokenizer.get_vocab().items() if tok.startswith("[")
        }


# ── Variant 5: FP16 + GPU top-K + max_length=256 ─────────────────────────────


def _make_max_length_encoder(N):
    """Factory for FP16 encoders with custom max_length."""

    class _Enc(FP16GpuTopkEncoder):
        name = f"fp16_max{N}"

        def encode_batch(self, texts):
            import torch
            tokens = self.tokenizer(
                texts, max_length=N, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                output = self.model(**tokens)
            logits = output.logits.float()
            vecs = torch.log1p(torch.relu(logits))
            vecs = vecs * tokens["attention_mask"].unsqueeze(-1)  # mask padding
            vecs = torch.max(vecs, dim=1).values
            top_w, top_idx = torch.topk(vecs, SPARSE_TOP_K, dim=1)
            top_w_cpu = top_w.cpu().numpy()
            top_idx_cpu = top_idx.cpu().numpy()
            results = []
            skip, id2tok = self._skip_token_ids, self.id_to_token
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

    return _Enc


FP16Short128Encoder = _make_max_length_encoder(128)
FP16Short128Encoder.name = "fp16_max128"
FP16Short256Encoder = _make_max_length_encoder(256)
FP16Short384Encoder = _make_max_length_encoder(384)
FP16Short384Encoder.name = "fp16_max384"


# ── Variant: FP16 + GPU top-K + async tokenization (background thread) ───────


class FP16AsyncTokenizeEncoder(FP16GpuTopkEncoder):
    """Overlap CPU tokenization with GPU compute via a 1-slot lookahead queue.

    Tokenizations are submitted to a background thread and pulled off a FIFO
    queue by encode_batch. Caller submits batch i (via pre_load_next) before
    calling encode_batch — encode_batch always pulls the head of the queue, so
    order is preserved.
    """

    name = "fp16_async_tok"

    def __init__(self, model_name, device, max_length=MAX_DOC_LENGTH):
        super().__init__(model_name, device)
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        self._tok_pool = ThreadPoolExecutor(max_workers=1)
        self._tok_queue = deque()
        self._max_length = max_length

    def _tokenize(self, texts):
        return self.tokenizer(
            texts, max_length=self._max_length, padding=True, truncation=True, return_tensors="pt"
        )

    def encode_batch(self, texts):
        import torch
        # Pull next tokenization from queue (must have been pre-loaded)
        if self._tok_queue:
            tokens = self._tok_queue.popleft().result()
        else:
            tokens = self._tokenize(texts)

        # non_blocking only helps with pinned memory; HF tokenizer output isn't
        # pinned, so the flag is a no-op here. Kept for clarity.
        tokens = tokens.to(self.device)

        with torch.inference_mode():
            output = self.model(**tokens)

        logits = output.logits.float()
        vecs = torch.log1p(torch.relu(logits))
        vecs = vecs * tokens["attention_mask"].unsqueeze(-1)  # mask padding
        vecs = torch.max(vecs, dim=1).values
        top_w, top_idx = torch.topk(vecs, SPARSE_TOP_K, dim=1)

        top_w_cpu = top_w.cpu().numpy()
        top_idx_cpu = top_idx.cpu().numpy()
        results = []
        skip, id2tok = self._skip_token_ids, self.id_to_token
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

    def pre_load_next(self, texts):
        """Append a tokenization job to the background queue."""
        self._tok_queue.append(self._tok_pool.submit(self._tokenize, texts))


def _make_async_max_length_encoder(N):
    """Async tokenization + custom max_length encoder factory."""

    class _Enc(FP16AsyncTokenizeEncoder):
        name = f"fp16_async_max{N}"

        def __init__(self, model_name, device):
            super().__init__(model_name, device, max_length=N)

    return _Enc


FP16AsyncMax128Encoder = _make_async_max_length_encoder(128)
FP16AsyncMax256Encoder = _make_async_max_length_encoder(256)
FP16AsyncMax384Encoder = _make_async_max_length_encoder(384)


# ── Variant: FP16 + GPU top-K + async tok + SDPA attention ───────────────────


class FP16AsyncSDPAEncoder(FP16AsyncTokenizeEncoder):
    """Force `attn_implementation='sdpa'` so PyTorch routes attention through
    scaled_dot_product_attention (FlashAttention / mem-efficient kernels under
    the hood). Zero install — built into PyTorch 2.6.
    """

    name = "fp16_async_sdpa"

    def __init__(self, model_name, device, max_length=MAX_DOC_LENGTH):
        # Re-init by hand so we can pass attn_implementation
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, attn_implementation="sdpa"
        )
        if device == "cuda":
            self.model = self.model.half()
        self.model.to(device).eval()
        self.id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self._skip_token_ids = {
            tok_id for tok, tok_id in self.tokenizer.get_vocab().items() if tok.startswith("[")
        }
        self._tok_pool = ThreadPoolExecutor(max_workers=1)
        self._tok_queue = deque()
        self._max_length = max_length


def _make_sdpa_max_length_encoder(N):
    class _Enc(FP16AsyncSDPAEncoder):
        name = f"fp16_async_sdpa_max{N}"

        def __init__(self, model_name, device):
            super().__init__(model_name, device, max_length=N)

    return _Enc


FP16AsyncSDPAMax128Encoder = _make_sdpa_max_length_encoder(128)
FP16AsyncSDPAMax256Encoder = _make_sdpa_max_length_encoder(256)
FP16AsyncSDPAMax384Encoder = _make_sdpa_max_length_encoder(384)


# ── Variant: ONNX Runtime (CUDA EP) + GPU top-K + async tok ──────────────────


class ONNXEncoder:
    """SPLADE inference via ONNX Runtime CUDA Execution Provider.

    The SPLADE post-processing (log1p(relu(logits)) → max-pool → top-K) is
    performed in PyTorch on the same GPU, since ONNX export of the masked LM
    head produces a (B, S, V) logits tensor that we still need to reduce.
    """

    name = "onnx_cuda"

    def __init__(self, model_name, device, max_length=256, providers=None, fp16=False):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        from pathlib import Path

        self.device = device
        self._max_length = max_length
        self._fp16 = fp16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Lazy-export to a local cache dir to avoid re-converting each run
        cache_name = "splade_onnx_fp16" if fp16 else "splade_onnx"
        cache = Path(f"/workspaces/sfu-library-mcp-training/models/{cache_name}")
        cache.mkdir(parents=True, exist_ok=True)

        if not (cache / "model.onnx").exists():
            import torch as _torch
            from optimum.onnxruntime import ORTModelForMaskedLM
            from transformers import AutoModelForMaskedLM
            import tempfile
            print(f"  Exporting {model_name} to ONNX{' (FP16)' if fp16 else ''} (one-time, ~1 min)...")
            if fp16:
                m = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=_torch.float16)
                with tempfile.TemporaryDirectory() as td:
                    m.save_pretrained(td)
                    self.tokenizer.save_pretrained(td)
                    tmp_model = ORTModelForMaskedLM.from_pretrained(td, export=True)
            else:
                tmp_model = ORTModelForMaskedLM.from_pretrained(model_name, export=True)
            tmp_model.save_pretrained(str(cache))
            print(f"  Exported → {cache}")

        # Load with raw ORT InferenceSession (avoids optimum's iobinding which
        # crashes with cudaErrorIllegalAddress on ORT 1.26 + CUDA 12.4).
        # We pass numpy arrays via session.run() instead.
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Two ways to pass providers in ORT:
        #   1. providers=[("Name", {options}), ...] alone (tuple form)
        #   2. providers=["Name", ...] AND provider_options=[{...}, ...]
        # If a caller already wrapped (Name, opts) tuples, use form 1.
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if any(isinstance(p, tuple) for p in providers):
            self.session = ort.InferenceSession(
                str(cache / "model.onnx"),
                sess_options=sess_opts,
                providers=providers,
            )
        else:
            provider_opts = [{"device_id": 0} if p == "CUDAExecutionProvider" else {} for p in providers]
            self.session = ort.InferenceSession(
                str(cache / "model.onnx"),
                sess_options=sess_opts,
                providers=providers,
                provider_options=provider_opts,
            )
        self._input_names = [i.name for i in self.session.get_inputs()]
        # Look at output to find the logits tensor name
        self._logits_name = self.session.get_outputs()[0].name

        self.id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self._skip_token_ids = {
            tok_id for tok, tok_id in self.tokenizer.get_vocab().items() if tok.startswith("[")
        }

        self._tok_pool = ThreadPoolExecutor(max_workers=1)
        self._tok_queue = deque()

    def _tokenize(self, texts):
        return self.tokenizer(
            texts, max_length=self._max_length, padding=True, truncation=True, return_tensors="np"
        )

    def encode_batch(self, texts):
        import torch
        import numpy as np

        if self._tok_queue:
            tokens = self._tok_queue.popleft().result()
        else:
            tokens = self._tokenize(texts)

        # Build feed dict for raw ORT — numpy arrays (int64 IDs)
        feed = {}
        for name in self._input_names:
            arr = tokens[name]
            if arr.dtype != np.int64:
                arr = arr.astype(np.int64)
            feed[name] = arr

        # Run on whatever EP loaded successfully (CUDA preferred)
        outputs = self.session.run([self._logits_name], feed)
        logits_np = outputs[0]  # (B, S, V) np.float32

        # Move to GPU for the post-processing; if no GPU, do it on CPU
        if torch.cuda.is_available():
            logits = torch.from_numpy(logits_np).to("cuda", non_blocking=False)
            attention_mask = torch.from_numpy(tokens["attention_mask"]).to("cuda")
        else:
            logits = torch.from_numpy(logits_np)
            attention_mask = torch.from_numpy(tokens["attention_mask"])

        # FP16 ONNX returns half-precision logits; up-cast for the SPLADE math
        if logits.dtype == torch.float16:
            logits = logits.float()

        vecs = torch.log1p(torch.relu(logits))
        vecs = vecs * attention_mask.unsqueeze(-1)  # mask padding
        vecs = torch.max(vecs, dim=1).values
        top_w, top_idx = torch.topk(vecs, SPARSE_TOP_K, dim=1)

        top_w_cpu = top_w.cpu().numpy()
        top_idx_cpu = top_idx.cpu().numpy()
        results = []
        skip, id2tok = self._skip_token_ids, self.id_to_token
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

    def pre_load_next(self, texts):
        self._tok_queue.append(self._tok_pool.submit(self._tokenize, texts))


def _make_onnx_max_length_encoder(N, providers=None, name=None, fp16=False):
    class _Enc(ONNXEncoder):
        def __init__(self, model_name, device):
            super().__init__(model_name, device, max_length=N, providers=providers, fp16=fp16)

    _Enc.name = name or (f"onnx_cuda_fp16_max{N}" if fp16 else f"onnx_cuda_max{N}")
    return _Enc


ONNXMax128Encoder = _make_onnx_max_length_encoder(128)
ONNXMax256Encoder = _make_onnx_max_length_encoder(256)
ONNXMax384Encoder = _make_onnx_max_length_encoder(384)
ONNXMax512Encoder = _make_onnx_max_length_encoder(512)
ONNXFp16Max128Encoder = _make_onnx_max_length_encoder(128, fp16=True)
ONNXFp16Max256Encoder = _make_onnx_max_length_encoder(256, fp16=True)


def _make_onnx_trt_max_length_encoder(N, fp16=True):
    """ONNX with TensorRT EP — pads every batch to fixed (batch, N) shape so TRT
    can reuse the same engine. Without this, TRT rebuilds per new shape combo
    and throughput collapses.

    First inference builds & caches the TRT engine (30-60s). Subsequent
    runs use the cached engine and are very fast.
    """

    class _Enc(ONNXEncoder):
        name = f"onnx_trt_fp16_max{N}" if fp16 else f"onnx_trt_max{N}"

        def __init__(self, model_name, device):
            import os
            os.makedirs("/workspaces/sfu-library-mcp-training/models/trt_engine_cache", exist_ok=True)
            trt_opts = {
                "device_id": 0,
                "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "/workspaces/sfu-library-mcp-training/models/trt_engine_cache",
            }
            providers = [
                ("TensorrtExecutionProvider", trt_opts),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            super().__init__(model_name, device, max_length=N, providers=providers, fp16=fp16)

        def _tokenize(self, texts):
            # Force every batch to (batch, N) — fixed shape so TRT engine is reused.
            return self.tokenizer(
                texts,
                max_length=self._max_length,
                padding="max_length",  # ← key difference vs CUDA EP path
                truncation=True,
                return_tensors="np",
            )

    return _Enc


ONNXTrtFp16Max128Encoder = _make_onnx_trt_max_length_encoder(128)
ONNXTrtFp16Max256Encoder = _make_onnx_trt_max_length_encoder(256)
ONNXTrtFp16Max384Encoder = _make_onnx_trt_max_length_encoder(384)


# ── Variant: ONNX TRT with zero-copy IO binding (logits stay on GPU) ─────────


class ONNXTrtIOBindingEncoder:
    """TRT EP with explicit IO binding — input + output tensors live on GPU,
    no numpy round-trip, post-processing operates directly on the GPU buffer.

    This is the version that should match or beat PyTorch SDPA for ONNX/TRT.
    """

    def __init__(self, model_name, device, max_length=256, batch_size=64, vocab_size=30522, fp16=True):
        import os
        import torch
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque
        from pathlib import Path

        self.device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._fp16 = fp16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        cache_name = "splade_onnx_fp16" if fp16 else "splade_onnx"
        cache = Path(f"/workspaces/sfu-library-mcp-training/models/{cache_name}")
        if not (cache / "model.onnx").exists():
            raise RuntimeError(
                f"Run an ONNX export first: {cache}/model.onnx not found. "
                "Use the onnx_trt_fp16_max* variants once to populate it."
            )

        os.makedirs("/workspaces/sfu-library-mcp-training/models/trt_engine_cache", exist_ok=True)
        trt_opts = {
            "device_id": 0,
            "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "/workspaces/sfu-library-mcp-training/models/trt_engine_cache",
        }

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(cache / "model.onnx"),
            sess_options=sess_opts,
            providers=[
                ("TensorrtExecutionProvider", trt_opts),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._logits_name = self.session.get_outputs()[0].name

        # ONNX output is always FP32 even when weights are FP16 (the model
        # casts back to fp32 before returning). So output buffer must be FP32.
        self._out_dtype = torch.float32
        # Pre-allocate fixed-shape input + output tensors on GPU (reused every batch)
        self._in_input_ids = torch.zeros(batch_size, max_length, dtype=torch.int64, device="cuda")
        self._in_attn_mask = torch.zeros(batch_size, max_length, dtype=torch.int64, device="cuda")
        self._in_token_type = torch.zeros(batch_size, max_length, dtype=torch.int64, device="cuda")
        self._out_logits = torch.empty(batch_size, max_length, vocab_size, dtype=self._out_dtype, device="cuda")

        self._iobinding = self.session.io_binding()
        # Bind inputs — note: ORT expects np dtype names, ints map directly
        import numpy as np
        for name, tensor in [
            ("input_ids", self._in_input_ids),
            ("attention_mask", self._in_attn_mask),
            ("token_type_ids", self._in_token_type),
        ]:
            if name in self._input_names:
                self._iobinding.bind_input(
                    name=name,
                    device_type="cuda",
                    device_id=0,
                    element_type=np.int64,
                    shape=list(tensor.shape),
                    buffer_ptr=tensor.data_ptr(),
                )
        self._iobinding.bind_output(
            name=self._logits_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float32,
            shape=list(self._out_logits.shape),
            buffer_ptr=self._out_logits.data_ptr(),
        )

        self.id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self._skip_token_ids = {
            tok_id for tok, tok_id in self.tokenizer.get_vocab().items() if tok.startswith("[")
        }

        self._tok_pool = ThreadPoolExecutor(max_workers=1)
        self._tok_queue = deque()

    def _tokenize(self, texts):
        # Always fixed shape (batch_size, max_length) for the bound buffers
        toks = self.tokenizer(
            texts,
            max_length=self._max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return toks

    def encode_batch(self, texts):
        import torch

        if self._tok_queue:
            tokens = self._tok_queue.popleft().result()
        else:
            tokens = self._tokenize(texts)

        # Copy tokenized ids into the bound GPU tensors (in place)
        actual_b = tokens["input_ids"].shape[0]
        if actual_b != self._batch_size:
            # Last partial batch — pad with PAD tokens (id=0) to match bound shape
            pad_b = self._batch_size - actual_b
            for k in ("input_ids", "attention_mask", "token_type_ids"):
                if k in tokens:
                    pad = torch.zeros(pad_b, self._max_length, dtype=tokens[k].dtype)
                    tokens[k] = torch.cat([tokens[k], pad], dim=0)

        self._in_input_ids.copy_(tokens["input_ids"], non_blocking=False)
        self._in_attn_mask.copy_(tokens["attention_mask"], non_blocking=False)
        if "token_type_ids" in tokens and "token_type_ids" in self._input_names:
            self._in_token_type.copy_(tokens["token_type_ids"], non_blocking=False)
        else:
            self._in_token_type.zero_()

        # Run inference — output already bound to self._out_logits on GPU
        self.session.run_with_iobinding(self._iobinding)

        # Post-processing operates directly on the GPU buffer (no copy)
        logits = self._out_logits  # (B, S, V) on GPU
        if logits.dtype == torch.float16:
            logits = logits.float()
        attention_mask = self._in_attn_mask
        vecs = torch.log1p(torch.relu(logits))
        vecs = vecs * attention_mask.unsqueeze(-1)  # mask padding
        vecs = torch.max(vecs, dim=1).values
        top_w, top_idx = torch.topk(vecs, SPARSE_TOP_K, dim=1)

        # Only sync to CPU at the end — and only for the actual batch (not padded slots)
        top_w_cpu = top_w[:actual_b].cpu().numpy()
        top_idx_cpu = top_idx[:actual_b].cpu().numpy()
        results = []
        skip, id2tok = self._skip_token_ids, self.id_to_token
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

    def pre_load_next(self, texts):
        self._tok_queue.append(self._tok_pool.submit(self._tokenize, texts))


def _make_onnx_trt_iobinding_encoder(N, fp16=True, batch_size=64):
    class _Enc(ONNXTrtIOBindingEncoder):
        def __init__(self, model_name, device):
            super().__init__(model_name, device, max_length=N, batch_size=batch_size, fp16=fp16)

    _Enc.name = f"onnx_trt_iob_fp16_max{N}" if fp16 else f"onnx_trt_iob_max{N}"
    return _Enc


ONNXTrtIoBindingFp16Max128Encoder = _make_onnx_trt_iobinding_encoder(128)
ONNXTrtIoBindingFp16Max256Encoder = _make_onnx_trt_iobinding_encoder(256)
ONNXTrtIoBindingFp16Max384Encoder = _make_onnx_trt_iobinding_encoder(384)


# ── Variant: same but force eager attention (control, to isolate SDPA effect) ─


class FP16AsyncEagerEncoder(FP16AsyncTokenizeEncoder):
    name = "fp16_async_eager"

    def __init__(self, model_name, device, max_length=MAX_DOC_LENGTH):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from concurrent.futures import ThreadPoolExecutor
        from collections import deque

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, attn_implementation="eager"
        )
        if device == "cuda":
            self.model = self.model.half()
        self.model.to(device).eval()
        self.id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self._skip_token_ids = {
            tok_id for tok, tok_id in self.tokenizer.get_vocab().items() if tok.startswith("[")
        }
        self._tok_pool = ThreadPoolExecutor(max_workers=1)
        self._tok_queue = deque()
        self._max_length = max_length


# ── Variant 6: FP16 + GPU top-K + torch.compile ──────────────────────────────


class FP16CompileEncoder(FP16GpuTopkEncoder):
    """Compile only the BERT body (avoids transformers @wraps decorator bug
    that breaks torch.compile of the whole MLM wrapper).
    """

    name = "fp16_compile"

    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        import torch
        try:
            # Wrap just the BERT body — the MLM head is small, no need to compile
            self.model.bert = torch.compile(self.model.bert, dynamic=True, mode="default")
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
    has_lookahead = hasattr(encoder, "pre_load_next")
    # Prime lookahead with first batch
    if has_lookahead and texts:
        encoder.pre_load_next(texts[:batch_size])
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if has_lookahead:
            # Submit next-batch tokenization before encoding this one
            next_start = i + batch_size
            if next_start < len(texts):
                encoder.pre_load_next(texts[next_start : next_start + batch_size])
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
        default=["baseline", "fp16_gpu_topk", "fp16_async_tok", "fp16_max384",
                 "fp16_max256", "fp16_async_max256"],
        choices=["baseline", "fp16", "fp16_gpu_topk", "bf16_gpu_topk",
                 "fp16_max128", "fp16_max256", "fp16_max384",
                 "fp16_async_tok", "fp16_async_max128",
                 "fp16_async_max256", "fp16_async_max384", "fp16_compile",
                 "fp16_async_sdpa", "fp16_async_sdpa_max128",
                 "fp16_async_sdpa_max256", "fp16_async_sdpa_max384",
                 "fp16_async_eager",
                 "onnx_cuda_max128", "onnx_cuda_max256",
                 "onnx_cuda_max384", "onnx_cuda_max512",
                 "onnx_cuda_fp16_max128", "onnx_cuda_fp16_max256",
                 "onnx_trt_fp16_max128", "onnx_trt_fp16_max256", "onnx_trt_fp16_max384",
                 "onnx_trt_iob_fp16_max128", "onnx_trt_iob_fp16_max256", "onnx_trt_iob_fp16_max384"],
    )
    parser.add_argument("--larger-batch", type=int, default=None,
                        help="Also re-run the best variant at this batch size (e.g. 128, 192, 256)")
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Allow TF32 path on the big MLM-head matmul (vocab=30k)
        torch.set_float32_matmul_precision("high")
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
        "bf16_gpu_topk": BF16GpuTopkEncoder,
        "fp16_max128": FP16Short128Encoder,
        "fp16_max256": FP16Short256Encoder,
        "fp16_max384": FP16Short384Encoder,
        "fp16_async_tok": FP16AsyncTokenizeEncoder,
        "fp16_async_max128": FP16AsyncMax128Encoder,
        "fp16_async_max256": FP16AsyncMax256Encoder,
        "fp16_async_max384": FP16AsyncMax384Encoder,
        "fp16_async_sdpa": FP16AsyncSDPAEncoder,
        "fp16_async_sdpa_max128": FP16AsyncSDPAMax128Encoder,
        "fp16_async_sdpa_max256": FP16AsyncSDPAMax256Encoder,
        "fp16_async_sdpa_max384": FP16AsyncSDPAMax384Encoder,
        "fp16_async_eager": FP16AsyncEagerEncoder,
        "fp16_compile": FP16CompileEncoder,
        "onnx_cuda_max128": ONNXMax128Encoder,
        "onnx_cuda_max256": ONNXMax256Encoder,
        "onnx_cuda_max384": ONNXMax384Encoder,
        "onnx_cuda_max512": ONNXMax512Encoder,
        "onnx_cuda_fp16_max128": ONNXFp16Max128Encoder,
        "onnx_cuda_fp16_max256": ONNXFp16Max256Encoder,
        "onnx_trt_fp16_max128": ONNXTrtFp16Max128Encoder,
        "onnx_trt_fp16_max256": ONNXTrtFp16Max256Encoder,
        "onnx_trt_fp16_max384": ONNXTrtFp16Max384Encoder,
        "onnx_trt_iob_fp16_max128": ONNXTrtIoBindingFp16Max128Encoder,
        "onnx_trt_iob_fp16_max256": ONNXTrtIoBindingFp16Max256Encoder,
        "onnx_trt_iob_fp16_max384": ONNXTrtIoBindingFp16Max384Encoder,
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

        dps, total, sample, all_out = benchmark_variant(
            enc, docs, args.batch_size, sort_by_length=False, warmup=2
        )

        # Reference correctness: compare to baseline (if it was run)
        if variant == "baseline":
            baseline_out = all_out
            correctness = 1.0
        elif baseline_out is not None:
            ovs = [overlap(all_out[i], baseline_out[i]) for i in range(min(50, len(all_out)))]
            correctness = sum(ovs) / max(len(ovs), 1)
        else:
            correctness = float("nan")

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
