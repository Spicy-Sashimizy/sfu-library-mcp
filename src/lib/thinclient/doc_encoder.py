"""Batched GPU SPLADE document encoder — extracted from the retired
scripts/splade_indexer.py so monthly thin-client index rebuilds can encode
docs without any OpenSearch machinery.

The math is byte-identical to what produced the live corpus's sparse_field:
log1p(relu(logits)) * attention_mask, max-pooled over sequence, top-256 terms,
weights rounded to 4 decimals, "[...]" special tokens dropped.

Backends: TensorRT IO-binding (fastest, measured 3.1k docs/s end-to-end on the
RTX 4070) with automatic fallback to eager PyTorch FP16. Query-side encoding
(single text, CPU ONNX) stays in lib.opensearch_retriever.encode_splade.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("sfu_library_mcp")

DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"
DEFAULT_BATCH_SIZE = 64
SPARSE_TOP_K = 256            # keep top-K terms per doc (prune noise)
MAX_DOC_LENGTH = 128          # Splade_PP_en_v1 card §6d: trained at doc=128
SPARSE_WEIGHT_THRESHOLD = 0.01
VOCAB_SIZE = 30522            # bert-base-uncased

_REPO_ROOT = Path(__file__).resolve().parents[3]
ONNX_CACHE_DIR = _REPO_ROOT / "models" / "splade_onnx_fp16"
TRT_ENGINE_CACHE = _REPO_ROOT / "models" / "trt_engine_cache"


def _ensure_onnx_export(model_name: str) -> Path:
    """Export the SPLADE model to ONNX (FP16) on first use; cache afterwards."""
    onnx_path = ONNX_CACHE_DIR / "model.onnx"
    sentinel = ONNX_CACHE_DIR / ".export_complete"

    if onnx_path.exists():
        try:
            size = onnx_path.stat().st_size
        except OSError:
            size = 0
        if size > 1_000_000 and sentinel.exists():
            return onnx_path
        logger.warning("ONNX cache at %s incomplete — re-exporting", ONNX_CACHE_DIR)
        import shutil as _shutil
        _shutil.rmtree(ONNX_CACHE_DIR, ignore_errors=True)

    logger.info("Exporting %s to ONNX FP16 (one-time, ~1 min)...", model_name)
    import shutil
    import tempfile

    import torch
    from optimum.onnxruntime import ORTModelForMaskedLM
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tmp_export = ONNX_CACHE_DIR.with_suffix(".tmp_export")
    if tmp_export.exists():
        shutil.rmtree(tmp_export)
    tmp_export.mkdir(parents=True)

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=torch.float16)
        with tempfile.TemporaryDirectory() as tmp_pt:
            model.save_pretrained(tmp_pt)
            tok.save_pretrained(tmp_pt)
            ort_model = ORTModelForMaskedLM.from_pretrained(tmp_pt, export=True)
            ort_model.save_pretrained(str(tmp_export))

        exported_onnx = tmp_export / "model.onnx"
        if not exported_onnx.exists() or exported_onnx.stat().st_size < 1_000_000:
            raise RuntimeError(f"ONNX export produced bad file at {exported_onnx}")

        if ONNX_CACHE_DIR.exists():
            shutil.rmtree(ONNX_CACHE_DIR)
        ONNX_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
        os.rename(str(tmp_export), str(ONNX_CACHE_DIR))
        sentinel.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        logger.info("ONNX FP16 saved → %s", ONNX_CACHE_DIR)
    except Exception:
        logger.exception("ONNX export failed — cache not promoted")
        raise
    return onnx_path


class SpladeDocEncoder:
    """Eager PyTorch encoder. Used as fallback when TRT EP is unavailable."""

    backend = "pytorch_fp16"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = MAX_DOC_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Loading SPLADE model '%s' on %s (eager PyTorch FP16) ...",
                    model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, attn_implementation="sdpa")
        if self.device == "cuda":
            self.model = self.model.half()
            torch.set_float32_matmul_precision("high")
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.vocab = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._skip_token_ids = {tid for tok, tid in self.vocab.items() if tok.startswith("[")}

    def encode_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Encode a batch of texts into sparse dicts {token: weight}."""
        import torch

        tokens = self.tokenizer(
            texts, max_length=self.max_length, padding=True, truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output = self.model(**tokens)

        # SPLADE: log(1+ReLU(x)) * attention_mask, then max-pool over sequence.
        logits = output.logits.float()
        vecs = torch.log1p(torch.relu(logits))
        vecs = vecs * tokens["attention_mask"].unsqueeze(-1)
        vecs = torch.max(vecs, dim=1).values  # (B, V)

        top_w, top_idx = torch.topk(vecs, SPARSE_TOP_K, dim=1)
        return self._build_sparse_dicts(top_w.cpu().numpy(), top_idx.cpu().numpy())

    def _build_sparse_dicts(self, top_w_np, top_idx_np) -> list[dict[str, float]]:
        results = []
        skip = self._skip_token_ids
        id2tok = self.id_to_token
        for i in range(top_w_np.shape[0]):
            d = {}
            for j in range(top_w_np.shape[1]):
                w = float(top_w_np[i, j])
                if w <= SPARSE_WEIGHT_THRESHOLD:
                    break  # sorted descending — remaining are below threshold too
                idx = int(top_idx_np[i, j])
                if idx in skip:
                    continue
                tok = id2tok.get(idx)
                if tok:
                    d[tok] = round(w, 4)
            results.append(d)
        return results


class SpladeDocEncoderTRT(SpladeDocEncoder):
    """ONNX Runtime + TensorRT EP encoder with zero-copy IO binding."""

    backend = "onnxruntime_trt_iob_fp16"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = MAX_DOC_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        import numpy as np
        import onnxruntime as ort
        import torch
        from transformers import AutoTokenizer

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if self.device != "cuda":
            raise RuntimeError("TRT backend requires CUDA")

        logger.info("Loading SPLADE model '%s' (ONNX + TensorRT IO-binding) ...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.vocab = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._skip_token_ids = {tid for tok, tid in self.vocab.items() if tok.startswith("[")}

        onnx_path = _ensure_onnx_export(model_name)
        TRT_ENGINE_CACHE.mkdir(parents=True, exist_ok=True)
        trt_opts = {
            "device_id": 0,
            "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(TRT_ENGINE_CACHE),
        }
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Keep ORT's CPU pool tiny and non-spinning — the TRT EP does the work
        # on-GPU and a busy-spinning full-width pool starves the GPU (measured:
        # 3.5k → 960 docs/s regression). Tune via SFU_ORT_INTRA_THREADS.
        sess_opts.intra_op_num_threads = int(os.environ.get("SFU_ORT_INTRA_THREADS", "2"))
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=[
                ("TensorrtExecutionProvider", trt_opts),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._logits_name = self.session.get_outputs()[0].name
        logger.info("ORT active providers: %s", self.session.get_providers())

        self._in_input_ids = torch.zeros(batch_size, max_length, dtype=torch.int64, device="cuda")
        self._in_attn_mask = torch.zeros(batch_size, max_length, dtype=torch.int64, device="cuda")
        self._in_token_type = torch.zeros(batch_size, max_length, dtype=torch.int64, device="cuda")
        self._out_logits = torch.empty(batch_size, max_length, VOCAB_SIZE,
                                       dtype=torch.float32, device="cuda")

        self._iobinding = self.session.io_binding()
        for name, tensor in [
            ("input_ids", self._in_input_ids),
            ("attention_mask", self._in_attn_mask),
            ("token_type_ids", self._in_token_type),
        ]:
            if name in self._input_names:
                self._iobinding.bind_input(
                    name=name, device_type="cuda", device_id=0,
                    element_type=np.int64, shape=list(tensor.shape),
                    buffer_ptr=tensor.data_ptr(),
                )
        self._iobinding.bind_output(
            name=self._logits_name, device_type="cuda", device_id=0,
            element_type=np.float32, shape=list(self._out_logits.shape),
            buffer_ptr=self._out_logits.data_ptr(),
        )
        logger.info("TRT engine cache: %s (1st batch builds ~30-60s)", TRT_ENGINE_CACHE)

    def encode_batch(self, texts: list[str]) -> list[dict[str, float]]:
        import torch

        tokens = self.tokenizer(
            texts, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        actual_b = tokens["input_ids"].shape[0]
        if actual_b != self.batch_size:
            pad_b = self.batch_size - actual_b
            for k in ("input_ids", "attention_mask", "token_type_ids"):
                if k in tokens:
                    pad = torch.zeros(pad_b, self.max_length, dtype=tokens[k].dtype)
                    tokens[k] = torch.cat([tokens[k], pad], dim=0)

        self._in_input_ids.copy_(tokens["input_ids"], non_blocking=False)
        self._in_attn_mask.copy_(tokens["attention_mask"], non_blocking=False)
        if "token_type_ids" in tokens and "token_type_ids" in self._input_names:
            self._in_token_type.copy_(tokens["token_type_ids"], non_blocking=False)
        else:
            self._in_token_type.zero_()

        self.session.run_with_iobinding(self._iobinding)

        logits = self._out_logits  # (B, S, V) FP32 on GPU
        vecs = torch.log1p(torch.relu(logits))
        vecs = vecs * self._in_attn_mask.unsqueeze(-1)
        vecs = torch.max(vecs, dim=1).values
        top_w, top_idx = torch.topk(vecs, SPARSE_TOP_K, dim=1)
        return self._build_sparse_dicts(top_w[:actual_b].cpu().numpy(),
                                        top_idx[:actual_b].cpu().numpy())


def make_doc_encoder(
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    max_length: int = MAX_DOC_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    backend: str = "auto",
) -> SpladeDocEncoder:
    """backend='auto' tries TRT first, falls back to eager PyTorch FP16."""
    import torch
    if backend == "pytorch" or device == "cpu" or not torch.cuda.is_available():
        return SpladeDocEncoder(model_name, device, max_length, batch_size)
    if backend == "trt":
        return SpladeDocEncoderTRT(model_name, device, max_length, batch_size)
    try:
        return SpladeDocEncoderTRT(model_name, device, max_length, batch_size)
    except Exception as e:
        logger.warning("TRT backend unavailable (%s) — falling back to PyTorch FP16", e)
        return SpladeDocEncoder(model_name, device, max_length, batch_size)
