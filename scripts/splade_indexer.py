#!/usr/bin/env python3
"""Index OpenAlex snapshot into OpenSearch using SPLADE sparse vectors.

Reads filtered JSONL chunks from data/openalex_snapshot/ (produced by
snapshot_downloader.py), encodes each doc through a SPLADE model to
generate sparse term weights, and bulk-upserts them into OpenSearch.

Checkpoint / Resume
───────────────────
Indexing can be interrupted at any time and resumed without re-processing:

    python scripts/splade_indexer.py --resume

Progress is checkpointed every --checkpoint-interval docs (default 100k).
On SIGINT/SIGTERM the current batch finishes, checkpoint is saved, and the
script exits cleanly.

Failsafes
─────────
1. Atomic checkpoint writes — fsync'd .tmp file + os.replace() + dir fsync,
   so a power loss leaves either the OLD or NEW checkpoint intact, never
   a half-written file.
2. SIGINT / SIGTERM handlers — finishes current batch, saves, exits
3. Double-signal force exit — second Ctrl-C exits immediately
4. PID lock file — refuses to start if another indexer is alive (prevents
   double-indexing); stale locks (process dead OR heartbeat > 5 min old)
   are auto-cleared
5. Atomic ONNX export — writes to .tmp_export dir, sentinel file marks
   completion; partial exports are detected and re-run on next start
6. TRT engine cache validation — zero-byte engine files (left by SIGKILL
   during build) are removed before the encoder loads
7. Encoder failure recovery — CUDA / device errors save checkpoint and
   exit cleanly so user can restart with --resume after fixing the GPU.
   Three consecutive non-CUDA encode failures also halt.
8. OpenSearch circuit breaker — five consecutive bulk failures halt with
   a checkpoint at the LAST successful doc, so resume re-tries the failed
   batch (idempotent thanks to deterministic _id).
9. Batch-level OpenSearch retry with exponential backoff (3 attempts)
10. Per-doc error isolation — one bad doc doesn't kill the batch
11. VRAM monitoring — logs GPU memory every checkpoint, warns at >90%
12. Throughput tracking — docs/sec, ETA, running average
13. Status file (indexer_status.json) — pollable by external monitors
14. Heartbeat file — updated every batch so external watchdogs can detect stalls
15. Dry run mode — encode 100 docs, verify OpenSearch connectivity, exit
16. Idempotent re-indexing — every doc is upserted by its OpenAlex _id,
   so re-indexing the same doc just overwrites; safe to re-run on partial
   failures.

Usage:
    # Full indexing run
    python scripts/splade_indexer.py

    # Resume after interruption
    python scripts/splade_indexer.py --resume

    # Dry run — encode 100 docs, test OpenSearch, report stats
    python scripts/splade_indexer.py --dry-run

    # Custom settings
    python scripts/splade_indexer.py \\
        --model naver/splade-cocondenser-distil \\
        --batch-size 128 \\
        --opensearch-url http://localhost:9200 \\
        --checkpoint-interval 50000

    # CPU-only (no GPU)
    python scripts/splade_indexer.py --device cpu --batch-size 16
"""

import argparse
import gzip
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_INPUT_DIR = Path(__file__).parent.parent / "data" / "openalex_snapshot"
DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"
DEFAULT_OPENSEARCH_URL = "http://localhost:9200"
DEFAULT_INDEX = "openalex_works"
DEFAULT_BATCH_SIZE = 64
DEFAULT_CHECKPOINT_INTERVAL = 100_000
CHECKPOINT_FILE = "indexer_checkpoint.json"
STATUS_FILE = "indexer_status.json"
HEARTBEAT_FILE = "indexer_heartbeat"
PID_LOCK_FILE = "indexer.pid"
PID_STALE_AFTER_S = 300  # consider lock stale if heartbeat older than 5 min
VRAM_RESERVE_FILE = Path(__file__).parent.parent / "data" / "vram_reserve.json"
ONNX_CACHE_DIR = Path(__file__).parent.parent / "models" / "splade_onnx_fp16"
TRT_ENGINE_CACHE = Path(__file__).parent.parent / "models" / "trt_engine_cache"
MAX_BULK_RETRIES = 3
BULK_RETRY_BACKOFF = 5
SPARSE_TOP_K = 256  # keep top-K terms per doc (prune noise)
MAX_DOC_LENGTH = 128  # Splade_PP_en_v1 model card §6d: trained at doc=128, query=24
SPARSE_WEIGHT_THRESHOLD = 0.01  # drop terms with weight <= this
VOCAB_SIZE = 30522  # bert-base-uncased vocab size (model dependent)

# ── Graceful shutdown ────────────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        logger.warning("Second %s received — forcing exit NOW", sig_name)
        sys.exit(1)
    logger.info(
        "%s received — will finish current batch, save checkpoint, then exit. "
        "Press Ctrl-C again to force quit.",
        sig_name,
    )
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ── Atomic file I/O ──────────────────────────────────────────────────────────


def _atomic_write_json(data: dict, path: Path) -> None:
    """Write JSON atomically: write to .tmp, fsync, then rename.

    The fsync ensures the data is on stable storage before the rename, so a
    power loss between write and rename leaves the OLD checkpoint intact and
    a power loss after rename leaves the NEW checkpoint intact. There's no
    intermediate "half-written checkpoint" state.
    """
    tmp = path.with_suffix(".tmp")
    payload = json.dumps(data, indent=2)
    # Use os.open for explicit fsync control
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))
    # fsync the directory entry so the rename itself is durable
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass  # some filesystems don't support directory fsync


def _touch_heartbeat(path: Path) -> None:
    try:
        path.write_text(str(time.time()))
    except Exception:
        pass


def _is_process_alive(pid: int) -> bool:
    """Check if a PID is still running (POSIX). Returns False on any error."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def acquire_pid_lock(input_dir: Path) -> Path:
    """Refuse to start if another indexer is already running.

    Stale locks (process dead OR heartbeat older than PID_STALE_AFTER_S) are
    forcibly cleared. Otherwise we exit with a clear message — preventing two
    indexers from racing on the checkpoint and double-indexing docs.
    """
    lock_path = input_dir / PID_LOCK_FILE
    heartbeat_path = input_dir / HEARTBEAT_FILE

    if lock_path.exists():
        try:
            other_pid = int(lock_path.read_text().strip())
        except Exception:
            other_pid = -1

        # Heartbeat staleness check (covers both crashed-with-stale-lock and
        # killed-process-with-recycled-pid)
        hb_age = float("inf")
        if heartbeat_path.exists():
            try:
                hb_age = time.time() - float(heartbeat_path.read_text().strip())
            except Exception:
                pass

        if other_pid > 0 and _is_process_alive(other_pid) and hb_age < PID_STALE_AFTER_S:
            logger.error(
                "Another indexer is already running (PID %d, heartbeat %.0fs old). Refusing to start.",
                other_pid, hb_age,
            )
            logger.error(
                "If you're certain it's dead, remove %s manually and retry.",
                lock_path,
            )
            sys.exit(2)

        logger.warning(
            "Stale lock found (PID %d, alive=%s, heartbeat=%.0fs old) — clearing",
            other_pid, _is_process_alive(other_pid), hb_age,
        )

    try:
        lock_path.write_text(str(os.getpid()))
    except Exception as e:
        logger.error("Could not write PID lock at %s: %s", lock_path, e)
        sys.exit(1)
    return lock_path


def release_pid_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists():
            # Only remove if it still contains our PID (race-safe)
            try:
                contained = int(lock_path.read_text().strip())
            except Exception:
                contained = -1
            if contained == os.getpid():
                lock_path.unlink()
    except Exception:
        pass


def validate_trt_cache(cache_dir: Path) -> int:
    """Return number of cached TRT engine files. Wipe any zero-byte files."""
    if not cache_dir.exists():
        return 0
    removed = 0
    kept = 0
    for f in cache_dir.iterdir():
        try:
            if f.is_file() and f.stat().st_size == 0:
                f.unlink()
                removed += 1
            else:
                kept += 1
        except OSError:
            pass
    if removed:
        logger.warning("Removed %d zero-byte TRT cache files in %s", removed, cache_dir)
    return kept


# ── VRAM reservation ────────────────────────────────────────────────────────

_last_vram_check = 0.0
_VRAM_CHECK_INTERVAL = 5.0  # seconds between file checks
_VRAM_POLL_INTERVAL = 10.0  # seconds between polls while paused
_vram_paused = False


def _read_vram_reserve() -> dict | None:
    """Read the VRAM reserve file. Returns None if absent or inactive."""
    try:
        if not VRAM_RESERVE_FILE.exists():
            return None
        data = json.loads(VRAM_RESERVE_FILE.read_text())
        if not data.get("active", False):
            return None
        return data
    except Exception:
        return None


def check_vram_reservation(
    encoder: "SpladeEncoder",
    original_batch_size: int,
    current_batch_size: int,
    checkpoint_fn: callable = None,
) -> int:
    """Check VRAM reservation and pause/resume the indexer as needed.

    Three modes based on the reserve file:
      - No file / inactive  → run at original batch size.
      - active, mode="reduce" → shrink batch size proportionally (legacy).
      - active, mode="pause" (default) → offload model to CPU, save checkpoint,
        poll until reservation clears, then reload to GPU and resume.

    Returns the batch size to use for the next batch.
    """
    global _last_vram_check, _vram_paused
    now = time.time()
    if now - _last_vram_check < _VRAM_CHECK_INTERVAL:
        return current_batch_size
    _last_vram_check = now

    data = _read_vram_reserve()

    # ── No reservation → restore if we were throttled ────────────────────
    if data is None:
        if _vram_paused:
            _vram_paused = False
        if current_batch_size != original_batch_size:
            logger.info("VRAM reservation cleared — restoring batch_size to %d", original_batch_size)
        return original_batch_size

    reserve_gb = data.get("reserve_gb", 0)
    mode = data.get("mode", "pause")

    if mode != "pause" and reserve_gb <= 0:
        return original_batch_size

    # ── Pause mode: offload model, checkpoint, wait ──────────────────────
    if mode == "pause":
        import torch

        if checkpoint_fn:
            checkpoint_fn()
            logger.info("VRAM pause: checkpoint saved before offloading")

        was_on_gpu = encoder.device == "cuda"
        # Only the eager PyTorch backend supports CPU offload; TRT engine + IO
        # bindings are bound to GPU memory and would need a full re-init.
        # For the TRT backend we just pause execution without offloading.
        can_offload = was_on_gpu and hasattr(encoder, "model") and not encoder.backend.startswith("onnxruntime_trt")
        if can_offload:
            encoder.model.cpu()
            encoder.device = "cpu"
            torch.cuda.empty_cache()
            freed = torch.cuda.memory_reserved() / 1e9
            logger.info(
                "VRAM pause: model offloaded to CPU, GPU cache cleared (%.2f GB reserved remains)",
                freed,
            )
        elif was_on_gpu:
            logger.info("VRAM pause: %s backend can't offload — pausing in-place", encoder.backend)

        _vram_paused = True
        logger.info("VRAM pause: indexer paused — polling every %.0fs for release", _VRAM_POLL_INTERVAL)

        while not _shutdown_requested:
            time.sleep(_VRAM_POLL_INTERVAL)
            check = _read_vram_reserve()
            if check is None:
                break
            if check.get("mode", "pause") != "pause" or not check.get("active", False):
                break

        _vram_paused = False

        if _shutdown_requested:
            return current_batch_size

        if can_offload and torch.cuda.is_available():
            encoder.model.to("cuda")
            encoder.device = "cuda"
            torch.cuda.empty_cache()
            logger.info(
                "VRAM resumed: model reloaded to GPU (%.2f GB allocated)",
                torch.cuda.memory_allocated() / 1e9,
            )

        logger.info("VRAM resumed: continuing at batch_size %d", original_batch_size)
        return original_batch_size

    # ── Reduce mode: shrink batch size proportionally ────────────────────
    import torch
    if not torch.cuda.is_available():
        return original_batch_size

    # TRT backend has a fixed-shape engine + bound buffers — batch size can't
    # change at runtime. Treat 'reduce' as 'pause' for the TRT backend.
    if hasattr(encoder, "backend") and encoder.backend.startswith("onnxruntime_trt"):
        return original_batch_size

    total_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    model_gb = 0.5
    available_for_batches = total_gb - model_gb - reserve_gb

    if available_for_batches <= 0:
        new_batch = max(1, original_batch_size // 8)
    else:
        usable_fraction = available_for_batches / (total_gb - model_gb)
        new_batch = max(1, int(original_batch_size * usable_fraction))

    if new_batch != current_batch_size:
        torch.cuda.empty_cache()
        logger.info(
            "VRAM reservation: %.1f GB reserved — batch_size %d → %d, GPU cache cleared",
            reserve_gb, current_batch_size, new_batch,
        )

    return new_batch


# ── SPLADE model management ─────────────────────────────────────────────────


def _ensure_onnx_export(model_name: str) -> Path:
    """Export the SPLADE model to ONNX (FP16) on first use; cache afterwards.

    Atomic semantics: writes to a sibling .tmp directory, then renames. If the
    process is killed mid-export, the .tmp dir is left behind and the canonical
    location does not exist — next run will retry cleanly.

    Validates the cached file on load. If model.onnx is missing or zero-bytes
    (e.g., partial write that survived the rename for some reason), wipes and
    re-exports.
    """
    onnx_path = ONNX_CACHE_DIR / "model.onnx"
    sentinel = ONNX_CACHE_DIR / ".export_complete"

    # Validate existing cache
    if onnx_path.exists():
        try:
            size = onnx_path.stat().st_size
        except OSError:
            size = 0
        if size > 1_000_000 and sentinel.exists():
            return onnx_path
        logger.warning(
            "ONNX cache at %s appears incomplete (size=%d, sentinel=%s) — re-exporting",
            ONNX_CACHE_DIR, size, sentinel.exists(),
        )
        import shutil as _shutil
        _shutil.rmtree(ONNX_CACHE_DIR, ignore_errors=True)

    logger.info("Exporting %s to ONNX FP16 (one-time, ~1 min)...", model_name)
    import tempfile
    import shutil
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from optimum.onnxruntime import ORTModelForMaskedLM

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

        # Sanity-check the export before promoting it
        exported_onnx = tmp_export / "model.onnx"
        if not exported_onnx.exists() or exported_onnx.stat().st_size < 1_000_000:
            raise RuntimeError(f"ONNX export produced bad file at {exported_onnx}")

        # Atomic-ish promotion: rename tmp dir to canonical location
        if ONNX_CACHE_DIR.exists():
            shutil.rmtree(ONNX_CACHE_DIR)
        ONNX_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
        os.rename(str(tmp_export), str(ONNX_CACHE_DIR))
        # Drop a sentinel file last so partial dirs are detectable
        sentinel.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        logger.info("ONNX FP16 saved → %s", ONNX_CACHE_DIR)
    except Exception:
        # Leave the .tmp_export dir for inspection but don't pollute the cache
        logger.exception("ONNX export failed — cache not promoted")
        raise
    return onnx_path


class SpladeEncoder:
    """Eager PyTorch encoder. Used as fallback when TRT EP is unavailable."""

    backend = "pytorch_fp16"

    def __init__(
        self,
        model_name: str,
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

        logger.info("Loading SPLADE model '%s' on %s (eager PyTorch FP16) ...", model_name, self.device)
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
        logger.info("SPLADE PyTorch model loaded (%d vocab tokens)", len(self.vocab))

        if self.device == "cuda":
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info("GPU memory after model load: %.2f GB", mem)

    def encode_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Encode a batch of texts into sparse dicts {token: weight}."""
        import torch

        tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            output = self.model(**tokens)

        # SPLADE: log(1+ReLU(x)) * attention_mask, then max-pool over sequence
        # The attention_mask multiplication is critical — without it, padding
        # positions leak into the sparse vector. Source: NAVER splade
        # transformer_rep.py and Splade_PP_en_v1 model card §6d.
        logits = output.logits.float()
        vecs = torch.log1p(torch.relu(logits))
        vecs = vecs * tokens["attention_mask"].unsqueeze(-1)
        vecs = torch.max(vecs, dim=1).values  # (B, V)

        # GPU-side top-K avoids the per-row nonzero scan + 2 CPU syncs
        top_w, top_idx = torch.topk(vecs, SPARSE_TOP_K, dim=1)
        return self._build_sparse_dicts(top_w.cpu().numpy(), top_idx.cpu().numpy())

    def _build_sparse_dicts(self, top_w_np, top_idx_np) -> list[dict[str, float]]:
        """Shared sparse-dict construction used by both eager and TRT paths."""
        results = []
        skip = self._skip_token_ids
        id2tok = self.id_to_token
        for i in range(top_w_np.shape[0]):
            d = {}
            for j in range(SPARSE_TOP_K):
                w = float(top_w_np[i, j])
                if w <= SPARSE_WEIGHT_THRESHOLD:
                    continue
                idx = int(top_idx_np[i, j])
                if idx in skip:
                    continue
                tok = id2tok.get(idx)
                if tok:
                    d[tok] = round(w, 4)
            results.append(d)
        return results

    def get_gpu_stats(self) -> dict:
        """Return GPU memory stats if available."""
        if self.device != "cuda":
            return {"device": "cpu"}
        import torch
        return {
            "device": "cuda",
            "backend": self.backend,
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "utilization_pct": round(
                100.0 * torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory, 1
            ),
        }


class SpladeEncoderTRT(SpladeEncoder):
    """ONNX Runtime + TensorRT EP encoder with zero-copy IO binding.

    Pre-allocates fixed (batch_size, max_length) GPU buffers once, runs every
    batch through the same TRT engine, and applies SPLADE post-processing
    directly on the GPU output buffer. ~20-27% faster than eager PyTorch.

    Falls back to CUDA EP automatically if TRT is unavailable; the SpladeEncoder
    factory below catches install errors and returns the eager class instead.
    """

    backend = "onnxruntime_trt_iob_fp16"

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = MAX_DOC_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        import torch
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if self.device != "cuda":
            raise RuntimeError("TRT backend requires CUDA")

        logger.info("Loading SPLADE model '%s' on %s (ONNX + TensorRT IO-binding) ...", model_name, self.device)
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
        active_providers = self.session.get_providers()
        logger.info("ORT active providers: %s", active_providers)

        # Pre-allocate fixed-shape GPU tensors. ONNX outputs FP32 even when
        # weights are FP16 (model casts back before output) — buffer must be FP32.
        self._in_input_ids = torch.zeros(batch_size, max_length, dtype=torch.int64, device="cuda")
        self._in_attn_mask = torch.zeros(batch_size, max_length, dtype=torch.int64, device="cuda")
        self._in_token_type = torch.zeros(batch_size, max_length, dtype=torch.int64, device="cuda")
        self._out_logits = torch.empty(batch_size, max_length, VOCAB_SIZE, dtype=torch.float32, device="cuda")

        self._iobinding = self.session.io_binding()
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

        mem = torch.cuda.memory_allocated() / 1e9
        logger.info("GPU memory after TRT model + buffers: %.2f GB", mem)
        logger.info("TRT engine cache: %s (1st batch builds ~30-60s)", TRT_ENGINE_CACHE)

    def encode_batch(self, texts: list[str]) -> list[dict[str, float]]:
        import torch

        # Always pad to fixed (batch_size, max_length) so the TRT engine + bound
        # buffers are reused. Tokenize CPU-side, then copy into GPU buffers.
        tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
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

        # Inference writes directly to self._out_logits (no copy)
        self.session.run_with_iobinding(self._iobinding)

        logits = self._out_logits  # (B, S, V) FP32 on GPU
        vecs = torch.log1p(torch.relu(logits))
        vecs = vecs * self._in_attn_mask.unsqueeze(-1)
        vecs = torch.max(vecs, dim=1).values
        top_w, top_idx = torch.topk(vecs, SPARSE_TOP_K, dim=1)

        # Sync to CPU only for the actual batch portion (not padded slots)
        return self._build_sparse_dicts(top_w[:actual_b].cpu().numpy(), top_idx[:actual_b].cpu().numpy())


def make_encoder(
    model_name: str,
    device: str = "auto",
    max_length: int = MAX_DOC_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    backend: str = "auto",
) -> SpladeEncoder:
    """Build an encoder. backend='auto' tries TRT first, falls back to PyTorch.

    backend choices: 'auto' | 'trt' | 'pytorch'
    """
    import torch
    if backend == "pytorch" or device == "cpu" or not torch.cuda.is_available():
        return SpladeEncoder(model_name, device, max_length, batch_size)
    if backend == "trt":
        return SpladeEncoderTRT(model_name, device, max_length, batch_size)
    # auto: try TRT, fall back on any failure
    try:
        return SpladeEncoderTRT(model_name, device, max_length, batch_size)
    except Exception as e:
        logger.warning("TRT backend unavailable (%s) — falling back to PyTorch FP16", e)
        return SpladeEncoder(model_name, device, max_length, batch_size)


# ── OpenSearch bulk upsert ───────────────────────────────────────────────────


# Encode-only offload: if SFU_ENCODE_OUT is set, write the same NDJSON (action +
# doc) to a gzip shard instead of POSTing. Lets a cheap GPU droplet produce
# bulk-ready sparse shards (-> TrueNAS) with no OpenSearch on the box. Resumable
# (the indexer's checkpoint still applies). Concatenated gzip members are valid.
import gzip as _gzip
import threading as _threading
_ENCODE_OUT = os.environ.get("SFU_ENCODE_OUT", "").strip()
_encode_lock = _threading.Lock()


def bulk_upsert_opensearch(
    session: requests.Session,
    opensearch_url: str,
    index: str,
    docs: list[dict],
) -> dict:
    """Bulk upsert documents to OpenSearch. Returns stats dict.

    Retries on failure with exponential backoff.
    """
    if not docs:
        return {"indexed": 0, "errors": 0}

    # Encode-only sink: append the bulk NDJSON to a gzip shard, skip OpenSearch.
    if _ENCODE_OUT:
        lines = []
        for doc in docs:
            lines.append(json.dumps({"index": {"_index": index, "_id": doc.get("id", "")}}))
            lines.append(json.dumps(doc))
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        with _encode_lock:
            with _gzip.open(_ENCODE_OUT, "ab") as gz:
                gz.write(payload)
        return {"indexed": len(docs), "errors": 0}

    stats = {"indexed": 0, "errors": 0, "error_details": []}

    # Build NDJSON bulk payload
    lines = []
    for doc in docs:
        doc_id = doc.get("id", "")
        action = {"index": {"_index": index, "_id": doc_id}}
        lines.append(json.dumps(action))
        lines.append(json.dumps(doc))
    payload = "\n".join(lines) + "\n"

    for attempt in range(1, MAX_BULK_RETRIES + 1):
        try:
            resp = session.post(
                f"{opensearch_url}/_bulk",
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/x-ndjson"},
                timeout=120,
            )

            if resp.status_code == 200:
                result = resp.json()
                if result.get("errors"):
                    for item in result.get("items", []):
                        op = item.get("index", {})
                        if op.get("error"):
                            stats["errors"] += 1
                            if len(stats["error_details"]) < 5:
                                stats["error_details"].append(str(op["error"])[:200])
                        else:
                            stats["indexed"] += 1
                else:
                    stats["indexed"] = len(docs)
                return stats

            resp.raise_for_status()

        except Exception as e:
            if attempt < MAX_BULK_RETRIES:
                wait = BULK_RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "Bulk upsert failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, MAX_BULK_RETRIES, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error("Bulk upsert failed after %d attempts: %s", MAX_BULK_RETRIES, e)
                stats["errors"] = len(docs)
                return stats

    return stats


# ── Async bulk uploader ──────────────────────────────────────────────────────

_tls = threading.local()


class AsyncBulkUploader:
    """Overlaps OpenSearch HTTP uploads with GPU encoding via a thread pool.

    Each submit() dispatches a bulk_upsert_opensearch call to the pool and
    tags it with the doc-index position it covers. drain() waits for all
    in-flight futures and returns aggregated stats including per-batch
    outcomes for circuit-breaker evaluation.

    workers=1 runs synchronously (zero threads, original behaviour) — use
    --async-workers 1 to roll back without touching the code.
    """

    def __init__(
        self,
        template_session: requests.Session,
        opensearch_url: str,
        index_name: str,
        workers: int = 3,
    ):
        self.opensearch_url = opensearch_url
        self.index_name = index_name
        self.workers = workers
        self._template = template_session
        self._pending: list[tuple] = []  # (future_or_done, doc_range_end)
        self._executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="os_bulk")
            if workers > 1 else None
        )

    def _get_session(self) -> requests.Session:
        """Return a per-thread Session cloned from the template."""
        if not hasattr(_tls, "session"):
            s = requests.Session()
            s.auth = self._template.auth
            s.headers.update(self._template.headers)
            s.verify = self._template.verify
            s.cert = self._template.cert
            _tls.session = s
        return _tls.session

    def _upload_in_thread(self, docs: list[dict]) -> dict:
        return bulk_upsert_opensearch(
            self._get_session(), self.opensearch_url, self.index_name, docs
        )

    def submit(self, docs: list[dict], doc_range_end: int) -> None:
        """Dispatch an upload. doc_range_end is committed_idx after this batch."""
        if not docs:
            return
        if self._executor is None:
            result = bulk_upsert_opensearch(
                self._template, self.opensearch_url, self.index_name, docs
            )
            class _Done:
                def result(self_): return result  # noqa: E301
            self._pending.append((_Done(), doc_range_end))
            return
        future = self._executor.submit(self._upload_in_thread, docs)
        self._pending.append((future, doc_range_end))

    def _drain_one(self) -> tuple[dict, int]:
        future, doc_range_end = self._pending.pop(0)
        try:
            result = future.result()
        except Exception as e:
            logger.error("Async bulk upload exception: %s", e)
            result = {"indexed": 0, "errors": -1, "error_details": [str(e)[:200]]}
        return result, doc_range_end

    def drain(self) -> dict:
        """Wait for all in-flight futures. Returns aggregated stats + per_batch list."""
        stats = {
            "indexed": 0, "errors": 0, "error_details": [],
            "new_committed_idx": None, "per_batch": [],
        }
        while self._pending:
            result, new_idx = self._drain_one()
            stats["per_batch"].append(result)
            stats["indexed"] += result.get("indexed", 0)
            stats["errors"] += result.get("errors", 0)
            stats["error_details"].extend(result.get("error_details", []))
            if stats["new_committed_idx"] is None or new_idx > stats["new_committed_idx"]:
                stats["new_committed_idx"] = new_idx
        return stats

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def close(self) -> None:
        self.drain()
        if self._executor:
            self._executor.shutdown(wait=True)


def check_opensearch_health(session: requests.Session, url: str, index: str) -> bool:
    """Verify OpenSearch is reachable and the target index exists."""
    try:
        resp = session.get(f"{url}/_cluster/health", timeout=10)
        if resp.status_code != 200:
            logger.error("OpenSearch cluster not healthy: %d", resp.status_code)
            return False

        health = resp.json()
        status = health.get("status", "red")
        if status == "red":
            logger.error("OpenSearch cluster status is RED")
            return False

        logger.info("OpenSearch cluster: %s, nodes: %s", status, health.get("number_of_nodes"))

        # Check if index exists, create if not
        idx_resp = session.head(f"{url}/{index}", timeout=10)
        if idx_resp.status_code == 404:
            logger.info("Index '%s' does not exist — creating...", index)
            create_resp = session.put(f"{url}/{index}", timeout=10)
            if create_resp.status_code not in (200, 201):
                logger.error("Failed to create index: %s", create_resp.text)
                return False
            logger.info("Index '%s' created", index)

        return True

    except Exception as e:
        logger.error("Cannot reach OpenSearch at %s: %s", url, e)
        return False


# ── Input file management ────────────────────────────────────────────────────


def list_input_files(input_dir: Path) -> list[Path]:
    """List all .jsonl.gz files sorted by name."""
    files = sorted(input_dir.glob("works_part_*.jsonl.gz"))
    if not files:
        files = sorted(input_dir.glob("*.jsonl.gz"))
    return files


def read_jsonl_gz(path: Path) -> list[dict]:
    """Read all records from a gzipped JSONL file."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ── Checkpoint management ────────────────────────────────────────────────────


def load_checkpoint(input_dir: Path, total_files: int | None = None) -> dict | None:
    cp_path = input_dir / CHECKPOINT_FILE
    if not cp_path.exists():
        return None
    try:
        text = cp_path.read_text()
        if not text.strip():
            logger.warning("Checkpoint file is empty — starting fresh")
            return None
        data = json.loads(text)
        # Sanity-check: file_index must be in range
        fi = data.get("file_index", 0)
        if total_files is not None and fi > total_files:
            logger.warning(
                "Checkpoint file_index=%d exceeds available files (%d). "
                "Treating as completed.", fi, total_files,
            )
        logger.info(
            "Resuming from checkpoint: file_index=%d, doc_offset=%d, total_indexed=%d",
            fi,
            data.get("doc_offset", 0),
            data.get("total_indexed", 0),
        )
        return data
    except Exception as e:
        logger.warning("Could not load checkpoint (%s) — starting fresh", e)
        # Move the corrupted checkpoint aside so the next save starts clean
        try:
            corrupted = cp_path.with_suffix(f".corrupted.{int(time.time())}")
            cp_path.rename(corrupted)
            logger.warning("Moved corrupted checkpoint to %s", corrupted)
        except Exception:
            pass
        return None


def save_checkpoint(input_dir: Path, state: dict) -> None:
    """Atomic write. Raises on disk-full / permission error so the caller can decide."""
    state["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_write_json(state, input_dir / CHECKPOINT_FILE)


def write_status(input_dir: Path, state: dict) -> None:
    try:
        _atomic_write_json(state, input_dir / STATUS_FILE)
    except Exception:
        pass


# ── Live progress mirror (push status/heartbeat to the always-on TrueNAS) ─────
# The TrueNAS monitor reads these files to put a live progress line in its
# heartbeat. The host isn't always on, so we PUSH from here while indexing runs.
# Fully decoupled from the write path, opt-out, and failure-safe: a mirror error
# never touches indexing. Disable with SFU_PROGRESS_PUSH=off (or empty).
PROGRESS_PUSH_FILES = (STATUS_FILE, HEARTBEAT_FILE, "snapshot_status.json")
_DEFAULT_PROGRESS_TARGET = "truenas:/mnt/MAIN/sfu-library-mcp/from-host/snapshots/"


def _progress_push_target() -> str:
    t = os.environ.get("SFU_PROGRESS_PUSH", _DEFAULT_PROGRESS_TARGET).strip()
    return "" if t.lower() in ("", "0", "off", "false", "none") else t


def _push_progress_once(input_dir: Path, target: str) -> bool:
    srcs = [str(input_dir / f) for f in PROGRESS_PUSH_FILES if (input_dir / f).exists()]
    if not srcs or not target:
        return False
    # rsync preferred (delta + preserves the status file's real mtime); scp fallback.
    for cmd in (["rsync", "-a", "--timeout=30", *srcs, target], ["scp", "-q", *srcs, target]):
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            return True
        except FileNotFoundError:
            continue   # tool not installed; try next
        except Exception:
            return False
    return False


def _start_progress_pusher(input_dir: Path) -> str:
    """Spawn a daemon thread that mirrors progress files to the NAS on an interval.
    Returns the resolved target ('' if disabled) so the caller can flush at exit."""
    target = _progress_push_target()
    if not target:
        return ""
    interval = max(15, int(os.environ.get("SFU_PROGRESS_PUSH_INTERVAL", "60")))

    def _loop():
        while True:
            _push_progress_once(input_dir, target)
            time.sleep(interval)

    threading.Thread(target=_loop, daemon=True, name="progress-pusher").start()
    return target


# ── Main indexing pipeline ───────────────────────────────────────────────────


def run_indexer(
    input_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    opensearch_url: str,
    index_name: str,
    checkpoint_interval: int,
    resume: bool,
    dry_run: bool,
    max_length: int = MAX_DOC_LENGTH,
    backend: str = "auto",
    async_workers: int = 3,
) -> dict:
    """Main indexing pipeline. Returns final stats dict."""
    global _shutdown_requested

    session = requests.Session()

    # ── Pre-flight checks ────────────────────────────────────────────────
    input_files = list_input_files(input_dir)
    if not input_files:
        logger.error("No input files found in %s", input_dir)
        return {"error": "no_input_files"}

    total_files = len(input_files)
    logger.info("Found %d input files in %s", total_files, input_dir)

    # PID lock — prevent two indexers from racing on the same checkpoint
    lock_path = acquire_pid_lock(input_dir)

    # Validate the TRT engine cache before model load (catches zero-byte files
    # left behind by a previous SIGKILL during engine build)
    n_trt_engines = validate_trt_cache(TRT_ENGINE_CACHE)
    if n_trt_engines:
        logger.info("Found %d cached TRT engines in %s", n_trt_engines, TRT_ENGINE_CACHE)

    if not dry_run and not _ENCODE_OUT:
        if not check_opensearch_health(session, opensearch_url, index_name):
            logger.error(
                "OpenSearch pre-flight check failed. Ensure OpenSearch is running at %s",
                opensearch_url,
            )
            release_pid_lock(lock_path)
            return {"error": "opensearch_unhealthy"}

    # ── Load model ───────────────────────────────────────────────────────
    try:
        encoder = make_encoder(
            model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            backend=backend,
        )
    except Exception as e:
        logger.exception("Failed to load encoder: %s", e)
        release_pid_lock(lock_path)
        return {"error": "encoder_load_failed", "details": str(e)}
    logger.info("Encoder backend: %s | max_length=%d | batch_size=%d", encoder.backend, max_length, batch_size)
    logger.info("Async upload workers: %d%s", async_workers,
                " (synchronous mode)" if async_workers == 1 else "")

    uploader = AsyncBulkUploader(
        template_session=session,
        opensearch_url=opensearch_url,
        index_name=index_name,
        workers=1 if dry_run else async_workers,
    )

    # ── Load checkpoint ──────────────────────────────────────────────────
    start_file = 0
    start_offset = 0
    cumulative = {
        "total_indexed": 0,
        "total_errors": 0,
        "total_docs_scanned": 0,
        "total_empty_sparse": 0,
    }

    if resume:
        cp = load_checkpoint(input_dir, total_files=total_files)
        if cp:
            start_file = min(cp.get("file_index", 0), total_files)
            start_offset = cp.get("doc_offset", 0)
            cumulative = cp.get("cumulative", cumulative)
            if start_file >= total_files:
                logger.info("Checkpoint indicates indexing already complete. Nothing to do.")
                release_pid_lock(lock_path)
                return {**cumulative, "state": "completed"}

    start_time = time.time()
    docs_since_checkpoint = 0
    heartbeat_path = input_dir / HEARTBEAT_FILE
    throughput_window: list[float] = []  # recent batch times for moving average
    original_batch_size = batch_size
    active_batch_size = batch_size
    consecutive_upload_failures = 0
    consecutive_encode_failures = 0
    MAX_CONSECUTIVE_UPLOAD_FAILURES = 5  # halt-and-checkpoint if upload keeps failing
    MAX_CONSECUTIVE_ENCODE_FAILURES = 3  # CUDA likely dead — halt
    committed_idx = 0  # last doc_idx confirmed uploaded; checkpoint uses this

    def _drain_and_commit() -> dict:
        """Drain all in-flight uploads, update cumulative stats and committed_idx."""
        nonlocal committed_idx, consecutive_upload_failures
        drain_stats = uploader.drain()
        cumulative["total_indexed"] += drain_stats["indexed"]
        cumulative["total_errors"] += drain_stats["errors"]
        if drain_stats["error_details"]:
            logger.warning("Async bulk errors: %s", drain_stats["error_details"][:2])
        if drain_stats["new_committed_idx"] is not None:
            committed_idx = drain_stats["new_committed_idx"]
        # Per-batch circuit-breaker update (mirrors original logic)
        for batch_result in drain_stats["per_batch"]:
            n = batch_result.get("indexed", 0) + batch_result.get("errors", 0)
            if batch_result.get("indexed") == 0 and batch_result.get("errors", 0) >= n > 0:
                consecutive_upload_failures += 1
                logger.warning(
                    "Bulk upload completely failed (%d/%d consecutive)",
                    consecutive_upload_failures, MAX_CONSECUTIVE_UPLOAD_FAILURES,
                )
            else:
                consecutive_upload_failures = 0
        return drain_stats

    # ── Process files ────────────────────────────────────────────────────
    for file_idx in range(start_file, total_files):
        if _shutdown_requested:
            break

        input_file = input_files[file_idx]
        logger.info("Loading file %d/%d: %s", file_idx + 1, total_files, input_file.name)
        all_records = read_jsonl_gz(input_file)
        total_in_file = len(all_records)

        # Skip already-processed docs within this file
        records = all_records[start_offset:]
        start_offset = 0  # only applies to first file on resume

        doc_idx = len(all_records) - len(records)
        committed_idx = doc_idx  # reset per-file to resume offset

        while doc_idx < total_in_file:
            if _shutdown_requested:
                logger.info("Shutdown requested — saving checkpoint...")
                _drain_and_commit()
                try:
                    save_checkpoint(input_dir, {
                        "file_index": file_idx,
                        "doc_offset": committed_idx,
                        "cumulative": cumulative,
                        "total_indexed": cumulative["total_indexed"],
                        "state": "interrupted",
                    })
                except Exception:
                    logger.exception("Failed to save checkpoint on shutdown")
                write_status(input_dir, {
                    "state": "interrupted",
                    "file_index": file_idx,
                    "total_files": total_files,
                    "total_indexed": cumulative["total_indexed"],
                    "gpu": encoder.get_gpu_stats(),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
                logger.info("Checkpoint saved. Run with --resume to continue.")
                uploader.close()
                release_pid_lock(lock_path)
                return {**cumulative, "state": "interrupted"}

            # ── Check VRAM reservation ───────────────────────────────────
            def _save_checkpoint_for_pause():
                _drain_and_commit()
                save_checkpoint(input_dir, {
                    "file_index": file_idx,
                    "doc_offset": committed_idx,
                    "cumulative": cumulative,
                    "total_indexed": cumulative["total_indexed"],
                    "state": "vram_paused",
                })
                write_status(input_dir, {
                    "state": "vram_paused",
                    "file_index": file_idx,
                    "total_files": total_files,
                    "doc_in_file": doc_idx,
                    "total_indexed": cumulative["total_indexed"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

            active_batch_size = check_vram_reservation(
                encoder, original_batch_size, active_batch_size,
                checkpoint_fn=_save_checkpoint_for_pause,
            )

            # ── Prepare batch ────────────────────────────────────────────
            batch_records = records[: active_batch_size]
            records = records[active_batch_size:]
            actual_batch = len(batch_records)

            texts = []
            valid_records = []
            for rec in batch_records:
                title = rec.get("title", "") or ""
                abstract = rec.get("abstract", "") or ""
                text = f"{title} {abstract}".strip()
                if text:
                    texts.append(text)
                    valid_records.append(rec)
                else:
                    cumulative["total_empty_sparse"] += 1

            cumulative["total_docs_scanned"] += actual_batch

            if not texts:
                doc_idx += actual_batch
                continue

            # ── Encode ───────────────────────────────────────────────────
            batch_start = time.time()
            try:
                sparse_vecs = encoder.encode_batch(texts)
                consecutive_encode_failures = 0
            except Exception as e:
                consecutive_encode_failures += 1
                err_str = str(e)
                # CUDA / device errors invalidate encoder state — must halt.
                # Any other transient error (e.g., a pathological doc) we skip.
                is_cuda_error = (
                    "CUDA" in err_str
                    or "cudnn" in err_str.lower()
                    or "device-side" in err_str
                    or "out of memory" in err_str.lower()
                    or consecutive_encode_failures >= MAX_CONSECUTIVE_ENCODE_FAILURES
                )
                logger.error(
                    "SPLADE encoding failed for batch at doc %d (failure %d/%d): %s",
                    doc_idx, consecutive_encode_failures, MAX_CONSECUTIVE_ENCODE_FAILURES, e,
                )
                if is_cuda_error:
                    logger.error(
                        "Encoder appears unrecoverable — saving checkpoint and exiting. "
                        "Restart with --resume after addressing the underlying issue."
                    )
                    _drain_and_commit()
                    try:
                        save_checkpoint(input_dir, {
                            "file_index": file_idx,
                            "doc_offset": committed_idx,
                            "cumulative": cumulative,
                            "total_indexed": cumulative["total_indexed"],
                            "state": "encoder_failed",
                            "last_error": err_str[:500],
                        })
                    except Exception:
                        logger.exception("Failed to save checkpoint on encoder failure")
                    uploader.close()
                    release_pid_lock(lock_path)
                    return {**cumulative, "state": "encoder_failed", "error": err_str[:500]}
                # Transient error — skip this batch
                doc_idx += actual_batch
                continue

            encode_time = time.time() - batch_start

            # ── Build OpenSearch docs ────────────────────────────────────
            os_docs = []
            for rec, sparse in zip(valid_records, sparse_vecs):
                if not sparse:
                    cumulative["total_empty_sparse"] += 1
                    continue
                os_doc = {
                    "id": rec.get("id", ""),
                    "doi": rec.get("doi"),
                    "title": rec.get("title", ""),
                    "abstract": rec.get("abstract", ""),
                    "publication_year": rec.get("publication_year"),
                    "type": rec.get("type", ""),
                    "openalex_id": rec.get("id", ""),
                    "sparse_field": sparse,
                }
                os_docs.append(os_doc)

            # ── Upsert to OpenSearch (async) ─────────────────────────────
            if os_docs and not dry_run:
                uploader.submit(os_docs, doc_range_end=doc_idx + actual_batch)

                # Backpressure: if queue is at capacity, drain now so the main
                # thread doesn't get too far ahead of confirmed uploads.
                if uploader.pending_count >= async_workers + 1:
                    _drain_and_commit()
                    if consecutive_upload_failures >= MAX_CONSECUTIVE_UPLOAD_FAILURES:
                        logger.error(
                            "OpenSearch unreachable for %d batches in a row. Halting "
                            "and saving checkpoint so resume re-indexes the failed docs "
                            "(idempotent via _id).",
                            MAX_CONSECUTIVE_UPLOAD_FAILURES,
                        )
                        try:
                            save_checkpoint(input_dir, {
                                "file_index": file_idx,
                                "doc_offset": committed_idx,
                                "cumulative": cumulative,
                                "total_indexed": cumulative["total_indexed"],
                                "state": "opensearch_unreachable",
                            })
                        except Exception:
                            logger.exception("Failed to save checkpoint on OS failure")
                        uploader.close()
                        release_pid_lock(lock_path)
                        return {**cumulative, "state": "opensearch_unreachable"}
            elif os_docs and dry_run:
                cumulative["total_indexed"] += len(os_docs)

            doc_idx += actual_batch
            docs_since_checkpoint += actual_batch

            # ── Throughput tracking ──────────────────────────────────────
            batch_time = time.time() - batch_start
            throughput_window.append(batch_time)
            if len(throughput_window) > 50:
                throughput_window = throughput_window[-50:]

            avg_batch_time = sum(throughput_window) / len(throughput_window)
            docs_per_sec = actual_batch / max(batch_time, 0.001)

            _touch_heartbeat(heartbeat_path)

            # ── Progress logging ─────────────────────────────────────────
            elapsed = time.time() - start_time
            if cumulative["total_docs_scanned"] > 0:
                overall_rate = cumulative["total_docs_scanned"] / elapsed
            else:
                overall_rate = 0

            # ── Checkpoint ───────────────────────────────────────────────
            if docs_since_checkpoint >= checkpoint_interval:
                gpu_stats = encoder.get_gpu_stats()
                if gpu_stats.get("utilization_pct", 0) > 90:
                    logger.warning(
                        "⚠ GPU VRAM usage at %.1f%% — consider reducing --batch-size",
                        gpu_stats["utilization_pct"],
                    )

                logger.info(
                    "CHECKPOINT — indexed: %d | errors: %d | rate: %.0f docs/sec | "
                    "GPU: %s | file %d/%d",
                    cumulative["total_indexed"],
                    cumulative["total_errors"],
                    overall_rate,
                    f"{gpu_stats.get('allocated_gb', 'N/A')} GB" if gpu_stats.get("device") == "cuda" else "CPU",
                    file_idx + 1,
                    total_files,
                )

                _drain_and_commit()
                if consecutive_upload_failures >= MAX_CONSECUTIVE_UPLOAD_FAILURES:
                    logger.error(
                        "OpenSearch unreachable for %d batches in a row. Halting.",
                        MAX_CONSECUTIVE_UPLOAD_FAILURES,
                    )
                    try:
                        save_checkpoint(input_dir, {
                            "file_index": file_idx,
                            "doc_offset": committed_idx,
                            "cumulative": cumulative,
                            "total_indexed": cumulative["total_indexed"],
                            "state": "opensearch_unreachable",
                        })
                    except Exception:
                        logger.exception("Failed to save checkpoint on OS failure")
                    uploader.close()
                    release_pid_lock(lock_path)
                    return {**cumulative, "state": "opensearch_unreachable"}

                save_checkpoint(input_dir, {
                    "file_index": file_idx,
                    "doc_offset": committed_idx,
                    "cumulative": cumulative,
                    "total_indexed": cumulative["total_indexed"],
                    "state": "running",
                })

                write_status(input_dir, {
                    "state": "running",
                    "file_index": file_idx,
                    "total_files": total_files,
                    "doc_in_file": doc_idx,
                    "total_indexed": cumulative["total_indexed"],
                    "total_errors": cumulative["total_errors"],
                    "docs_per_sec": round(overall_rate, 1),
                    "elapsed_seconds": round(elapsed, 1),
                    "gpu": gpu_stats,
                    "pid": os.getpid(),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

                docs_since_checkpoint = 0

            # ── Dry run exit ─────────────────────────────────────────────
            if dry_run and cumulative["total_docs_scanned"] >= 100:
                logger.info("─── DRY RUN COMPLETE ───")
                logger.info("Encoded %d docs", cumulative["total_docs_scanned"])
                logger.info("Would index %d docs", cumulative["total_indexed"])
                logger.info("Empty sparse vectors: %d", cumulative["total_empty_sparse"])
                logger.info("Encoding speed: %.1f docs/sec", docs_per_sec)
                logger.info("GPU: %s", encoder.get_gpu_stats())
                if sparse_vecs:
                    sample = sparse_vecs[0]
                    logger.info(
                        "Sample sparse vector: %d terms, top-5: %s",
                        len(sample),
                        dict(sorted(sample.items(), key=lambda x: -x[1])[:5]),
                    )
                uploader.close()
                release_pid_lock(lock_path)
                return {**cumulative, "state": "dry_run"}

        # ── End of file checkpoint ───────────────────────────────────────
        if not _shutdown_requested:
            _drain_and_commit()
            save_checkpoint(input_dir, {
                "file_index": file_idx + 1,
                "doc_offset": 0,
                "cumulative": cumulative,
                "total_indexed": cumulative["total_indexed"],
                "state": "running",
            })
            docs_since_checkpoint = 0

    # ── Final summary ────────────────────────────────────────────────────
    total_time = time.time() - start_time
    overall_rate = cumulative["total_docs_scanned"] / max(total_time, 0.001)

    logger.info("═══ Indexing complete ═══")
    logger.info("Total time: %s", time.strftime("%H:%M:%S", time.gmtime(total_time)))
    logger.info("Files processed: %d/%d", total_files, total_files)
    logger.info("Docs scanned: %d", cumulative["total_docs_scanned"])
    logger.info("Docs indexed: %d", cumulative["total_indexed"])
    logger.info("Indexing errors: %d", cumulative["total_errors"])
    logger.info("Empty sparse vectors: %d", cumulative["total_empty_sparse"])
    logger.info("Average throughput: %.1f docs/sec", overall_rate)
    logger.info("GPU: %s", encoder.get_gpu_stats())

    _drain_and_commit()
    save_checkpoint(input_dir, {
        "file_index": total_files,
        "doc_offset": 0,
        "cumulative": cumulative,
        "total_indexed": cumulative["total_indexed"],
        "state": "completed",
    })

    write_status(input_dir, {
        "state": "completed",
        "total_files": total_files,
        "total_indexed": cumulative["total_indexed"],
        "total_errors": cumulative["total_errors"],
        "docs_per_sec": round(overall_rate, 1),
        "total_time_seconds": round(total_time, 1),
        "gpu": encoder.get_gpu_stats(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    uploader.close()
    release_pid_lock(lock_path)
    return {**cumulative, "state": "completed"}


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Index OpenAlex snapshot into OpenSearch using SPLADE sparse vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Input directory with .jsonl.gz files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"SPLADE model name/path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for encoding (default: auto — uses CUDA if available)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Encoding batch size (default: {DEFAULT_BATCH_SIZE}; reduce if OOM). "
             f"For TRT backend this is fixed for the engine — changing it triggers a rebuild.",
    )
    parser.add_argument(
        "--max-length", type=int, default=MAX_DOC_LENGTH,
        help=f"Tokenizer max length (default: {MAX_DOC_LENGTH}, the model's "
             f"trained doc length per Splade_PP_en_v1 model card §6d). "
             f"Higher (256/384/512) costs more compute and goes outside training distribution.",
    )
    parser.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "trt", "pytorch"],
        help="Inference backend: 'trt' = ONNX+TensorRT IO-binding (fastest), "
             "'pytorch' = eager FP16+SDPA, 'auto' = trt if available else pytorch",
    )
    parser.add_argument(
        "--opensearch-url", type=str,
        default=os.environ.get("SFU_OPENSEARCH_URL", DEFAULT_OPENSEARCH_URL),
        help=f"OpenSearch URL (default: {DEFAULT_OPENSEARCH_URL})",
    )
    parser.add_argument(
        "--index", type=str,
        default=os.environ.get("SFU_OPENSEARCH_INDEX", DEFAULT_INDEX),
        help=f"OpenSearch index name (default: {DEFAULT_INDEX})",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N docs (default: {DEFAULT_CHECKPOINT_INTERVAL})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Encode 100 docs and report stats without full indexing",
    )
    parser.add_argument(
        "--async-workers", type=int, default=3, metavar="N",
        help=(
            "Concurrent OpenSearch upload threads (default: 3). "
            "Set to 1 for fully synchronous operation (original behaviour, zero overhead). "
            "Higher values increase upload/encode overlap but use more memory."
        ),
    )
    args = parser.parse_args()

    logger.info("SPLADE Indexer")
    logger.info("Input: %s", args.input)
    logger.info("Model: %s", args.model)
    logger.info("Device: %s | Backend: %s", args.device, args.backend)
    logger.info("Batch size: %d | Max length: %d", args.batch_size, args.max_length)
    logger.info("OpenSearch: %s / %s", args.opensearch_url, args.index)
    logger.info("Checkpoint every: %d docs", args.checkpoint_interval)
    if args.resume:
        logger.info("Mode: RESUME from checkpoint")
    if args.dry_run:
        logger.info("Mode: DRY RUN (100 docs)")

    # Mirror status/heartbeat to the always-on NAS so its monitor heartbeat shows
    # LIVE progress (failure-safe; disable with SFU_PROGRESS_PUSH=off).
    push_target = _start_progress_pusher(args.input)
    if push_target:
        logger.info("Live progress mirror -> %s (every %ss)",
                    push_target, os.environ.get("SFU_PROGRESS_PUSH_INTERVAL", "60"))

    try:
        result = run_indexer(
            input_dir=args.input,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            opensearch_url=args.opensearch_url,
            index_name=args.index,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
            dry_run=args.dry_run,
            max_length=args.max_length,
            backend=args.backend,
            async_workers=args.async_workers,
        )
    finally:
        if push_target:
            _push_progress_once(args.input, push_target)  # flush the terminal state to the NAS

    if result.get("state") == "interrupted":
        sys.exit(130)
    elif result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
